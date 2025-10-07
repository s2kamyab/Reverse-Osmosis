# ===== Ensemble MISO "J(x)" minimization using your data pipeline =====
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.termination import get_termination
import copy, random

# -----------------------------
# 0) Utilities (reuse your seeds)
# -----------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# 1) Data loading (your function)
# -----------------------------
def load_and_prepare_data(file1, file2):
    df1 = pd.DataFrame()  # pd.read_excel(file1, header=0)  # left blank in your snippet
    df2 = pd.read_excel(file2, header=0)
    df = pd.concat([df1, df2], ignore_index=True)
    df['mem_encoded'] = LabelEncoder().fit_transform(df['Membrane '])
    df = df.drop(columns=['Membrane '])
    X = df.iloc[:, list(range(0, 4)) + [-1]].values   # 5 inputs
    y = df.iloc[:, 4:6].values                        # [Energy, Recovery]
    y[:, -1] = 1 - y[:, -1]                           # now y[:,1] = (1 - recovery)
    return X, y                                       # X: (N,5) ; y: (N,2) with columns [Energy, (1-Recovery)]

# -----------------------------
# 2) Split & scale for J
# -----------------------------
def normalize_split_data_for_J(X, energy_and_one_minus_recovery, recovery_floor=1e-6):
    energy = energy_and_one_minus_recovery[:, 0]
    one_minus_recovery = np.clip(energy_and_one_minus_recovery[:, 1], recovery_floor, None)
    J = energy / one_minus_recovery

    X_train, X_val, y_train, y_val = train_test_split(X, J, test_size=0.2, random_state=42)

    x_scaler = StandardScaler().fit(X_train)
    X_train_s = x_scaler.transform(X_train)
    X_val_s   = x_scaler.transform(X_val)

    y_scaler = StandardScaler().fit(y_train.reshape(-1,1))
    y_train_s = y_scaler.transform(y_train.reshape(-1,1)).ravel()
    y_val_s   = y_scaler.transform(y_val.reshape(-1,1)).ravel()

    return X_train_s, X_val_s, y_train_s, y_val_s, x_scaler, y_scaler

# -----------------------------
# 3) Datasets & model
# -----------------------------
class SupervisedRegDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class MLPRegressor(nn.Module):
    def __init__(self, in_dim=5, hidden=(8, 16, 8), dropout=0.10):
        super().__init__()
        h = list(hidden)
        layers = []
        dims = [in_dim] + h + [1]
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
            if dropout > 0 and i < len(dims)-3:
                layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

# -----------------------------
# 4) Train one model (early stop)
# -----------------------------
def train_one_model(model, train_ds, val_ds, batch_size=32, max_epochs=300,
                    lr=3e-3, weight_decay=1e-4, patience=25, device='cpu'):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val = float('inf')
    wait = 0

    for epoch in range(1, max_epochs+1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        sched.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val

# -----------------------------
# 5) Ensemble (bootstrap bagging)
# -----------------------------
def _bootstrap_idx(n, m, rng):
    return rng.integers(0, n, size=m)  # with replacement

def train_ensemble_J(
    X_train_s, y_train_s, X_val_s, y_val_s,
    n_models=7, bootstrap_frac=0.9, seed=123, device='cpu'
):
    rng = np.random.default_rng(seed)
    train_ds_full = SupervisedRegDataset(X_train_s, y_train_s)
    val_ds        = SupervisedRegDataset(X_val_s,   y_val_s)
    ensemble = []
    m = int(round(bootstrap_frac * len(train_ds_full)))
    for i in range(n_models):
        idx = _bootstrap_idx(len(train_ds_full), m, rng)
        sub_ds = Subset(train_ds_full, idx.tolist())
        model = MLPRegressor(in_dim=5, hidden=(32,64,32), dropout=0.10)
        model, val_mse = train_one_model(model, sub_ds, val_ds, device=device)
        print(f"[J-ensemble] model {i+1}/{n_models}, best val MSE (scaled): {val_mse:.4f}")
        ensemble.append(model.to('cpu').eval())
    return ensemble

# -----------------------------
# 6) Ensemble inference helpers
# -----------------------------
@torch.no_grad()
def ensemble_predict_J(ensemble, X, x_scaler, y_scaler, discrete_last_var=True):
    """
    X: (N,5) in original units (same as input to x_scaler.fit)
    Returns: (mean_J, std_J) in original J units
    """
    X = np.array(X, dtype=np.float32).copy()
    if discrete_last_var:
        X[:, -1] = np.round(X[:, -1])  # membrane -> {0,1}

    Xs = x_scaler.transform(X)
    xt = torch.tensor(Xs, dtype=torch.float32)

    preds_s = []
    for m in ensemble:
        ps = m(xt).cpu().numpy()
        preds_s.append(ps)
    preds_s = np.stack(preds_s, axis=0)     # (E, N)
    preds   = y_scaler.inverse_transform(preds_s.T)  # (N, E)
    mean = np.mean(preds, axis=1)
    std  = np.std(preds,  axis=1)
    return mean, std

# -----------------------------
# 7) pymoo Problem to minimize J
# -----------------------------
class JEnsembleProblem(ElementwiseProblem):
    def __init__(self, ensemble, x_scaler, y_scaler, xl, xu, discrete_last_var=True):
        super().__init__(n_var=5, n_obj=1, n_constr=0, xl=xl, xu=xu)
        self.ensemble = ensemble
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.discrete_last_var = discrete_last_var

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.array(x, dtype=np.float32).reshape(1, -1)
        mean, _ = ensemble_predict_J(self.ensemble, x, self.x_scaler, self.y_scaler,
                                     discrete_last_var=self.discrete_last_var)
        out["F"] = np.array([mean[0]], dtype=np.float64)

# -----------------------------
# 8) Driver that uses YOUR data
# -----------------------------
if __name__ == "__main__":
    set_seed(42)

    # --- Load your data exactly as you already do ---
    X, y_E_and_1mR = load_and_prepare_data('AG.xlsx', 'AK.xlsx')  # X: (N,5); y: [Energy, (1-Recovery)]

    # --- Build J and make a split/scalers for J ---
    X_train_s, X_val_s, y_train_s, y_val_s, x_scaler_J, y_scaler_J = normalize_split_data_for_J(
        X, y_E_and_1mR, recovery_floor=1e-6
    )

    # --- Train ensemble on J ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ensemble_J = train_ensemble_J(
        X_train_s, y_train_s, X_val_s, y_val_s,
        n_models=7, bootstrap_frac=0.9, seed=123, device=device
    )

    # --- Define bounds in the SAME INPUT SPACE as the optimizer sees ---
    # We'll optimize in ORIGINAL UNITS (not scaled).
    X_train_orig = x_scaler_J.inverse_transform(X_train_s)
    xmins = X_train_orig.min(axis=0)
    xmaxs = X_train_orig.max(axis=0)
    # Force membrane bounds to [0,1] exactly:
    xl = np.array([xmins[0], xmins[1], xmins[2], xmins[3], 0.0], dtype=np.float32)
    xu = np.array([xmaxs[0], xmaxs[1], xmaxs[2], xmaxs[3], 1.0], dtype=np.float32)

    problem = JEnsembleProblem(
        ensemble=ensemble_J,
        x_scaler=x_scaler_J,
        y_scaler=y_scaler_J,
        xl=xl, xu=xu,
        discrete_last_var=True
    )

    # --- Optimize J with a GA (robust for black-box + rounding) ---
    algorithm = GA(pop_size=120, eliminate_duplicates=True)
    termination = get_termination("n_evals", 5000)

    res = minimize(problem, algorithm, termination, seed=1, verbose=True)

    # --- Report solution in ORIGINAL UNITS ---
    x_star = res.X if res.X.ndim == 1 else res.X[0]
    mean_J, std_J = ensemble_predict_J(ensemble_J, x_star.reshape(1,-1), x_scaler_J, y_scaler_J, discrete_last_var=True)

    best_df = pd.DataFrame([{
        'feed_flow':   x_star[0],
        'temperature': x_star[1],
        'salinity':    x_star[2],
        'pressure':    x_star[3],
        'membrane':    int(np.round(x_star[4])),
        'J_mean':      float(mean_J[0]),
        'J_std':       float(std_J[0])
    }])

    print("\n=== Best J Solution (ensemble) ===")
    print(best_df)
    best_df.to_csv("best_single_objective_J_ensemble.csv", index=False)
