# modular_nsga2_analysis.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap, BoundaryNorm
import random
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.problem import ElementwiseProblem
# from pymoo.algorithms.soo.nonconvex.cmaes import CMAES  # optional alternative
from pymoo.termination import get_termination
# ================================
# Data Loading and Preprocessing
# ================================
def load_and_prepare_data(file1, file2):
    df1 = pd.DataFrame()#pd.read_excel(file1, header=0)
    df2 = pd.read_excel(file2, header=0)
    df = pd.concat([df1, df2], ignore_index=True)
    df['mem_encoded'] = LabelEncoder().fit_transform(df['Membrane '])
    df = df.drop(columns=['Membrane '])
    X = df.iloc[:, list(range(0, 4)) + [-1]].values
    y = df.iloc[:, 4:6].values
    y[:, -1] = 1 - y[:, -1]  # 1 - recovery
    return X, y


def normalize_split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train_norm = scaler_x.fit_transform(X_train)
    X_test_norm = scaler_x.transform(X_test)
    y_train_norm = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_norm = scaler_y.transform(y_test.reshape(-1, 1))
    return X_train_norm, X_test_norm, y_train_norm, y_test_norm, scaler_x, scaler_y


def set_seed(seed=42):
    torch.manual_seed(seed)                # Set PyTorch seed
    np.random.seed(seed)                   # Set NumPy seed
    random.seed(seed)                      # Set built-in Python seed
    torch.cuda.manual_seed(seed)           # If using CUDA
    torch.cuda.manual_seed_all(seed)       # For multi-GPU (if applicable)

    # For reproducible GPU results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ================================
# Model Definition
# ================================
class NeuralNet_REC(nn.Module):
    def __init__(self):

        super(NeuralNet_REC, self).__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(5, 6),
        #     nn.ReLU(),
        #     nn.Linear(6, 8),
        #     nn.ReLU(),
        #     nn.Linear(8, 2)
        # )
        self.net = nn.Sequential(
        nn.Linear(5, 32),
        # nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.1),

        nn.Linear(32, 64),
        # nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.1),

        nn.Linear(64, 32),
        nn.ReLU(),

        nn.Linear(32, 1)
    )

    def forward(self, x):
        return self.net(x)
    

class NeuralNet_EC(nn.Module):
    def __init__(self):

        super(NeuralNet_EC, self).__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(5, 6),
        #     nn.ReLU(),
        #     nn.Linear(6, 8),
        #     nn.ReLU(),
        #     nn.Linear(8, 2)
        # )
        self.net = nn.Sequential(
        nn.Linear(5, 32),
        # nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.1),

        nn.Linear(32, 64),
        # nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.1),

        nn.Linear(64, 32),
        nn.ReLU(),

        nn.Linear(32, 1)
    )

    def forward(self, x):
        return self.net(x)


# ================================
# Training Routine
# ================================
def train_model(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, num_epochs=500):
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=16, shuffle=True)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,  weight_decay=1e-5)

    test_mse_list = []
    train_mse_list = []

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor)
            test_loss = criterion(y_pred, y_test_tensor)
            test_mse_list.append(test_loss.item())
            train_loss = criterion(model(X_train_tensor), y_train_tensor)
            train_mse_list.append(train_loss.item())

    return model, test_mse_list, train_mse_list


# ================================
# Single-objective Optimization
# ================================

# ---------- Utility: handle optional scalers ----------
def maybe_inverse(scaler, y):
    if scaler is None:
        return y
    # y can be shape (N,1) or (N,)
    y2 = y.reshape( -1 , 1)
    return scaler.inverse_transform(y2).ravel()

# ---------- Problem definition ----------
class TwoModelRatioProblem(ElementwiseProblem):
    """
    Minimize J(x) = Energy(x) / Recovery(x)
    - Two independent models with 5 inputs each.
    - Optional output scalers for each model.
    - Last decision var can be a categorical 'membrane' in {0,1} (rounded).
    - Adds a small penalty to avoid division by tiny Recovery.
    """
    def __init__(
        self,
        model_energy,
        model_recovery,
        x_lows, x_highs,
        scaler_energy=None,
        scaler_recovery=None,
        discrete_last_var=True,
        recovery_floor=1e-6,      # prevents div-by-zero & crazy ratios
        penalty_small_recovery=0.0 # set >0 to softly discourage tiny recovery
    ):
        super().__init__(n_var=5, n_obj=1, n_constr=0, xl=x_lows, xu=x_highs)
        self.model_energy = model_energy.eval()
        self.model_recovery = model_recovery.eval()
        self.scaler_energy = scaler_energy
        self.scaler_recovery = scaler_recovery
        self.discrete_last_var = discrete_last_var
        self.recovery_floor = recovery_floor
        self.penalty_small_recovery = penalty_small_recovery

        # No grads during evaluation
        for p in self.model_energy.parameters():  p.requires_grad_(False)
        for p in self.model_recovery.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def _predict_models(self, x_np):
        # x_np: shape (5,)
        # Round the last var if it's categorical (e.g., membrane)
        if self.discrete_last_var:
            x_np = x_np.copy()
            x_np[-1] = np.round(x_np[-1])

        x_t = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)  # (1,5)

        e_hat = self.model_energy(x_t).cpu().numpy().ravel()  # (1,) or (1,k) -> ravel
        r_hat = self.model_recovery(x_t).cpu().numpy().ravel()

        # If models output normalized values, invert them
        e = maybe_inverse(self.scaler_energy, e_hat)[0]
        r = maybe_inverse(self.scaler_recovery, r_hat)[0]
        return float(e), float(r), x_np

    def _evaluate(self, x, out, *args, **kwargs):
        energy, recovery, x_used = self._predict_models(np.array(x, dtype=np.float32))
        # Guard against tiny/negative recovery (if model extrapolates)
        rec = max(recovery, self.recovery_floor)

        J = energy / (1-rec)  # since recovery model predicts 1 - recovery

        # Optional soft penalty if recovery is very small (helps steer search)
        if self.penalty_small_recovery > 0.0 and recovery < 0.05:  # tune threshold to your scale
            J += self.penalty_small_recovery * (0.05 - recovery)

        out["F"] = np.array([J], dtype=np.float64)
# ---------- Driver ----------
def run_single_objective_ratio_optimization(
    model_energy,
    model_recovery,
    X_train_norm,                   # used to define bounds
    scaler_x=None,                  # optional input scaler (if you want to work in original space)
    scaler_energy=None,
    scaler_recovery=None,
    pop_size=120,
    n_gen=250,
    discrete_last_var=True
):
    """
    Minimizes Energy/Recovery over 5-D decision vector.
    By default we optimize in the (already normalized) space defined by X_train_norm.
    If you prefer original units: pass scaler_x and set bounds in original space instead.
    """
    # Define bounds from normalized training cloud (same style as your NSGA-2)
    x_mins = np.min(X_train_norm, axis=0)
    x_maxs = np.max(X_train_norm, axis=0)
    # Example: keep first 4 features continuous, and clamp last "membrane" to [0,1]
    xl = np.array([x_mins[0], x_mins[1], x_mins[2], x_mins[3], 0.0], dtype=np.float32)
    xu = np.array([x_maxs[0], x_maxs[1], x_maxs[2], x_maxs[3], 1.0], dtype=np.float32)

    problem = TwoModelRatioProblem(
        model_energy=model_energy,
        model_recovery=model_recovery,
        x_lows=xl,
        x_highs=xu,
        scaler_energy=scaler_energy,
        scaler_recovery=scaler_recovery,
        discrete_last_var=discrete_last_var,
        recovery_floor=1e-6,
        penalty_small_recovery=0.0,
    )

    # Strong single-objective global optimizer (robust to non-smoothness from rounding)
    algorithm = DE(pop_size=pop_size, CR=0.9, F=0.8)
    # For smooth continuous variables you can try CMA-ES (often very powerful):
    # algorithm = CMAES(x0=(xl + xu)/2.0, sigma=0.2*(xu - xl), verbose=True)

    res = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", n_gen),
        seed=42,
        verbose=True
    )

    # Prepare a neat result record (optionally map membrane to labels)
    X_opt = res.X if res.X.ndim == 1 else res.X[0]
    # Recompute outputs for the winner
    with torch.no_grad():
        e_opt, r_opt, X_used = problem._predict_models(np.array(X_opt, dtype=np.float32))
    J_opt = e_opt / max((1-r_opt), 1e-6) # model predicts 1 - recovery
    X_used = np.squeeze(scaler_x_REC.inverse_transform(X_used.reshape(1,-1)))
    df = pd.DataFrame([{
        'feed_flow':   X_used[0],
        'temperature': X_used[1],
        'salinity':    X_used[2],
        'pressure':    X_used[3],
        'membrane':    int(np.round(X_used[4])) if discrete_last_var else X_used[4],
        'Energy':      e_opt,
        'Recovery':    r_opt,
        'Energy/Recovery': J_opt
    }])

    df.to_csv("best_single_objective_solution.csv", index=False)
    return df, res
# ================================
# Visualization
# ================================
def plot_solution_space(x, y, Z, lbl):
    # Step 1: Create a grid over the region
    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), 200),
        np.linspace(y.min(), y.max(), 200)
    )

    XY = np.column_stack((x, y))
    # Step 2: Interpolate Z values onto the grid
    grid_z = griddata(XY, Z, (grid_x, grid_y), method='linear')
    # Step 3: Plot filled contours
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=50, cmap='jet')#'viridis')
    cbar = plt.colorbar(contour)
    cbar.set_label(lbl, fontsize=14)  # ⬅️ set the label font size
    plt.xlabel('Energy Consumption (Watt)',fontsize=14)
    plt.ylabel('Recovery (%)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    # plt.axis('equal')
    # plt.ylim([0, 70])
    plt.grid(True)

def plot_pareto(x, y,z, res):
    # Step 1: Create a grid over the region
    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), 200),
        np.linspace(y.min(), y.max(), 200)
    )

    XY = np.column_stack((x, y))
    # Step 2: Interpolate Z values onto the grid
    grid_z = griddata(XY, z, (grid_x, grid_y), method='linear')
    # Step 3: Plot filled contours
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(grid_x, grid_y, grid_z,levels=1,alpha=0.3, cmap='tab10',label='Solution Space')#'viridis')
    # cbar = plt.colorbar(contour)
    # cbar.set_label(lbl, fontsize=14)  # ⬅️ set the label font size
    plt.scatter(res['Energy'], (1 - res['Recovery'])*100, c='blue', alpha=0.2,s=100, label='Pareto Front Optimal Set')
    plt.xlabel('Energy Consumption (Watt)',fontsize=14)
    plt.ylabel('Recovery (%)', fontsize=14)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True)
    # plt.axis('equal')
    # plt.ylim([0, 70])
    plt.grid(True)


    
def plot_solution_space_discerete(x, y, Z, lbl):
    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), 200),
        np.linspace(y.min(), y.max(), 200)
    )
    
    XY = np.column_stack((x, y))
    # Step 2: Interpolate Z values onto the grid
    grid_z = griddata(XY, Z, (grid_x, grid_y))

    # Define two levels: e.g., Z < 0.5 and Z >= 0.5

    levels = [0,0.5, 1]  # two regions: below and above 0.5

    # Create a 2-color colormap
    colors = ['blue', 'orange']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=len(colors))

    # Plot
    fig, ax = plt.subplots()
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap, norm=norm, method='cubic')

    # Colorbar with two ticks
    cbar = plt.colorbar(contour, ticks=[0.25, 0.75])  # midpoints of bins
    cbar.ax.set_yticklabels(['AG', 'AK'])
    cbar.set_label(lbl, fontsize=14)  # ⬅️ set the label font size
    plt.xlabel('Energy Consumption (Watt)',fontsize=14)
    plt.ylabel('Recovery (%)', fontsize=14)
    # plt.ylim([0, 70])
    plt.grid(True)
    # plt.show()
# ================================
# RSM Modeling
# ================================
def RSM_AG_EC(X1, X2, X3, X4):
    EC = (
        -193.561
        - 8.487 * X1
        + 10.673 * X2
        + 0.012 * X3
        - 0.599 * X4
        - 0.172 * X1 * X2
        + 0.001 * X1 * X3
        + 0.122 * X1 * X4
        - 0.0003 * X2 * X3
        + 0.005 * X2 * X4
        + 1.295e-05 * X3 * X4
        + 1.935 * (X1 ** 2)
        - 0.125 * (X2 ** 2)
        - 1.191e-06 * (X3 ** 2)
        + 0.005 * (X4 ** 2)
    )
    return EC

def RSM_AG_RE(X1, X2, X3, X4):
    Re = (
        121.679
        - 7.894 * X1
        - 4.908 * X2
        - 0.0222 * X3
        + 0.559 * X4
        + 0.011 * X1 * X2
        + 8.586e-04 * X1 * X3
        - 0.036 * X1 * X4
        - 3.664e-05 * X2 * X3
        + 2.618e-03 * X2 * X4
        - 1.369e-05 * X3 * X4
        + 0.586 * (X1 ** 2)
        + 0.071 * (X2 ** 2)
        + 2.688e-06 * (X3 ** 2)
        - 7.671e-04 * (X4 ** 2)
    )
    return Re
def RSM_AK_EC(X1, X2, X3, X4):
    EC = (
        -1904.670
        + 55.448 * X1
        + 99.315 * X2
        + 0.034 * X3
        + 0.139 * X4
        + 0.173 * X1 * X2
        - 0.006 * X1 * X3
        + 0.092 * X1 * X4
        - 0.0002 * X2 * X3
        + 0.009 * X2 * X4
        - 0.0001 * X3 * X4
        - 2.315 * (X1 ** 2)
        - 1.419 * (X2 ** 2)
        + 1.393e-06 * (X3 ** 2)
        + 0.003 * (X4 ** 2)
    )
    return EC

def RSM_AK_RE(X1, X2, X3, X4):
    Re = (
        -211.810
        - 22.533 * X1
        + 14.703 * X2
        - 4.222e-03 * X3
        + 0.677 * X4
        - 0.165 * X1 * X2
        + 1.627e-03 * X1 * X3
        - 0.108 * X1 * X4
        - 3.280e-04 * X2 * X3
        + 9.771e-03 * X2 * X4
        - 4.296e-05 * X3 * X4
        + 2.869 * (X1 ** 2)
        - 0.190 * (X2 ** 2)
        + 9.787e-07 * (X3 ** 2)
        + 3.941e-05 * (X4 ** 2)
    )
    return Re
def generate_random_df(N, col_ranges):
    """
    Generate a DataFrame with N rows and 5 columns,
    where each column's values are drawn from a specified (min, max) range.

    Parameters:
        N (int): Number of rows
        col_ranges (list of tuples): List of 5 (min, max) tuples for each column

    Returns:
        pd.DataFrame: Randomly generated DataFrame
    """
    assert len(col_ranges) == 5, "Must provide exactly 5 column ranges"

    data = {
        f'feature_{i+1}': np.random.uniform(low, high, N)
        for i, (low, high) in enumerate(col_ranges)
    }

    return pd.DataFrame(data)

# ================================
# Main
# ================================
if __name__ == '__main__':
    X, y = load_and_prepare_data('AG.xlsx', 'AK.xlsx')
    X_train_norm_EC, X_test_norm_EC, y_train_norm_EC, y_test_norm_EC, scaler_x_EC, scaler_y_EC = normalize_split_data(X, y[:,-2]) # energy consumption
    X_train_norm_REC, X_test_norm_REC, y_train_norm_REC, y_test_norm_REC, scaler_x_REC, scaler_y_REC = normalize_split_data(X, y[:,-1]) # recovery
    # Convert to tensors
    X_train_tensor_EC = torch.tensor(X_train_norm_EC, dtype=torch.float32)
    y_train_tensor_EC = torch.tensor(y_train_norm_EC, dtype=torch.float32)
    X_test_tensor_EC = torch.tensor(X_test_norm_EC, dtype=torch.float32)
    y_test_tensor_EC = torch.tensor(y_test_norm_EC, dtype=torch.float32)

    X_train_tensor_REC = torch.tensor(X_train_norm_REC, dtype=torch.float32)
    y_train_tensor_REC = torch.tensor(y_train_norm_REC, dtype=torch.float32)
    X_test_tensor_REC = torch.tensor(X_test_norm_REC, dtype=torch.float32)
    y_test_tensor_REC = torch.tensor(y_test_norm_REC, dtype=torch.float32)

    # Train model
    set_seed(42)
    model_EC = NeuralNet_EC()
    model_REC = NeuralNet_REC()
    model_EC, test_mse_list_EC, train_mse_list_EC = train_model(model_EC, X_train_tensor_EC, y_train_tensor_EC, X_test_tensor_EC, y_test_tensor_EC)
    model_REC, test_mse_list_REC, train_mse_list_REC = train_model(model_REC, X_train_tensor_REC, y_train_tensor_REC, X_test_tensor_REC, y_test_tensor_REC)
    # print(train_mse_list[-1], test_mse_list[-1])
    # # Save the model
    # torch.save(model.state_dict(), 'model.pth')
    # model_EC.load_state_dict(torch.load('model_EC.pth'))  # load weights
    # model_REC.load_state_dict(torch.load('model_REC.pth'))  # load weights
    # Plot MSE
    # plt.figure(figsize=(8, 6))
    # plt.plot(test_mse_list,linewidth=3, label='Test MSE')
    # plt.plot(train_mse_list,linewidth=3, label='Train MSE')
    # plt.legend(fontsize=14)
    # plt.title('MSE over Epochs')
    # plt.xlabel('epoch',fontsize=14)
    # plt.ylabel('MSE (loss)', fontsize=14)
    # plt.grid(True)
    # plt.savefig('learning_curve.png')
    # # plt.show()
    # Assume you already have:
    # model_energy: torch.nn.Module (5 -> 1)
    # model_recovery: torch.nn.Module (5 -> 1)
    # X_train_norm: np.ndarray with shape (N,5) in the same input space you want to optimize
    # Optional scalers for outputs if your models predict normalized values:
    # scaler_energy, scaler_recovery = StandardScaler().fit(...), ...

    best_df, res = run_single_objective_ratio_optimization(
        model_energy=model_EC,
        model_recovery=model_REC,
        X_train_norm=X_train_norm_EC,
        scaler_x=None,                 # pass if you want to optimize in original units
        scaler_energy=scaler_y_EC,   # or None if already in real units
        scaler_recovery=scaler_y_REC,
        pop_size=150,
        n_gen=300,
        discrete_last_var=True
    )

    print(best_df)

    # best_df = pd.DataFrame([{
    #     'feed_flow':   X_used[0],
    #     'temperature': X_used[1],
    #     'salinity':    X_used[2],
    #     'pressure':    X_used[3],
    #     'membrane':    int(np.round(X_used[4])) if discrete_last_var else X_used[4],
    #     'Energy':      e_opt,
    #     'Recovery':    r_opt,
    #     'Energy/Recovery': J_opt
    # }])
    # NSGA-II optimization
    # df_res_EC = run_nsga2(model_EC, scaler_x_EC, scaler_y_EC, X_train_norm_EC)# model predicts 1 - recovery
    # df_res_REC = run_nsga2(model_REC, scaler_x_REC, scaler_y_REC, X_train_norm_REC)
    ########################################################################
    input_data_norm_EC = np.column_stack((best_df.iloc[:,0], best_df.iloc[:,1], best_df.iloc[:,2], best_df.iloc[:,3], best_df.iloc[:,4]))#scaler_x.transform(np.column_stack((Z1, Z2, Z3, Z4, Z5)))
    input_data_EC = torch.tensor(input_data_norm_EC, dtype=torch.float32)

    #  ####################################### Step 3: Feed inputs into the model (assume model is already defined and loaded)
    model_EC.eval()
    model_REC.eval()
    with torch.no_grad():
        output_EC = model_EC(input_data_EC)  # expect shape: [n_samples, 2]
        output_REC = model_REC(input_data_EC)  # expect shape: [n_samples, 2]

    # Step 4: Convert output to NumPy and plot
    output_np_EC = output_EC.cpu().numpy()
    output_np_EC = scaler_y_EC.inverse_transform(output_np_EC)

    output_np_REC = output_REC.cpu().numpy()
    output_np_REC = scaler_y_REC.inverse_transform(output_np_REC)
    # output_np[:, 1] = (1 - output_np[:, 1])*100
    # df_res['Energy Consumption'] = output_np_EC[:, 0]
    # df_res_EC['Recovery'] = output_np_EC[:, 1]

    # df_res_REC['Energy Consumption'] = output_np_REC[:, 0]
    # df_res_REC['Recovery'] = output_np_REC[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(best_df['Energy'], (1-best_df['Recovery'])*100, c='green', alpha=0.6)
    plt.xlabel('Energy Consumption (Watt)',fontsize=14)
    plt.ylabel('Recovery (%)', fontsize=14)
    plt.title('Optimal Set')
    plt.grid(True)


    plt.figure(figsize=(8, 6))
    plt.scatter(best_df['Energy'], (1-best_df['Recovery'])*100, c='blue', alpha=0.6)
    plt.xlabel('Energy Consumption (Watt)',fontsize=14)
    plt.ylabel('Recovery (%)', fontsize=14)
    plt.title('Optimal Set')
    plt.grid(True)

    # plt.show()

    from scipy.interpolate import griddata
    N = 1000
    # Example usage
    tt = scaler_x_REC.inverse_transform(X_train_norm_REC)#X_train_norm_EC#scaler_x.inverse_transform(X_train_norm)
    ranges =  list(zip(tt.min(axis=0), tt.max(axis=0)))#[(-1, 1), (0, 10), (5, 15), (-100, -50), (1, 2)]
    ranges2 = [ranges[0], ranges[1], ranges[2], ranges[3], (0, 1)]
    df_rnd = generate_random_df(N, col_ranges=ranges2)
    print(df_rnd.head())
    # df1 = pd.read_excel('AG_with_theoretical.xlsx', header=0)
    # df1['mem_encoded'] = LabelEncoder().fit_transform(df1['Membrane '])
    # df1 = df1.drop(columns=['Membrane '])
    # x = df_rnd.iloc[:,4]
    # y = df1.iloc[:,5]*100
    # input_data_np = np.array(df1.iloc[:, list(range(0, 4)) + [-1]])

    Z1 = df_rnd.iloc[:,0]#df1['feed_flow']
    Z2 = df_rnd.iloc[:,1]#df1['temperature'] 
    Z3 = df_rnd.iloc[:,2]#df1['salinity']
    Z4 = df_rnd.iloc[:,3]#df1['pressure']
    Z5 = df_rnd.iloc[:,-1].apply(lambda x: 0 if x < 0.5 else 1)#.apply(lambda x: 0 if x < 1 else 1)#df1['membrane'].apply(lambda x: 0 if x < 1 else 1)#.apply(lambda x: 0 if x == 'Membrane AG' else  1)#res.X[:, 4].apply(lambda x: 0 if x == 'Membrane AG' else  1)
    Z6 = df_rnd.iloc[:,-1].apply(lambda x: 0 if x < 1 else 1)#input_data_np[:,-1].apply(lambda x: 0 if x < 0.2 else 1)#df1['membrane'].apply(lambda x: 0 if x < 0.2 else 1)#.apply(lambda x: 0 if x == 'Membrane AG' else  1)#res.X[:, 4].apply(lambda x: 0 if x == 'Membrane AG' else  1)

    ########################################################################
    input_data_norm = np.column_stack((Z1, Z2, Z3, Z4, Z5))#scaler_x.transform(np.column_stack((Z1, Z2, Z3, Z4, Z5)))
    input_data = torch.tensor(input_data_norm, dtype=torch.float32)

    #  ####################################### Step 3: Feed inputs into the model (assume model is already defined and loaded)
    model_EC.eval()
    model_REC.eval()
    with torch.no_grad():
        output_EC = model_EC(input_data)  # expect shape: [n_samples, 2]
        output_REC = model_REC(input_data)  # expect shape: [n_samples, 2]

    # Step 4: Convert output to NumPy and plot
    output_np_EC = output_EC.cpu().numpy()
    output_np_EC = scaler_y_EC.inverse_transform(output_np_EC)
    output_np_REC = output_REC.cpu().numpy()
    output_np_REC = scaler_y_REC.inverse_transform(output_np_REC)
    output_np_REC[:, 0] = (1 - output_np_REC[:, 0])*100
    x = output_np_EC[:, 0]
    y = output_np_REC[:, 0]

    plot_solution_space(x, y, Z1, 'Feed flow (LPM)')
    plt.savefig('feed_flow.png')
    plot_solution_space(x, y, Z2, 'Temperature (C)')
    plt.savefig('temperature.png')
    plot_solution_space(x, y, Z3, 'Salinity (PPM)')
    plt.savefig('salinity.png')
    plot_solution_space(x, y, Z4, 'Pressure (Psi)')
    plt.savefig('pressure.png')
    plot_solution_space_discerete(x, y, Z5, 'Membrane')
    plt.savefig('membrane.png')
    plot_pareto(x, y,Z6,best_df)
    plt.savefig('pareto.png')


    ####################################### AG_with_theoretical
    df1 = pd.read_excel('AG_with_theoretical.xlsx', header=0)
    df1['mem_encoded'] = LabelEncoder().fit_transform(df1['Membrane '])
    df1 = df1.drop(columns=['Membrane '])
    plt.figure(figsize=(8, 6))
    # plt.scatter(df1.iloc[:,5], (df1.iloc[:,6])*100, c='black',label='Actual', alpha=0.6)
    plt.scatter(df1.iloc[:,9], (df1.iloc[:,8]*100), alpha = 0.2, c='m')#, label='Theoretical')
    theo_np = np.column_stack((np.array(df1.iloc[:, 9]), np.array(df1.iloc[:, 8]*100)))
    hull = ConvexHull(theo_np)
    for simplex in hull.simplices:
        plt.plot(theo_np[simplex, 0], theo_np[simplex, 1], 'm-')
    plt.fill(theo_np[hull.vertices, 0], theo_np[hull.vertices, 1], edgecolor='m', fill=True,alpha = 0.2, label='Theoritical Region',facecolor='m')


    x = df1.iloc[:,4]
    y = df1.iloc[:,5]*100
    input_data_np = np.array(df1.iloc[:, list(range(0, 4)) + [-1]])
    ##################### RSM modeling ###############################
    rsm_ag_ec = RSM_AG_EC(input_data_np[:, 0], input_data_np[:,1], input_data_np[:,2], input_data_np[:,3])
    rsm_ag_re = RSM_AG_RE(input_data_np[:, 0], input_data_np[:,1], input_data_np[:,2], input_data_np[:,3])
    out_rsm_np = np.column_stack((rsm_ag_ec, rsm_ag_re))

    plt.scatter(rsm_ag_ec , rsm_ag_re, alpha=0.1, c='blue', edgecolors='b')#, label='RSM Output')
    hull = ConvexHull(out_rsm_np)
    for simplex in hull.simplices:
        plt.plot(out_rsm_np[simplex, 0], out_rsm_np[simplex, 1], 'b-')
    plt.fill(out_rsm_np[hull.vertices, 0], out_rsm_np[hull.vertices, 1], edgecolor='blue', fill=True, label='RSM Region' , facecolor='blue', alpha=0.2)


    ########################################################################
    input_data_norm = scaler_x_REC.transform(input_data_np)
    input_data = torch.tensor(input_data_norm, dtype=torch.float32)

    #  ####################################### Step 3: Feed inputs into the model (assume model is already defined and loaded)
    model_EC.eval()
    model_REC.eval()
    with torch.no_grad():
        output_EC = model_EC(input_data)  # expect shape: [n_samples, 2]
        output_REC = model_REC(input_data)  # expect shape: [n_samples, 2]

    # Step 4: Convert output to NumPy and plot
    output_np_EC = output_EC.cpu().numpy()
    output_np_EC = scaler_y_EC.inverse_transform(output_np_EC)
    # output_np_EC[:, 0] = (1 - output_np_EC[:, 0])*100  # 1 - recovery

    output_np_REC = output_REC.cpu().numpy()
    output_np_REC = scaler_y_REC.inverse_transform(output_np_REC)
    output_np_REC[:, 0] = (1 - output_np_REC[:, 0])*100  # 1 - recovery
    # outpu_np = scaler.inverse_transform(output_np)
    plt.scatter(output_np_EC[:, 0], output_np_REC[:, 0], alpha=0.1, c='green', edgecolors='k')#, label='Modeled Output')
    # Compute convex hull
    tt = np.concatenate((output_np_EC, output_np_REC), axis=1)
    hull = ConvexHull(tt)
    for simplex in hull.simplices:
        plt.plot(tt[simplex, 0], tt[simplex, 1], 'g-')
    plt.fill(tt[hull.vertices, 0], tt[hull.vertices, 1], edgecolor='green', fill=True, label='ANN Region' , facecolor='green', alpha=0.4)

################################################## Actual #############################################3
    
    XY = np.column_stack((x, y))
    # Step 2: Interpolate Z values onto the grid
    # grid_z = griddata(XY, Z6, (grid_x, grid_y), method='linear')
    # Step 3: Plot filled contours
    # plt.figure(figsize=(8, 6))
    # Compute convex hull
    hull = ConvexHull(XY)
    # Plot
    plt.scatter(XY[:, 0], XY[:, 1],alpha=0.2, c ='k')#,label='Actual Points')
    for simplex in hull.simplices:
        plt.plot(XY[simplex, 0], XY[simplex, 1], 'k-')
    plt.fill(XY[hull.vertices, 0], XY[hull.vertices, 1], edgecolor='black', fill=True, label='Actual Region', facecolor='black', alpha=0.2)
    plt.legend(fontsize=14, loc='upper right')
    # plt.axis('equal')
    # contour = plt.contourf(grid_x, grid_y*100, grid_z,levels=1,alpha=0.2, cmap='tab10')#'viridis')
    # Create proxy line for contour
    # contour_proxy = Line2D([0], [0], color='blue', label='Modeled')

    plt.xlabel('Energy Consumption (Watt)',fontsize=14)
    plt.ylabel('Recovery (%)', fontsize=14)
    plt.ylim([0, 100])
    plt.grid(True)
    # plt.legend(fontsize=14, loc='upper left', handles=[contour_proxy])
    plt.savefig('cmp_spaces_AG.png')
    # plt.show()


#################################################AK_with_theoretical####################################33
    df2 = pd.read_excel('AK_with_theoretical.xlsx', header=0)   
    df2['mem_encoded'] = LabelEncoder().fit_transform(df2['Membrane '])
    df2 = df2.drop(columns=['Membrane '])
    plt.figure(figsize=(8, 6))
    # plt.scatter(df2.iloc[:,5], (df2.iloc[:,6])*100, c='black',label='Actual', alpha=0.6)
    # plt.scatter(df2.iloc[:,9], (df2.iloc[:,8]*100), alpha = 0.2, c='m', label='Theoretical')
    # theo_np = np.column_stack((np.array(df2.iloc[:, 9]), np.array(df2.iloc[:, 8]*100)))
    # hull = ConvexHull(theo_np)
    # for simplex in hull.simplices:
    #     plt.plot(theo_np[simplex, 0], theo_np[simplex, 1], 'm-')
    # plt.fill(theo_np[hull.vertices, 0], theo_np[hull.vertices, 1], edgecolor='m', fill=True,alpha = 0.2, label='Theoritical Region',facecolor='m')
    x = df2.iloc[:,4]
    y = df2.iloc[:,5]*100
    
    input_data_np = np.array(df2.iloc[:, list(range(0, 4)) + [-1]])
    ####################################### AG_with_theoretical

    # plt.figure(figsize=(8, 6))
    # plt.scatter(df1.iloc[:,5], (df1.iloc[:,6])*100, c='black',label='Actual', alpha=0.6)
    plt.scatter(df2.iloc[:,9], (df2.iloc[:,8]*100), alpha = 0.2, c='m')#, label='Theoretical')
    theo_np = np.column_stack((np.array(df1.iloc[:, 9]), np.array(df1.iloc[:, 8]*100)))
    hull = ConvexHull(theo_np)
    for simplex in hull.simplices:
        plt.plot(theo_np[simplex, 0], theo_np[simplex, 1], 'm-')
    plt.fill(theo_np[hull.vertices, 0], theo_np[hull.vertices, 1], edgecolor='m', fill=True,alpha = 0.2, label='Theoritical Region',facecolor='m')


    ##################### RSM modeling ###############################
    rsm_ak_ec = RSM_AK_EC( input_data_np[:, 0], input_data_np[:,1], input_data_np[:,2], input_data_np[:,3])
    rsm_ak_re = RSM_AK_RE( input_data_np[:, 0], input_data_np[:,1], input_data_np[:,2], input_data_np[:,3])
    out_rsm_np = np.column_stack((rsm_ak_ec, rsm_ak_re))

    plt.scatter(rsm_ak_ec , rsm_ak_re, alpha=0.1, c='blue', edgecolors='b')#, label='RSM Output')
    hull = ConvexHull(out_rsm_np)
    for simplex in hull.simplices:
        plt.plot(out_rsm_np[simplex, 0], out_rsm_np[simplex, 1], 'b-')
    plt.fill(out_rsm_np[hull.vertices, 0], out_rsm_np[hull.vertices, 1], edgecolor='blue', fill=True, label='RSM Region' , facecolor='blue', alpha=0.2)


    ########################################################################
    input_data_norm = scaler_x_REC.transform(input_data_np)
    input_data = torch.tensor(input_data_norm, dtype=torch.float32)
    # Step 3: Feed inputs into the model (assume model is already defined and loaded)
    model_EC.eval()
    model_REC.eval()
    with torch.no_grad():
        output_EC = model_EC(input_data)  # expect shape: [n_samples, 2]
        output_REC = model_REC(input_data)  # expect shape: [n_samples, 2]

    # Step 4: Convert output to NumPy and plot
    output_np_EC = output_EC.cpu().numpy()
    output_np_EC = scaler_y_EC.inverse_transform(output_np_EC)
    

    output_np_REC = output_REC.cpu().numpy()
    output_np_REC = scaler_y_REC.inverse_transform(output_np_REC)
    output_np_REC[:, 0] = (1 - output_np_REC[:, 0])*100  # 1 - recovery
    # outpu_np = scaler.inverse_transform(output_np)
    plt.scatter(output_np_EC[:, 0], output_np_REC[:, 0], alpha=0.1, c='green', edgecolors='k')#, label='ANN Output')
    # Compute convex hull
    hull = ConvexHull(np.concatenate((output_np_EC, output_np_REC), axis=1))
    for simplex in hull.simplices:
        plt.plot(output_np_EC[simplex, 0], output_np_REC[simplex, 0], 'g-')
    plt.fill(output_np_EC[hull.vertices, 0], output_np_REC[hull.vertices, 0], edgecolor='green', fill=True, label='ANN Region' , facecolor='green', alpha=0.4)


    
    XY = np.column_stack((x, y))
    # Step 2: Interpolate Z values onto the grid
    # grid_z = griddata(XY, Z6, (grid_x, grid_y), method='linear')
    # Step 3: Plot filled contours
    # plt.figure(figsize=(8, 6))
    # Compute convex hull
    hull = ConvexHull(XY)
    # Plot
    plt.scatter(XY[:, 0], XY[:, 1],alpha=0.2, c ='k')#,label='Actual Points')
    for simplex in hull.simplices:
        plt.plot(XY[simplex, 0], XY[simplex, 1], 'k-')
    plt.fill(XY[hull.vertices, 0], XY[hull.vertices, 1], edgecolor='black', fill=True, label='Actual Region', facecolor='black', alpha=0.2)
    plt.legend(fontsize=14, loc='upper right')
    # plt.axis('equal')
    # contour = plt.contourf(grid_x, grid_y*100, grid_z,levels=1,alpha=0.2, cmap='tab10')#'viridis')
    # Create proxy line for contour
    # contour_proxy = Line2D([0], [0], color='blue', label='Modeled')

    plt.xlabel('Energy Consumption (Watt)',fontsize=14)
    plt.ylabel('Recovery (%)', fontsize=14)
    plt.ylim([0, 100])
    plt.grid(True)
    # plt.legend(fontsize=14, loc='upper left', handles=[contour_proxy])
    plt.savefig('cmp_spaces_AK.png')
    plt.show()

    

    


        
