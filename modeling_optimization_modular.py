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
# ================================
# Data Loading and Preprocessing
# ================================
def load_and_prepare_data(file1, file2):
    df1 = pd.read_excel(file1, header=0)
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
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm, y_train, y_test, scaler

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
class NeuralNet(nn.Module):
    def __init__(self):

        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 6),
            nn.ReLU(),
            nn.Linear(6, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.net(x)


# ================================
# Training Routine
# ================================
def train_model(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, num_epochs=500):
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=16, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
# Multi-objective Optimization
# ================================
class TorchModelProblem(Problem):
    def __init__(self, model, input_mins, input_maxs):
        super().__init__(n_var=5, n_obj=2, n_constr=0, xl=input_mins, xu=input_maxs)
        self.model = model

    def _evaluate(self, X, out, *args, **kwargs):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_tensor).numpy()
        out["F"] = outputs


def run_nsga2(model, scaler, X_train_norm):
    x_mins = np.min(X_train_norm, axis=0)
    x_maxs = np.max(X_train_norm, axis=0)
    input_mins = np.array([x_mins[0], x_mins[1], x_mins[2], x_mins[3], 0])
    input_maxs = np.array([x_maxs[0], x_maxs[1], x_maxs[2], x_maxs[3], 1])

    problem = TorchModelProblem(model, input_mins, input_maxs)
    algorithm = NSGA2(pop_size=100)

    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', 200),
        seed=1,
        verbose=True
    )

    res_X_inverse = scaler.inverse_transform(res.X)
    df_res = pd.DataFrame(res_X_inverse, columns=['feed_flow', 'temperature', 'salinity', 'pressure', 'membrane'])
    # df_res['membrane'] = df_res['membrane']#.apply(lambda x: 'Membrane AG' if x < 0.5 else 'Membrane AK')
    df_res['Energy Consumption'] = res.F[:, 0]
    df_res['Recovery'] = 1 - res.F[:, 1]
    df_res.to_csv("nsga2_solutions.csv", index=False)
    return df_res
# ================================
# Visualization
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
    contour = plt.contourf(grid_x, grid_y*100, grid_z, levels=50, cmap='jet')#'viridis')
    cbar = plt.colorbar(contour)
    cbar.set_label(lbl, fontsize=14)  # ⬅️ set the label font size
    plt.xlabel('Energy Consumption (Watt)',fontsize=14)
    plt.ylabel('Recovery (%)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    # plt.axis('equal')
    plt.ylim([0, 70])
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
    contour = plt.contourf(grid_x, grid_y*100, grid_z,levels=1,alpha=0.3, cmap='tab10',label='Solution Space')#'viridis')
    # cbar = plt.colorbar(contour)
    # cbar.set_label(lbl, fontsize=14)  # ⬅️ set the label font size
    plt.scatter(res['Energy Consumption'], res['Recovery']*100, c='blue', alpha=0.6, label='Pareto Front Optimal Set')
    plt.xlabel('Energy Consumption (Watt)',fontsize=14)
    plt.ylabel('Recovery (%)', fontsize=14)
    plt.legend(fontsize=14, loc='upper left')
    plt.grid(True)
    # plt.axis('equal')
    plt.ylim([0, 70])
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
    contour = ax.contourf(grid_x, grid_y*100, grid_z, levels=levels, cmap=cmap, norm=norm, method='cubic')

    # Colorbar with two ticks
    cbar = plt.colorbar(contour, ticks=[0.25, 0.75])  # midpoints of bins
    cbar.ax.set_yticklabels(['AG', 'AK'])
    cbar.set_label(lbl, fontsize=14)  # ⬅️ set the label font size
    plt.xlabel('Energy Consumption (Watt)',fontsize=14)
    plt.ylabel('Recovery (%)', fontsize=14)
    plt.ylim([0, 70])
    plt.grid(True)
    # plt.show()
# ================================
# Main
# ================================
if __name__ == '__main__':
    X, y = load_and_prepare_data('AG.xlsx', 'AK.xlsx')
    X_train_norm, X_test_norm, y_train, y_test, scaler = normalize_split_data(X, y)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Train model
    set_seed(42)
    model = NeuralNet()
    model, test_mse_list, train_mse_list = train_model(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
    # Save the model
    torch.save(model.state_dict(), 'model.pth')
    # Plot MSE
    plt.figure(figsize=(8, 6))
    plt.plot(test_mse_list,linewidth=3, label='Test MSE')
    plt.plot(train_mse_list,linewidth=3, label='Train MSE')
    plt.legend(fontsize=14)
    plt.title('MSE over Epochs')
    plt.grid(True)
    plt.savefig('learning_curve.png')
    # plt.show()

    # NSGA-II optimization
    df_res = run_nsga2(model, scaler, X_train_norm)
    plt.figure(figsize=(8, 6))
    plt.scatter(df_res['Energy Consumption'], df_res['Recovery']*100, c='green', alpha=0.6)
    plt.xlabel('Energy Consumption (Watt)',fontsize=14)
    plt.ylabel('Recovery (%)', fontsize=14)
    plt.title('Pareto Front Optimal Set')
    plt.grid(True)

    # plt.show()

    from scipy.interpolate import griddata
    N = 500
    x = df_res['Energy Consumption']
    y = df_res['Recovery']
    Z1 = df_res['feed_flow']
    Z2 = df_res['temperature'] 
    Z3 = df_res['salinity']
    Z4 = df_res['pressure']
    Z5 = df_res['membrane'].apply(lambda x: 0 if x < 1 else 1)#.apply(lambda x: 0 if x == 'Membrane AG' else  1)#res.X[:, 4].apply(lambda x: 0 if x == 'Membrane AG' else  1)
    Z6 = df_res['membrane'].apply(lambda x: 0 if x < 0.2 else 1)#.apply(lambda x: 0 if x == 'Membrane AG' else  1)#res.X[:, 4].apply(lambda x: 0 if x == 'Membrane AG' else  1)

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
    plot_pareto(x, y,Z6,df_res)
    plt.savefig('pareto.png')


    # AG_with_theoretical
    df1 = pd.read_excel('AG_with_theoretical.xlsx', header=0)
    plt.figure(figsize=(8, 6))
    plt.scatter(df1.iloc[:,5], (df1.iloc[:,6])*100, c='green',label='Actual', alpha=0.6)
    plt.scatter(df1.iloc[:,10], (1 - df1.iloc[:,9])*100, c='blue', label='Theoretical',alpha=0.6)
    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), 200),
        np.linspace(y.min(), y.max(), 200)
    )
    plt.legend(fontsize=14, loc='upper left')
    XY = np.column_stack((x, y))
    # Step 2: Interpolate Z values onto the grid
    grid_z = griddata(XY, Z6, (grid_x, grid_y), method='linear')
    # Step 3: Plot filled contours
    # plt.figure(figsize=(8, 6))
    contour = plt.contourf(grid_x, grid_y*100, grid_z,levels=1,alpha=0.2, cmap='tab10')#'viridis')
    # Create proxy line for contour
    contour_proxy = Line2D([0], [0], color='blue', label='Modeled')

    plt.xlabel('Energy Consumption (Watt)',fontsize=14)
    plt.ylabel('Recovery (%)', fontsize=14)
    plt.grid(True)
    # plt.legend(fontsize=14, loc='upper left', handles=[contour_proxy])
    plt.savefig('cmp_spaces.png')
    plt.show()



        
