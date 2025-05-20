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
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train_norm = scaler_x.fit_transform(X_train)
    X_test_norm = scaler_x.transform(X_test)
    y_train_norm = scaler_y.fit_transform(y_train)
    y_test_norm = scaler_y.transform(y_test)
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
class NeuralNet(nn.Module):
    def __init__(self):

        super(NeuralNet, self).__init__()
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

        nn.Linear(32, 2)
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
# Multi-objective Optimization
# ================================
from pymoo.core.problem import ElementwiseProblem
class TorchModelProblem(ElementwiseProblem):
    def __init__(self, model, input_mins, input_maxs):
        super().__init__(n_var=5, n_obj=2, n_constr=0, xl=input_mins, xu=input_maxs)
        self.model = model


    def _evaluate(self, X, out, *args, **kwargs):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_tensor).numpy()
        out["F"] = outputs


def run_nsga2(model, scaler_x, scaler_y, X_train_norm):
    x_mins = np.min(X_train_norm, axis=0)
    x_maxs = np.max(X_train_norm, axis=0)
    input_mins = np.array([x_mins[0], x_mins[1], x_mins[2], x_mins[3], 0])
    input_maxs = np.array([x_maxs[0], x_maxs[1], x_maxs[2], x_maxs[3], 1])

    problem = TorchModelProblem(model,  input_mins, input_maxs)
    algorithm = NSGA2(pop_size=500)

    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', 200),
        seed=1,
        verbose=True
    )

    res_X_inverse = res.X#scaler_x.inverse_transform(res.X)
    df_res = pd.DataFrame(res_X_inverse, columns=['feed_flow', 'temperature', 'salinity', 'pressure', 'membrane'])
    # df_res['membrane'] = df_res['membrane']#.apply(lambda x: 'Membrane AG' if x < 0.5 else 'Membrane AK')
    tt = scaler_y.inverse_transform(res.F)
    df_res['Energy Consumption'] = tt[:, 0]
    df_res['Recovery'] = tt[:, 1]
    df_res.to_csv("nsga2_solutions.csv", index=False)
    return df_res
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
    plt.scatter(res['Energy Consumption'], (1 - res['Recovery'])*100, c='blue', alpha=0.2,s=100, label='Pareto Front Optimal Set')
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
    X_train_norm, X_test_norm, y_train_norm, y_test_norm, scaler_x, scaler_y = normalize_split_data(X, y)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_norm, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_norm, dtype=torch.float32)

    # Train model
    set_seed(42)
    model = NeuralNet()
    # model, test_mse_list, train_mse_list = train_model(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
    # print(train_mse_list[-1], test_mse_list[-1])
    # # Save the model
    # torch.save(model.state_dict(), 'model.pth')
    model.load_state_dict(torch.load('model.pth'))  # load weights
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

    # NSGA-II optimization
    df_res = run_nsga2(model, scaler_x, scaler_y, X_train_norm)# model predicts 1 - recovery
    ########################################################################
    input_data_norm = np.column_stack((df_res.iloc[:,0], df_res.iloc[:,1], df_res.iloc[:,2], df_res.iloc[:,3], df_res.iloc[:,4]))#scaler_x.transform(np.column_stack((Z1, Z2, Z3, Z4, Z5)))
    input_data = torch.tensor(input_data_norm, dtype=torch.float32)

    #  ####################################### Step 3: Feed inputs into the model (assume model is already defined and loaded)
    model.eval()
    with torch.no_grad():
        output = model(input_data)  # expect shape: [n_samples, 2]

    # Step 4: Convert output to NumPy and plot
    output_np = output.cpu().numpy()
    output_np = scaler_y.inverse_transform(output_np)
    # output_np[:, 1] = (1 - output_np[:, 1])*100
    df_res['Energy Consumption'] = output_np[:, 0]
    df_res['Recovery'] = output_np[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(df_res['Energy Consumption'], (1-df_res['Recovery'])*100, c='green', alpha=0.6)
    plt.xlabel('Energy Consumption (Watt)',fontsize=14)
    plt.ylabel('Recovery (%)', fontsize=14)
    plt.title('Pareto Front Optimal Set')
    plt.grid(True)

    # plt.show()

    from scipy.interpolate import griddata
    N = 1000
    # Example usage
    tt = X_train_norm#scaler_x.inverse_transform(X_train_norm)
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
    model.eval()
    with torch.no_grad():
        output = model(input_data)  # expect shape: [n_samples, 2]

    # Step 4: Convert output to NumPy and plot
    output_np = output.cpu().numpy()
    output_np = scaler_y.inverse_transform(output_np)
    output_np[:, 1] = (1 - output_np[:, 1])*100
    x = output_np[:, 0]
    y = output_np[:, 1]

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
    input_data_norm = scaler_x.transform(input_data_np)
    input_data = torch.tensor(input_data_norm, dtype=torch.float32)

    #  ####################################### Step 3: Feed inputs into the model (assume model is already defined and loaded)
    model.eval()
    with torch.no_grad():
        output = model(input_data)  # expect shape: [n_samples, 2]

    # Step 4: Convert output to NumPy and plot
    output_np = output.cpu().numpy()
    output_np = scaler_y.inverse_transform(output_np)
    output_np[:, 1] = (1 - output_np[:, 1])*100  # 1 - recovery
    # outpu_np = scaler.inverse_transform(output_np)
    plt.scatter(output_np[:, 0], output_np[:, 1], alpha=0.1, c='green', edgecolors='k')#, label='Modeled Output')
    # Compute convex hull
    hull = ConvexHull(output_np)
    for simplex in hull.simplices:
        plt.plot(output_np[simplex, 0], output_np[simplex, 1], 'g-')
    plt.fill(output_np[hull.vertices, 0], output_np[hull.vertices, 1], edgecolor='green', fill=True, label='ANN Region' , facecolor='green', alpha=0.4)

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
    df2 = pd.read_excel('AK.xlsx', header=0)   
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
    input_data_norm = scaler_x.transform(input_data_np)
    input_data = torch.tensor(input_data_norm, dtype=torch.float32)
    # Step 3: Feed inputs into the model (assume model is already defined and loaded)
    model.eval()
    with torch.no_grad():
        output = model(input_data)  # expect shape: [n_samples, 2]

    # Step 4: Convert output to NumPy and plot
    output_np = output.cpu().numpy()
    output_np = scaler_y.inverse_transform(output_np)
    output_np[:, 1] = (1 - output_np[:, 1])*100  # 1 - recovery
    # outpu_np = scaler.inverse_transform(output_np)
    plt.scatter(output_np[:, 0], output_np[:, 1], alpha=0.1, c='green', edgecolors='k')#, label='ANN Output')
    # Compute convex hull
    hull = ConvexHull(output_np)
    for simplex in hull.simplices:
        plt.plot(output_np[simplex, 0], output_np[simplex, 1], 'g-')
    plt.fill(output_np[hull.vertices, 0], output_np[hull.vertices, 1], edgecolor='green', fill=True, label='ANN Region' , facecolor='green', alpha=0.4)


    
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

    

    


        
