#! C:\Users\shima\Documents\Leili\Second Paper-20250507T162628Z-1-001\Second Paper\Pezhman Abbasi\leili_paper_venv\Scripts\python.exe
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
# === Load and Combine Data ===
file1 = 'AG.xlsx'
file2 = 'AK.xlsx'

df1 = pd.read_excel(file1, header=0)
df2 = pd.read_excel(file2, header=0)

# Encode the string column

# Now df is all numeric
# Combine both datasets
df = pd.concat([df1, df2], ignore_index=True)
df['mem_encoded'] = LabelEncoder().fit_transform(df['Membrane '])# there is a space after Membrane

# Drop the original string column if needed
df = df.drop(columns=['Membrane '])

# === Prepare Features and Labels ===
X = df.iloc[:, list(range(0, 4)) + [-1]].values#df.iloc[:, 0:5].values  # 5 input features
y = df.iloc[:, 4:6].values  # 2 output features
y[:,-1]= 1 - y[:,-1] # 1 - recovery
# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# === Normalize Features ===
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# === Create DataLoader ===
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# === Define the Neural Network ===
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 6),
            nn.ReLU(),
            nn.Linear(6, 8),
            nn.ReLU(),
            nn.Linear(8, 2)  # 2 outputs
        )

    def forward(self, x):
        return self.net(x)

model = NeuralNet()

# === Define Loss and Optimizer ===
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === Train the Model ===
num_epochs = 500
test_mse_list =[]
train_mse_list =[]
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        test_loss = criterion(y_pred, y_test_tensor)
        test_mse_list.append(test_loss.item())
        predictions_tr = model(X_train_tensor)
        train_loss = criterion(predictions_tr, y_train_tensor)
        train_mse_list.append(train_loss.item())
# === Evaluate on Test Data ===
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor).item()
    
print(f"\nTest MSE Loss: {test_loss:.4f}")
plt.figure()
plt.plot(range(1, num_epochs + 1), test_mse_list, label='Test MSE')
plt.plot(range(1, num_epochs + 1), train_mse_list, label='Train MSE')
plt.legend()
plt.title("Test/train MSE vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Test MSE")
# plt.ylim([0, 20])
plt.grid(True)

# Scatter plot
plt.figure()
plt.scatter(y_test[:,0], predictions[:,0], alpha=0.6)
# plt.scatter(y_test[:,1], predictions[:,1], alpha=0.6)
plt.plot([y_test[:,0].min(), y_test[:,0].max()], [y_test[:,0].min(), y_test[:,0].max()], 'r--')  # y = x line
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs True Values - Power")
plt.grid(True)

plt.figure()
# plt.scatter(y_test[:,0], predictions[:,0], alpha=0.6)
plt.scatter(y_test[:,1], predictions[:,1], alpha=0.6)
plt.plot([y_test[:,1].min(), y_test[:,1].max()], [y_test[:,1].min(), y_test[:,1].max()], 'r--')  # y = x line
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs True Values - Recovery")
plt.grid(True)








# === Save the Model ===
torch.save(model.state_dict(), 'leili_model.pth') 


# Assume min/max arrays for each of the 5 inputs
allowed_values = [
    [0, 1],      # membrane: 2 discrete options
    [3, 4, 5, 6],       # feed flow: 4 options
    [30, 40], # Temprature: 2 options
    [2000, 2500, 3000],      # salinity: 3 options
    [50, 75, 100, 125, 150, 175]          # pressure: binary
]


x_mins = np.min(X_train_norm ,axis=0)
x_maxs = np.max(X_train_norm ,axis=0)
input_mins = np.array([x_mins[0], x_mins[1], x_mins[2], x_mins[3]])
input_maxs = np.array([x_maxs[0], x_maxs[1], x_maxs[2], x_maxs[3]])

# Step 1: Randomly sample N input points in 5D
N = 5000
inputs = np.concatenate([np.random.uniform(low=input_mins, high=input_maxs, size=(N, 4)), np.random.randint(0, 2, size=(N,1))], axis=1)

# Step 2: Pass through the model
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
with torch.no_grad():
    outputs = model(inputs_tensor).numpy()  # shape (N, 2)

# Step 3: Plot outputs in 2D
plt.figure(figsize=(6, 6))
plt.scatter(outputs[:, 0], 1 - outputs[:, 1], alpha=0.5, s=10)
plt.xlabel("Power")
plt.ylabel("Recovery")
plt.title("Model Output Region in 2D")
plt.grid(True)
# plt.axis('equal')
plt.ylim([-0.5, 1])



import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# Step 1: Define your multi-objective problem

# Load or define your trained PyTorch model
# model = YourModel()
# model.load_state_dict(torch.load('your_model.pth'))
# model.eval()
x_mins = np.min(X_train_norm ,axis=0)
x_maxs = np.max(X_train_norm ,axis=0)
input_mins = np.array([x_mins[0], x_mins[1], x_mins[2], x_mins[3], 0])# np.random.randint(0, 2)])
input_maxs = np.array([x_maxs[0], x_maxs[1], x_maxs[2], x_maxs[3], 1])#np.random.randint(0, 2)])
# input_mins = scaler.transform(input_mins.reshape(1, -1)).flatten()
# input_maxs = scaler.transform(input_maxs.reshape(1, -1)).flatten()
# Define PyMoo problem using the torch model
class TorchModelProblem(Problem):
    def __init__(self, model):
        super().__init__(n_var=5, n_obj=2, n_constr=0, xl=input_mins, xu=input_maxs)
        self.model = model

    def _evaluate(self, X, out, *args, **kwargs):
        # Convert input array to torch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Pass through model (no gradient needed)
        with torch.no_grad():
            outputs = self.model(X_tensor).numpy()

        # Each row of `outputs` is [f1, f2]
        out["F"] = outputs

# Setup and run NSGA-II
pop_size=100
problem = TorchModelProblem(model)
algorithm = NSGA2(pop_size=100)

res = minimize(
    problem,
    algorithm,
    termination=('n_gen', 200),
    seed=1,
    verbose=True
)

# Plot Pareto front
res_X_inverse = scaler.inverse_transform(res.X)
df_res = pd.DataFrame(res_X_inverse)
df_res['Power'] = res.F[:, 0]
df_res['Recovery'] = 1 - res.F[:, 1]
df_res.to_csv("nsga2_solutions.csv", index=False)
plt.figure(figsize=(6, 6))
plt.scatter(outputs[:, 0], 1-outputs[:, 1], alpha=0.5, s=10)
plt.scatter(df_res['Power'], df_res['Recovery'], alpha=0.8, s=10, c='green')
plt.xlabel("Power")
plt.ylabel("Recovery")
Scatter().add(res.F).show()

# class TorchModelProblem_discrete(Problem):
#     def __init__(self, model, allowed_values):
#         self.allowed_values = allowed_values
#         n_var = len(allowed_values)
#         n_obj = 2
        
#         # Variable bounds are based on index in each value list
#         xl = np.zeros(n_var, dtype=int)
#         xu = np.array([len(vals) - 1 for vals in allowed_values], dtype=int)
        
#         super().__init__(
#             n_var=n_var,
#             n_obj=n_obj,
#             n_constr=0,
#             xl=xl,
#             xu=xu,
#             vtype=int
#         )

#         self.model = model

#     def _evaluate(self, X, out, *args, **kwargs):
#         # X contains indices → map to actual values
#         real_inputs = np.array([
#             [self.allowed_values[j][int(i)] for j, i in enumerate(row)]
#             for row in X
#         ])
#         real_inputs_norm = scaler.transform(real_inputs)
#         X_tensor = torch.tensor(real_inputs_norm, dtype=torch.float32)

#         with torch.no_grad():
#             outputs = self.model(X_tensor).numpy()

#         out["F"] = outputs


# Step 2: Run NSGA-II
# problem = TorchModelProblem(model, allowed_values)
# algorithm = NSGA2(pop_size=100)

# res = minimize(
#     problem,
#     algorithm,
#     termination=('n_gen', 200),  # number of generations
#     seed=1,
#     save_history=True,
#     verbose=True
# )

# Step 3: Visualize Pareto front
# Scatter().add(res.F).show()
