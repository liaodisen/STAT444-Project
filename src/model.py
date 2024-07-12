from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn import datasets, linear_model
import torch
import torch.nn as nn
import torch.optim as optim

# Define PyTorch Regression Model
class RegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_model(model_name, input_dim=None):
    if model_name == 'svm':
        model = SVR(kernel='linear')
    elif model_name == 'rf':
        model = RandomForestRegressor(max_depth=20)
    elif model_name == 'mlp':
        return RegressionModel(input_dim)
    elif model_name == 'linear':
        return linear_model.LinearRegression()
    elif model_name == 'ridge':
        return linear_model.Ridge()
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    return model

# Function to get optimizer for PyTorch model
def get_optimizer(model, learning_rate=1e-3):
    return optim.Adam(model.parameters(), lr=learning_rate)
