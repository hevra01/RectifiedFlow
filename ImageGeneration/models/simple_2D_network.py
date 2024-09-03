import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils, layers

@utils.register_model(name='Simple2DNetwork')
class Simple2DNetwork(nn.Module):
    def __init__(self, config):
        super(Simple2DNetwork, self).__init__()

        input_dim = config.model.input_dim
        hidden_dim = config.model.hidden_dim
        output_dim = config.model.output_dim
        
        # Define a simple feedforward network
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # Second hidden layer
        self.fc3 = nn.Linear(hidden_dim, output_dim) # Output layer

    def forward(self, x, time_cond):
        temb = layers.get_timestep_embedding(time_cond)
        x = torch.cat([x, temb], dim=1)

        # Apply the layers with ReLU activations
        x = F.relu(self.fc1(x))  # Apply ReLU after first hidden layer
        x = F.relu(self.fc2(x))  # Apply ReLU after second hidden layer
        x = self.fc3(x)          # Output layer without activation (for regression tasks)
        return x

