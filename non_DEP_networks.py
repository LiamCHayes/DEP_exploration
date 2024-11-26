"""
Neural networks to critique DEP networks
"""

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple critic network
class SimpleCritic(nn.Module):
    def __init__(self, action_size, observation_size):
        super(SimpleCritic, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(action_size + observation_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)

    def forward(self, states, actions):
        actions = torch.squeeze(actions, dim=1)
        ntw_in = torch.cat((states, actions), dim=1)
        return self.network(ntw_in)
    
# Simple actor network
class SimpleActor(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleActor, self).__init__()

        # Layers
        self.L1 = nn.Linear(input_size, 512).to(device)
        self.L2 = nn.Linear(512, 256).to(device)
        self.L3 = nn.Linear(256, 128).to(device)
        self.L4 = nn.Linear(128, 64).to(device)
        self.L5 = nn.Linear(64, output_size).to(device)
        self.activation = nn.Tanh()

    def forward(self, x):
        y = self.L1(x)
        y = self.activation(y)
        y = self.L2(y)
        y = self.activation(y)
        y = self.L3(y)
        y = self.activation(y)
        y = self.L4(y)
        y = self.activation(y)
        y = self.L5(y)
        y = self.activation(y)
        return y
