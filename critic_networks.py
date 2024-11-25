"""
Neural networks to critique DEP networks
"""

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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