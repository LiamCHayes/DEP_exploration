"""
A neural network with DEP layers
"""

from typing import Any, Mapping
import torch
import torch.nn as nn

from DEP import DEP, BatchedDEP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Single DEP layer
class DEPLayer(BatchedDEP):
    def __init__(self, input_size, output_size):
        super(DEPLayer, self).__init__(13, 1000, 0.0025, 5.25, 1, device, output_size, input_size)
        self.optimizer = torch.optim.Adam(self.M.parameters(), lr=0.01)

# Network where DEP is part of the input
class FirstLayerDEP(nn.Module):
    def __init__(self, input_size, output_size):
        super(FirstLayerDEP, self).__init__()

        # Layers
        self.DEPlayer = DEPLayer(input_size, output_size)
        self.L1 = nn.Linear(input_size + output_size, 512).to(device)
        self.L2 = nn.Linear(512, 256).to(device)
        self.L3 = nn.Linear(256, 128).to(device)
        self.L4 = nn.Linear(128, 64).to(device)
        self.L5 = nn.Linear(64, output_size).to(device)
        self.activation = nn.Tanh()

    def forward(self, x):
        dep_output = self.DEPlayer.step(x)
        ntw_in = torch.concat((x, dep_output), dim=1)
        y = self.L1(ntw_in)
        y = self.activation(y)
        y = self.L2(y)
        y = self.activation(y)
        y = self.L3(y)
        y = self.activation(y)
        y = self.L4(y)
        y = self.activation(y)
        y = self.L5(y)
        y = self.activation(y)
        return y, dep_output

    def forward_no_step(self, x, dep_output):
        """
        Does a forward pass on the network without stepping DEP layer
        Used for network updates when sampling from replay buffer
        """
        dep_output = torch.squeeze(dep_output, dim=1)
        ntw_in = torch.concat((x, dep_output), dim=1)
        y = self.L1(ntw_in)
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

    def only_dep(self, x):
        """
        Returns only the DEP forward pass without the NN 
        """
        return self.DEPlayer.step(x)

    def reset_dep(self):
        self.DEPlayer.reset()

    def set_dep_model(self, model):
        self.DEPlayer.set_model(model)

    def learn_dep_model(self, observation):
        """
        Does gradient descent on the DEP inverse prediction model M
        
        args:
            observation: The current observation
        """
        # Check if there has been a learning step
        if self.DEPlayer.C is not None:
            # Zero grad
            self.DEPlayer.optimizer.zero_grad()

            # Compute loss
            prev_action = self.DEPlayer.memory[-self.DEPlayer.delta_t][1]
            Mx = self.DEPlayer.M(torch.tensor(observation, dtype=torch.float32).to(device))
            loss = torch.sum((Mx - prev_action)**2)

            # Update network
            loss.backward()
            self.DEPlayer.optimizer.step()
            loss = loss.item()
        else:
            loss = 0
        
        return loss

    def load_state_dict(self, state_dict: Mapping[str, Any], dep_state_dict: Mapping[str, Any],  strict: bool = True, assign: bool = False):
        """Load the network state dict as well as the DEP model state dict"""
        self.DEPlayer.M.load_state_dict(dep_state_dict)
        return super().load_state_dict(state_dict, strict, assign)
