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
        super(DEPLayer, self).__init__(8, 0.5, 0.0025, 5.25, 1, device, output_size, input_size)
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

# Deep network implementation of DEP - An actor with access to DEP
class DEPActor(nn.Module):
    """
    A neural network implementation of DEP with:
    - Deep network imitation controller matrix
    - Deep model matrix
    - Classic batched DEP controller
    """
    def __init__(self, input_size, output_size, imitation_lr):
        super(DEPActor, self).__init__()

        # DEP controller
        self.dep_controller = DEPLayer(input_size, output_size)
        self.dep_controller._batch_size = 1

        # For keeping track of dimensions
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = self.dep_controller.tau + self.dep_controller.delta_t

        # Takes in a batch of observations and "classifies" them
        self.state_encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 1)
        ).to(device)

        # Takes the classified states and gives an action
        self.network_controller = nn.Sequential(
            nn.Linear(self.batch_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, output_size),
            nn.Tanh()
        ).to(device)

        # For computing the loss and doing the imitation update step
        self.last_action = None
        self.last_dep_action = None
        self.lr = imitation_lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.im_loss = None

    def forward(self, observation):
        """
        Forward pass on the network and the unbatched dep_controller

        Observation is a single observation of shape (1, self.input_size)

        The unbatched dep_controller step is done to 
        1. Store the observation in the memory
        2. Get the dep action for the imitation learning of dep
        3. Return an action even if the memory isn't big enough
        """
        # Do DEP step
        dep_action = self.dep_controller.step(observation)

        if len(self.dep_controller.memory) < self.batch_size:
            self.last_dep_action = dep_action
            return dep_action

        # Format the DEP memory to give to the network
        observations = self.batch_observations()

        # Encode states, returns a (batch_size, 1) tensor
        encoded_states = self.state_encoder(observations)

        # Turn the (batch_size, 1) tensor into a linear input
        lin_input = encoded_states.squeeze()
        print(lin_input)

        # Get action
        action = self.network_controller(lin_input)

        # Store actions to compute loss
        self.last_dep_action = dep_action
        self.last_action = action

        # Imitation update step
        loss = self.imitation_loss()
        self.imitation_step(loss)
        self.im_loss = loss

        return action.unsqueeze(0)
    
    def only_network(self, batched_obs):
        """
        Forward pass on only the network for the RL step
        """
        # Encode states, returns a (batch_size, 1) tensor
        encoded_states = self.state_encoder(batched_obs)

        # Turn the (batch_size, 1) tensor into a linear input
        encoded_states = torch.squeeze(encoded_states, 2)
        lin_input = encoded_states.squeeze()

        # Get action
        action = self.network_controller(lin_input)

        return action
    
    def only_dep(self, observation):
        """
        Forward pass on only DEP
        """
        dep_action = self.dep_controller.step(observation)
        return dep_action
        
    def imitation_loss(self):
        """
        Returns the imitation learning loss for approximating DEP
        """
        if self.last_action is None or self.last_dep_action is None:
            return 0.
        else:
            sq_err = (self.last_action - self.last_dep_action)**2
            return torch.sum(sq_err)
    
    def imitation_step(self, loss):
        """
        Takes a small step in the direction of the DEP action
        """
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

    def batch_observations(self):
        """
        Batch observations to put into the network
        """
        observations = []
        for m in self.dep_controller.memory:
            observations.append(torch.tensor(m[0], dtype=torch.float32))
        observations = torch.stack(observations)

        return observations

    def reset(self):
        """
        Resets the DEP controller
        """
        self.dep_controller.reset()
