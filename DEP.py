"""
Unbatched DEP agent implementation
"""

# Imports and Setup
import torch
from collections import deque

from utils import print_and_pause


# Classic DEP implementation
class DEP:
    """
    Implements an unbatched DEP controller in pytorch

    args:
        tau (int): Number of previous timesteps to update the controller with
        kappa (float): Normalization scaling factor
        beta (float): Scale for the bias
        sigma (float): Scale for the action
        delta_t (int): Number of timesteps to compute the change in state over
        device: Device to store computations in (CPU or GPU)
        action_size (int): size of y
        observation_size (int): size of x

    attributes:
        tau (int): Number of previous timesteps to update the controller with
        kappa (float): Normalization scaling factor
        delta_t (int): Number of timesteps to compute the change in state over
        C (torch.Tensor in R2): Current controller matrix
        C_normalized (torch.Tensor in R2): Current normalized controller matrix
        M (torch.Tensor in R2): Inverse prediction model matrix
        memory (Collections.deque): Memory of the previous states
        x_smoothed (torch.Tensor in R): Moving average of the observations
        _timestep (int): Current timestep index
        _device: Device to store computations in (CPU or GPU)
        _action_size (int): size of y
        _observation_size (int): size of x
    """
    def __init__(self, tau, kappa, beta, sigma, delta_t, device, action_size, observation_size):
        # Scalars
        self.tau = tau
        self.kappa = kappa
        self.beta = beta
        self.sigma = sigma
        self.delta_t = delta_t

        # Matrices
        self.C = None
        self.C_normalized = None
        self.M = -torch.eye(action_size, observation_size, requires_grad=True).to(device)
        self.h = torch.zeros((action_size,)).to(device)

        # Memory
        self.memory = deque(maxlen = self.tau + self.delta_t)
        self.x_smoothed = None

        # Internal attributes
        self._timestep = 0
        self._device = device
        self._action_size = action_size
        self._observation_size = observation_size

    def step(self, x, numpy=True):
        """
        Does a single DEP step

        args:
            x (array of size self._observation_size): Current state
            numpy (bool): Return a numpy array if true, else return a torch tensor stored on self._device

        returns:
            y: DEP action corresponding to x
        """
        # Convert x to a tensor if it isn't already
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self._device)

        # Smooth the observation x
        if self.x_smoothed is None:
            self.x_smoothed = x
        else:
            self.x_smoothed = (x - self.x_smoothed) / 2

        # Learning step if there are enough memories in the buffer
        if len(self.memory) > 2 + self.delta_t:
            self._learn_controller()
        
        # Get action y
        if self.C == None:
            y = torch.tanh(torch.rand(self._action_size)).to(self._device)
        else:
            y = self._get_action()

        # Add state action pair (x, y) to memory
        self.memory.append((self.x_smoothed, y))

        # Update timestep and return action
        self._timestep += 1
        if numpy:
            y = y.cpu().detach().numpy()
        return y

    def set_model(self, model):
        """
        Initialize the model matrix from something that's not an identity
        
        args:
            model (torch tensor): Matrix of the shape (self._action_size, self._observation_size)
        """
        self.M = model

    def reset(self):
        """
        Resets the memory, controller matrices, and timestep while keeping the model matrix the same
        """
        self.C = None
        self.C_normalized = None
        self._timestep = 0
        self.memory = deque(maxlen = self.tau + self.delta_t)

    def _learn_controller(self):
        """
        Updates the controller matrix based on the state x
        """
        # Compute the controller matrix C
        self.C = torch.zeros((self._action_size, self._observation_size)).to(self._device)

        update_set = range(1, min(self.tau, self._timestep - self.delta_t))
        for s in update_set:
            chi = self.memory[-s][0] - self.memory[-(s+1)][0]
            nu = self.memory[-(s+self.delta_t)][0] - self.memory[-(s+self.delta_t+1)][0]
            mu = torch.matmul(self.M, chi)
            self.C = + self.C + torch.einsum('j, k->jk', mu, nu)

        # Normalize C
        self.C_normalized = self.C

        # Compute bias h
        self.h = torch.tanh(self.memory[-1][1] * self.beta)/2 * self.h * 0.001

    def _get_action(self):
        """
        Forward pass for the one-layer DEP network (controller matrix with tanh activation)
        """
        q = torch.matmul(self.C_normalized, self.x_smoothed)
        q = q / (torch.norm(q) + 1e-5)
        y = torch.tanh(self.kappa * q + self.h) * self.sigma
        return y

# DEP implementation with a deep inverse prediction model
class DEPDeepModel(DEP):
    """
    Same as DEP but with a multi-layer model network
    """
    def __init__(self, tau, kappa, beta, sigma, delta_t, device, action_size, observation_size):
        super(DEPDeepModel, self).__init__(tau, kappa, beta, sigma, delta_t, device, action_size, observation_size)
        self.set_model(MNetwork(observation_size, action_size, device))

    def _learn_controller(self):
        """
        Updates the controller matrix based on the state x
        """
        # Compute the controller matrix C
        self.C = torch.zeros((self._action_size, self._observation_size)).to(self._device)

        update_set = range(1, min(self.tau, self._timestep - self.delta_t))
        for s in update_set:
            chi = self.memory[-s][0] - self.memory[-(s+1)][0]
            nu = self.memory[-(s+self.delta_t)][0] - self.memory[-(s+self.delta_t+1)][0]
            with torch.no_grad():
                mu = self.M(chi)
            self.C = self.C + torch.einsum('j, k->jk', mu, nu)

        # Normalize C
        self.C_normalized = self.C

        # Compute bias h
        self.h = torch.tanh(self.memory[-1][1] * self.beta)/2 * self.h * 0.001

class MNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size, device):
        super(MNetwork, self).__init__()
        self.L1 = torch.nn.Linear(input_size, output_size).to(device)
        self.L2 = torch.nn.Linear(output_size, output_size).to(device)
        self.L3 = torch.nn.Linear(output_size, output_size).to(device)
        self.activation = torch.nn.ReLU().to(device)

    def forward(self, x):
        x = self.L1(x)
        x = self.activation(x)
        x = self.L2(x)
        x = self.activation(x)
        x = self.L3(x)

        return x

# DEP with deep prediction model that can handle batches
class BatchedDEP(DEPDeepModel):
    """
    Implements a batched version of DEP for use in a Neural Network
    """
    def __init__(self, tau, kappa, beta, sigma, delta_t, device, action_size, observation_size):
        super(BatchedDEP, self).__init__(tau, kappa, beta, sigma, delta_t, device, action_size, observation_size)
        self._batch_size = None

    def step(self, x):
        """
        Does a single DEP step

        args:
            x (array of size (batch_size, self._observation_size)): Current state

        returns:
            y (tensor of size (batch_size, self._action_size)): DEP action corresponding to x
        """
        # Convert x to a tensor if it isn't already
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self._device)

        # Set batch size
        self._batch_size = x.shape[0]

        # Smooth the observation x
        if self.x_smoothed is None:
            self.x_smoothed = x
        else:
            self.x_smoothed = (x - self.x_smoothed) / 2

        # Learning step if there are enough memories in the buffer
        if len(self.memory) > 2 + self.delta_t:
            self._learn_controller()
        
        # Get action y
        if self.C == None:
            y = torch.tanh(torch.rand((self._batch_size, self._action_size))).to(self._device)
        else:
            y = self._get_action()

        # Add state action pair (x, y) to memory
        self.memory.append((self.x_smoothed, y))

        # Update timestep and return action
        self._timestep += 1
        return y
    
    def _learn_controller(self):
        """
        Updates the controller matrix based on the state x
        """
        # Compute the controller matrix C
        self.C = torch.zeros((self._batch_size, self._action_size, self._observation_size)).to(self._device)

        update_set = range(1, min(self.tau, self._timestep - self.delta_t))
        for s in update_set:
            chi = self.memory[-s][0] - self.memory[-(s+1)][0]
            nu = self.memory[-(s+self.delta_t)][0] - self.memory[-(s+self.delta_t+1)][0]
            with torch.no_grad():
                mu = self.M(chi)
            self.C = self.C + torch.einsum('ij, ik->ijk', mu, nu)

        # Normalize C
        self.C_normalized = self.C

        # Compute bias h
        self.h = torch.tanh(self.memory[-1][1] * self.beta)/2 * self.h * 0.001

    def _get_action(self):
        """
        Forward pass for the one-layer DEP network (controller matrix with tanh activation)
        """
        q = torch.einsum('ijk, ik -> ij', self.C_normalized, self.x_smoothed)
        q = q / (torch.norm(q) + 1e-5)
        y = torch.tanh(self.kappa * q + self.h) * self.sigma
        return y
