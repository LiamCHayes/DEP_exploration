"""
Unbatched DEP agent implementation
"""

# Imports and Setup
import torch
from collections import deque

# Classes and functions
class DEP:
    """
    Implements an unbatched DEP controller in pytorch

    args:
        tau (int): Number of previous timesteps to update the controller with
        kappa (float): Normalization scaling factor
        delta_t (int): Number of timesteps to compute the change in state over
        _device: Device to store computations in (CPU or GPU)

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
    """
    def __init__(self, tau, kappa, delta_t, _device):
        # Scalars
        self.tau = tau
        self.kappa = kappa
        self.delta_t = delta_t

        # Matrices
        self.C = None
        self.C_normalized = None
        self.M = None

        # Memory
        self.memory = deque(maxlen = self.tau + self.delta_t)
        self.x_smoothed = None

        # Internal attributes
        self._timestep = 0
        self._device = _device

    def step(self, x):
        """
        Does a single DEP step

        args:
            x: Current state

        returns:
            y: DEP action corresponding to x
        """
        # Convert x to a tensor if it isn't already
        if isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Smooth the observation x
        if self.x_smoothed is None:
            self.x_smoothed = x
        else:
            self.x_smoothed = (x - self.x_smoothed) / 2

        # Learning step
        self._learn_controller()
        
        # Get action y
        y = self._get_action()

        # Add state action pair (x, y) to memory
        self.memory.append((x, y))

        pass

    def _learn_controller(self):
        """
        Updates the controller matrix based on the state x
        """
        # Compute the controller matrix C

        # Normalize C 

        # Compute bias h

        pass

    def _get_action(self):
        """
        Forward pass for the one-layer DEP network (controller matrix with tanh activation)
        """
        pass