"""
A neural network with DEP layers
"""

from typing import Any, Mapping
import torch
import torch.nn as nn
from collections import deque
import random

from DEP import DEP, BatchedDEP
from ThompsonSamplers import ThomsonSampling

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

# Network that controls when to turn on DEP
class ActorControlsDEP(nn.Module):
    """
    A neural network that can cotnrol when DEP is being turned on. 

    """
    def __init__(self, position_size, velocity_size, action_size):
        super(ActorControlsDEP, self).__init__()

        # DEP controller
        self.dep_controller = DEPLayer(position_size, action_size)
        self.dep_controller._batch_size = 1

        # DEP threshold sampler and memory
        arms = torch.linspace(0, 1, 10)
        self.thresh_sampler = ThomsonSampling(arms, window_size=100, prior_mean=10)
        self.threshold = None
        self.dep_choice = None
        self.current_state = None
        self.memory = self.ReplayBuffer(256)

        # Network that chooses whether DEP gets turned on 
        self.decide_dep = nn.Sequential(
            nn.Linear(position_size+velocity_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to(device)
        self.decide_dep_optimizer = torch.optim.Adam(self.decide_dep.parameters(), lr = 0.001)
        self.dep_target = nn.Sequential(
            nn.Linear(position_size+velocity_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to(device)

        # Network that predicts an action based on positions and velocities
        self.decide_action = nn.Sequential(
            nn.Linear(position_size+velocity_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_size),
            nn.Tanh()
        ).to(device)
        self.action_target = nn.Sequential(
            nn.Linear(position_size+velocity_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_size),
            nn.Tanh()
        ).to(device)

        # Timesteps/countdowns
        self.dep_countdown = 0
        self.t = 0

    def forward(self, positions, velocities):
        """
        First step determines whether DEP gets turned on or not
        If DEP gets turned on:
            Calculate for how long and do DEP for that long
        Else:
            Predict an action
        """
        # Update timestep
        self.t += 1

        # Do DEP to store in the memory
        dep_action = self.dep_controller.step(positions)

        # Get and store current state
        self.current_state = torch.concat((positions, velocities), dim=0)

        if self.dep_countdown == 0:
            # Decide if we will do DEP
            dep_prob = self.decide_dep(self.current_state)
            self.threshold = self.thresh_sampler.select_arm()

            self.dep_choice = int(dep_prob > self.threshold)

            if self.dep_choice == 1:
                self.dep_countdown = 5 # In the future, make a network or sampler to tune this
            else:
                # Do network action
                action = self.decide_action(self.current_state)
                action = action.unsqueeze(0)
        
        # Return DEP action if the network says so
        if self.dep_countdown > 0:
            self.dep_choice = 1
            action = dep_action
            self.dep_countdown -= 1

        return action, self.dep_choice
    
    def only_network(self, state):
        """
        Only forward pass on the actiond decision network
        """
        action = self.decide_action(state)

        return action

    def update_dep_network(self):
        """
        Updates the policy and target networks for the dep decision
        """
        batch_size = self.memory.len() if self.memory.len() < 32 else 32
        batch = self.memory.sample(batch_size)

        # Based on the current states
        current_q = self.decide_dep(batch[0])

        # Based on the next states
        t_q = self.dep_target(batch[3]) > self.threshold
        target_q = batch[2] + 0.95 * t_q

        # Compute loss
        loss = torch.nn.functional.mse_loss(current_q, target_q)

        # Update step
        self.decide_dep_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.decide_dep.parameters(), 1)
        self.decide_dep_optimizer.step()
    
    def update_memories(self, reward, next_state, done):
        """
        Updates memory and the sampler for DEP threshold
        """
        # Update sampler
        self.thresh_sampler.update(self.threshold, reward)

        # Update memory
        self.memory.add(self.current_state, self.dep_choice, reward, next_state, done)

    def update_targets(self):
        """
        Updates targets for the networks
        """
        self.action_target.load_state_dict(self.decide_action.state_dict())
        self.dep_target.load_state_dict(self.decide_dep.state_dict())

    def only_dep(self, observation):
        """
        Forward pass on only DEP
        """
        dep_action = self.dep_controller.step(observation)
        return dep_action

    def reset(self):
        """
        Resets the DEP controller
        """
        self.dep_controller.reset()

    class ReplayBuffer:
        def __init__(self, maxlen):
            self.memory = deque(maxlen=maxlen)

        def add(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))

        def sample(self, batch_size):
            batch = random.sample(self.memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to tensors
            actions = [torch.tensor(a, dtype=torch.float32).to(device) for a in actions]
            rewards = [torch.tensor(r, dtype=torch.float32).to(device) for r in rewards]
            dones = [torch.tensor(d, dtype=torch.int32).to(device) for d in dones]

            return torch.stack(states), torch.stack(actions), torch.stack(rewards), torch.stack(next_states), torch.stack(dones)

        def len(self):
            return len(self.memory)
