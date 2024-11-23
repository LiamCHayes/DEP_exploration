"""
Pure DEP control of the cheetah
"""

import numpy as np
import torch
from dm_control import suite
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from DEP import DEP
from utils import make_video, see_live
from utils import print_and_pause

# Load environment and DEP controller
env = suite.load(domain_name="cheetah", task_name="run")

tau = 13
kappa = 1
beta = 0.1
sigma = 0.5
delta_t = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_size = env.action_spec().shape[0]
observation_size = env.observation_spec()['position'].shape[0]
dep_controller = DEP(tau, kappa, beta, sigma, delta_t, device, action_size, observation_size)

# Initialize lists to track DEP
frames = []
weights = []

# Do simulation
num_steps = 200
time_step = env.reset()
for t in tqdm(range(num_steps)):
    # Get action and do it
    observation = time_step.observation['position']
    action = dep_controller.step(observation)
    time_step = env.step(action)

    # Render and capture frame
    frame = env.physics.render()
    frames.append(frame)

    # Save DEP weights and biases
    if dep_controller.C_normalized is not None:
        w = dep_controller.C_normalized.cpu().numpy()
        weights.append(w)
    
    # Break if episode is over
    if time_step.last():
        break

# Visualize
see_live(frames)
see_live(weights)
