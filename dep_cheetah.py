"""
Pure DEP control of the cheetah
"""

import numpy as np
import torch
from dm_control import suite
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from DEP import DEP, DEPDeepModel
from utils import make_video, see_live
from utils import print_and_pause

# Load environment and DEP controller
env = suite.load(domain_name="cheetah", task_name="run")

tau = 40
kappa = 1000
beta = 0.002
sigma = 1
delta_t = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_size = env.action_spec().shape[0]
observation_size = env.observation_spec()['position'].shape[0]
dep_controller = DEP(tau, kappa, beta, sigma, delta_t, device, action_size, observation_size)
dep_controller.set_model(torch.load("dep_backprop_results/init_ep10_model_matrix.pt"))

# Initialize lists to track DEP
frames = []
weights = []

# Do simulation
num_steps = 500
time_step = env.reset()
for t in tqdm(range(num_steps)):
    # Get action and do it
    observation = time_step.observation['position']
    action = dep_controller.step(observation)
    time_step = env.step(action)

    # Adjust camera positon
    agent_pos = env.physics.named.data.xpos['torso']
    env.physics.named.data.cam_xpos['side'][0] = agent_pos[0]
    
    # Render frame and save
    frame = env.physics.render(camera_id = 'side')
    frames.append(frame)

    # Save DEP weights
    if dep_controller.C_normalized is not None:
        w = dep_controller.C_normalized.cpu().detach().numpy()
        weights.append(w)
    
    # Break if episode is over
    if time_step.last():
        break

# Visualize
see_live(frames)
see_live(weights)
