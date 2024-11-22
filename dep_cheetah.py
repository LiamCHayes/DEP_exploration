"""
Pure DEP control of the cheetah
"""

import numpy as np
import torch
from dm_control import suite

from DEP import DEP
from utils import make_video
from utils import print_and_pause

# Load environment and DEP controller
env = suite.load(domain_name="cheetah", task_name="run")

tau = 40
kappa = 1000
beta = 0.002
sigma = 1
delta_t = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_size = env.action_spec().shape[0]
observation_size = env.observation_spec()['position'].shape[0]
dep_controller = DEP(tau, kappa, beta, sigma, delta_t, device, action_size, observation_size)

# Do simulation
time_step = env.reset()
frames = []
while not time_step.last():
    # Get action and do it
    observation = time_step.observation['position']
    action = dep_controller.step(observation)
    time_step = env.step(action)

    # Render and capture frame
    frame = env.physics.render()
    frames.append(frame)

make_video(frames, "init_run")

