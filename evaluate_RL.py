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
from non_DEP_networks import SimpleActor
from dep_networks import DEPActor
from utils import make_video, see_live
from utils import print_and_pause

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load environment and DEP controller
env = suite.load(domain_name="cheetah", task_name="run")
action_size = env.action_spec().shape[0]
observation_size = env.observation_spec()['position'].shape[0]

model = DEPActor(observation_size, action_size, 0.001)
weights = torch.load("deep_dep_results/test/model_matrix.pth")
model.load_state_dict(weights)

# Initialize lists to track DEP
frames = []

# Do simulation
num_steps = 500
time_step = env.reset()
for t in tqdm(range(num_steps)):
    # Get action and do it
    observation = time_step.observation['position']
    observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
    action = model(observation_tensor)
    action = action.cpu().detach().numpy()
    time_step = env.step(action)

    # Adjust camera positon
    agent_pos = env.physics.named.data.xpos['torso']
    env.physics.named.data.cam_xpos['side'][0] = agent_pos[0]
    
    # Render frame and save
    frame = env.physics.render(camera_id = 'side')
    frames.append(frame)
    
    # Break if episode is over
    if time_step.last():
        break

# Visualize
see_live(frames)
