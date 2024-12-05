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
from dep_networks import DEPActor, ActorControlsDEP
from utils import make_video, see_live
from utils import print_and_pause

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load environment and DEP controller
env = suite.load(domain_name="cheetah", task_name="run")
action_size = env.action_spec().shape[0]
position_size = env.observation_spec()['position'].shape[0]
velocity_size = env.observation_spec()['velocity'].shape[0]

model = ActorControlsDEP(position_size, velocity_size, action_size)
weights = torch.load("dep_actor_results/init/model_matrix.pth")
model.load_state_dict(weights)

# Initialize lists to track DEP
frames = []

# Do simulation
num_steps = 500
time_step = env.reset()
for t in tqdm(range(num_steps)):
    # Get action and do it
    position = time_step.observation['position']
    velocity = time_step.observation['velocity']
    position_tensor = torch.tensor(position, dtype=torch.float32).to(device)
    velocity_tensor = torch.tensor(velocity, dtype=torch.float32).to(device)
    action, dep_choice = model.forward(position_tensor, velocity_tensor)
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
