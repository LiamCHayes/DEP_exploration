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

actor = DEPActor(observation_size, action_size, 1e-3)

# Do simulation
num_episodes = 1001
num_steps = 500
time_step = env.reset()
episode_loss = []
for e in range(num_episodes):
    print('\nEpisode ', e)

    time_step = env.reset()
    actor.reset()
    total_loss = 0
    frames = []
    for t in tqdm(range(num_steps)):
        # Get action and do it
        observation = time_step.observation['position']
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
        action = actor.forward(observation_tensor)
        action = action.cpu().detach().numpy()
        time_step = env.step(action)

        if actor.im_loss is not None:
            total_loss += actor.im_loss.item()

        if e % 100 == 0:
            # Adjust camera positon
            agent_pos = env.physics.named.data.xpos['torso']
            env.physics.named.data.cam_xpos['side'][0] = agent_pos[0]
            
            # Render frame and save
            frame = env.physics.render(camera_id = 'side')
            frames.append(frame)
        
        # Break if episode is over
        if time_step.last():
            break

    episode_loss.append(total_loss)
    print('Loss: ', total_loss)

    if e % 100 == 0:
        # Visualize
        see_live(frames)

        # Save model
        torch.save(actor.state_dict(), 'deep_dep_results/imitation.pth')
