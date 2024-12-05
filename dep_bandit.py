"""
Tuning DEP parameters with Thompson sampling
"""

import numpy as np
import torch
from dm_control import suite
import torch
from collections import deque
from tqdm import tqdm
import pandas as pd

from DEP import DEP
from ThompsonSamplers import ThomsonSampling
from utils import make_video, see_live
from utils import print_and_pause

# Load environment
env = suite.load(domain_name="cheetah", task_name="run")

# Invariant DEP args
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_size = env.action_spec().shape[0]
observation_size = env.observation_spec()['position'].shape[0]
tau = 8
beta = 0.0025
sigma = 5.25
delta_t = 1

# Thompson sampling
kappa_arms = np.concatenate((np.linspace(0, 1, 5), np.arange(2, 5)))

# Training loop
num_episodes = 100
kappa_sampler = ThomsonSampling(kappa_arms, 50)
kappas = []
rewards = []
for e in range(num_episodes):
    print(f'\nEpisode {e}')

    # DEP controller
    tau = int(tau)
    delta_t = int(delta_t)
    kappa = kappa_sampler.select_arm()
    dep_controller = DEP(tau, kappa, beta, sigma, delta_t, device, action_size, observation_size)

    # Loop things
    num_steps = 500
    num_samples = 5

    # Run simulation num_samples times per parameter combination
    tot_rew = 0
    time_step = env.reset()
    for t in range(num_steps):
        # Get action and do it
        observation = time_step.observation['position']
        action = dep_controller.step(observation)
        time_step = env.step(action)

        # Measure reward
        tot_rew += time_step.reward

        # Break if episode is over
        if time_step.last():
            break
    
    # Update samplers
    mean_rew = tot_rew / num_samples
    kappa_sampler.update(kappa, mean_rew)

    # Print and save metrics
    print(f'Kappa: {kappa}')
    print(f'Mean reward: {mean_rew}')

    kappas.append(kappa)
    rewards.append(mean_rew)

# Save metrics 
data = np.array([kappas, rewards])
cols=['kappa', 'reward']
df = pd.DataFrame(data).transpose()
df.columns = cols
df.to_csv(f'thompson_sampling_results/kappa_sampling.csv')  
