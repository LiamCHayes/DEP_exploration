"""
Grid search over parameters for the cheetah
"""

import numpy as np
import torch
from dm_control import suite
import torch
from tqdm import tqdm
import itertools
import pandas as pd

from DEP import DEP
from utils import make_video, see_live
from utils import print_and_pause

# Load environment
env = suite.load(domain_name="cheetah", task_name="run")

# Initialize DEP args that we are not tuning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_size = env.action_spec().shape[0]
observation_size = env.observation_spec()['position'].shape[0]

# Create grid to search over
tau = np.linspace(1, 50, 5, dtype=np.int32)
kappa = [1, 10, 100, 1000, 10000]
beta = np.linspace(0, 0.01, 5)
sigma = np.linspace(0.5, 10, 5)
delta_t = np.arange(1, 6, dtype=np.int32)

grid = list(itertools.product(tau, kappa, beta, sigma, delta_t))

# Initialize list to store reward data
avg_reward = []

# Do grid search
for param_set in tqdm(grid):
    # DEP controller
    tau, kappa, beta, sigma, delta_t = param_set
    tau = int(tau)
    delta_t = int(delta_t)
    dep_controller = DEP(tau, kappa, beta, sigma, delta_t, device, action_size, observation_size)

    # Loop things
    num_steps = 300
    num_samples = 5

    # Run simulation num_samples times per parameter combination
    tot_rew = 0
    for s in range(num_samples):
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

    avg_reward.append(tot_rew / num_samples)

# Make dataframe and save it
data = np.array(list(zip(*grid))).T
data = np.hstack((data, np.array(avg_reward)[np.newaxis, :].T))
cols = ['tau', 'kappa', 'beta', 'sigma', 'delta_t']
df = pd.DataFrame(data, columns=cols)
df.to_csv('metrics/grid_rewards.csv')
