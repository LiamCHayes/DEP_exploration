"""
Use RL on the deep model matrix to find the most useful basin of attraction
"""

import numpy as np
from dm_control import suite
import torch
from tqdm import tqdm
import argparse
from learning_utils import DataTrackers, Memories
import time

from DEP import DEPDeepModel, MNetwork
from non_DEP_networks import SimpleCritic
from utils import make_video, see_live, print_and_pause

# Argparser
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='Name of this run', required=True)
    parser.add_argument('-e', '--episodes', help='Number of episodes to train for', required=True)
    args = parser.parse_args()
    return args

args = argparser()

# Load env
env = suite.load(domain_name="cheetah", task_name="run")

# Controllers, networks, and optimizers
tau = 8
kappa = 0.5
beta = 0.0025
sigma = 5.25
delta_t = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_size = env.action_spec().shape[0]
observation_size = env.observation_spec()['position'].shape[0]

dep_controller = DEPDeepModel(tau, kappa, beta, sigma, delta_t, device, action_size, observation_size)
dep_target = DEPDeepModel(tau, kappa, beta, sigma, delta_t, device, action_size, observation_size)
dep_target.M.load_state_dict(dep_controller.M.state_dict())

critic = SimpleCritic(action_size, observation_size)
critic_target = SimpleCritic(action_size, observation_size)
critic_target.load_state_dict(critic.state_dict())

lr = 1e-3
dep_optim = torch.optim.Adam(dep_controller.M.parameters(), lr=lr)
critic_optim = torch.optim.Adam(critic.parameters(), lr=lr)

# Training loop things
metrics = DataTrackers.TrainingLoopTracker("rewards", "actor_loss", "critic_loss")
num_episodes = int(args.episodes)
num_steps = 500
progress_report_freq = 10
update_freq = 50
batch_size = 512
gamma = 0.95

# Replay buffer
memory = Memories.ReplayBuffer(100000)

print("\nFilling replay buffer...")
while memory.len() <= batch_size:
    # Reset episode things
    time_step = env.reset()
    dep_controller.reset()

    for t in range(num_steps):
        # Get action and do it
        observation = time_step.observation['position']
        action = dep_controller.step(observation)
        time_step = env.step(action)

        # Store experience in memory
        if len(dep_controller.memory) == tau + delta_t:
            reward = time_step.reward
            next_observation = time_step.observation['position']
            done = time_step.last()
            dep_memory = dep_controller.memory
            memory.add(observation, action, reward, next_observation, done, dep_memory)

# Training loop
for e in range(num_episodes):
    print("\nEpisode ", e)
    
    # Reset episode things
    time_step = env.reset()
    dep_controller.reset()
    total_reward = 0
    total_actor_loss = 0
    total_critic_loss = 0

    # Determine if we are reporting or not
    if e % progress_report_freq == 0:
        reporting = True
        frames = []
    else:
        reporting = False

    # Run the episode
    for t in tqdm(range(num_steps)):
        # Get action and do it
        observation = time_step.observation['position']
        action = dep_controller.step(observation)
        time_step = env.step(action)
        etime = time.time()

        # Store experience in memory
        if len(dep_controller.memory) == tau + delta_t:
            reward = time_step.reward
            next_observation = time_step.observation['position']
            done = time_step.last()
            dep_memory = dep_controller.memory
            memory.add(observation, action, reward, next_observation, done, dep_memory)
        etime = time.time()

        # Render and capture frame if we are making a progress report
        if reporting:
            # Adjust camera
            agent_pos = env.physics.named.data.xpos['torso']
            env.physics.named.data.cam_xpos['side'][0] = agent_pos[0]

            # Render and save frame
            frame = env.physics.render(camera_id = 'side')
            frames.append(frame)
            etime = time.time()

        ## Learning step
        # Sample from memories
        states, actions, rewards, next_states, dones, dep_memories = memory.sample(batch_size)

        # Get batch of target actions
        target_actions = []
        for i, m in enumerate(dep_memories):
            dep_target.memory = m
            target_actions.append(dep_target.step(next_states[i]))
        
        # Convert states to network inputs
        next_states_ntw = torch.stack([torch.tensor(n, dtype=torch.float32).to(device) for n in next_states])
        states_ntw = torch.stack([torch.tensor(s, dtype=torch.float32).to(device) for s in states])
        actions_ntw = torch.stack([torch.tensor(a, dtype=torch.float32).to(device) for a in actions])
        target_actions_ntw = torch.stack([torch.tensor(t, dtype=torch.float32).to(device) for t in target_actions])
        dones = torch.stack([torch.tensor(d, dtype=torch.float32).to(device) for d in dones])
        rewards = torch.stack([torch.tensor(r, dtype=torch.float32).to(device) for r in rewards])

        # Compute critic loss
        target_q = critic_target(next_states_ntw, target_actions_ntw)
        q_targets = rewards + (1-dones) * gamma * target_q.squeeze(1)
        q_values = critic(states_ntw, actions_ntw).squeeze(1)
        critic_loss = torch.nn.MSELoss()(q_values, q_targets)

        # Update critic
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        # Compute actor loss
        # Need to use a copy of the dep controller to not mess up its memory
        dep_controller_copy = DEPDeepModel(tau, kappa, beta, sigma, delta_t, device, action_size, observation_size)
        dep_controller_copy.M.load_state_dict(dep_controller.M.state_dict())
        real_actions = []
        for i, m in enumerate(dep_memories):
            dep_controller_copy.memory = m
            real_actions.append(dep_controller_copy.step(states[i]))
        real_actions_ntw = torch.stack([torch.tensor(r, dtype=torch.float32).to(device) for r in real_actions])
        actor_loss = -critic(states_ntw, real_actions_ntw).mean()
        
        # Update actor
        dep_optim.zero_grad()
        actor_loss.backward()
        dep_optim.step()

        # Update targets every once in a while
        if e % update_freq == 0:
            critic_target.load_state_dict(critic.state_dict())
            dep_target.M.load_state_dict(dep_controller.M.state_dict())

        ## End learning step

        # Update reward and loss
        total_reward += time_step.reward
        total_actor_loss += actor_loss.item()
        total_critic_loss += critic_loss.item()

        # Break if episode is over
        if time_step.last():
            break

    # Data tracking
    print('Total reward: ', total_reward)
    print('Total actor loss: ', total_actor_loss)
    print('Total critic loss: ', total_critic_loss)
    metrics.update(total_reward, total_actor_loss, total_critic_loss)
    metrics.save_metrics(f"RL_model_matrix/{args.name}/metrics.csv")

    # Make a progress report video once in a while
    if reporting:
        see_live(frames)
        make_video(frames, f"RL_model_matrix/{args.name}/ep{e}_progress_report")
        torch.save(dep_controller.M, f'RL_model_matrix/{args.name}/ep{e}_model_matrix.pt')
    
    if e == num_episodes-1:
        torch.save(dep_controller.M, f'RL_model_matrix/{args.name}/final_model_matrix.pt')
