"""
Deep reinforcement learning with DEP as part of the network input
"""

from dm_control import suite
import torch
from collections import deque
import numpy as np
from tqdm import tqdm
import random
import pandas as pd
import argparse

from non_DEP_networks import SimpleCritic
from dep_networks import DEPActor

from utils import make_video, print_and_pause

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Networks, DEP, and learning parameters
action_size = env.action_spec().shape[0]
observation_size = env.observation_spec()['position'].shape[0]

actor = DEPActor(observation_size, action_size, 1e-5)
critic = SimpleCritic(action_size, observation_size)

actor_target = DEPActor(observation_size, action_size, 1e-3)
actor_target.load_state_dict(state_dict = actor.state_dict())
critic_target = SimpleCritic(action_size, observation_size)
critic_target.load_state_dict(critic.state_dict())

lr = 1e-4
actor_adam = torch.optim.Adam(actor.parameters(), lr=lr)
critic_adam = torch.optim.Adam(critic.parameters(), lr=lr)

# Replay buffer
class ReplayBuffer:
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)

    def add(self, state, action, batched_obs, reward, next_state, done): 
        self.memory.append((state, action, batched_obs, reward, next_state, done))

    def sample(self, batch_size):
        # Sample
        batch = random.sample(self.memory, batch_size)
        states, actions, batched_obss, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = [torch.tensor(s, dtype=torch.float32).to(device) for s in states]
        actions = [torch.tensor(a, dtype=torch.float32).to(device) for a in actions]
        batched_obss = [torch.tensor(o, dtype=torch.float32).to(device) for o in batched_obss]
        rewards = [torch.tensor(r, dtype=torch.float32).to(device) for r in rewards]
        next_states = [torch.tensor(n, dtype=torch.float32).to(device) for n in next_states]
        dones = [torch.tensor(d, dtype=torch.int32).to(device) for d in dones]

        return torch.stack(states), torch.stack(actions), torch.stack(batched_obss), torch.stack(rewards), torch.stack(next_states), torch.stack(dones)
    
    def len(self):
        return len(self.memory)

# Training loop variables
episode_reward = []
episode_actor_loss = []
episode_critic_loss = []
num_episodes = int(args.episodes)
num_steps = 500
progress_report_freq = 1000
memory = ReplayBuffer(maxlen=1000000)
batch_size = 512
gamma = 0.95
update_freq = 75
dep_probability = 0.75
dep_length = 10
dep_countdown = 0

# Fill the replay buffer with DEP experiences
while memory.len() < batch_size:
    print("\nFilling replay buffer...")
    time_step = env.reset()

    for t in range(num_steps):
        # Get action and do it
        observation = time_step.observation['position']
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
        action = actor.only_dep(observation_tensor)
        action = action.cpu().detach().numpy()
        time_step = env.step(action)

        # Store experience in memory
        batched_obs = actor.batch_observations()
        if batched_obs.shape[0] == actor.batch_size:
            memory.add(observation, action, batched_obs, time_step.reward, time_step.observation['position'], time_step.last())

# Training loop
for e in range(num_episodes):
    print('\nEpisode ', e)

    # Reset episode things
    time_step = env.reset()
    actor.reset()
    total_reward = 0
    total_actor_loss = 0
    total_critic_loss = 0
    if e % 500 == 0:
        dep_probability -= 0.01

    # Determine if we are reporting or not
    if e % progress_report_freq == 0:
        reporting = True
        frames = []
    else:
        reporting = False

    # Run the episode
    for t in range(num_steps):
        # Get the observation
        observation = time_step.observation['position']

        # Determine whether we want to do DEP or actor network action
        if dep_countdown == 0:
            start_dep = bool(np.random.binomial(1, dep_probability, 1))
            dep_countdown = dep_length if start_dep else 0

        # Do DEP action or network action
        if start_dep:
            action = actor.only_dep(observation)
            dep_countdown -= 1
        else:            
            observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
            action = actor.forward(observation_tensor)
        
        # Do the action
        action = action.cpu().detach().numpy()
        time_step = env.step(action)

        # Store experience in memory
        batched_obs = actor.batch_observations()
        if batched_obs.shape[0] == actor.batch_size:
            memory.add(observation, action, batched_obs, time_step.reward, time_step.observation['position'], time_step.last())

        # Render and capture frame if we are making a progress report
        if reporting:
            # Adjust camera
            agent_pos = env.physics.named.data.xpos['torso']
            env.physics.named.data.cam_xpos['side'][0] = agent_pos[0]

            # Render and save frame
            frame = env.physics.render(camera_id = 'side')
            frames.append(frame)

        # Update actor and critic
        states, actions, batched_obss, rewards, next_states, dones = memory.sample(batch_size)

        target_actions = actor_target.only_network(batched_obss)
        target_q_values = critic_target(next_states, target_actions)
        q_targets = rewards + (1 - dones) * gamma * target_q_values.squeeze(1)
        q_values = critic(states, actions).squeeze(1)

        critic_loss = torch.nn.MSELoss()(q_values, q_targets)

        critic_adam.zero_grad()
        critic_loss.backward()
        critic_adam.step()

        actor_loss = -critic(states, actor.only_network(batched_obss)).mean()

        actor_adam.zero_grad()
        actor_loss.backward()
        actor_adam.step()

        # Update targets every once in a while
        if e % update_freq == 0:
            critic_target.load_state_dict(critic.state_dict())
            actor_target.load_state_dict(actor.state_dict())

        # Update stats
        total_actor_loss += actor_loss.item()
        total_critic_loss += critic_loss.item()
        total_reward += time_step.reward

    # Print episode stats
    print("Reward: ", total_reward)
    print("Actor Loss: ", total_actor_loss)
    print("Critic Loss: ", total_critic_loss)
    episode_actor_loss.append(total_actor_loss)
    episode_critic_loss.append(total_critic_loss)
    episode_reward.append(total_reward)

    # Make a progress report video once in a while
    if reporting:
        make_video(frames, f"deep_dep_results/{args.name}/ep{e}_progress_report")
        torch.save(actor.state_dict(), f'deep_dep_results/{args.name}/ep{e}_model_matrix.pth')

    # Save metrics 
    data = np.array([episode_reward, episode_actor_loss, episode_critic_loss])
    cols=['reward', 'actor_loss', 'critic_loss']
    df = pd.DataFrame(data).transpose()
    df.columns = cols
    df.to_csv(f'deep_dep_results/{args.name}/metrics.csv')  

# Save model
make_video(frames, f"deep_dep_results/{args.name}/ep{e}_progress_report")
torch.save(actor.state_dict(), f'deep_dep_results/{args.name}/model_matrix.pth')
