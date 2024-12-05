"""
Deep reinforcement learning with DEP as part of the network input
"""

from dm_control import suite
import torch
from collections import deque
import numpy as np
import random
import pandas as pd
import argparse

from non_DEP_networks import SimpleCritic
from dep_networks import ActorControlsDEP

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
position_size = env.observation_spec()['position'].shape[0]
velocity_size = env.observation_spec()['velocity'].shape[0]

actor = ActorControlsDEP(position_size, velocity_size, action_size)
actor_optimizer = torch.optim.Adam(actor.decide_action.parameters(), lr=0.001)

reward_net = SimpleCritic(action_size, position_size + velocity_size)
reward_net_optimizer = torch.optim.Adam(reward_net.parameters(), lr=0.001)

# Replay buffer
class ReplayBuffer:
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)

    def add(self, state, action, reward, next_state, done): 
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Sample
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        actions = [torch.tensor(a, dtype=torch.float32).to(device) for a in actions]
        rewards = [torch.tensor(r, dtype=torch.float32).to(device) for r in rewards]
        dones = [torch.tensor(d, dtype=torch.int32).to(device) for d in dones]

        return torch.stack(states), torch.stack(actions), torch.stack(rewards), torch.stack(next_states), torch.stack(dones)
    
    def len(self):
        return len(self.memory)

# Training loop variables
episode_reward = []
episode_actor_loss = []
episode_critic_loss = []
dep_choice_proportion = []
avg_dep_threshold = []
num_episodes = int(args.episodes)
num_steps = 500
progress_report_freq = 100
memory = ReplayBuffer(maxlen=10000)
batch_size = 32
gamma = 0.95
update_freq = 75

# Fill the replay buffer with DEP experiences
while memory.len() < batch_size:
    print("\nFilling replay buffer...")
    time_step = env.reset()
    actor.reset()

    for t in range(num_steps):
        # Get observation
        position = time_step.observation['position']
        velocity = time_step.observation['velocity']
        position = torch.tensor(position, dtype=torch.float32).to(device)
        velocity = torch.tensor(velocity, dtype=torch.float32).to(device)

        # Do dep action
        observation_tensor = position.unsqueeze(0).to(device)
        action = actor.only_dep(observation_tensor)
        action = action.cpu().detach().numpy()
        time_step = env.step(action)

        # Store experience in memory
        current_state = torch.concat((position, velocity))
        next_position = torch.tensor(time_step.observation['position'], dtype=torch.float32).to(device)
        next_velocity = torch.tensor(time_step.observation['velocity'], dtype=torch.float32).to(device)
        next_state = torch.concat((next_position, next_velocity))
        memory.add(current_state, action, time_step.reward, next_state, time_step.last())

# Training loop
for e in range(num_episodes):
    print('\nEpisode ', e)

    # Reset things for the episodes
    time_step = env.reset()
    actor.reset()
    total_reward = 0
    total_actor_loss = 0
    total_critic_loss = 0
    total_dep_choice = 0
    total_dep_threshold = 0

    # Determine if we are reporting or not
    if e % progress_report_freq == 0:
        reporting = True
        frames = []
    else:
        reporting = False

    # Run the episode
    for t in range(num_steps):
        # Get the observation
        position = time_step.observation['position']
        velocity = time_step.observation['velocity']
        position = torch.tensor(position, dtype=torch.float32).to(device)
        velocity = torch.tensor(velocity, dtype=torch.float32).to(device)

        # Do action
        with torch.no_grad():
            action, dep_choice = actor.forward(position, velocity)
        action = action.cpu().detach().numpy()
        time_step = env.step(action)

        # Store things in memory
        current_state = torch.concat((position, velocity))
        next_position = torch.tensor(time_step.observation['position'], dtype=torch.float32).to(device)
        next_velocity = torch.tensor(time_step.observation['velocity'], dtype=torch.float32).to(device)
        next_state = torch.concat((next_position, next_velocity))

        memory.add(current_state, action, time_step.reward, next_state, time_step.last())

        # Update dep network
        actor.update_memories(time_step.reward, next_state, time_step.last())
        actor.update_dep_network()
        if e % update_freq == 0:
            actor.update_targets()

        # Render and capture frame if we are making a progress report
        if reporting:
            # Adjust camera
            agent_pos = env.physics.named.data.xpos['torso']
            env.physics.named.data.cam_xpos['side'][0] = agent_pos[0]

            # Render and save frame
            frame = env.physics.render(camera_id = 'side')
            frames.append(frame)
        
        # Sample batch
        states, actions, rewards, next_states, dones = memory.sample(batch_size)

        # Update critic
        predicted_rewards = reward_net.forward(states, actions)
        critic_loss = torch.nn.MSELoss()(predicted_rewards, rewards)

        reward_net_optimizer.zero_grad()
        critic_loss.backward()
        reward_net_optimizer.step()

        # Update actor
        actor_loss = -reward_net.forward(states, actor.only_network(states)).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Update stats
        total_actor_loss += actor_loss.item()
        total_critic_loss += critic_loss.item()
        total_reward += time_step.reward
        total_dep_choice += dep_choice
        total_dep_threshold += actor.threshold

    # Print episode stats
    print("Reward: ", total_reward)
    print("Actor Loss: ", total_actor_loss)
    print("Critic Loss: ", total_critic_loss)
    episode_actor_loss.append(total_actor_loss)
    episode_critic_loss.append(total_critic_loss)
    episode_reward.append(total_reward)
    dep_choice_proportion.append(total_dep_choice/num_steps)
    avg_dep_threshold.append(total_dep_threshold/num_steps)

    # Make a progress report video once in a while
    if reporting:
        make_video(frames, f"dep_actor_results/{args.name}/ep{e}_progress_report")
        torch.save(actor.state_dict(), f'dep_actor_results/{args.name}/ep{e}_model_matrix.pth')

# Save metrics 
data = np.array([episode_reward, episode_actor_loss, episode_critic_loss, dep_choice_proportion, avg_dep_threshold])
cols=['reward', 'actor_loss', 'critic_loss', 'dep_proportion', 'dep_threshold']
df = pd.DataFrame(data).transpose()
df.columns = cols
df.to_csv(f'dep_actor_results/{args.name}/metrics.csv')  

# Save model
make_video(frames, f"dep_actor_results/{args.name}/ep{e}_progress_report")
torch.save(actor.state_dict(), f'dep_actor_results/{args.name}/model_matrix.pth')
