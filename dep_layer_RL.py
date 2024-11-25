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

from dep_networks import FirstLayerDEP
from critic_networks import SimpleCritic

from utils import make_video

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load env
env = suite.load(domain_name="cheetah", task_name="run")

# Networks and learning parameters
action_size = env.action_spec().shape[0]
observation_size = env.observation_spec()['position'].shape[0]

actor = FirstLayerDEP(observation_size, action_size)
critic = SimpleCritic(action_size, observation_size)

actor_target = FirstLayerDEP(observation_size, action_size)
actor_target.load_state_dict(state_dict = actor.state_dict(), dep_state_dict = actor.DEPlayer.M.state_dict())
critic_target = SimpleCritic(action_size, observation_size)
critic_target.load_state_dict(critic.state_dict())

lr = 0.01
actor_adam = torch.optim.Adam(actor.parameters(), lr=lr)
critic_adam = torch.optim.Adam(critic.parameters(), lr=lr)

# Replay buffer
class ReplayBuffer:
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)

    def add(self, state, action, dep_output, reward, next_state, done):
        self.memory.append((state, action, dep_output, reward, next_state, done))

    def sample(self, batch_size):
        # Sample
        batch = random.sample(self.memory, batch_size)
        states, actions, dep_outputs, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = [torch.tensor(s, dtype=torch.float32).to(device) for s in states]
        actions = [torch.tensor(a, dtype=torch.float32).to(device) for a in actions]
        dep_outputs = [torch.tensor(d, dtype=torch.float32).to(device) for d in dep_outputs]
        rewards = [torch.tensor(r, dtype=torch.float32).to(device) for r in rewards]
        next_states = [torch.tensor(n, dtype=torch.float32).to(device) for n in next_states]
        dones = [torch.tensor(d, dtype=torch.int32).to(device) for d in dones]

        return torch.stack(states), torch.stack(actions), torch.stack(dep_outputs), torch.stack(rewards), torch.stack(next_states), torch.stack(dones)
    
    def len(self):
        return len(self.memory)

# Training loop variables
episode_reward = []
episode_loss = []
num_episodes = 2001
num_steps = 500
progress_report_freq = 100
memory = ReplayBuffer(maxlen=1000)
batch_size = 32
gamma = 0.95
update_freq = 50

# Fill the replay buffer with DEP experiences
while memory.len() < batch_size:
    print("\nFilling replay buffer...")
    time_step = env.reset()
    actor.reset_dep()

    for t in range(num_steps):
        # Get action and do it
        observation = time_step.observation['position']
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
        action = actor.only_dep(observation_tensor).cpu().detach().numpy()
        time_step = env.step(action)

        # Store experience in memory
        memory.add(observation, action, action, time_step.reward, time_step.observation['position'], time_step.last())

# Training loop
for e in range(num_episodes):
    print('\nEpisode ', e)

    # Reset episode things
    time_step = env.reset()
    actor.reset_dep()
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
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
        action, dep_output = actor(observation_tensor)
        action = action.cpu().detach().numpy()
        dep_output = dep_output.cpu().detach().numpy()
        time_step = env.step(action)

        # Render and capture frame if we are making a progress report
        if reporting:
            # Adjust camera
            agent_pos = env.physics.named.data.xpos['torso']
            env.physics.named.data.cam_xpos['side'][0] = agent_pos[0]

            # Render and save frame
            frame = env.physics.render(camera_id = 'side')
            frames.append(frame)

        # Update the DEP inverse prediction model
        dep_model_loss = actor.learn_dep_model(observation)

        # Store experience in memory
        memory.add(observation, action, dep_output, time_step.reward, time_step.observation['position'], time_step.last())

        # Update actor and critic
        states, actions, dep_outputs, rewards, next_states, dones = memory.sample(batch_size)

        target_actions = actor_target.forward_no_step(next_states, dep_outputs)
        target_q_values = critic_target(next_states, target_actions)
        q_targets = rewards + (1 - dones) * gamma * target_q_values.squeeze(1)
        q_values = critic(states, actions).squeeze(1)

        critic_loss = torch.nn.MSELoss()(q_values, q_targets)

        critic_adam.zero_grad()
        critic_loss.backward()
        critic_adam.step()

        actor_loss = -critic(states, actor.forward_no_step(states, dep_outputs)).mean()

        actor_adam.zero_grad()
        actor_loss.backward()
        actor_adam.step()

        # Update targets every once in a while
        if e % update_freq == 0:
            critic_target.load_state_dict(critic.state_dict())
            actor_target.load_state_dict(actor.state_dict(), actor.DEPlayer.M.state_dict())

        # Update stats
        total_actor_loss += actor_loss.item()
        total_critic_loss += critic_loss.item()
        total_reward += time_step.reward

    # Print episode stats
    print("Episode: ", e)
    print("Reward: ", total_reward)
    print("Actor Loss: ", total_actor_loss)
    print("Critic Loss: ", total_critic_loss)

    # Make a progress report video once in a while
    if reporting:
        make_video(frames, f"dep_layer_results/init/ep{e}_progress_report")
        torch.save(actor.state_dict(), f'dep_layer_results/init/ep{e}_model_matrix.pth')

# Save metrics 
data = np.array([episode_reward, episode_loss])
cols=['reward', 'loss']
df = pd.DataFrame(data).transpose()
df.columns = cols
df.to_csv('dep_layer_results/init/metrics.csv')  

# Save model
make_video(frames, f"dep_layer_results/init/ep{e}_progress_report")
torch.save(actor.state_dict(), 'dep_layer_results/init/model_matrix.pth')
