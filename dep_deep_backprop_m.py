"""
Use gradient descent to find the best M for maximizing the reward
"""

from dm_control import suite
import torch
from tqdm import tqdm
import pandas as pd
import argparse

from DEP import DEPDeepModel
from utils import make_video, see_live
from utils import print_and_pause

# Argparser
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='Name of this run', required=True)
    parser.add_argument('-e', '--episodes', help='Number of episodes to train for', required=True)
    args = parser.parse_args()
    return args

args = argparser()

# Load environment and dep controller
env = suite.load(domain_name="cheetah", task_name="run")

# Load up DEP controller
tau = 40
kappa = 1000
beta = 0.002
sigma = 1
delta_t = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_size = env.action_spec().shape[0]
observation_size = env.observation_spec()['position'].shape[0]

dep_controller = DEPDeepModel(tau, kappa, beta, sigma, delta_t, device, action_size, observation_size)

# Optimizer things
lr = 0.01
optimizer = torch.optim.Adam(dep_controller.M.parameters(), lr=lr)

# Training loop variables
episode_reward = []
episode_loss = []
num_episodes = int(args.episodes)
num_steps = 300
progress_report_freq = 10

# Training loop
for e in range(num_episodes):
    print("\nEpisode ", e)

    # Reset episode things
    time_step = env.reset()
    dep_controller.reset()
    total_reward = 0
    total_loss = 0

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

        # Render and capture frame if we are making a progress report
        if reporting:
            # Adjust camera
            agent_pos = env.physics.named.data.xpos['torso']
            env.physics.named.data.cam_xpos['side'][0] = agent_pos[0]

            # Render and save frame
            frame = env.physics.render(camera_id = 'side')
            frames.append(frame)

        # Update model matrix if there has been a learning step
        if dep_controller.C is not None:
            # Zero grad
            optimizer.zero_grad()

            # Compute loss 
            prev_action = dep_controller.memory[-delta_t][1]
            Mx = dep_controller.M(torch.tensor(observation, dtype=torch.float32).to(device))
            loss = torch.sum((Mx - prev_action)**2)

            # Update network
            loss.backward()
            optimizer.step()
        else:
            loss = 0

        # Update reward and loss
        total_reward += time_step.reward
        total_loss += loss

        # Break if episode is over
        if time_step.last():
            break

    # Episode ended things
    print('Total reward: ', total_reward)
    print('Total loss: ', total_loss.item())
    episode_reward.append(total_reward)
    episode_loss.append(total_loss)

    # Make a progress report video once in a while
    if reporting:
        make_video(frames, f"dep_deep_backprop_results/{args.name}/ep{e}_progress_report")
        torch.save(dep_controller.M, f'dep_deep_backprop_results/{args.name}/ep{e}_model_matrix.pt')

# Save episode rewards and losses
data = [episode_reward, episode_loss]
cols = ['reward', 'loss']
df = pd.DataFrame(data)
df.transpose()
df.columns = cols
df.to_csv(f'dep_deep_backprop_results/{args.name}/metrics.csv')  

# Save DEP parameters
data = [tau, kappa, beta, sigma, delta_t]
cols = ['tau', 'kappa', 'beta', 'sigma', 'delat_t']
df = pd.DataFrame(data, columns=cols, index=False)
df.to_csv(f'dep_deep_backprop_results/{args.name}/dep_parameters.csv')

# Save model matrix
torch.save(dep_controller.M, f'dep_deep_backprop_results/{args.name}/model_matrix.pt')
