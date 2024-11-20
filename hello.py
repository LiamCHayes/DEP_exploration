from dm_control import suite
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

print_domains = False

if print_domains:
    for domain_name, task_name in suite.BENCHMARKING:
        print("Domain name: ", domain_name)
        print("Task name: ", task_name)
        print("\n")

# Load environment
env = suite.load(domain_name="cheetah", task_name="run")

# Do simulation
action_spec = env.action_spec()
time_step = env.reset()
frames = []
while not time_step.last():
    # Get action and do it
    action = np.random.uniform(action_spec.minimum, action_spec.maximum, size = action_spec.shape)
    time_step = env.step(action)

    # Render and capture frame
    frame = env.physics.render()
    frames.append(frame)

    # Print metrics
    print(time_step.reward, time_step.discount, time_step.observation)

# Visualize the episode
height, width, layers = frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_name = 'videos/output_test.avi'
fps = 30

out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

for frame in frames:
    out.write(frame)

out.release()
cv2.destroyAllWindows()
