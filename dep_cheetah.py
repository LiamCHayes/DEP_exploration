"""
Pure DEP control of the cheetah
"""

from dm_control import suite
import numpy as np

# Load environment
env = suite.load(domain_name="cheetah", task_name="run")

# Do simulation
action_spec = env.action_spec()
time_step = env.reset()
frames = []
while not time_step.last():
    # Get action and do it
    action = None ## TODO ## Get DEP action
    time_step = env.step(action)

    # Render and capture frame
    frame = env.physics.render()
    frames.append(frame)

