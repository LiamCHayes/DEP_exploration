"""
Utility functions
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Visualize the episode
def make_video(frames, video_name):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_name = f'videos/{video_name}.avi'
    fps = 30

    out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()

def see_live(weights_list):
    fig, ax = plt.subplots()
    im = ax.imshow(weights_list[0], cmap='viridis', animated=True)
    for m in weights_list:
        im.set_array(m)
        plt.draw()
        plt.pause(0.02)


def print_and_pause(variable, message=""):
    print(f"\n{message}")
    print(variable)
    input("\nPrinted and paused")
