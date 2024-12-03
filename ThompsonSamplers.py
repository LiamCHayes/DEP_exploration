"""
Classes for Thompson samplers"""

import numpy as np
from collections import deque

class ThomsonSampling:
    """
    Class to do Thomson sampling on a single parameter
    """
    def __init__(self, arms, window_size):
        self.arms = arms
        self.n_arms = len(arms)
        self.window_size = window_size
        self.arm_distributions = [deque(maxlen=window_size) for _ in range(self.n_arms)]

    def select_arm(self):
        """
        Sample expected reward from each arm and select the arm with the highest expected reward
        """
        # Sample each arm 
        arm_reward = np.empty(self.n_arms)
        for i, a in enumerate(self.arms):
            arm_distribution = self.arm_distributions[i]
            if len(arm_distribution) == 0:
                arm_reward[i] = np.random.normal(33, 13, 1).item()
            else:
                mean = np.mean(arm_distribution)
                std = np.std(arm_distribution)
                arm_reward[i] = np.random.normal(mean, std, 1).item()
        
        # Choose the one with the highest reward
        arm_idx = np.argmax(arm_reward)
        return self.arms[arm_idx]

    def update(self, arm, reward):
        """
        Observe the reward and update the prior of the arm
        """
        arm_idx = np.where(self.arms == arm)[0].item()
        self.arm_distributions[arm_idx].append(reward)

## TODO ## adapt this to multivariate
class MVThomsonSampling:
    """
    Class to do Thomson sampling on a multiple parameters
    """
    def __init__(self, arms, window_size):
        self.arms = arms
        self.n_arms = len(arms)
        self.window_size = window_size
        self.arm_distributions = [deque(maxlen=window_size) for _ in range(self.n_arms)]

    def select_arm(self):
        """
        Sample expected reward from each arm and select the arm with the highest expected reward
        """
        # Sample each arm 
        arm_reward = np.empty(self.n_arms)
        for i, a in enumerate(self.arms):
            arm_distribution = self.arm_distributions[i]
            if len(arm_distribution) == 0:
                arm_reward[i] = np.random.normal(33, 13, 1).item()
            else:
                mean = np.mean(arm_distribution)
                std = np.std(arm_distribution)
                arm_reward[i] = np.random.normal(mean, std, 1).item()
        
        # Choose the one with the highest reward
        arm_idx = np.argmax(arm_reward)
        return self.arms[arm_idx]

    def update(self, arm, reward):
        """
        Observe the reward and update the prior of the arm
        """
        arm_idx = np.where(self.arms == arm)[0].item()
        self.arm_distributions[arm_idx].append(reward)