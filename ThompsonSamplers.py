"""
Classes for Thompson samplers"""

import numpy as np
from collections import deque

class ThomsonSampling:
    """
    Class to do Thomson sampling on a single parameter

    args :
        arms (list of values): List of possible arm values we want to sample
        window_size (int): size of previous interactions we want to take into account
    """
    def __init__(self, arms, window_size, prior_mean):
        self.arms = arms
        self.n_arms = len(arms)
        self.window_size = window_size
        self.arm_distributions = [deque(maxlen=window_size) for _ in range(self.n_arms)]
        self.prior_mean = prior_mean

    def select_arm(self):
        """
        Sample expected reward from each arm and select the arm with the highest expected reward
        """
        # Sample each arm 
        arm_reward = np.empty(self.n_arms)
        for i, a in enumerate(self.arms):
            arm_distribution = self.arm_distributions[i]
            if len(arm_distribution) == 0:
                arm_reward[i] = np.random.normal(self.prior_mean, 1, 1).item()
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

    args:
        arms (list of numpy arrays): list of arms to tune over, all arrays in the list must be the same size
        window_size (int): size of previous interactions we want to take into account

    attributes:
        arms (list of numpy arrays)
        n_arms (int): size of numpy arrays in arms
        v_vars (int): number of arrays in arms
        window_size (int): size of previous interactions we want to take into account
    """
    def __init__(self, arms, window_size):
        self.arms = arms
        self.n_arms = len(arms[0])
        self.n_vars = len(arms)
        self.window_size = window_size
        self.arm_distributions = [[deque(maxlen=window_size) for _ in range(self.n_arms)] for _ in range(self.n_vars)]

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

if __name__ == '__main__':
    arms = [np.arange(4), np.arange(4)]
    sampler = MVThomsonSampling(arms, window_size=10)