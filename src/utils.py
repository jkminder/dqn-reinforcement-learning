import random

import gym
import torch
import torch.nn as nn

import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(obs, env, prev=None):
    """
    Performs necessary observation preprocessing. Use of prev_stack is required for Pong-v0, since we used
    observation stacking.
    """
    if env in ['CartPole-v0', 'CartPole-v1']:
        return torch.tensor(obs, device=device).float().unsqueeze(0)
    elif env in ['Pong-v0']:
        # Normalise image to 0-1
        obs = torch.tensor(obs, device=device).float().unsqueeze(0)
        obs /= 255
        # Create obs stack
        if prev is None:
            # Initial stack
            obs_stack = torch.cat(config.Pong["obs_stack_size"] * [obs]).unsqueeze(0).to(device)
        else:
            # Continuous stack is created by updating the prev stack
            obs_stack = torch.cat((prev[:, 1:, ...], obs.unsqueeze(1)), dim=1).to(device)
        return obs_stack
    else:
        raise ValueError('Please add necessary observation preprocessing instructions to preprocess() in utils.py.')
