import random
from collections import namedtuple, deque
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", "obs action next_obs reward done")

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward, done):
        """Add elements to memory"""
        self.memory.append(Transition(obs.squeeze(), action, next_obs.squeeze(), reward, done))

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a Transition
        """
        sample = random.sample(self.memory, batch_size)
        return Transition(*zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.anneal_type = env_config["anneal_type"]
        self.n_actions = env_config["n_actions"]

        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.steps_done = -1


    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def load_state(self, other):
        self.load_state_dict(other.state_dict())

    def get_epsilon(self):
        if self.anneal_type == 'exp':
            return self.eps_end + (self.eps_start  -  self.eps_end) * \
                        math.exp(-1. * self.steps_done /  self.anneal_length)
        elif self.anneal_type == 'lin':
            if self.steps_done >= self.anneal_length:
                return self.eps_end
            else:
                return self.eps_start - (self.eps_start - self.eps_end) * (self.steps_done / self.anneal_length)
        else:
            raise NotImplementedError

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # Compute exploiting actions
        with torch.no_grad():
            q_values = self.forward(observation)
        actions = torch.argmax(q_values, dim=1)

        # Add exploration
        # only non exploiting calls will anneal epsilon
        if not exploit:
            eps = self.get_epsilon()
            roll = torch.rand(observation.shape[0])
            exploration_indices = roll < eps
            actions[exploration_indices] = torch.randint(high=self.n_actions, size=[sum(exploration_indices)], device=device)
            self.steps_done += observation.shape[0]
        return actions

def optimize(dqn, target_dqn, memory, optimizer, grad_clip = False):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # Sample batch
    batch = memory.sample(dqn.batch_size)

    obs = torch.stack(batch.obs).to(device)
    actions = torch.tensor(batch.action, device=device)
    next_obs = torch.stack(batch.next_obs).to(device)
    reward = torch.tensor(batch.reward, device=device)
    terminal = torch.tensor(batch.done, device=device)

    all = dqn.forward(obs)
    actions_b = actions.unsqueeze(1)
    q_values = all.gather(1, actions_b).squeeze()

    max_next_q_values = target_dqn.forward(next_obs).max(1)[0]
    q_value_targets = reward + ~terminal * target_dqn.gamma * max_next_q_values

    # Compute loss.
    loss = F.mse_loss(q_values, q_value_targets)

    # Perform gradient descent.
    # Zeros gradients before running backwards pass
    optimizer.zero_grad()

    loss.backward()

    # Gradient Clipping
    if grad_clip:
        for param in dqn.parameters():
            param.grad.data.clamp_(-1, 1)

    optimizer.step()

    return loss.item()
