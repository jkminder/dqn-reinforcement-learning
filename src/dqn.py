import random
from collections import namedtuple, deque
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", "obs action next_obs reward")

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward):
        action = torch.tensor([[action]], device=device)
        reward = torch.tensor([reward], device=device)
        #done = torch.tensor(done, device=device)
        self.memory.append(Transition(obs, action, next_obs, reward))

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

    def get_epsilon(self):
        self.steps_done += 1
        if self.anneal_type == 'exp':
            return self.eps_end + (self.eps_start  -  self.eps_end) * \
                        math.exp(-1. * self.steps_done /  self.anneal_length)
        elif self.anneal_type == 'lin':
            if self.steps_done > self.anneal_length:
                return self.eps_end
            else:
                return self.eps_start - (self.eps_start - self.eps_end) * (self.steps_done / self.anneal_length)
        else:
            raise NotImplmentedError

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        if not exploit:
            # only non exploiting calls will anneal epsilon
            eps = self.get_epsilon()
        else:
            eps = 0
        if random.random() > eps or exploit:
            with torch.no_grad():
                return self(observation).max(1)[1].view(1,1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

def optimize(dqn, target_dqn, memory, optimizer, grad_clip = False):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # Sample batch
    batch = memory.sample(dqn.batch_size)

    obs_batch = torch.cat(batch.obs).to(device)
    actions_batch = torch.cat(batch.action).to(device)
    non_final_next_obs_batch = torch.cat([obs for obs in batch.next_obs if obs is not None]).to(device)
    rewards_batch = torch.cat(batch.reward).to(device)

    # create Terminated Mask
    term_mask = torch.tensor(tuple(map(lambda obs: obs is not None, batch.next_obs)))

    # Calculate q values
    q_values = dqn(obs_batch).gather(1, actions_batch)

    # Calculate q value targets
    non_final_qt_max = target_dqn(non_final_next_obs_batch).max(1)[0].detach()

    q_values_targets = torch.zeros(dqn.batch_size, device=device)
    q_values_targets[term_mask] = non_final_qt_max
    q_values_targets = (dqn.gamma * q_values_targets) + rewards_batch

    # Compute loss.
    loss = F.mse_loss(q_values, q_values_targets.unsqueeze(1))

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
