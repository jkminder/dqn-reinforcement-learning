import argparse

import gym
import torch
import torch.nn as nn
import copy
from os import path, mkdir

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize
from statistics import Statistics
from train import train, TrainingState
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0', 'CartPole-v1'])
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')
parser.add_argument('--training_runs', type=int, default=10, help='Number of training runs per configuration.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole,
    'CartPole-v1': config.CartPole
}

# Grid is currently hardcoded
# TODO: Move to param or external file
# train_freq, target_update_freq, anneal_length, batch_size, lr
params = ["train_frequency", "target_update_frequency", "anneal_length", "batch_size", "lr"]
grid = [
    (1,100,10**4,32,1e-4),(2,100,10**4,32,1e-4),(3,100,10**4,32,1e-4),(4,100,10**4,32,1e-4),(5,100,10**4,32,1e-4), # train_freq
    (1,50,10**4,32,1e-4), (1,100,10**4,32,1e-4),(1,150,10**4,32,1e-4), (1,200,10**4,32,1e-4), (1,250,10**4,32,1e-4), (1,300,10**4,32,1e-4), (1,350,10**4,32,1e-4), # target_update_freq
    (1,30,10**4*(1/2),32,1e-4), (1,30,10**4*2,32,1e-4), # anneal length
    (1,100,10**4,32,1e-2), (1,100,10**4,32,1e-3), (1,100,10**4,32,1e-5), (1,100,10**4,32,1e-6), # lr
    (1,100,10**4,64,1e-4), (1,100,10**4,128,1e-4) #batxh size
]
grid_it = 0

def update_config(env_config):
    global grid, grid_it

    print("Updated Configs:           ")
    for i, key in enumerate(params):
        env_config[key] = grid[grid_it][i]
        print(f" - {key}:", grid[grid_it][i], end = "")

    print()

    grid_it += 1

    return env_config


if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]


    # Initialize statistics and training state
    training_state = TrainingState(path="./", model_path="./models", statistics="./statistics",
                                   persist=False)

    stats_columns = []
    stats_columns.extend(params)
    stats_columns.extend(["episode","iterations", "mean_return", "loss"])
    stats = Statistics(stats_columns, "./grid_search.csv")

    training_state.stats = stats

    # Start training
    while (grid_it < len(grid)):
        env_config = update_config(env_config)

        for i in range(args.training_runs):
            print(f"## Training run {i+1}/10             ")
            train(env, args.evaluation_episodes, args.evaluate_freq, env_config, training_state, verbose=1)

        # save it each iteration
        stats.save("grid_search.csv")
    # Close environment after training is completed.
    env.close()
