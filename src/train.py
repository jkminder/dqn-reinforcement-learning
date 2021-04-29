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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0', 'CartPole-v1'])
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole,
    'CartPole-v1': config.CartPole
}

def train(env, eval_episodes, eval_freq, env_config, stats = None, save_model = True, verbose = 2):
    if verbose == 1:
        end = "\r"
    elif verbose == 2:
        end = "\n"
    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    dqn.train()

    # Create and initialize target Q-network.
    target_dqn = DQN(env_config=env_config).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    for episode in range(env_config['n_episodes']):
        done = False

        obs = preprocess(env.reset(), env=env.spec.id).unsqueeze(0)

        iteration = 0

        if stats:
            stats.start_episode()
            stats.log("episode", episode)

        while not done:
            iteration += 1

            obs_prev = obs

            # Get action from DQN.
            action = dqn.act(obs)

            # Act in the true environment.
            obs, reward, done, info = env.step(action.item())

            # Preprocess incoming observation.
            if not done:
                obs = preprocess(obs, env=env.spec.id).unsqueeze(0)
            else:
                obs = None

            # Add the transition to the replay memory. Remember to convert
            memory.push(obs_prev, action, obs, reward)

            # Run optimization every env_config["train_frequency"] steps.
            if iteration % env_config['train_frequency'] == 0:
                loss = optimize(dqn, target_dqn, memory, optimizer, env_config['grad_clip'])
                stats.log_iteration("loss", loss)

            # Update the target network every env_config["target_update_frequency"] steps.
            if iteration % env_config['target_update_frequency'] == 0:
                target_dqn.load_state_dict(dqn.state_dict())

        stats.log("iterations", iteration)
        # Evaluate the current agent.
        if episode % eval_freq == 0 or episode == env_config['n_episodes']-1:
            mean_return = evaluate_policy(dqn, env, env_config, n_episodes=eval_episodes)

            if stats:
                stats.log("mean_return", mean_return)
            if verbose > 0:
                print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}', end=end)

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                if verbose > 1:
                    print('Best performance so far! Saving model.')


                if save_model:
                    # Test if models dir exists
                    if not path.isdir("models"):
                        mkdir('models')

                    torch.save(dqn, f'models/{env.spec.id}_best.pt')
    if stats:
        stats.finalize()

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]

    # Initialize statistics
    stats = Statistics({}, ["episode","iterations", "mean_return", "loss"])

    # Start training
    train(env, args.evaluation_episodes, args.evaluate_freq, env_config, stats)

    stats.save("test.csv")
    # Close environment after training is completed.
    env.close()
