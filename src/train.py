import argparse

import gym
import torch
import numpy as np
from os import path, mkdir
import random

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, QN, ReplayMemory, optimize
from statistics import Statistics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0', 'CartPole-v1', 'Pong-v0'])
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')
parser.add_argument('--stats', type=str, default=None, help='Path to statistics folder', nargs='?')
parser.add_argument('--models', type=str, default=None, help='Path to model folder', nargs='?')
parser.add_argument('--episodes', type=int, default=None, help='Number of training episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole,
    'CartPole-v1': config.CartPole,
    'Pong-v0': config.Pong
}
def create_network(env_name, env_config):
    if env_name in ['CartPole-v0', 'CartPole-v1']:
        return QN(env_config=env_config)
    elif env_name in ['Pong-v0']:
        return DQN(env_config=env_config)

def create_env(env_name):
    env = gym.make(env_name)
    if env_name in ['CartPole-v0', 'CartPole-v1']:
        return env
    elif env_name in ['Pong-v0']:
        # Import required wrapper
        from gym.wrappers import AtariPreprocessing
        return AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)

def train(env, eval_episodes, eval_freq, env_config, stats = None, save_stats = None, save_model = None, verbose = 2):
    if verbose == 1:
        end = "\r"
    elif verbose >= 2:
        end = "\n"
    # Initialize action mapping, if available
    action_map = env_config.get('action_map')

    # Initialize deep Q-networks.
    dqn = create_network(env.spec.id, env_config).to(device)
    dqn.train()

    # Create and initialize target Q-network.
    target_dqn = create_network(env.spec.id, env_config).to(device)
    target_dqn.load_state(dqn)
    target_dqn.eval()

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    total_rewards = np.zeros(env_config['n_episodes'])
    steps = np.zeros(env_config['n_episodes'])

    for episode in range(env_config['n_episodes']):
        iteration = 0
        done = False
        reward_sum = 0
        # preprocessing of first observation
        obs = preprocess(env.reset(), env=env.spec.id)

        if stats:
            stats.start_episode()
            stats.log_config(env_config)
            stats.log("episode", episode)

        while not done:
            if verbose > 2 and iteration % 100 == 0:
                print(f"Episode {episode}/{env_config['n_episodes']} Iteration {iteration}      ", end="\r")

            # Get action from DQN.
            action = dqn.act(obs, exploit=False).item()

            # Remapping actions if needed
            if action_map is not None:
                env_action = action_map[action]
            else:
                env_action = action
            obs_prev = obs

            # Act in the true environment.
            obs, reward, done, info = env.step(env_action)

            reward_sum += reward

            # Preprocess incoming observation.
            obs = preprocess(obs, env=env.spec.id, prev=obs_prev)

            # Add the transition to the replay memory. Remember to convert
            memory.push(obs_prev, action, obs, reward, done)

            # Run optimization every env_config["train_frequency"] steps.
            if iteration % env_config['train_frequency'] == 0:
                loss = optimize(dqn, target_dqn, memory, optimizer, env_config['grad_clip'])
                if stats:
                    stats.log_iteration("loss", loss)

            # Update the target network every env_config["target_update_frequency"] steps.
            if iteration % env_config['target_update_frequency'] == 0:
                target_dqn.load_state(dqn)

            iteration += 1

        if stats:
            stats.log("iterations", iteration)
            stats.log("reward", reward_sum)

        total_rewards[episode] = reward_sum
        steps[episode] = iteration

        # Evaluate the current agent.
        if episode % eval_freq == 0 or episode == env_config['n_episodes']-1:
            mean_return = evaluate_policy(dqn, env, env_config, n_episodes=eval_episodes)

            if stats:
                stats.log("eval_mean_return", mean_return)
            if verbose > 0:
                print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}              ', end=end)

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                if verbose > 1:
                    print('Best performance so far! Saving model.')

                if save_model is not None:
                    torch.save(dqn, f'{save_model}/{env.spec.id}_best.pt')

        # Save stats after each episode
        if stats and save_stats is not None:
            stats.save(stats_path)

    if stats:
        stats.finalize()

    return total_rewards, steps


if __name__ == '__main__':
    print(f"Training on device: {device}")
    args = parser.parse_args()

    # Initialize environment and config.
    env = create_env(args.env)
    env_config = ENV_CONFIGS[args.env]

    # Update config if required
    if args.episodes is not None:
        env_config['n_episodes'] = args.episodes

    # Make things reproducible.
    torch.manual_seed(seed=0)
    random.seed(a=0)
    env.seed(seed=0)

    # Initialize statistics
    stats_columns = []
    stats_columns.extend(["episode","iterations", "reward", "eval_mean_return"])
    stats = Statistics(stats_columns)
    stats_path = ""
    if args.stats is None:
        stats = None
    else:
        stats_path = path.join(args.stats, "train.csv")

    # Start training
    total_rewards, steps = train(env, args.evaluation_episodes, args.evaluate_freq, env_config,
                                 stats=stats, save_stats=stats_path, save_model=args.models, verbose=3)

    if stats is not None:
        stats.save(stats_path)

    # Close environment after training is completed.
    env.close()
