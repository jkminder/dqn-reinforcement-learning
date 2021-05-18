import argparse

import gym
import torch
import json
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
parser.add_argument('--evaluation_episodes', type=int, default=10, help='Number of evaluation episodes.', nargs='?')
parser.add_argument('--stats', type=str, default=None, help='Path to statistics folder', nargs='?')
parser.add_argument('--models', type=str, default=None, help='Path to model folder', nargs='?')
parser.add_argument('--episodes', type=int, default=None, help='Number of training episodes.', nargs='?')
parser.add_argument('--pretrained', type=str, default=None, help='Path to a pretrained model, that is used as starting point.', nargs='?')
parser.add_argument('--training_state_path', type=str, default="./", help='Path to the dir where the training state should be saved.', nargs='?')
parser.add_argument('--recover_training_state', help='Will recover last training state', action='store_true')
parser.add_argument('--persist', help="If activated a model will be regularly saved (the performance of it doesn't matter)", action='store_true')

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


class TrainingState:
    """
    This class represents the current state of the traininig and carries important information
    about the progress. It can be used to retake training, when it was interrupted
    """
    def __init__(self, path="./", model_path=None, statistics=None, persist=True):
        """
        :param path: Where to save the training state
        :param model_path: Path to the model directory
        :param statistics: Statistics objectt
        :param persist: If True the model will be saved regularly (every eval freq)
        """

        self.model_path = model_path
        self.stats = statistics
        self.path = path
        self.steps_done = -1
        self.episode = 0
        self.persist = persist
        if model_path is None and persist:
            raise Exception("If persist is activated you must provide a model path")

    def update(self, steps_done, episode):
        self.episode = episode
        self.steps_done = steps_done

    def save(self):
        data = {
            "episode": self.episode,
            "steps_done": self.steps_done,
            "model_path": self.model_path,
            "persist": self.persist
        }
        if self.stats is not None:
            data["statistics_path"] = self.stats.filepath

        with open(path.join(self.path, "training_state.json"), 'w') as outfile:
            json.dump(data, outfile)

    @staticmethod
    def create(state_dir):
        with open(path.join(state_dir, "training_state.json")) as json_file:
            data = json.load(json_file)
            stats = Statistics.load(data["statistics_path"])
            training_config = TrainingState(state_dir, data["model_path"], stats, bool(data["persist"]))
            training_config.episode = data["episode"]+1
            training_config.steps_done = data["steps_done"]
            training_config.model_path = data["model_path"]
            print(f"Loaded training state - episode: {training_config.episode}, steps: {training_config.steps_done}")

            return training_config


def train(env, eval_episodes, eval_freq, env_config, training_state, pretrained_path = None , verbose = 2):
    if verbose == 1:
        end = "\r"
    elif verbose >= 2:
        end = "\n"

    stats = training_state.stats

    # Initialize action mapping, if available
    action_map = env_config.get('action_map')

    # Initialize deep Q-networks.
    if pretrained_path is not None:
        # Load a pretrained model
        print(f"Loaded pretrained model {pretrained_path}")
        dqn = torch.load(pretrained_path, map_location=device)
    else:
        dqn = create_network(env.spec.id, env_config).to(device)
    dqn.train()

    # update dqn training state
    dqn.steps_done = training_state.steps_done

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

    for episode in range(training_state.episode, env_config['n_episodes']):
        iteration = 0
        done = False
        reward_sum = 0
        # preprocessing of first observation
        obs = preprocess(env.reset(), env=env.spec.id)

        # handle statistics and training state
        training_state.episode = episode
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

        # Update training state
        training_state.steps_done = dqn.steps_done

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

                if training_state.model_path is not None:
                    if verbose > 1:
                        print('Best performance so far! Saving model.')
                    torch.save(dqn, f'{training_state.model_path}/{env.spec.id}_best.pt')

            # Persist model if required
            if training_state.persist:
                torch.save(dqn, f'{training_state.model_path}/{env.spec.id}_curr.pt')

        # Save stats and training state after each episode
        if stats and stats.filepath is not None:
            stats.save()
        training_state.save()

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
    stats = None
    if args.stats is not None:
        stats_columns = []
        stats_columns.extend(["episode", "iterations", "reward", "eval_mean_return"])
        stats = Statistics(stats_columns, path.join(args.stats, "train.csv"))


    # Initialize training state
    if args.recover_training_state:
        training_state = TrainingState.create(args.training_state_path)
        pretrained = f'{training_state.model_path}/{env.spec.id}_curr.pt' if training_state.model_path and not args.pretrained else args.pretrained
        if not path.isfile(pretrained):
            pretrained = None
    else:
        training_state = TrainingState(path=args.training_state_path, model_path=args.models, statistics=stats, persist=args.persist)
        pretrained = args.pretrained

    # Start training
    total_rewards, steps = train(env, args.evaluation_episodes, args.evaluate_freq,
                                 env_config, training_state, pretrained_path=pretrained, verbose=3)

    if stats is not None:
        stats.save()

    # Close environment after training is completed.
    env.close()
