from typing import List, Dict, Any
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

import wandb
import gymnasium as gym
from gymnasium import Env

import athlete
from athlete.agent import Agent

# These imports register the according algorithms
import td3_see
import sac_see

# This import registers the custom environments
import environments

import configuration

#######################################################
#######################################################
### SCROLL DOWN TO THE BOTTOM FOR THE MAIN FUNCTION ###
#######################################################
#######################################################


def average_agent_info(
    agent_infos: List[Dict[str, Any]], prefix: str = ""
) -> Dict[str, float]:
    """Averages a list of dictionaries grouped by their keys. Only considers int and float values.

    Args:
        agent_infos (List[Dict[str, Any]]): A list of dictionaries containing agent information.
        prefix (str, optional): Prefix to add to the keys of the averaged dictionary. Defaults to "".

    Returns:
        Dict[str, float]: A dictionary containing the averaged values for each key, optionally with additional prefix.
    """
    averaged_info = {}
    collected_values = defaultdict(list)
    for info in agent_infos:
        for key, value in info.items():
            if isinstance(value, (int, float)):  # Only collect numeric values
                collected_values[key].append(value)

    # Then calculate averages for each key
    for key, values in collected_values.items():
        if values:  # Make sure we don't divide by zero
            averaged_info[prefix + key] = sum(values) / len(values)

    return averaged_info


def run_experiment(
    env: Env,
    agent: Agent,
    training_duration: int,
    eval_freq: int,
    num_eval_episodes: int,
) -> None:
    evaluation_env = deepcopy(env)

    """
    Run an experiment with the given environment and agent.
    We assume a wandb session has been initialized.

    Args:
        env (Env): The environment to run the experiment in.
        agent (Agent): The agent to use in the experiment.
        training_duration (int): The total duration of the training in environment steps.
        eval_freq (int): The frequency of evaluation according to environment steps.
        eval_duration (int): The duration of each evaluation in environment steps.
        seed (Optional[int]): The random seed for the experiment. Defaults to None.
    """
    # Initialize the environment and agent
    observation, _ = env.reset()
    action, agent_info = agent.reset_step(observation)

    rewards = []
    agent_infos = []

    for training_step in tqdm(range(training_duration)):
        observation, reward, terminated, truncated, _ = env.step(action)
        action, agent_info = agent.step(observation, reward, terminated, truncated)

        rewards.append(reward)
        agent_infos.append(agent_info)

        if training_step % eval_freq == 0:
            # set agent to eval mode
            agent.eval()
            evaluation_logging_dict = evaluate_agent(
                evaluation_env,
                agent,
                num_eval_episodes=num_eval_episodes,
            )
            # set agent back to train mode
            agent.train()

            if not (terminated or truncated):
                # Because wandb can't log twice in the same step
                wandb.log(evaluation_logging_dict, step=training_step)

        if terminated or truncated:
            observation, _ = env.reset()
            action, agent_info = agent.reset_step(observation)

            # logging
            _return = sum(rewards)
            episode_length = len(rewards)
            log_agent_info = average_agent_info(agent_infos, prefix="training/")

            logging_dict = {
                "training/return": _return,
                "training/episode_length": episode_length,
                **log_agent_info,
            }

            if training_step % eval_freq == 0:
                # Because wandb can't log twice in the same step
                logging_dict.update(evaluation_logging_dict)

            wandb.log(
                logging_dict,
                step=training_step,
            )

            rewards = []
            agent_infos = []


def evaluate_agent(
    evaluation_env: Env,
    agent: Agent,
    num_eval_episodes: int,
) -> Dict[str, float]:
    """
    Evaluate the agent in the given environment.

    Args:
        evaluation_env (Env): The environment to evaluate the agent in.
        agent (Agent): The agent to evaluate.
        num_eval_episodes (int): The number of evaluation episodes.

    Returns:
        Dict[str, float]: A dictionary containing logging information such as average return and episode length.
    """

    returns = []
    episode_lengths = []
    agent_infos = []

    for episode in range(num_eval_episodes):
        rewards = []
        observation, _ = evaluation_env.reset()
        action, agent_info = agent.reset_step(observation)
        agent_infos.append(agent_info)
        done = False

        while not done:
            observation, reward, terminated, truncated, _ = evaluation_env.step(action)
            action, agent_info = agent.step(observation, reward, terminated, truncated)
            rewards.append(reward)
            agent_infos.append(agent_info)

            done = terminated or truncated

        returns.append(sum(rewards))
        episode_lengths.append(len(rewards))

    log_agent_info = average_agent_info(agent_infos, prefix="evaluation/")

    logging_dict = {
        "evaluation/return": sum(returns) / len(returns),
        "evaluation/episode_length": sum(episode_lengths) / len(episode_lengths),
        **log_agent_info,
    }

    return logging_dict


def create_environment(environment_name: str, reward_setting: str) -> Env:
    """
    Base on the environment name and reward setting, create the actual environment following predefined configurations.

    Args:
        environment_name (str): The name of the environment to create.
        reward_setting (str): The reward setting to use.

    Returns:
        Env: The created environment.
    """

    search_key = (environment_name, reward_setting)
    if search_key not in configuration.environment_map:
        raise ValueError(
            f"Invalid environment name and reward setting combination: {search_key}. "
            f"Choose from {list(configuration.environment_map.keys())}."
        )
    env = gym.make(configuration.environment_map[search_key])
    return env


def create_agent(environment_name: str, env: Env, method: str) -> Agent:
    """
    Create an agent based on the environment and method following predefined configurations.

    Args:
        environment (Env): The environment to use.
        method (str): The method to use.

    Returns:
        Agent: The created agent.
    """
    search_key = (method, environment_name)
    if search_key not in configuration.algorithm_map:
        raise ValueError(
            f"Invalid method and environment combination: {search_key}. "
            f"Choose from {list(configuration.algorithm_map.keys())}."
        )
    algorithm_id, hyperparameters = configuration.algorithm_map[search_key]
    agent = athlete.make(
        observation_space=env.observation_space,
        action_space=env.action_space,
        algorithm_id=algorithm_id,
        **hyperparameters,
    )

    return agent


def setup_and_run_experiment(
    environment_name: str,
    reward_setting: str,
    method: str,
) -> None:
    """
    Set up and run an experiment with the given environment, reward setting, and method.

    Args:
        environment_name (str): The name of the environment to use. Choose from:
            "Pendulum-v1", "LocalOptimumCar-v0", "HalfCheetah-v5", "Ant-v5", "Hopper-v5", "Swimmer-v5", "FetchPickAndPlace-v4", "LargePointMaze-v3"
        reward_setting (str): The reward setting to use. Choose from:
            "dense", "sparse", "adverse"
        method (str): The method to use. Choose from:
            "SAC", "TD3", "SAC+SEE", "TD3+SEE", "SAC+SEE w/o maximum update", "SAC+SEE w/o conditioning", "SAC+SEE w/o mixing", "TD3+SEE w/o maximum update", "TD3+SEE w/o conditioning", "TD3+SEE w/o mixing"
    """

    if environment_name not in configuration.environments:
        raise ValueError(
            f"Invalid environment name: {environment_name}. Choose from {configuration.environments}."
        )
    if reward_setting not in configuration.reward_settings:
        raise ValueError(
            f"Invalid reward setting: {reward_setting}. Choose from {configuration.reward_settings}."
        )
    if method not in configuration.methods:
        raise ValueError(
            f"Invalid method: {method}. Choose from {configuration.methods}."
        )

    environment = create_environment(
        environment_name=environment_name,
        reward_setting=reward_setting,
    )
    agent = create_agent(
        environment_name=environment_name,
        env=environment,
        method=method,
    )

    training_duration, eval_freq, num_eval_episodes = (
        configuration.experiment_settings_map[environment_name]
    )

    # Initialize wandb for logging
    wandb.init(
        project="SEE",
        name=f"{method}_{reward_setting}_{environment_name}",
    )

    run_experiment(
        env=environment,
        agent=agent,
        training_duration=training_duration,
        eval_freq=eval_freq,
        num_eval_episodes=num_eval_episodes,
    )


if __name__ == "__main__":
    """Modify this to run specific experiments
    Choices
    environment_name: "Pendulum-v1", "LocalOptimumCar-v0", "HalfCheetah-v5", "Ant-v5", "Hopper-v5", "Swimmer-v5", "FetchPickAndPlace-v4", "LargePointMaze-v3",
    reward_setting: "dense", "sparse", "adverse"
    method: "SAC", "TD3", "SAC+SEE", "TD3+SEE", "SAC+SEE w/o maximum update", "SAC+SEE w/o conditioning", "SAC+SEE w/o mixing", "TD3+SEE w/o maximum update", "TD3+SEE w/o conditioning", "TD3+SEE w/o mixing"
    """
    setup_and_run_experiment(
        environment_name="Pendulum-v1",
        reward_setting="adverse",
        method="SAC+SEE",
    )
