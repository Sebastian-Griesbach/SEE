from typing import Dict, SupportsFloat, Optional

import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper, RewardWrapper

from gymnasium_robotics.envs.fetch.pick_and_place import MujocoFetchPickAndPlaceEnv
from gymnasium_robotics.envs.maze.point_maze import PointMazeEnv
from gymnasium_robotics.envs.maze import maps


# Wrapper and entry points for modified gymnasium robotics environments


class RewardPlusOne(RewardWrapper):
    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        return reward + 1.0


class GoalObservationConverter(ObservationWrapper):
    def __init__(
        self,
        env,
        observation_key: str = "observation",
        goal_key: str = "desired_goal",
        achieved_goal_key: Optional[
            str
        ] = None,  # only adds achieved goal if this key is given
    ):
        """Constructor for the observation wrapper.

        Args:
            env: Environment to be wrapped.
        """
        super().__init__(env)
        self.observation_key = observation_key
        self.goal_key = goal_key
        self.achieved_goal_key = achieved_goal_key
        # this only works for flat dimensions
        lows = [
            self.observation_space[self.observation_key].low,
            self.observation_space[self.goal_key].low,
        ]
        highs = [
            self.observation_space[self.observation_key].high,
            self.observation_space[self.goal_key].high,
        ]

        if self.achieved_goal_key is not None:
            lows.append(self.observation_space[self.achieved_goal_key].low)
            highs.append(self.observation_space[self.achieved_goal_key].high)

        combined_low = np.concatenate(lows)
        combined_high = np.concatenate(highs)
        combined_observation_space = gym.spaces.Box(
            low=combined_low,
            high=combined_high,
            dtype=self.observation_space[self.observation_key].dtype,
        )
        self.observation_space = combined_observation_space

    def observation(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        to_concatenate = [
            observation[self.observation_key],
            observation[self.goal_key],
        ]

        if self.achieved_goal_key is not None:
            to_concatenate.append(observation[self.achieved_goal_key])

        combined_observation = np.concatenate(to_concatenate)
        return combined_observation


def make_goal_observation_converter_wrapped_fetch_pick_and_place_env(**kwargs):
    env = MujocoFetchPickAndPlaceEnv(**kwargs)
    env = GoalObservationConverter(env)
    # Transforms pseudo sparse setting (-1 everywhere and 0 on goal, to true sparse setting (0 everywhere and 1 on goal)
    env = RewardPlusOne(env)
    return env


def make_goal_observation_converter_wrapped_dense_fetch_pick_and_place_env(**kwargs):
    env = MujocoFetchPickAndPlaceEnv(reward_type="dense", **kwargs)
    env = GoalObservationConverter(env)
    return env


def make_goal_observation_converter_wrapped_adverse_fetch_pick_and_place_env(**kwargs):
    env = MujocoFetchPickAndPlaceEnv(reward_type="adverse", **kwargs)
    env = GoalObservationConverter(env)
    env = RewardPlusOne(env)
    return env


def make_goal_observation_converter_wrapped_large_point_maze_env(**kwargs):
    if "maze_map" not in kwargs:
        kwargs["maze_map"] = maps.LARGE_MAZE_DIVERSE_GR
    env = PointMazeEnv(**kwargs)
    env = GoalObservationConverter(env)
    return env


def make_goal_observation_converter_wrapped_dense_large_point_maze_env(**kwargs):
    if "maze_map" not in kwargs:
        kwargs["maze_map"] = maps.LARGE_MAZE_DIVERSE_GR
    env = PointMazeEnv(reward_type="dense", **kwargs)
    env = GoalObservationConverter(env)
    return env


def make_goal_observation_converter_wrapped_adverse_large_point_maze_env(**kwargs):
    if "maze_map" not in kwargs:
        kwargs["maze_map"] = maps.LARGE_MAZE_DIVERSE_GR
    env = PointMazeEnv(reward_type="adverse", **kwargs)
    env = GoalObservationConverter(env)
    return env
