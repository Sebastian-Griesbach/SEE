from typing import Optional, Callable, Any, Tuple, Dict

import torch
from gymnasium.spaces import Box
import numpy as np

from athlete.policy.policy_builder import Policy
from athlete.global_objects import StepTracker, RNGHandler
from athlete.function import numpy_to_tensor, tensor_to_numpy

INFO_KEY_UNSCALED_ACTION = "unscaled_action"
INFO_POLICY_ID = "policy_id"


class AlternatingTrainingPolicy(Policy):
    """This policy alternatingly chooses actions of exploration and exploitation policy.
    Tis is used for the ablation study.
    """

    def __init__(
        self,
        exploration_actor: torch.nn.Module,
        exploitation_actor: torch.nn.Module,
        action_space: Box,
        post_replay_buffer_preprocessing: Optional[Callable[[Any], Any]] = None,
    ):
        """Initialize the alternating policy.

        Args:
            exploration_actor (torch.nn.Module): Actor to get exploration actions.
            exploitation_actor (torch.nn.Module): Actor to get exploitation actions.
            action_space (Box): Action space used to sample random actions during warmup.
            post_replay_buffer_preprocessing (Optional[Callable[[Any], Any]], optional): List of preprocessing functions to
            use on the observations before passing them to the actor.. Defaults to None.
        """
        super().__init__()

        self.exploration_actor = exploration_actor
        self.exploitation_actor = exploitation_actor
        self.post_replay_buffer_preprocessing = post_replay_buffer_preprocessing

        self.action_space = action_space
        self.action_scales = (action_space.high - action_space.low) / 2
        self.action_offsets = (action_space.high + action_space.low) / 2

        self.module_device = next(self.exploration_actor.parameters()).device
        self.step_tracker = StepTracker.get_instance()
        self.random_number_generator = RNGHandler.get_random_number_generator()

    def act(self, observation: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate action based on observation. Returns alternatingly exploration
            and exploitation action.

        Args:
            observation (np.ndarray): Observation to base action on.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Action and additional information, unscaled action and policy id.
                0 is exploration, 1 is exploitation.
        """

        if not self.step_tracker.warmup_is_done:
            random_action = self.random_number_generator.random(
                size=self.action_space.shape
            )
            random_scaled_action = (
                random_action * 2 - 1
            ) * self.action_scales + self.action_offsets
            return random_scaled_action, {INFO_KEY_UNSCALED_ACTION: random_action}

        observation = np.expand_dims(observation, axis=0)
        if self.post_replay_buffer_preprocessing is not None:
            observation = self.post_replay_buffer_preprocessing(observation)
        observation = numpy_to_tensor(observation, device=self.module_device)

        if self.step_tracker.total_interactions % 2 == 0:
            action = self.exploration_actor(observation)
            policy_id = 0
        else:
            action = self.exploitation_actor(observation)
            policy_id = 1
        action = tensor_to_numpy(action).squeeze(0)
        scaled_action = action * self.action_scales + self.action_offsets
        return scaled_action, {
            INFO_KEY_UNSCALED_ACTION: action,
            INFO_POLICY_ID: policy_id,
        }
