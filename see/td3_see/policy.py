from typing import Optional, Callable, Any, Tuple, Dict

import torch
from gymnasium.spaces import Box
import numpy as np

from athlete.policy.policy_builder import PolicyBuilder, Policy
from athlete.global_objects import StepTracker, RNGHandler
from athlete.function import numpy_to_tensor, tensor_to_numpy
from athlete.algorithms.ddpg.policy import DDPGEvaluationPolicy

from module import FingerprintingConditionedContinuousQValueFunction
from policy import AlternatingTrainingPolicy

INFO_KEY_UNSCALED_ACTION = "unscaled_action"
INFO_POLICY_ID = "policy_id"


class TD3SEETrainingPolicy(Policy):
    """The training policy for the TD3+SEE algorithm.
    It handles the mixing of the exploration and exploitation policies according to their relative Q-values.
    It scales the actions to the action space of the environment and implements a warmup phase where random actions are taken.
    """

    def __init__(
        self,
        exploration_actor: torch.nn.Module,
        exploration_critic: FingerprintingConditionedContinuousQValueFunction,
        exploitation_actor: torch.nn.Module,
        exploitation_critic: torch.nn.Module,
        exploration_advantage_mixture: float,
        action_selection_temperature: float,
        action_space: Box,
        post_replay_buffer_preprocessing: Optional[Callable[[Any], Any]] = None,
    ):
        """Initializes the TD3SEETrainingPolicy.

        Args:
            exploration_actor (torch.nn.Module): The exploration actor network.
            exploration_critic (FingerprintingConditionedContinuousQValueFunction): The exploration critic network to use for the relative advantage.
            exploitation_actor (torch.nn.Module): The exploitation actor network.
            exploitation_critic (torch.nn.Module): The exploitation critic network to use for the relative advantage.
            exploration_advantage_mixture (float): The mixture coefficient for the exploration advantage, 0.5 means that the relative scales stay the same.
            action_selection_temperature (float): The temperature for the Boltzmann distribution selecting one of the two candidate actions.
            action_space (Box): The action space of the environment.
            post_replay_buffer_preprocessing (Optional[Callable[[Any], Any]], optional): A function to preprocess the observation before passing it to the actor. Defaults to None.
        """
        super().__init__()

        self.exploration_actor = exploration_actor
        self.exploration_critic = exploration_critic
        self.exploitation_actor = exploitation_actor
        self.exploitation_critic = exploitation_critic
        self.exploration_advantage_mixture = exploration_advantage_mixture
        self.action_selection_temperature = action_selection_temperature
        self.post_replay_buffer_preprocessing = post_replay_buffer_preprocessing

        self.action_space = action_space
        self.action_scales = (action_space.high - action_space.low) / 2
        self.action_offsets = (action_space.high + action_space.low) / 2

        self.module_device = next(self.exploration_actor.parameters()).device
        self.step_tracker = StepTracker.get_instance()
        self.random_number_generator = RNGHandler.get_random_number_generator()

    def act(self, observation: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Returns the action to take given the observation.

        Args:
            observation (np.ndarray): The observation from the environment.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: The action to take and a dictionary with additional information
            including the unscaled action and the policy id (where 0 is exploration and 1 is exploitation).
        """
        if not self.step_tracker.warmup_is_done:
            random_action = self.random_number_generator.random(
                size=self.action_space.shape
            )
            random_scaled_action = (
                random_action * 2 - 1
            ) * self.action_scales + self.action_offsets  # scales the random action to the action space
            return random_scaled_action, {INFO_KEY_UNSCALED_ACTION: random_action}

        observation = np.expand_dims(observation, axis=0)
        if self.post_replay_buffer_preprocessing is not None:
            observation = self.post_replay_buffer_preprocessing(observation)
        observation = numpy_to_tensor(observation, device=self.module_device)

        action, policy_id = self._sample_mixed_policy(observation)

        action = tensor_to_numpy(action)
        scaled_action = action * self.action_scales + self.action_offsets

        return scaled_action, {
            INFO_KEY_UNSCALED_ACTION: action,
            INFO_POLICY_ID: policy_id,
        }

    def _sample_mixed_policy(
        self,
        observation: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Samples an action based on the relative Q-values of the exploration and exploitation policies.

        Args:
            observation (torch.Tensor): Observation to use for the action selection.

        Returns:
            Tuple[torch.Tensor, int]: The selected action and the policy id (0 for exploration, 1 for exploitation).
        """
        with torch.no_grad():
            # get candidate actions
            exploration_action = self.exploration_actor(observation)
            exploitation_action = self.exploitation_actor(observation)

            stacked_actions = torch.cat(
                [exploration_action, exploitation_action], dim=0
            )

            repeated_observations = observation.repeat(2, 1)

            exploration_q_values = self.exploration_critic(
                observations=repeated_observations,
                actions=stacked_actions,
                conditioning_module=self.exploitation_critic,
            )

            exploration_advantage = exploration_q_values[0] - exploration_q_values[1]

            # Calculated exploitation advantage
            exploitation_q_values = self.exploitation_critic(
                repeated_observations, stacked_actions
            )

            exploitation_advantage = exploitation_q_values[1] - exploitation_q_values[0]

            # Select action
            stacked_advantages = torch.cat(
                [exploration_advantage, exploitation_advantage], dim=0
            )

            boltzmann_probabilities = torch.softmax(
                stacked_advantages / self.action_selection_temperature, dim=0
            )

            action_distribution = torch.distributions.Categorical(
                boltzmann_probabilities
            )

            selected_action_index = action_distribution.sample()
            selected_action = stacked_actions[selected_action_index]
        return selected_action, selected_action_index.item()


class TD3SEEPolicyBuilder(PolicyBuilder):
    """This class is responsible for building the TD3+SEE training and evaluation policies."""

    def __init__(
        self,
        exploration_actor: torch.nn.Module,
        exploration_critic: FingerprintingConditionedContinuousQValueFunction,
        exploitation_actor: torch.nn.Module,
        exploitation_critic: torch.nn.Module,
        exploration_advantage_mixture: float,
        action_selection_temperature: float,
        action_space: Box,
        post_replay_buffer_preprocessing: Optional[Callable[[Any], Any]] = None,
        use_alternating_training_policy: bool = False,
    ):
        """Initializes the TD3SEEPolicyBuilder.

        Args:
            exploration_actor (torch.nn.Module): The exploration actor network.
            exploration_critic (FingerprintingConditionedContinuousQValueFunction): The exploration critic network to use for the relative advantage.
            exploitation_actor (torch.nn.Module): The exploitation actor network.
            exploitation_critic (torch.nn.Module): The exploitation critic network to use for the relative advantage.
            exploration_advantage_mixture (float): The mixture coefficient for the exploration advantage, 0.5 means that the relative scales stay the same.
            action_selection_temperature (float): The temperature for the Boltzmann distribution selecting one of the two candidate actions.
            action_space (Box): The action space of the environment.
            post_replay_buffer_preprocessing (Optional[Callable[[Any], Any]], optional): A function to preprocess the observation before passing it to the actor. Defaults to None.
            use_alternating_training_policy (bool, optional): Whether to use the alternating training policy instead. This is used for the ablation study w/o mixing. Defaults to False.
        """
        super().__init__()
        self.exploration_actor = exploration_actor
        self.exploration_critic = exploration_critic
        self.exploitation_actor = exploitation_actor
        self.exploitation_critic = exploitation_critic
        self.exploration_advantage_mixture = exploration_advantage_mixture
        self.action_selection_temperature = action_selection_temperature
        self.action_space = action_space
        self.post_replay_buffer_preprocessing = post_replay_buffer_preprocessing
        self.use_alternating_training_policy = use_alternating_training_policy

    def build_training_policy(self) -> Policy:
        """Builds the training policy for the TD3+SEE algorithm.

        Returns:
            Policy: The training policy for the TD3+SEE algorithm.
        """
        if self.use_alternating_training_policy:
            return AlternatingTrainingPolicy(
                exploitation_actor=self.exploration_actor,
                exploration_actor=self.exploitation_actor,
                action_space=self.action_space,
                post_replay_buffer_preprocessing=self.post_replay_buffer_preprocessing,
            )
        return TD3SEETrainingPolicy(
            exploration_actor=self.exploration_actor,
            exploration_critic=self.exploration_critic,
            exploitation_actor=self.exploitation_actor,
            exploitation_critic=self.exploitation_critic,
            exploration_advantage_mixture=self.exploration_advantage_mixture,
            action_selection_temperature=self.action_selection_temperature,
            action_space=self.action_space,
            post_replay_buffer_preprocessing=self.post_replay_buffer_preprocessing,
        )

    def build_evaluation_policy(self) -> Policy:
        """Builds the evaluation policy for the TD3+SEE algorithm.
            This is the same as a regular TD3 policy which in turn is the same as DDPG's policy.

        Returns:
            Policy: The evaluation policy for the TD3+SEE algorithm.
        """
        return DDPGEvaluationPolicy(
            actor=self.exploitation_actor,
            action_space=self.action_space,
            post_replay_buffer_preprocessing=self.post_replay_buffer_preprocessing,
        )

    @property
    def requires_rebuild_on_policy_change(self) -> bool:
        """Whether the policy needs to be rebuilt when the policy changes."""
        return False
