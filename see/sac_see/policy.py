from typing import Optional, Callable, Any, Tuple, Dict

import torch
from gymnasium.spaces import Box
import numpy as np

from athlete.policy.policy_builder import PolicyBuilder, Policy
from athlete.global_objects import StepTracker, RNGHandler
from athlete.function import numpy_to_tensor, tensor_to_numpy
from athlete.algorithms.sac.module import SACActor
from athlete.algorithms.sac.policy import SACEvaluationPolicy


from module import FingerprintingConditionedContinuousQValueFunction
from policy import AlternatingTrainingPolicy

INFO_KEY_UNSCALED_ACTION = "unscaled_action"
INFO_POLICY_ID = "policy_id"


class SACSEETrainingPolicy(Policy):
    """Training policy for SAC+SEE. It mixes the exploration and exploitation policy based on the
    relative advantage of the two policies.
    It also scales the actions according to the action space of the environment and implements
    a warmup period where it samples random actions.
    """

    def __init__(
        self,
        exploration_actor: SACActor,
        exploration_critic_1: FingerprintingConditionedContinuousQValueFunction,
        exploration_critic_2: FingerprintingConditionedContinuousQValueFunction,
        exploitation_actor: SACActor,
        exploitation_critic_1: torch.nn.Module,
        exploitation_critic_2: torch.nn.Module,
        exploration_advantage_mixture: float,
        action_selection_temperature: float,
        action_space: Box,
        post_replay_buffer_preprocessing: Optional[Callable[[Any], Any]] = None,
    ):
        """Initializes the SACSEETrainingPolicy.

        Args:
            exploration_actor (SACActor): Actor to propose exploration actions.
            exploration_critic_1 (FingerprintingConditionedContinuousQValueFunction): First exploration critic.
            exploration_critic_2 (FingerprintingConditionedContinuousQValueFunction): Second exploration critic.
            exploitation_actor (SACActor): Actor to propose exploitation actions.
            exploitation_critic_1 (torch.nn.Module): First exploitation critic.
            exploitation_critic_2 (torch.nn.Module): Second exploitation critic.
            exploration_advantage_mixture (float): Mixture coefficient for exploration advantage. Can be used to
                rescale both advantage functions if needed. relative scales remains unchanged with 0.5.
            action_selection_temperature (float): Temperature of the Boltzmann distribution used to decide between
                both candidate actions.
            action_space (Box): Action space of the environment used to determine the action scaling
            post_replay_buffer_preprocessing (Optional[Callable[[Any], Any]], optional): Preprocessing function to apply to the
                observations before passing them to the policy. Defaults to None.
        """
        super().__init__()

        self.exploration_actor = exploration_actor
        self.exploration_critic_1 = exploration_critic_1
        self.exploration_critic_2 = exploration_critic_2
        self.exploitation_actor = exploitation_actor
        self.exploitation_critic_1 = exploitation_critic_1
        self.exploitation_critic_2 = exploitation_critic_2
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
        """Samples an action from the mixed policy based on the observation.
            Random actions are sampled during the warmup period.

        Args:
            observation (np.ndarray): The observation from the environment.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: The sampled action and additional information including the
                unscaled action and the policy id, that proposed the action. (0 for exploration, 1 for exploitation)
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
        """Samples an action from the mixed policy according to the relative advantages and a
            Boltzmann distribution.

        Args:
            observation (torch.Tensor): The observation from the environment.

        Returns:
            Tuple[torch.Tensor, int]: The sampled action and the policy id (0 for exploration, 1 for exploitation).
        """
        with torch.no_grad():
            # get candidate actions
            exploration_action = self.exploration_actor(observation)
            exploitation_action = self.exploitation_actor(observation)

            stacked_actions = torch.cat(
                [exploration_action, exploitation_action], dim=0
            )

            repeated_observations = observation.repeat(2, 1)

            # The q value prediction here imitates here how the actor loss is calculated
            # such that the q-values are as alined with the actor as possible,
            # this require more compute and might not be necessary, but this is how it was done
            # for the experiments in the paper

            exploration_q_values_1_embedding_1 = self.exploration_critic_1(
                observations=repeated_observations,
                actions=stacked_actions,
                conditioning_module=self.exploitation_critic_1,
            )

            exploration_q_values_2_embedding_1 = self.exploration_critic_2(
                observations=repeated_observations,
                actions=stacked_actions,
                conditioning_module=self.exploitation_critic_1,
            )
            exploration_q_values_1_embedding_2 = self.exploration_critic_1(
                observations=repeated_observations,
                actions=stacked_actions,
                conditioning_module=self.exploitation_critic_2,
            )
            exploration_q_values_2_embedding_2 = self.exploration_critic_2(
                observations=repeated_observations,
                actions=stacked_actions,
                conditioning_module=self.exploitation_critic_2,
            )

            stacked_predictions = torch.stack(
                [
                    exploration_q_values_1_embedding_1,
                    exploration_q_values_2_embedding_1,
                    exploration_q_values_1_embedding_2,
                    exploration_q_values_2_embedding_2,
                ],
                dim=0,
            )
            exploration_q_values = torch.min(stacked_predictions, dim=0).values

            exploration_advantage = exploration_q_values[0] - exploration_q_values[1]

            # Calculated exploitation advantage
            exploitation_q_values_1 = self.exploitation_critic_1(
                repeated_observations, stacked_actions
            )

            exploitation_q_values_2 = self.exploitation_critic_2(
                repeated_observations, stacked_actions
            )
            exploitation_q_values = torch.min(
                torch.stack([exploitation_q_values_1, exploitation_q_values_2], dim=0),
                dim=0,
            ).values

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


class SACSEEPolicyBuilder(PolicyBuilder):
    """Builder for the SAC+SEE policy responsible for creating the training and evaluation policies."""

    def __init__(
        self,
        exploration_actor: torch.nn.Module,
        exploration_critic_1: FingerprintingConditionedContinuousQValueFunction,
        exploration_critic_2: FingerprintingConditionedContinuousQValueFunction,
        exploitation_actor: torch.nn.Module,
        exploitation_critic_1: torch.nn.Module,
        exploitation_critic_2: torch.nn.Module,
        exploration_advantage_mixture: float,
        action_selection_temperature: float,
        action_space: Box,
        post_replay_buffer_preprocessing: Optional[Callable[[Any], Any]] = None,
        use_alternating_training_policy: bool = False,
    ):
        """Initializes the SACSEEPolicyBuilder.
        This class is responsible for creating the training and evaluation policies for the SAC+SEE algorithm.

        Args:
            exploration_actor (torch.nn.Module): The actor network of the exploration policy.
            exploration_critic_1 (FingerprintingConditionedContinuousQValueFunction): The first critic network of the exploration policy.
            exploration_critic_2 (FingerprintingConditionedContinuousQValueFunction): The second critic network of the exploration policy.
            exploitation_actor (torch.nn.Module): The actor network of the exploitation policy.
            exploitation_critic_1 (torch.nn.Module): The first critic network of the exploitation policy.
            exploitation_critic_2 (torch.nn.Module): The second critic network of the exploitation policy.
            exploration_advantage_mixture (float): The mixture coefficient for the exploration advantage.
                This coefficient can be used to rescale both advantage functions if needed.
                The relative scales remain unchanged with 0.5.
            action_selection_temperature (float): The temperature of the Boltzmann distribution used to decide between both candidate actions.
            action_space (Box): The action space of the environment used to determine the action scaling.
            post_replay_buffer_preprocessing (Optional[Callable[[Any], Any]], optional): Preprocessing function to apply to the observations before passing them to the policy. Defaults to None.
            use_alternating_training_policy (bool, optional): Whether to use the alternating training policy instead. This is used for the ablation study w/o mixing. Defaults to False.
        """
        super().__init__()
        self.exploration_actor = exploration_actor
        self.exploration_critic_1 = exploration_critic_1
        self.exploration_critic_2 = exploration_critic_2
        self.exploitation_actor = exploitation_actor
        self.exploitation_critic_1 = exploitation_critic_1
        self.exploitation_critic_2 = exploitation_critic_2
        self.exploration_advantage_mixture = exploration_advantage_mixture
        self.action_selection_temperature = action_selection_temperature
        self.action_space = action_space
        self.post_replay_buffer_preprocessing = post_replay_buffer_preprocessing
        self.use_alternating_training_policy = use_alternating_training_policy

    def build_training_policy(self) -> Policy:
        """Builds the training policy for SAC+SEE."""
        if self.use_alternating_training_policy:
            return AlternatingTrainingPolicy(
                exploitation_actor=self.exploration_actor,
                exploration_actor=self.exploitation_actor,
                action_space=self.action_space,
                post_replay_buffer_preprocessing=self.post_replay_buffer_preprocessing,
            )
        return SACSEETrainingPolicy(
            exploration_actor=self.exploration_actor,
            exploration_critic_1=self.exploration_critic_1,
            exploration_critic_2=self.exploration_critic_2,
            exploitation_actor=self.exploitation_actor,
            exploitation_critic_1=self.exploitation_critic_1,
            exploitation_critic_2=self.exploitation_critic_2,
            exploration_advantage_mixture=self.exploration_advantage_mixture,
            action_selection_temperature=self.action_selection_temperature,
            action_space=self.action_space,
            post_replay_buffer_preprocessing=self.post_replay_buffer_preprocessing,
        )

    def build_evaluation_policy(self) -> Policy:
        """Builds the evaluation policy for SAC+SEE. Which is simply a SAC evaluation policy."""
        return SACEvaluationPolicy(
            actor=self.exploitation_actor,
            action_space=self.action_space,
            post_replay_buffer_preprocessing=self.post_replay_buffer_preprocessing,
        )

    @property
    def requires_rebuild_on_policy_change(self) -> bool:
        """Whether the policy needs to be rebuilt when the policy changes."""
        return False
