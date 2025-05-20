import torch
from typing import Dict, Callable

from athlete import constants
from athlete.update.common import TorchFrequentGradientUpdate

from function import calculate_bellmann_target, calculate_maximum_target
from module import FingerprintingConditionedContinuousQValueFunction


class TD3SEECriticUpdate(TorchFrequentGradientUpdate):
    """This component is responsible for updating the exploration critics of TD3+SEE."""

    CRITIC_LOSS_TAG = "exploration_critic_loss"

    SAVE_FILE_NAME = "td3_see_critic_update"

    def __init__(
        self,
        data_sampler: Callable[[None], Dict[str, torch.tensor]],
        exploration_critic_1: FingerprintingConditionedContinuousQValueFunction,
        exploration_critic_2: FingerprintingConditionedContinuousQValueFunction,
        exploration_target_critic_1: FingerprintingConditionedContinuousQValueFunction,
        exploration_target_critic_2: FingerprintingConditionedContinuousQValueFunction,
        exploration_target_actor: torch.nn.Module,
        exploration_critic_optimizer: torch.optim.Optimizer,
        exploitation_critic: torch.nn.Module,
        exploitation_actor: torch.nn.Module,
        target_calculation_method: str = "maximum",
        discount_exploitation: float = 0.99,
        discount_exploration: float = 0.99,
        target_noise_std: float = 0.2,
        target_noise_clip: float = 0.5,
        exploitation_criteria: torch.nn.modules.loss._Loss = torch.nn.L1Loss(),
        exploration_criteria: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
        update_frequency: int = 1,
        number_of_updates: int = 1,
        multiply_number_of_updates_by_environment_steps: bool = False,
        log_tag: str = CRITIC_LOSS_TAG,
        gradient_max_norm: float = None,
        changes_policy: bool = True,
        save_file_name: str = SAVE_FILE_NAME,
    ) -> None:
        """Initializes the TD3+SEE critic update component.

        Args:
            data_sampler (Callable[[None], Dict[str, torch.tensor]]): A Function that returns data from the replay buffer.
            exploration_critic_1 (FingerprintingConditionedContinuousQValueFunction): First exploration critic to be updated.
            exploration_critic_2 (FingerprintingConditionedContinuousQValueFunction): Second exploration critic to be updated.
            exploration_target_critic_1 (FingerprintingConditionedContinuousQValueFunction): Target network for the first exploration critic.
            exploration_target_critic_2 (FingerprintingConditionedContinuousQValueFunction): Target network for the second exploration critic.
            exploration_target_actor (torch.nn.Module): Target network of the exploration actor.
            exploration_critic_optimizer (torch.optim.Optimizer): Optimizer for the exploration critics.
            exploitation_critic (torch.nn.Module): The exploitation critic to condition the exploration critics on.
            exploitation_actor (torch.nn.Module): The exploitation actor used to calculate the exploration reward.
            target_calculation_method (str, optional): Method to calculate the target for the exploration critics. Can be either "bellmann" or "maximum". Defaults to "maximum".
            discount_exploitation (float, optional): Discount factor for the exploitation objective. Defaults to 0.99.
            discount_exploration (float, optional): Discount factor for the exploration objective. Defaults to 0.99.
            target_noise_std (float, optional): The standard deviation of the noise added to the target actions. Defaults to 0.2.
            target_noise_clip (float, optional): The maximum absolute value of the noise added to the target actions. Defaults to 0.5.
            exploitation_criteria (torch.nn.modules.loss._Loss, optional): The loss function used to calculate the exploitation loss which is used as the exploration reward. Defaults to torch.nn.L1Loss().
            exploration_criteria (torch.nn.modules.loss._Loss, optional): The loss function used to calculate the exploration loss. Defaults to torch.nn.MSELoss().
            update_frequency (int, optional): The update frequency of the component according to the number of environment steps. If -1 the update will happen at the end of each episode. Defaults to 1.
            number_of_updates (int, optional): The number of updates to be performed at each update step. Defaults to 1.
            multiply_number_of_updates_by_environment_steps (bool, optional): Whether to multiply the number of updates by the number of environment steps since the last update. Defaults to False.
            log_tag (str, optional): The tag used for logging the exploration critic loss. Defaults to "exploration_critic_loss".
            gradient_max_norm (float, optional): The maximum norm of the gradients. If None, no gradient clipping is performed. Defaults to None.
            changes_policy (bool, optional): Whether the update changes the policy immediately. Defaults to True because this affects the relative advantage of the exploration and exploitation policies.
            save_file_name (str, optional): The name of the file to save the component. Defaults to "td3_see_critic_update".

        Raises:
            ValueError: If the target calculation method is not "bellmann" or "maximum".
        """
        super().__init__(
            optimizer=exploration_critic_optimizer,
            changes_policy=changes_policy,
            log_tag=log_tag,
            save_file_name=save_file_name,
            update_frequency=update_frequency,
            number_of_updates=number_of_updates,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
            gradient_max_norm=gradient_max_norm,
        )

        self.exploration_critic_1 = exploration_critic_1
        self.exploration_target_critic_1 = exploration_target_critic_1
        self.exploration_critic_2 = exploration_critic_2
        self.exploration_target_critic_2 = exploration_target_critic_2
        self.exploration_target_actor = exploration_target_actor
        self.data_sampler = data_sampler
        self.exploitation_critic = exploitation_critic
        self.exploitation_actor = exploitation_actor

        if target_calculation_method == "bellmann":
            self.target_calculation_function = calculate_bellmann_target
        elif target_calculation_method == "maximum":
            self.target_calculation_function = calculate_maximum_target
        else:
            raise ValueError("Invalid target calculation method")

        self.exploration_criteria = exploration_criteria

        self.discount_exploitation = discount_exploitation
        self.discount_exploration = discount_exploration
        self.exploitation_criteria = type(exploitation_criteria)(reduction="none")

        self.target_noise_std = target_noise_std
        self.target_noise_clip = target_noise_clip

    def calculate_loss(
        self,
    ) -> torch.Tensor:
        """Calculates the loss for the exploration critics.

        Returns:
            torch.Tensor: The loss for the exploration critics.
        """

        transition_batch_dictionary = self.data_sampler()

        observations = transition_batch_dictionary[constants.DATA_OBSERVATIONS]
        actions = transition_batch_dictionary[constants.DATA_ACTIONS]
        rewards = transition_batch_dictionary[constants.DATA_REWARDS]
        next_observations = transition_batch_dictionary[
            constants.DATA_NEXT_OBSERVATIONS
        ]
        terminateds = transition_batch_dictionary[constants.DATA_TERMINATEDS]

        not_terminateds = ~terminateds.type(torch.bool).reshape(-1)

        # Calculating td_errors of the exploitation critic for each transition as the exploration reward
        with torch.no_grad():
            exploitation_prediction = self.exploitation_critic(
                observations=observations, actions=actions
            )

            next_actions = self.exploitation_actor(next_observations[not_terminateds])

            next_q_value_predictions = self.exploitation_critic(
                observations=next_observations[not_terminateds], actions=next_actions
            )

            exploitation_target = torch.clone(rewards)
            exploitation_target[not_terminateds] += (
                self.discount_exploitation * next_q_value_predictions
            )

            exploration_reward = self.exploitation_criteria(
                exploitation_prediction, exploitation_target
            )

        # Exploration loss

        # prediction
        predicted_values_1 = self.exploration_critic_1(
            observations=observations,
            actions=actions,
            conditioning_module=self.exploitation_critic,
        )

        predicted_values_2 = self.exploration_critic_2(
            observations=observations,
            actions=actions,
            conditioning_module=self.exploitation_critic,
        )

        # target
        with torch.no_grad():
            target = self._calculate_target(
                exploration_reward=exploration_reward,
                next_observations=next_observations,
                not_terminateds=not_terminateds,
                conditioning_module=self.exploitation_critic,
            )

        loss_1 = self.exploration_criteria(predicted_values_1, target)
        loss_2 = self.exploration_criteria(predicted_values_2, target)
        return loss_1 + loss_2

    def _calculate_target(
        self,
        exploration_reward: torch.tensor,
        next_observations: torch.tensor,
        not_terminateds: torch.tensor,
        conditioning_module: torch.nn.Module,
    ) -> torch.Tensor:
        """Calculates the target for the exploration critics.
            This function mimics the TD3 critic target calculation by taking the minimum of the two target critics.

        Args:
            exploration_reward (torch.tensor): Batch of exploration rewards.
            next_observations (torch.tensor): Batch of next observations.
            not_terminateds (torch.tensor): Batch of negated termination flags.
            conditioning_module (torch.nn.Module): The exploitation critic to condition the exploration critics on.

        Returns:
            torch.Tensor: Batch of targets for the exploration critics.
        """

        next_actions = self.exploration_target_actor(next_observations[not_terminateds])

        clipped_noise = torch.clip(
            torch.randn_like(next_actions) * self.target_noise_std,
            min=-self.target_noise_clip,
            max=self.target_noise_clip,
        )
        noisy_next_actions = torch.clip(
            next_actions + clipped_noise,
            min=-1.0,
            max=1.0,  # Assuming that action space is normalized for the update and only scaled in the policy
        )
        next_predictions_1 = self.exploration_target_critic_1(
            observations=next_observations[not_terminateds],
            actions=noisy_next_actions,
            conditioning_module=conditioning_module,
        )

        next_predictions_2 = self.exploration_target_critic_2(
            observations=next_observations[not_terminateds],
            actions=noisy_next_actions,
            conditioning_module=conditioning_module,
        )

        min_next_predictions = torch.min(next_predictions_1, next_predictions_2)

        # Either regular bellmann target or maximum target
        target = self.target_calculation_function(
            rewards=exploration_reward,
            not_terminateds=not_terminateds,
            next_predictions=min_next_predictions,
            discount=self.discount_exploration,
        )
        return target


class TD3SEEActorUpdate(TorchFrequentGradientUpdate):
    """This component is responsible for updating the exploration actor of TD3+SEE."""

    ACTOR_LOSS_LOG_TAG = "exploration_actor_loss"

    SAVE_FILE_NAME = "td3_see_actor_update"

    def __init__(
        self,
        data_sampler: Callable[[None], Dict[str, torch.tensor]],
        exploration_critic: FingerprintingConditionedContinuousQValueFunction,
        exploration_actor: torch.nn.Module,
        exploitation_critic: torch.nn.Module,
        actor_optimizer: torch.nn.Module,
        update_frequency: int = 1,
        number_of_updates: int = 1,
        multiply_number_of_updates_by_environment_steps: bool = False,
        log_tag: str = ACTOR_LOSS_LOG_TAG,
        gradient_max_norm: float = None,
        changes_policy: bool = True,
        save_file_name: str = SAVE_FILE_NAME,
    ) -> None:
        """Initializes the TD3+SEE actor update component.

        Args:
            data_sampler (Callable[[None], Dict[str, torch.tensor]]): Function that returns data from the replay buffer.
            exploration_critic (FingerprintingConditionedContinuousQValueFunction): The exploration critic to calculate the actor loss.
            exploration_actor (torch.nn.Module): The exploration actor network to be updated.
            exploitation_critic (torch.nn.Module): The exploitation critic to condition the exploration critic on.
            actor_optimizer (torch.nn.Module): The optimizer for the exploration actor.
            update_frequency (int, optional): The update frequency of the component according to the number of environment steps. If -1 the update will happen at the end of each episode. Defaults to 1.
            number_of_updates (int, optional): The number of updates to be performed at each update step. Defaults to 1.
            multiply_number_of_updates_by_environment_steps (bool, optional): Whether to multiply the number of updates by the number of environment steps since the last update. Defaults to False.
            log_tag (str, optional): The tag used for logging the exploration actor loss. Defaults to "exploration_actor_loss".
            gradient_max_norm (float, optional): The maximum norm of the gradients. If None, no gradient clipping is performed. Defaults to None.
            changes_policy (bool, optional): Whether the update changes the policy immediately. Defaults to True.
            save_file_name (str, optional): The name of the file to save the component. Defaults to "td3_see_actor_update".
        """
        super().__init__(
            optimizer=actor_optimizer,
            changes_policy=changes_policy,
            log_tag=log_tag,
            save_file_name=save_file_name,
            update_frequency=update_frequency,
            number_of_updates=number_of_updates,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
            gradient_max_norm=gradient_max_norm,
        )

        self.exploration_critic = exploration_critic
        self.exploration_actor = exploration_actor
        self.exploitation_critic = exploitation_critic
        self.data_sampler = data_sampler

    def calculate_loss(
        self,
    ) -> torch.Tensor:
        """Calculates the loss for the exploration actor.

        Returns:
            torch.Tensor: The loss for the exploration actor.
        """

        batch_dictionary = self.data_sampler()

        observations = batch_dictionary[constants.DATA_OBSERVATIONS]

        actions = self.exploration_actor(observations)

        values = self.exploration_critic(
            observations=observations,
            actions=actions,
            conditioning_module=self.exploitation_critic,
        )

        loss = -values.mean()

        return loss
