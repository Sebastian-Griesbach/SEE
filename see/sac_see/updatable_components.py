import torch
from typing import Dict, Callable, Optional, Tuple

from athlete import constants
from athlete.update.common import TorchFrequentGradientUpdate
from athlete.algorithms.sac.module import SACActor


from function import calculate_bellmann_target, calculate_maximum_target
from module import FingerprintingConditionedContinuousQValueFunction


class SACSEECriticUpdate(TorchFrequentGradientUpdate):
    """This class implements the SAC-SEE critic update. It is responsible for updating the critic networks
    of the exploration objective.
    """

    CRITIC_LOSS_TAG = "exploration_critic_loss"
    SAVE_FILE_NAME = "sac_see_critic_update"

    def __init__(
        self,
        exploration_temperature: torch.nn.Parameter,
        exploitation_temperature: torch.nn.Parameter,
        exploration_critic_1: FingerprintingConditionedContinuousQValueFunction,
        exploration_critic_2: FingerprintingConditionedContinuousQValueFunction,
        exploration_target_critic_1: torch.nn.Module,
        exploration_target_critic_2: torch.nn.Module,
        exploration_actor: SACActor,
        exploration_critic_optimizer: torch.nn.Module,
        data_sampler: Callable[[None], Dict[str, torch.tensor]],
        exploitation_critic_1: torch.nn.Module,
        exploitation_critic_2: torch.nn.Module,
        exploitation_actor: SACActor,
        target_calculation_method: str = "maximum",
        discount_exploitation: float = 0.99,
        discount_exploration: float = 0.99,
        exploitation_criteria: torch.nn.modules.loss._Loss = torch.nn.L1Loss(),
        exploration_criteria: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
        number_of_updates: int = 1,
        log_tag: str = CRITIC_LOSS_TAG,
        gradient_max_norm: Optional[float] = None,
        changes_policy: bool = True,
        update_frequency: int = 1,
        multiply_number_of_updates_by_environment_steps: bool = False,
        save_file_name: str = SAVE_FILE_NAME,
    ) -> None:
        """Initializes the SAC-SEE critic update class. This class is responsible for updating the critic networks
        of the exploration objective.

        Args:
            exploration_temperature (torch.nn.Parameter): Temperature parameter used for the entropy bonus of the exploration objective.
            exploitation_temperature (torch.nn.Parameter): Temperature parameter used for the entropy bonus of the exploitation objective.
            exploration_critic_1 (FingerprintingConditionedContinuousQValueFunction): First critic network for the exploration objective to be updated.
            exploration_critic_2 (FingerprintingConditionedContinuousQValueFunction): Second critic network for the exploration objective to be updated.
            exploration_target_critic_1 (torch.nn.Module): First target critic network for the exploration objective.
            exploration_target_critic_2 (torch.nn.Module): Second target critic network for the exploration objective.
            exploration_actor (SACActor): Actor network for the exploration objective.
            exploration_critic_optimizer (torch.nn.Module): Optimizer for the exploration critic networks.
            data_sampler (Callable[[None], Dict[str, torch.tensor]]): Function that samples data from the replay buffer.
            exploitation_critic_1 (torch.nn.Module): First critic network for the exploitation objective.
            exploitation_critic_2 (torch.nn.Module): Second critic network for the exploitation objective.
            exploitation_actor (SACActor): Actor network for the exploitation objective.
            target_calculation_method (str, optional): Method used to calculate the target for the critic networks. Can be either "bellmann" or "maximum". Defaults to "maximum".
            discount_exploitation (float, optional): Discount factor for the exploitation objective. Defaults to 0.99.
            discount_exploration (float, optional): Discount factor for the exploration objective. Defaults to 0.99.
            exploitation_criteria (torch.nn.modules.loss._Loss, optional): Loss function used to calculate the exploration reward. Defaults to torch.nn.L1Loss().
            exploration_criteria (torch.nn.modules.loss._Loss, optional): Loss function used for the exploration objective. Defaults to torch.nn.MSELoss().
            number_of_updates (int, optional): Number of updates to perform during each update step. Defaults to 1.
            log_tag (str, optional): Log used for the loss. Defaults to "critic_loss".
            gradient_max_norm (Optional[float], optional): Gradient max norm for the gradient clipping. Defaults to None.
            changes_policy (bool, optional): Whether the update changes the policy immediately. Defaults to True (because this affects the relative advantage).
            update_frequency (int, optional): Update based on the number of environment steps. If -1 updates happen at the end of each episode. Defaults to 1.
            multiply_number_of_updates_by_environment_steps (bool, optional): Whether to multiply the number of updates by the number of environment steps since the last update. Defaults to False.
            save_file_name (str, optional): File name used for saving this component. Defaults to "sac_see_critic_update".

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
        self.exploration_temperature = exploration_temperature
        self.exploitation_temperature = exploitation_temperature
        self.exploration_critic_1 = exploration_critic_1
        self.exploration_target_critic_1 = exploration_target_critic_1
        self.exploration_critic_2 = exploration_critic_2
        self.exploration_target_critic_2 = exploration_target_critic_2
        self.exploration_actor = exploration_actor
        self.data_sampler = data_sampler
        self.exploitation_critic_1 = exploitation_critic_1
        self.exploitation_critic_2 = exploitation_critic_2
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

    def calculate_loss(
        self,
    ) -> torch.Tensor:
        """Calculates the loss for the exploration critics.
            This first calculates the td-error of the exploitation critics and then uses this
            as the reward for the exploration critics.

        Returns:
            torch.Tensor: The loss for the exploration critics.
        """

        batch_dictionary = self.data_sampler()

        observations = batch_dictionary[constants.DATA_OBSERVATIONS]
        actions = batch_dictionary[constants.DATA_ACTIONS]
        rewards = batch_dictionary[constants.DATA_REWARDS]
        next_observations = batch_dictionary[constants.DATA_NEXT_OBSERVATIONS]
        terminateds = batch_dictionary[constants.DATA_TERMINATEDS]

        not_terminateds = ~terminateds.type(torch.bool).reshape(-1)

        # We train both exploration critics on both exploitation critics embeddings,
        # we take the minimum over the exploration critics and use both embeddings individually therefore there are two rewards
        # this should help to learn better embeddings for the exploration critics as the target is connected to their input conditioning
        # in the actor update we simply take the minimum over everything as the exploitation also uses the minimum to update its actor, which is the policy we want to find the errors of

        # Calculating exploitation  td_errors for each transition
        with torch.no_grad():
            exploitation_prediction_1 = self.exploitation_critic_1(
                observations=observations, actions=actions
            )
            exploitation_prediction_2 = self.exploitation_critic_2(
                observations=observations, actions=actions
            )

            next_actions, next_log_probabilities = (
                self.exploitation_actor.get_action_and_log_prob(
                    next_observations[not_terminateds]
                )
            )

            # individual td-error calculation for both exploitation critic embeddings
            next_q_value_predictions_1 = self.exploitation_critic_1(
                observations=next_observations[not_terminateds], actions=next_actions
            )
            next_q_value_predictions_2 = self.exploitation_critic_2(
                observations=next_observations[not_terminateds], actions=next_actions
            )

            next_q_values_with_entropy_1 = (
                next_q_value_predictions_1
                - self.exploitation_temperature * next_log_probabilities
            )

            next_q_values_with_entropy_2 = (
                next_q_value_predictions_2
                - self.exploitation_temperature * next_log_probabilities
            )

            exploitation_target_1 = torch.clone(rewards)
            exploitation_target_1[not_terminateds] += (
                self.discount_exploitation * next_q_values_with_entropy_1
            )

            exploitation_target_2 = torch.clone(rewards)
            exploitation_target_2[not_terminateds] += (
                self.discount_exploitation * next_q_values_with_entropy_2
            )

            exploration_reward_1 = self.exploitation_criteria(
                exploitation_prediction_1, exploitation_target_1
            )
            exploration_reward_2 = self.exploitation_criteria(
                exploitation_prediction_2, exploitation_target_2
            )

        # prediction, this is computational expensive. Probably it would be sufficient to simply use one embedding
        # But this is how it was done for the experiments in the paper
        prediction_critic_1_embedding_1 = self.exploration_critic_1(
            observations=observations,
            actions=actions,
            conditioning_module=self.exploitation_critic_1,
        )
        prediction_critic_1_embedding_2 = self.exploration_critic_1(
            observations=observations,
            actions=actions,
            conditioning_module=self.exploitation_critic_2,
        )
        prediction_critic_2_embedding_1 = self.exploration_critic_2(
            observations=observations,
            actions=actions,
            conditioning_module=self.exploitation_critic_1,
        )
        prediction_critic_2_embedding_2 = self.exploration_critic_2(
            observations=observations,
            actions=actions,
            conditioning_module=self.exploitation_critic_2,
        )

        # target
        with torch.no_grad():
            target_embedding_1, target_embedding_2 = self._calculate_target(
                exploration_reward_1=exploration_reward_1,
                exploration_reward_2=exploration_reward_2,
                next_observations=next_observations,
                not_terminateds=not_terminateds,
            )

        # loss for each exploitation critic - exploration critic pair
        loss_critic_1_embedding_1 = self.exploration_criteria(
            prediction_critic_1_embedding_1, target_embedding_1
        )
        loss_critic_1_embedding_2 = self.exploration_criteria(
            prediction_critic_1_embedding_2, target_embedding_2
        )
        loss_critic_2_embedding_1 = self.exploration_criteria(
            prediction_critic_2_embedding_1, target_embedding_1
        )
        loss_critic_2_embedding_2 = self.exploration_criteria(
            prediction_critic_2_embedding_2, target_embedding_2
        )
        return (
            loss_critic_1_embedding_1
            + loss_critic_1_embedding_2
            + loss_critic_2_embedding_1
            + loss_critic_2_embedding_2
        )

    def _calculate_target(
        self,
        exploration_reward_1: torch.tensor,
        exploration_reward_2: torch.tensor,
        next_observations: torch.tensor,
        not_terminateds: torch.tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the target for the exploration critics.
        This is done by calculating the prediction for both exploitation critics with both target exploration critics,
        then taking the minimum over exploration critics, calculating individual targets with individual rewards for both
        exploitation critic conditioning. This aims to be a generalization of the SAC target calculation for the case
        with conditioned critics.

        Args:
            exploration_reward_1 (torch.tensor): Batch of exploration rewards according to first exploitation critic.
            exploration_reward_2 (torch.tensor): Batch of exploration rewards according to second exploitation critic.
            next_observations (torch.tensor): Batch of next observations.
            not_terminateds (torch.tensor): Batch of negated termination flags.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of two targets according to the two exploitation critics.
        """

        # Calculate the prediction for both exploitation critics with both target exploration critics,
        # take the minimum over exploration critics, calculate individual targets with individual rewards
        # such that we minimize over the results of a specific conditioning and have targets per conditioning
        next_actions, next_log_probabilities = (
            self.exploration_actor.get_action_and_log_prob(
                next_observations[not_terminateds]
            )
        )

        next_prediction_critic_1_embedding_1 = self.exploration_target_critic_1(
            observations=next_observations[not_terminateds],
            actions=next_actions,
            conditioning_module=self.exploitation_critic_1,
        )
        next_prediction_critic_1_embedding_2 = self.exploration_target_critic_1(
            observations=next_observations[not_terminateds],
            actions=next_actions,
            conditioning_module=self.exploitation_critic_2,
        )
        next_prediction_critic_2_embedding_1 = self.exploration_target_critic_2(
            observations=next_observations[not_terminateds],
            actions=next_actions,
            conditioning_module=self.exploitation_critic_1,
        )
        next_prediction_critic_2_embedding_2 = self.exploration_target_critic_2(
            observations=next_observations[not_terminateds],
            actions=next_actions,
            conditioning_module=self.exploitation_critic_2,
        )

        min_prediction_1 = torch.min(
            next_prediction_critic_1_embedding_1, next_prediction_critic_2_embedding_1
        )
        min_prediction_2 = torch.min(
            next_prediction_critic_1_embedding_2, next_prediction_critic_2_embedding_2
        )

        min_prediction_with_entropy_1 = (
            min_prediction_1 - self.exploration_temperature * next_log_probabilities
        )
        min_prediction_with_entropy_2 = (
            min_prediction_2 - self.exploration_temperature * next_log_probabilities
        )

        target_embedding_1 = self.target_calculation_function(
            rewards=exploration_reward_1,
            not_terminateds=not_terminateds,
            next_predictions=min_prediction_with_entropy_1,
            discount=self.discount_exploration,
        )

        target_embedding_2 = self.target_calculation_function(
            rewards=exploration_reward_2,
            not_terminateds=not_terminateds,
            next_predictions=min_prediction_with_entropy_2,
            discount=self.discount_exploration,
        )
        return target_embedding_1, target_embedding_2


class SACSEEActorUpdate(TorchFrequentGradientUpdate):
    """This class implements the update of the exploration actor of the SAC+SEE algorithm."""

    ACTOR_LOSS_LOG_TAG = "exploration_actor_loss"
    SAVE_FILE_NAME = "sac_see_actor_update"

    def __init__(
        self,
        temperature: torch.nn.Parameter,
        exploration_critic_1: torch.nn.Module,
        exploration_critic_2: torch.nn.Module,
        exploration_actor: SACActor,
        exploitation_critic_1: torch.nn.Module,
        exploitation_critic_2: torch.nn.Module,
        actor_optimizer: torch.nn.Module,
        data_sampler: Callable[[None], Dict[str, torch.tensor]],
        changes_policy: bool = True,
        update_frequency: int = 1,
        number_of_updates: int = 1,
        multiply_number_of_updates_by_environment_steps: bool = False,
        gradient_max_norm: Optional[float] = None,
        log_tag: str = ACTOR_LOSS_LOG_TAG,
        save_file_name: str = SAVE_FILE_NAME,
    ) -> None:
        """Initializes the SAC+SEE actor update class. This class
        is responsible for updating the actor network of the exploration objective.

        Args:
            temperature (torch.nn.Parameter): The exploration temperature of the entropy bonus.
            exploration_critic_1 (torch.nn.Module): First exploration critic network.
            exploration_critic_2 (torch.nn.Module): Second exploration critic network.
            exploration_actor (SACActor): The exploration actor network to be updated.
            exploitation_critic_1 (torch.nn.Module): First exploitation critic network.
            exploitation_critic_2 (torch.nn.Module): First exploitation critic network.
            actor_optimizer (torch.nn.Module): The optimizer for the exploration actor network.
            data_sampler (Callable[[None], Dict[str, torch.tensor]]): Function that samples data from the replay buffer.
            changes_policy (bool, optional): Whether the update changes the policy immediately. Defaults to True.
            update_frequency (int, optional): Update frequency based on the number of environment steps. If -1, updates happen at the end of each episode. Defaults to 1.
            number_of_updates (int, optional): Number of updates to perform during each update step. Defaults to 1.
            multiply_number_of_updates_by_environment_steps (bool, optional): Whether to multiply the number of updates by the number of environment steps since the last update. Defaults to False.
            gradient_max_norm (Optional[float], optional): Gradient max norm for the gradient clipping. Defaults to None.
            log_tag (str, optional): Log tag used for the loss. Defaults to "actor_loss".
            save_file_name (str, optional): File name used for saving this component. Defaults to "sac_see_actor_update".
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
        self.temperature = temperature
        self.exploration_critic_1 = exploration_critic_1
        self.exploration_critic_2 = exploration_critic_2
        self.exploration_actor = exploration_actor
        self.exploitation_critic_1 = exploitation_critic_1
        self.exploitation_critic_2 = exploitation_critic_2
        self.data_sampler = data_sampler

    def calculate_loss(
        self,
    ) -> torch.Tensor:
        """Calculates the loss for the exploration actor.
        This is done by calculating the value estimates with both exploration critics
        conditioned on both exploitation critics and then taking the minimum over teh four resulting values.
        This aims to be a generalization of the SAC actor update for the case with conditioned critics.

        Returns:
            torch.Tensor: The loss for the exploration actor.
        """
        batch_dictionary = self.data_sampler()

        observations = batch_dictionary[constants.DATA_OBSERVATIONS]

        actions, log_probabilities = self.exploration_actor.get_action_and_log_prob(
            observations
        )

        # prediction
        prediction_critic_1_embedding_1 = self.exploration_critic_1(
            observations=observations,
            actions=actions,
            conditioning_module=self.exploitation_critic_1,
        )
        prediction_critic_1_embedding_2 = self.exploration_critic_1(
            observations=observations,
            actions=actions,
            conditioning_module=self.exploitation_critic_2,
        )
        prediction_critic_2_embedding_1 = self.exploration_critic_2(
            observations=observations,
            actions=actions,
            conditioning_module=self.exploitation_critic_1,
        )
        prediction_critic_2_embedding_2 = self.exploration_critic_2(
            observations=observations,
            actions=actions,
            conditioning_module=self.exploitation_critic_2,
        )

        # taking the minimum over the four critic and embedding combinations

        stacked_predictions = torch.stack(
            [
                prediction_critic_1_embedding_1,
                prediction_critic_1_embedding_2,
                prediction_critic_2_embedding_1,
                prediction_critic_2_embedding_2,
            ],
            dim=0,
        )
        min_q_values = torch.min(stacked_predictions, dim=0).values
        loss = torch.mean((self.temperature * log_probabilities) - min_q_values)

        return loss
