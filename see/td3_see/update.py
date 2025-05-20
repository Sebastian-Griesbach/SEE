import torch
from torch import nn
from torch.optim import Optimizer
from typing import Callable, Dict, Any, Type, Optional, Tuple
import copy

from gymnasium.spaces import Space

from athlete import constants
from athlete.update.update_rule import UpdateRule, UpdatableComponent
from athlete.algorithms.ddpg.updatable_components import DDPGActorUpdate
from athlete.algorithms.td3.updatable_components import TD3CriticUpdate
from athlete.update.common import TargetNetUpdate, ReplayBufferUpdate
from athlete.saving.saveable_component import CompositeSaveableComponent
from athlete.function import create_transition_data_info, extract_data_from_batch
from athlete.update.buffer import EpisodicCPPReplayBuffer
from athlete.update.buffer_wrapper import PostBufferPreprocessingWrapper
from athlete.data_collection.provider import UpdateDataProvider

from td3_see.updatable_components import TD3SEECriticUpdate, TD3SEEActorUpdate


# Twin Delayed Deep Deterministic policy gradient + Stationary Error-seeking Exploration


class TD3SEEUpdateRule(UpdateRule, CompositeSaveableComponent):
    """This class is responsible for managing the updatable components of the TD3+SEE algorithm.
    It provides all required dependencies and also saves most of the stateful components upon
    checkpointing.
    """

    EXPLOITATION_CRITIC_LOSS_LOG_TAG = "exploitation_critic_loss"
    EXPLOITATION_ACTOR_LOSS_LOG_TAG = "exploitation_actor_loss"
    EXPLORATION_CRITIC_LOSS_LOG_TAG = "exploration_critic_loss"
    EXPLORATION_ACTOR_LOSS_LOG_TAG = "exploration_actor_loss"

    EXPLOITATION_CRITIC_UPDATE_SAVE_FILE_NAME = "exploitation_critic_update"
    EXPLOITATION_ACTOR_UPDATE_SAVE_FILE_NAME = "exploitation_actor_update"
    EXPLORATION_CRITIC_UPDATE_SAVE_FILE_NAME = "exploration_critic_update"
    EXPLORATION_ACTOR_UPDATE_SAVE_FILE_NAME = "exploration_actor_update"

    def __init__(
        self,
        exploitation_critic_1: nn.Module,
        exploitation_critic_2: nn.Module,
        exploitation_actor: nn.Module,
        exploration_critic_1: nn.Module,
        exploration_critic_2: nn.Module,
        exploration_actor: nn.Module,
        exploitation_discount: float,
        exploration_discount: float,
        observation_space: Space,
        action_space: Space,
        update_data_provider: UpdateDataProvider,
        exploitation_critic_optimizer_class: Type[Optimizer],
        exploitation_actor_optimizer_class: Type[Optimizer],
        exploration_critic_optimizer_class: Type[Optimizer],
        exploration_actor_optimizer_class: Type[Optimizer],
        exploitation_critic_optimizer_arguments: Dict[str, Any],
        exploitation_actor_optimizer_arguments: Dict[str, Any],
        exploration_critic_optimizer_arguments: Dict[str, Any],
        exploration_actor_optimizer_arguments: Dict[str, Any],
        exploitation_critic_criteria: torch.nn.modules.loss._Loss,
        exploration_critic_criteria: torch.nn.modules.loss._Loss,
        exploration_target_calculation_method: str,
        exploitation_critic_update_frequency: int,
        exploitation_critic_number_of_updates: int,
        exploitation_actor_update_frequency: int,  # Having a slower update rate for the target networks is part of the TD3 algorithm
        exploitation_actor_number_of_updates: int,
        exploration_critic_update_frequency: int,
        exploration_critic_number_of_updates: int,
        exploration_actor_update_frequency: int,
        exploration_actor_number_of_updates: int,
        multiply_number_of_updates_by_environment_steps: bool,
        exploitation_target_critic_update_frequency: int,
        exploitation_target_actor_update_frequency: int,
        exploration_target_critic_update_frequency: int,
        exploration_target_actor_update_frequency: int,
        exploitation_target_critic_tau: float,
        exploitation_target_actor_tau: float,
        exploration_target_critic_tau: float,
        exploration_target_actor_tau: float,
        exploitation_target_noise_std: float,
        exploitation_target_noise_clip: float,
        exploration_target_noise_std: float,
        exploration_target_noise_clip: float,
        exploitation_critic_gradient_max_norm: Optional[float],
        exploitation_actor_gradient_max_norm: Optional[float],
        exploration_critic_gradient_max_norm: Optional[float],
        exploration_actor_gradient_max_norm: Optional[float],
        replay_buffer_capacity: int,
        replay_buffer_mini_batch_size: int,
        additional_replay_buffer_arguments: Optional[Dict[str, Any]],
        post_replay_buffer_data_preprocessing: Optional[Dict[str, Callable]],
        device: str,
    ) -> None:
        """Initializes the TD3SEEUpdateRule class.

        Args:
            exploitation_critic_1 (nn.Module): First exploitation critic network.
            exploitation_critic_2 (nn.Module): Second exploitation critic network.
            exploitation_actor (nn.Module): Exploitation actor network.
            exploration_critic_1 (nn.Module): First exploration critic network.
            exploration_critic_2 (nn.Module): Second exploration critic network.
            exploration_actor (nn.Module): Exploration actor network.
            exploitation_discount (float): The discount factor used for the exploitation objective.
            exploration_discount (float): The discount factor used for the exploration objective.
            observation_space (Space): The observation space of the environment.
            action_space (Space): The action space of the environment.
            update_data_provider (UpdateDataProvider): The data provider used to get data from the DataCollector.
            exploitation_critic_optimizer_class (Type[Optimizer]): The optimizer class to use for the exploitation critics.
            exploitation_actor_optimizer_class (Type[Optimizer]): The optimizer class to use for the exploitation actor.
            exploration_critic_optimizer_class (Type[Optimizer]): The optimizer class to use for the exploration critics.
            exploration_actor_optimizer_class (Type[Optimizer]): The optimizer class to use for the exploration actor.
            exploitation_critic_optimizer_arguments (Dict[str, Any]): The arguments used to initialize the exploitation critic optimizer without the parameters.
            exploitation_actor_optimizer_arguments (Dict[str, Any]): The arguments used to initialize the exploitation actor optimizer without the parameters.
            exploration_critic_optimizer_arguments (Dict[str, Any]): The arguments used to initialize the exploration critic optimizer without the parameters.
            exploration_actor_optimizer_arguments (Dict[str, Any]): The arguments used to initialize the exploration actor optimizer without the parameters.
            exploitation_critic_criteria (torch.nn.modules.loss._Loss): The loss function used to calculate the loss for the exploitation critics.
            exploration_critic_criteria (torch.nn.modules.loss._Loss): The loss function used to calculate the loss for the exploration critics.
            exploration_target_calculation_method (str): The method used to calculate the target for the exploration critic. This can be "bellmann" or "maximum".
            exploitation_critic_update_frequency (int): The update frequency of the exploitation critics based on the number of steps in the environment. If -1 the update will happen at the end of each episode.
            exploitation_critic_number_of_updates (int): The number of updates of the exploitation critic to perform per update step.
            exploitation_actor_update_frequency (int): The update frequency of the exploitation actor based on the number of steps in the environment. If -1 the update will happen at the end of each episode.
            exploration_critic_update_frequency (int): The update frequency of the exploration critics based on the number of steps in the environment. If -1 the update will happen at the end of each episode.
            exploration_critic_number_of_updates (int): The number of updates of the exploration critic to perform per update step.
            exploration_actor_update_frequency (int): The update frequency of the exploration actor based on the number of steps in the environment. If -1 the update will happen at the end of each episode.
            exploration_actor_number_of_updates (int): The number of updates of the exploration actor to perform per update step.
            multiply_number_of_updates_by_environment_steps (bool): Whether to multiply the number of updates by the number of steps in the environment since the last update.
            exploitation_target_critic_update_frequency (int): The update frequency of the exploitation target critic based on the number of steps in the environment.
            exploitation_target_actor_update_frequency (int): The update frequency of the exploitation target actor based on the number of steps in the environment.
            exploration_target_critic_update_frequency (int): The update frequency of the exploration target critic based on the number of steps in the environment.
            exploration_target_actor_update_frequency (int): The update frequency of the exploration target actor based on the number of steps in the environment.
            exploitation_target_critic_tau (float): The soft update factor for the exploitation target critic. If None, a hard update will be performed.
            exploitation_target_actor_tau (float): The soft update factor for the exploitation target actor. If None, a hard update will be performed.
            exploration_target_critic_tau (float): The soft update factor for the exploration target critic. If None, a hard update will be performed.
            exploration_target_actor_tau (float): The soft update factor for the exploration target actor. If None, a hard update will be performed.
            exploitation_target_noise_std (float): The standard deviation of the noise added to the target actions for the exploitation critic.
            exploitation_target_noise_clip (float): The maximum absolute value of the noise added to the target actions for the exploitation critic.
            exploration_target_noise_std (float): The standard deviation of the noise added to the target actions for the exploration critic.
            exploration_target_noise_clip (float): The maximum absolute value of the noise added to the target actions for the exploration critic.
            exploitation_critic_gradient_max_norm (Optional[float]): The maximum norm of the gradients for the exploitation critic. If None, no gradient clipping will be performed.
            exploitation_actor_gradient_max_norm (Optional[float]): The maximum norm of the gradients for the exploitation actor. If None, no gradient clipping will be performed.
            exploration_critic_gradient_max_norm (Optional[float]): The maximum norm of the gradients for the exploration critic. If None, no gradient clipping will be performed.
            exploration_actor_gradient_max_norm (Optional[float]): The maximum norm of the gradients for the exploration actor. If None, no gradient clipping will be performed.
            replay_buffer_capacity (int): The capacity of the replay buffer.
            replay_buffer_mini_batch_size (int): The mini-batch size used to sample data from the replay buffer.
            additional_replay_buffer_arguments (Optional[Dict[str, Any]]): The additional arguments used to initialize the replay buffer.
            post_replay_buffer_data_preprocessing (Optional[Dict[str, Callable]]): A dictionary of functions that are applied to the data from the replay buffer. The keys are the names of the data fields to which the functions should be applied and the values are the functions themselves.
            device (str): The device to use for the training e.g. "cpu" or "cuda".
        """

        UpdateRule.__init__(self)
        CompositeSaveableComponent.__init__(self)

        # move modules to device
        self.exploitation_critic_1 = exploitation_critic_1.to(device)
        self.exploitation_critic_2 = exploitation_critic_2.to(device)
        self.exploitation_actor = exploitation_actor.to(device)
        self.exploration_critic_1 = exploration_critic_1.to(device)
        self.exploration_critic_2 = exploration_critic_2.to(device)
        self.exploration_actor = exploration_actor.to(device)

        # register as saveable components
        self.register_saveable_component(
            "exploitation_critic_1", self.exploitation_critic_1
        )
        self.register_saveable_component(
            "exploitation_critic_2", self.exploitation_critic_2
        )
        self.register_saveable_component("exploitation_actor", self.exploitation_actor)
        self.register_saveable_component(
            "exploration_critic_1", self.exploration_critic_1
        )
        self.register_saveable_component(
            "exploration_critic_2", self.exploration_critic_2
        )
        self.register_saveable_component("exploration_actor", self.exploration_actor)

        # Initialize optimizers
        self.exploitation_critic_optimizer = exploitation_critic_optimizer_class(
            [
                {
                    "params": exploitation_critic_1.parameters(),
                    **exploitation_critic_optimizer_arguments,
                },
                {
                    "params": exploitation_critic_2.parameters(),
                    **exploitation_critic_optimizer_arguments,
                },
            ]
        )

        self.exploitation_actor_optimizer = exploitation_actor_optimizer_class(
            params=exploitation_actor.parameters(),
            **exploitation_actor_optimizer_arguments
        )
        self.exploration_critic_optimizer = exploration_critic_optimizer_class(
            [
                {
                    "params": exploration_critic_1.parameters(),
                    **exploration_critic_optimizer_arguments,
                },
                {
                    "params": exploration_critic_2.parameters(),
                    **exploration_critic_optimizer_arguments,
                },
            ]
        )

        self.exploration_actor_optimizer = exploration_actor_optimizer_class(
            params=exploration_actor.parameters(),
            **exploration_actor_optimizer_arguments
        )

        # register as saveable components
        self.register_saveable_component(
            "exploitation_critic_optimizer", self.exploitation_critic_optimizer
        )
        self.register_saveable_component(
            "exploitation_actor_optimizer", self.exploitation_actor_optimizer
        )
        self.register_saveable_component(
            "exploration_critic_optimizer", self.exploration_critic_optimizer
        )
        self.register_saveable_component(
            "exploration_actor_optimizer", self.exploration_actor_optimizer
        )

        # Create Target Networks
        self.exploitation_target_critic_1 = copy.deepcopy(exploitation_critic_1).eval()
        self.exploitation_target_critic_2 = copy.deepcopy(exploitation_critic_2).eval()
        self.exploitation_target_actor = copy.deepcopy(exploitation_actor).eval()
        self.exploration_target_critic_1 = copy.deepcopy(exploration_critic_1).eval()
        self.exploration_target_critic_2 = copy.deepcopy(exploration_critic_2).eval()
        self.exploration_target_actor = copy.deepcopy(exploration_actor).eval()

        self.exploitation_target_critic_1.requires_grad_(False)
        self.exploitation_target_critic_2.requires_grad_(False)
        self.exploitation_target_actor.requires_grad_(False)
        self.exploration_target_critic_1.requires_grad_(False)
        self.exploration_target_critic_2.requires_grad_(False)
        self.exploration_target_actor.requires_grad_(False)

        # register as saveable components
        self.register_saveable_component(
            "exploitation_target_critic_1", self.exploitation_target_critic_1
        )
        self.register_saveable_component(
            "exploitation_target_critic_2", self.exploitation_target_critic_2
        )
        self.register_saveable_component(
            "exploitation_target_actor", self.exploitation_target_actor
        )
        self.register_saveable_component(
            "exploration_target_critic_1", self.exploration_target_critic_1
        )
        self.register_saveable_component(
            "exploration_target_critic_2", self.exploration_target_critic_2
        )
        self.register_saveable_component(
            "exploration_target_actor", self.exploration_target_actor
        )

        # Updatable components

        # Replay buffer update
        additional_arguments = {
            "next_of": constants.DATA_OBSERVATIONS,
        }
        if additional_replay_buffer_arguments is not None:
            additional_arguments.update(additional_replay_buffer_arguments)

        update_data_info = create_transition_data_info(
            observation_space=observation_space,
            action_space=action_space,
        )

        self.replay_buffer = EpisodicCPPReplayBuffer(
            capacity=replay_buffer_capacity,
            replay_buffer_info=update_data_info,
            additional_arguments=additional_arguments,
        )

        self.register_saveable_component("replay_buffer", self.replay_buffer)

        self.replay_buffer_update = ReplayBufferUpdate(
            update_data_provider=update_data_provider,
            replay_buffer=self.replay_buffer,
        )

        if post_replay_buffer_data_preprocessing is not None:
            sample_replay_buffer = PostBufferPreprocessingWrapper(
                replay_buffer=self.replay_buffer,
                post_replay_buffer_data_preprocessing=post_replay_buffer_data_preprocessing,
            )
        else:
            sample_replay_buffer = self.replay_buffer

        # Exploitation Critic Update
        critic_data_keys = list(update_data_info.keys())
        critic_data_sampler = lambda: extract_data_from_batch(
            sample_replay_buffer.sample(replay_buffer_mini_batch_size),
            keys=critic_data_keys,
            device=device,
        )

        self.exploitation_critic_update = TD3CriticUpdate(
            data_sampler=critic_data_sampler,
            critic_1=self.exploitation_critic_1,
            critic_2=self.exploitation_critic_2,
            target_critic_1=self.exploitation_target_critic_1,
            target_critic_2=self.exploitation_target_critic_2,
            critic_optimizer=self.exploitation_critic_optimizer,
            target_actor=self.exploitation_target_actor,
            discount=exploitation_discount,
            target_noise_std=exploitation_target_noise_std,
            target_noise_clip=exploitation_target_noise_clip,
            changes_policy=True,
            update_frequency=exploitation_critic_update_frequency,
            number_of_updates=exploitation_critic_number_of_updates,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
            criteria=exploitation_critic_criteria,
            log_tag=self.EXPLOITATION_CRITIC_LOSS_LOG_TAG,
            save_file_name=self.EXPLOITATION_CRITIC_UPDATE_SAVE_FILE_NAME,
            gradient_max_norm=exploitation_critic_gradient_max_norm,
        )

        self.register_saveable_component(
            "exploitation_critic_update", self.exploitation_critic_update
        )

        # Exploitation Actor Update
        actor_data_keys = [constants.DATA_OBSERVATIONS]
        actor_data_sampler = lambda: extract_data_from_batch(
            sample_replay_buffer.sample(replay_buffer_mini_batch_size),
            keys=actor_data_keys,
            device=device,
        )

        self.exploitation_actor_update = DDPGActorUpdate(
            data_sampler=actor_data_sampler,
            actor=self.exploitation_actor,
            actor_optimizer=self.exploitation_actor_optimizer,
            critic=self.exploitation_critic_1,
            changes_policy=True,
            update_frequency=exploitation_actor_update_frequency,
            number_of_updates=exploitation_actor_number_of_updates,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
            log_tag=self.EXPLOITATION_ACTOR_LOSS_LOG_TAG,
            save_file_name=self.EXPLOITATION_ACTOR_UPDATE_SAVE_FILE_NAME,
            gradient_max_norm=exploitation_actor_gradient_max_norm,
        )

        self.register_saveable_component(
            "exploitation_actor_update", self.exploitation_actor_update
        )

        # Exploration Critic Update

        self.exploration_critic_update = TD3SEECriticUpdate(
            data_sampler=critic_data_sampler,
            exploration_critic_1=self.exploration_critic_1,
            exploration_critic_2=self.exploration_critic_2,
            exploration_target_critic_1=self.exploration_target_critic_1,
            exploration_target_critic_2=self.exploration_target_critic_2,
            exploration_target_actor=self.exploration_target_actor,
            exploration_critic_optimizer=self.exploration_critic_optimizer,
            exploitation_critic=self.exploitation_critic_1,
            exploitation_actor=self.exploitation_actor,
            target_calculation_method=exploration_target_calculation_method,
            discount_exploitation=exploitation_discount,
            discount_exploration=exploration_discount,
            target_noise_std=exploration_target_noise_std,
            target_noise_clip=exploration_target_noise_clip,
            exploitation_criteria=torch.nn.L1Loss(),  # We use the absolute td-error of the exploitation critic as the target for the exploration critic
            exploration_criteria=exploration_critic_criteria,
            update_frequency=exploration_critic_update_frequency,
            number_of_updates=exploration_critic_number_of_updates,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
            log_tag=self.EXPLORATION_CRITIC_LOSS_LOG_TAG,
            gradient_max_norm=exploration_critic_gradient_max_norm,
            changes_policy=True,
            save_file_name=self.EXPLORATION_CRITIC_UPDATE_SAVE_FILE_NAME,
        )

        self.register_saveable_component(
            "exploration_critic_update", self.exploration_critic_update
        )

        # Exploration Actor Update
        self.exploration_actor_update = TD3SEEActorUpdate(
            data_sampler=actor_data_sampler,
            exploration_critic=self.exploration_critic_1,
            exploration_actor=self.exploration_actor,
            exploitation_critic=self.exploitation_critic_1,
            actor_optimizer=self.exploration_actor_optimizer,
            update_frequency=exploration_actor_update_frequency,
            number_of_updates=exploration_actor_number_of_updates,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
            log_tag=self.EXPLORATION_ACTOR_LOSS_LOG_TAG,
            gradient_max_norm=exploration_actor_gradient_max_norm,
            changes_policy=True,
            save_file_name=self.EXPLORATION_ACTOR_UPDATE_SAVE_FILE_NAME,
        )

        self.register_saveable_component(
            "exploration_actor_update", self.exploration_actor_update
        )

        # exploitation target critic_1 update
        self.exploitation_target_critic_1_update = TargetNetUpdate(
            source_net=self.exploitation_critic_1,
            target_net=self.exploitation_target_critic_1,
            tau=exploitation_target_critic_tau,
            update_frequency=exploitation_target_critic_update_frequency,
        )

        # exploitation target critic_2 update
        self.exploitation_target_critic_2_update = TargetNetUpdate(
            source_net=self.exploitation_critic_2,
            target_net=self.exploitation_target_critic_2,
            tau=exploitation_target_critic_tau,
            update_frequency=exploitation_target_critic_update_frequency,
        )

        # exploration target critic_1 update
        self.exploration_target_critic_1_update = TargetNetUpdate(
            source_net=self.exploration_critic_1,
            target_net=self.exploration_target_critic_1,
            tau=exploration_target_critic_tau,
            update_frequency=exploration_target_critic_update_frequency,
        )

        # exploration target critic_2 update
        self.exploration_target_critic_2_update = TargetNetUpdate(
            source_net=self.exploration_critic_2,
            target_net=self.exploration_target_critic_2,
            tau=exploration_target_critic_tau,
            update_frequency=exploration_target_critic_update_frequency,
        )

        # exploitation target actor update
        self.exploitation_target_actor_update = TargetNetUpdate(
            source_net=self.exploitation_actor,
            target_net=self.exploitation_target_actor,
            tau=exploitation_target_actor_tau,
            update_frequency=exploitation_target_actor_update_frequency,
        )

        # exploration target actor update
        self.exploration_target_actor_update = TargetNetUpdate(
            source_net=self.exploration_actor,
            target_net=self.exploration_target_actor,
            tau=exploration_target_actor_tau,
            update_frequency=exploration_target_actor_update_frequency,
        )

    @property
    def updatable_components(self) -> Tuple[UpdatableComponent, ...]:
        """Returns the updatable components of the TD3SEEUpdateRule in the order they should be updated in.

        Returns:
            Tuple[UpdatableComponent, ...]: A tuple of updatable components:
                1. Replay buffer update
                2. Exploitation critic update
                3. Exploitation actor update
                4. Exploration critic update
                5. Exploration actor update
                6. Exploitation target critic 1 update
                7. Exploitation target critic 2 update
                8. Exploration target critic 1 update
                9. Exploration target critic 2 update
        """
        return (
            self.replay_buffer_update,
            self.exploitation_critic_update,
            self.exploitation_actor_update,
            self.exploration_critic_update,
            self.exploration_actor_update,
            self.exploitation_target_critic_1_update,
            self.exploitation_target_critic_2_update,
            self.exploration_target_critic_1_update,
            self.exploration_target_critic_2_update,
            self.exploitation_target_actor_update,
            self.exploration_target_actor_update,
        )
