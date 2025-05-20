from typing import Callable, Dict, Any, Type, Optional, Tuple, Union
import copy

import torch
from torch import nn
from torch.optim import Optimizer

from gymnasium.spaces import Box

import numpy as np

from athlete import constants
from athlete.update.update_rule import UpdateRule, UpdatableComponent
from athlete.algorithms.sac.updatable_components import (
    SACCriticUpdate,
    SACActorUpdate,
    SACTemperatureUpdate,
)
from athlete.update.common import TargetNetUpdate, ReplayBufferUpdate
from athlete.saving.saveable_component import CompositeSaveableComponent
from athlete.function import create_transition_data_info, extract_data_from_batch
from athlete.update.buffer import EpisodicCPPReplayBuffer
from athlete.update.buffer_wrapper import PostBufferPreprocessingWrapper
from athlete.data_collection.provider import UpdateDataProvider
from athlete.algorithms.sac.module import SACActor

from sac_see.updatable_components import SACSEECriticUpdate, SACSEEActorUpdate
from module import FingerprintingConditionedContinuousQValueFunction


# Soft Actor Critic + Stationary Error-seeking Exploration


class SACSEEUpdateRule(UpdateRule, CompositeSaveableComponent):
    """This class is responsible for managing the updatable components of the SAC+SEE algorithm.
    It provides all required dependencies and also saves most of the stateful components upon
    checkpointing.
    """

    EXPLOITATION_CRITIC_LOSS_LOG_TAG = "exploitation_critic_loss"
    EXPLOITATION_ACTOR_LOSS_LOG_TAG = "exploitation_actor_loss"
    EXPLOITATION_TEMPERATURE_LOSS_LOG_TAG = "exploitation_temperature_loss"
    EXPLOITATION_TEMPERATURE_LOG_TAG = "exploitation_temperature"
    EXPLORATION_CRITIC_LOSS_LOG_TAG = "exploration_critic_loss"
    EXPLORATION_ACTOR_LOSS_LOG_TAG = "exploration_actor_loss"
    EXPLORATION_TEMPERATURE_LOSS_LOG_TAG = "exploration_temperature_loss"
    EXPLORATION_TEMPERATURE_LOG_TAG = "exploration_temperature"

    EXPLOITATION_CRITIC_UPDATE_SAVE_FILE_NAME = "exploitation_critic_update"
    EXPLOITATION_ACTOR_UPDATE_SAVE_FILE_NAME = "exploitation_actor_update"
    EXPLORATION_CRITIC_UPDATE_SAVE_FILE_NAME = "exploration_critic_update"
    EXPLORATION_ACTOR_UPDATE_SAVE_FILE_NAME = "exploration_actor_update"
    EXPLOITATION_TEMPERATURE_UPDATE_SAVE_FILE_NAME = "exploitation_temperature_update"
    EXPLORATION_TEMPERATURE_UPDATE_SAVE_FILE_NAME = "exploration_temperature_update"

    SETTING_AUTO = "auto"

    def __init__(
        self,
        exploitation_critic_1: nn.Module,
        exploitation_critic_2: nn.Module,
        exploitation_actor: SACActor,
        exploration_critic_1: FingerprintingConditionedContinuousQValueFunction,
        exploration_critic_2: FingerprintingConditionedContinuousQValueFunction,
        exploration_actor: SACActor,
        exploitation_discount: float,
        exploration_discount: float,
        observation_space: Box,
        action_space: Box,
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
        exploitation_temperature: Union[float, str],
        exploitation_target_entropy: Union[float, str],
        exploitation_initial_temperature: float,  # only used if temperature is "auto"
        exploration_temperature: Union[float, str],
        exploration_target_entropy: Union[float, str],
        exploration_initial_temperature: float,  # only used if temperature is "auto"
        exploration_target_calculation_method: str,
        exploitation_critic_update_frequency: int,
        exploitation_critic_number_of_updates: int,
        exploitation_actor_update_frequency: int,
        exploitation_actor_number_of_updates: int,
        exploration_critic_update_frequency: int,
        exploration_critic_number_of_updates: int,
        exploration_actor_update_frequency: int,
        exploration_actor_number_of_updates: int,
        multiply_number_of_updates_by_environment_steps: bool,
        exploitation_target_critic_update_frequency: int,
        exploration_target_critic_update_frequency: int,
        exploitation_target_critic_tau: float,
        exploration_target_critic_tau: float,
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
        """Initializes the SACSEEUpdateRule class.

        Args:
            exploitation_critic_1 (nn.Module): First critic network for the exploitation objective.
            exploitation_critic_2 (nn.Module): Second critic network for the exploitation objective.
            exploitation_actor (SACActor): Actor network for the exploitation objective.
            exploration_critic_1 (FingerprintingConditionedContinuousQValueFunction): First critic network for the exploration objective.
            exploration_critic_2 (FingerprintingConditionedContinuousQValueFunction): First critic network for the exploration objective.
            exploration_actor (SACActor): Actor network for the exploration objective.
            exploitation_discount (float): Discount factor for the exploitation objective.
            exploration_discount (float): Discount factor for the exploration objective.
            observation_space (Box): Observation space of the environment.
            action_space (Box): Action space of the environment.
            update_data_provider (UpdateDataProvider): Data provider used to get data from the DataCollector.
            exploitation_critic_optimizer_class (Type[Optimizer]): Optimizer class for the exploitation critic.
            exploitation_actor_optimizer_class (Type[Optimizer]): Optimizer class for the exploitation actor.
            exploration_critic_optimizer_class (Type[Optimizer]): Optimizer class for the exploration critic.
            exploration_actor_optimizer_class (Type[Optimizer]): Optimizer class for the exploration actor.
            exploitation_critic_optimizer_arguments (Dict[str, Any]): Initialization arguments for the exploitation critic optimizer, not including the parameters.
            exploitation_actor_optimizer_arguments (Dict[str, Any]): Initialization arguments for the exploitation actor optimizer, not including the parameters.
            exploration_critic_optimizer_arguments (Dict[str, Any]): Initialization arguments for the exploration critic optimizer, not including the parameters.
            exploration_actor_optimizer_arguments (Dict[str, Any]): Initialization arguments for the exploration actor optimizer, not including the parameters.
            exploitation_critic_criteria (torch.nn.modules.loss._Loss): Loss function used to calculate the loss of the exploitation critics.
            exploration_critic_criteria (torch.nn.modules.loss._Loss): Loss function used to calculate the loss of the exploration critics.
            exploitation_temperature (Union[float, str]): Temperature for the entropy bonus of the exploitation actor. If "auto", the temperature is learned according to the target entropy.
            exploitation_target_entropy (Union[float, str]): The target entropy for the exploitation actor. If "auto", the target entropy is set to -action_space.shape[0].
            exploitation_initial_temperature (float): The initial temperature for the exploitation actor. Only used if exploitation_temperature is "auto".
            exploration_target_entropy (Union[float, str]): The target entropy for the exploration actor. If "auto", the target entropy is set to -action_space.shape[0].
            exploration_initial_temperature (float): The initial temperature for the exploration actor. Only used if exploration_temperature is "auto".
            exploitation_critic_update_frequency (int): The update frequency of the exploitation critics by environment steps. If -1 the critics are updated at the end of each episode.
            exploitation_critic_number_of_updates (int): Tne number of update performed per update step.
            exploitation_actor_update_frequency (int): The update frequency of the exploitation actor by environment steps. If -1 the actor is updated at the end of each episode.
            exploitation_actor_number_of_updates (int): The number of update performed per update step.
            exploration_critic_update_frequency (int): The update frequency of the exploration critics by environment steps. If -1 the critics are updated at the end of each episode.
            exploration_critic_number_of_updates (int): The number of update performed per update step.
            exploration_actor_update_frequency (int): The update frequency of the exploration actor by environment steps. If -1 the actor is updated at the end of each episode.
            exploration_actor_number_of_updates (int): The number of update performed per update step.
            multiply_number_of_updates_by_environment_steps (bool): Whether to multiply the number of updates by the environment steps since the last update.
            exploitation_target_critic_update_frequency (int): The update frequency of the target critic networks by environment steps.
            exploration_target_critic_update_frequency (int): The update frequency of the target critic networks by environment steps.
            exploitation_target_critic_tau (float): The soft update factor for the target critic networks. If None a hard update is performed.
            exploration_target_critic_tau (float): The soft update factor for the target critic networks. If None a hard update is performed.
            exploitation_critic_gradient_max_norm (Optional[float]): The maximum norm for the gradients of the exploitation critics. If None no gradient clipping is performed.
            exploitation_actor_gradient_max_norm (Optional[float]): The maximum norm for the gradients of the exploitation actor. If None no gradient clipping is performed.
            exploration_critic_gradient_max_norm (Optional[float]): The maximum norm for the gradients of the exploration critics. If None no gradient clipping is performed.
            exploration_actor_gradient_max_norm (Optional[float]): The maximum norm for the gradients of the exploration actor. If None no gradient clipping is performed.
            replay_buffer_capacity (int): The capacity of the replay buffer.
            replay_buffer_mini_batch_size (int): The mini batch size of the replay buffer.
            additional_replay_buffer_arguments (Optional[Dict[str, Any]]): Additional arguments for the replay buffer.
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
        self.exploration_target_critic_1 = copy.deepcopy(exploration_critic_1).eval()
        self.exploration_target_critic_2 = copy.deepcopy(exploration_critic_2).eval()

        self.exploitation_target_critic_1.requires_grad_(False)
        self.exploitation_target_critic_2.requires_grad_(False)
        self.exploration_target_critic_1.requires_grad_(False)
        self.exploration_target_critic_2.requires_grad_(False)

        # register as saveable components
        self.register_saveable_component(
            "exploitation_target_critic_1", self.exploitation_target_critic_1
        )
        self.register_saveable_component(
            "exploitation_target_critic_2", self.exploitation_target_critic_2
        )
        self.register_saveable_component(
            "exploration_target_critic_1", self.exploration_target_critic_1
        )
        self.register_saveable_component(
            "exploration_target_critic_2", self.exploration_target_critic_2
        )

        # Temperature settings
        # Exploitation
        self.automatic_exploitation_temperature_update = (
            exploitation_temperature == self.SETTING_AUTO
        )

        if self.automatic_exploitation_temperature_update:
            # requires grad false because this is just a mirror of log_temperature which is being updated
            self.exploitation_temperature = torch.nn.Parameter(
                torch.tensor(exploitation_initial_temperature), requires_grad=False
            )
            self.exploitation_log_temperature = torch.nn.Parameter(
                torch.log(
                    torch.tensor(exploitation_initial_temperature, requires_grad=True)
                )
            )
            # for simplicity we use the same optimizer setting for the temperature as for the critic when needed
            self.exploitation_temperature_optimizer = (
                exploitation_critic_optimizer_class(
                    params=[self.exploitation_log_temperature],
                    **exploitation_critic_optimizer_arguments
                )
            )

            # We only need to save these if they are dynamically updated
            self.register_saveable_component(
                "exploitation_temperature", self.exploitation_temperature
            )
            self.register_saveable_component(
                "exploitation_log_temperature", self.exploitation_log_temperature
            )
            self.register_saveable_component(
                "exploitation_temperature_optimizer",
                self.exploitation_temperature_optimizer,
            )
        else:
            self.exploitation_temperature = torch.nn.Parameter(
                torch.log(torch.tensor(exploitation_temperature, requires_grad=False))
            )

        if exploitation_target_entropy == self.SETTING_AUTO:
            self.exploitation_target_entropy = float(
                -np.prod(action_space.shape).astype(np.float32)
            )
        else:
            self.exploitation_target_entropy = exploitation_target_entropy

        # Exploration
        self.automatic_exploration_temperature_update = (
            exploration_temperature == self.SETTING_AUTO
        )

        if self.automatic_exploration_temperature_update:
            # requires grad false because this is just a mirror of log_temperature which is being updated
            self.exploration_temperature = torch.nn.Parameter(
                torch.tensor(exploration_initial_temperature), requires_grad=False
            )
            self.exploration_log_temperature = torch.nn.Parameter(
                torch.log(
                    torch.tensor(exploration_initial_temperature, requires_grad=True)
                )
            )
            # for simplicity we use the same optimizer setting for the temperature as for the critic when needed
            self.exploration_temperature_optimizer = exploration_critic_optimizer_class(
                params=[self.exploration_log_temperature],
                **exploration_critic_optimizer_arguments
            )

            # We only need to save these if they are dynamically updated
            self.register_saveable_component(
                "exploration_temperature", self.exploration_temperature
            )
            self.register_saveable_component(
                "exploration_log_temperature", self.exploration_log_temperature
            )
            self.register_saveable_component(
                "exploration_temperature_optimizer",
                self.exploration_temperature_optimizer,
            )
        else:
            self.exploration_temperature = torch.nn.Parameter(
                torch.log(torch.tensor(exploration_temperature, requires_grad=False))
            )

        if exploration_target_entropy == self.SETTING_AUTO:
            self.exploration_target_entropy = float(
                -np.prod(action_space.shape).astype(np.float32)
            )
        else:
            self.exploration_target_entropy = exploration_target_entropy

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

        self.exploitation_critic_update = SACCriticUpdate(
            temperature=self.exploitation_temperature,
            actor=self.exploitation_actor,
            critic_1=self.exploitation_critic_1,
            critic_2=self.exploitation_critic_2,
            critic_optimizer=self.exploitation_critic_optimizer,
            target_critic_1=self.exploitation_target_critic_1,
            target_critic_2=self.exploitation_target_critic_2,
            data_sampler=critic_data_sampler,
            discount=exploitation_discount,
            criteria=exploitation_critic_criteria,
            update_frequency=exploitation_critic_update_frequency,
            number_of_updates=exploitation_critic_number_of_updates,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
            changes_policy=True,
            gradient_max_norm=exploitation_critic_gradient_max_norm,
            log_tag=self.EXPLOITATION_CRITIC_LOSS_LOG_TAG,
            save_file_name=self.EXPLOITATION_CRITIC_UPDATE_SAVE_FILE_NAME,
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

        self.exploitation_actor_update = SACActorUpdate(
            actor=self.exploitation_actor,
            actor_optimizer=self.exploitation_actor_optimizer,
            critic_1=self.exploitation_critic_1,
            critic_2=self.exploitation_critic_2,
            temperature=self.exploitation_temperature,
            data_sampler=actor_data_sampler,
            update_frequency=exploitation_actor_update_frequency,
            number_of_updates=exploitation_actor_number_of_updates,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
            changes_policy=True,
            log_tag=self.EXPLOITATION_ACTOR_LOSS_LOG_TAG,
            save_file_name=self.EXPLOITATION_ACTOR_UPDATE_SAVE_FILE_NAME,
            gradient_max_norm=exploitation_actor_gradient_max_norm,
        )

        self.register_saveable_component(
            "exploitation_actor_update", self.exploitation_actor_update
        )

        # Exploration Critic Update

        self.exploration_critic_update = SACSEECriticUpdate(
            exploration_temperature=self.exploration_temperature,
            exploitation_temperature=self.exploitation_temperature,
            exploration_critic_1=self.exploration_critic_1,
            exploration_critic_2=self.exploration_critic_2,
            exploration_target_critic_1=self.exploration_target_critic_1,
            exploration_target_critic_2=self.exploration_target_critic_2,
            exploration_actor=self.exploration_actor,
            exploration_critic_optimizer=self.exploration_critic_optimizer,
            data_sampler=critic_data_sampler,
            exploitation_critic_1=self.exploitation_critic_1,
            exploitation_critic_2=self.exploitation_critic_2,
            exploitation_actor=self.exploitation_actor,
            target_calculation_method=exploration_target_calculation_method,
            discount_exploitation=exploitation_discount,
            discount_exploration=exploration_discount,
            exploitation_criteria=torch.nn.L1Loss(),  # We use the absolute td-error of the exploitation critic as the target for the exploration critic
            exploration_criteria=exploration_critic_criteria,
            number_of_updates=exploration_critic_number_of_updates,
            log_tag=self.EXPLORATION_CRITIC_LOSS_LOG_TAG,
            gradient_max_norm=exploration_critic_gradient_max_norm,
            changes_policy=True,
            update_frequency=exploration_critic_update_frequency,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
            save_file_name=self.EXPLORATION_CRITIC_UPDATE_SAVE_FILE_NAME,
        )

        self.register_saveable_component(
            "exploration_critic_update", self.exploration_critic_update
        )

        # Exploration Actor Update
        self.exploration_actor_update = SACSEEActorUpdate(
            temperature=self.exploration_temperature,
            exploration_critic_1=self.exploration_critic_1,
            exploration_critic_2=self.exploration_critic_2,
            exploration_actor=self.exploration_actor,
            exploitation_critic_1=self.exploitation_critic_1,
            exploitation_critic_2=self.exploitation_critic_2,
            actor_optimizer=self.exploration_actor_optimizer,
            data_sampler=actor_data_sampler,
            changes_policy=True,
            update_frequency=exploration_actor_update_frequency,
            number_of_updates=exploration_actor_number_of_updates,
            multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
            log_tag=self.EXPLORATION_ACTOR_LOSS_LOG_TAG,
            save_file_name=self.EXPLORATION_ACTOR_UPDATE_SAVE_FILE_NAME,
            gradient_max_norm=exploration_actor_gradient_max_norm,
        )

        self.register_saveable_component(
            "exploration_actor_update", self.exploration_actor_update
        )

        # Exploitation Temperature Update
        if self.automatic_exploitation_temperature_update:
            self.exploitation_temperature_update = SACTemperatureUpdate(
                target_entropy=self.exploitation_target_entropy,
                log_temperature=self.exploitation_log_temperature,
                temperature=self.exploitation_temperature,
                temperature_optimizer=self.exploitation_temperature_optimizer,
                actor=self.exploitation_actor,
                data_sampler=actor_data_sampler,
                update_frequency=exploitation_actor_update_frequency,
                number_of_updates=exploitation_actor_number_of_updates,
                multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
                changes_policy=False,
                gradient_max_norm=None,
                temperature_log_tag=self.EXPLOITATION_TEMPERATURE_LOG_TAG,
                loss_log_tag=self.EXPLOITATION_TEMPERATURE_LOSS_LOG_TAG,
                save_file_name=self.EXPLOITATION_TEMPERATURE_UPDATE_SAVE_FILE_NAME,
            )

            self.register_saveable_component(
                "exploitation_temperature_update", self.exploitation_temperature_update
            )

        # Exploration Temperature Update
        if self.automatic_exploration_temperature_update:
            self.exploration_temperature_update = SACTemperatureUpdate(
                target_entropy=self.exploration_target_entropy,
                log_temperature=self.exploration_log_temperature,
                temperature=self.exploration_temperature,
                temperature_optimizer=self.exploration_temperature_optimizer,
                actor=self.exploration_actor,
                data_sampler=actor_data_sampler,
                update_frequency=exploration_actor_update_frequency,
                number_of_updates=exploration_actor_number_of_updates,
                multiply_number_of_updates_by_environment_steps=multiply_number_of_updates_by_environment_steps,
                changes_policy=False,
                gradient_max_norm=None,
                temperature_log_tag=self.EXPLORATION_TEMPERATURE_LOG_TAG,
                loss_log_tag=self.EXPLORATION_TEMPERATURE_LOSS_LOG_TAG,
                save_file_name=self.EXPLORATION_TEMPERATURE_UPDATE_SAVE_FILE_NAME,
            )

            self.register_saveable_component(
                "exploration_temperature_update", self.exploration_temperature_update
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

        # create updatable components set based on automatic temperature updates
        _updatable_components = [
            self.replay_buffer_update,
            self.exploitation_critic_update,
            self.exploitation_actor_update,
            self.exploration_critic_update,
            self.exploration_actor_update,
            self.exploitation_target_critic_1_update,
            self.exploitation_target_critic_2_update,
            self.exploration_target_critic_1_update,
            self.exploration_target_critic_2_update,
        ]
        if self.automatic_exploitation_temperature_update:
            _updatable_components.append(self.exploitation_temperature_update)
        if self.automatic_exploration_temperature_update:
            _updatable_components.append(self.exploration_temperature_update)

        self._updatable_components = tuple(_updatable_components)

    @property
    def updatable_components(self) -> Tuple[UpdatableComponent, ...]:
        """Returns the updatable components of the SACSEEUpdateRule.

        Returns:
            Tuple[UpdatableComponent, ...]: Tuple of updatable components:
                1. Replay buffer update
                2. Exploitation critic update
                3. Exploitation actor update
                4. Exploration critic update
                5. Exploration actor update
                6. Exploitation target critic_1 update
                7. Exploitation target critic_2 update
                8. Exploration target critic_1 update
                9. Exploration target critic_2 update
                10. Exploitation temperature update (if temperature is "auto")
                11. Exploration temperature update (if temperature is "auto")
        """
        return self._updatable_components
