# CONTINUE HERE

from typing import Dict, Any, Tuple

from gymnasium.spaces import Space, Box
import torch

import athlete
from athlete.data_collection.collector import DataCollector
from athlete.data_collection.transition import (
    ActionReplacementGymnasiumTransitionDataCollector,
)
from athlete.update.update_rule import UpdateRule
from athlete.policy.policy_builder import PolicyBuilder
from athlete.data_collection.provider import UpdateDataProvider
from athlete.module.torch.common import FCContinuousQValueFunction
from athlete.algorithms.sac.module import GaussianActor
from athlete import constants

from module import FingerprintingConditionedContinuousQValueFunction
from sac_see.policy import INFO_KEY_UNSCALED_ACTION, SACSEEPolicyBuilder
from sac_see.update import SACSEEUpdateRule

# This file defines the default configuration for
# Soft Actor Critic + Stable Error-seeking Exploration
# and registers the algorithm with the Athlete API.

ARGUMENT_EXPLOITATION_DISCOUNT = "exploitation_discount"
ARGUMENT_EXPLORATION_DISCOUNT = "exploration_discount"
ARGUMENT_EXPLOITATION_CRITIC_1_NETWORK_CLASS = "exploitation_critic_1_network_class"
ARGUMENT_EXPLOITATION_CRITIC_1_NETWORK_ARGUMENTS = (
    "exploitation_critic_network_arguments"
)
ARGUMENT_EXPLOITATION_CRITIC_2_NETWORK_CLASS = "exploitation_critic_2_network_class"
ARGUMENT_EXPLOITATION_CRITIC_2_NETWORK_ARGUMENTS = (
    "exploitation_critic_2_network_arguments"
)
ARGUMENT_EXPLOITATION_ACTOR_NETWORK_CLASS = "exploitation_actor_network_class"
ARGUMENT_EXPLOITATION_ACTOR_NETWORK_ARGUMENTS = "exploitation_actor_network_arguments"
ARGUMENT_EXPLORATION_CRITIC_1_NETWORK_CLASS = "exploration_critic_1_network_class"
ARGUMENT_EXPLORATION_CRITIC_1_NETWORK_ARGUMENTS = (
    "exploration_critic_1_network_arguments"
)
ARGUMENT_EXPLORATION_CRITIC_2_NETWORK_CLASS = "exploration_critic_2_network_class"
ARGUMENT_EXPLORATION_CRITIC_2_NETWORK_ARGUMENTS = (
    "exploration_critic_2_network_arguments"
)
ARGUMENT_EXPLORATION_ACTOR_NETWORK_CLASS = "exploration_actor_network_class"
ARGUMENT_EXPLORATION_ACTOR_NETWORK_ARGUMENTS = "exploration_actor_network_arguments"

ARGUMENT_EXPLOITATION_CRITIC_OPTIMIZER_CLASS = "exploitation_critic_optimizer_class"
ARGUMENT_EXPLOITATION_CRITIC_OPTIMIZER_ARGUMENTS = (
    "exploitation_critic_optimizer_arguments"
)
ARGUMENT_EXPLOITATION_ACTOR_OPTIMIZER_CLASS = "exploitation_actor_optimizer_class"
ARGUMENT_EXPLOITATION_ACTOR_OPTIMIZER_ARGUMENTS = (
    "exploitation_actor_optimizer_arguments"
)
ARGUMENT_EXPLORATION_CRITIC_OPTIMIZER_CLASS = "exploration_critic_optimizer_class"
ARGUMENT_EXPLORATION_CRITIC_OPTIMIZER_ARGUMENTS = (
    "exploration_critic_optimizer_arguments"
)
ARGUMENT_EXPLORATION_ACTOR_OPTIMIZER_CLASS = "exploration_actor_optimizer_class"
ARGUMENT_EXPLORATION_ACTOR_OPTIMIZER_ARGUMENTS = "exploration_actor_optimizer_arguments"
ARGUMENT_EXPLOITATION_TEMPERATURE = "exploitation_temperature"
ARGUMENT_EXPLOITATION_TARGET_ENTROPY = "exploitation_target_entropy"
ARGUMENT_EXPLOITATION_INITIAL_TEMPERATURE = "exploitation_initial_temperature"
ARGUMENT_EXPLORATION_TEMPERATURE = "exploration_temperature"
ARGUMENT_EXPLORATION_TARGET_ENTROPY = "exploration_target_entropy"
ARGUMENT_EXPLORATION_INITIAL_TEMPERATURE = "exploration_initial_temperature"
ARGUMENT_REPLAY_BUFFER_CAPACITY = "replay_buffer_capacity"
ARGUMENT_REPLAY_BUFFER_MINI_BATCH_SIZE = "replay_buffer_mini_batch_size"
ARGUMENT_EXPLOITATION_CRITIC_UPDATE_FREQUENCY = "exploitation_critic_update_frequency"
ARGUMENT_EXPLOITATION_CRITIC_NUMBER_OF_UPDATES = "exploitation_critic_number_of_updates"
ARGUMENT_EXPLOITATION_ACTOR_UPDATE_FREQUENCY = "exploitation_actor_update_frequency"
ARGUMENT_EXPLOITATION_ACTOR_NUMBER_OF_UPDATES = "exploitation_actor_number_of_updates"
ARGUMENT_EXPLORATION_CRITIC_UPDATE_FREQUENCY = "exploration_critic_update_frequency"
ARGUMENT_EXPLORATION_CRITIC_NUMBER_OF_UPDATES = "exploration_critic_number_of_updates"
ARGUMENT_EXPLORATION_ACTOR_UPDATE_FREQUENCY = "exploration_actor_update_frequency"
ARGUMENT_EXPLORATION_ACTOR_NUMBER_OF_UPDATES = "exploration_actor_number_of_updates"
ARGUMENT_MULTIPLY_NUMBER_OF_UPDATES_BY_ENVIRONMENT_STEPS = (
    "multiply_number_of_updates_by_environment_steps"
)
ARGUMENT_EXPLOITATION_TARGET_CRITIC_UPDATE_FREQUENCY = (
    "exploitation_target_critic_update_frequency"
)
ARGUMENT_EXPLOITATION_TARGET_CRITIC_TAU = "exploitation_target_critic_tau"
ARGUMENT_EXPLORATION_TARGET_CRITIC_UPDATE_FREQUENCY = (
    "exploration_target_critic_update_frequency"
)
ARGUMENT_EXPLORATION_TARGET_CRITIC_TAU = "exploration_target_critic_tau"
ARGUMENT_EXPLOITATION_CRITIC_CRITERIA = "exploitation_critic_criteria"
ARGUMENT_EXPLORATION_CRITIC_CRITERIA = "exploration_critic_criteria"
ARGUMENT_EXPLOITATION_CRITIC_GRADIENT_MAX_NORM = "exploitation_critic_gradient_max_norm"
ARGUMENT_EXPLOITATION_ACTOR_GRADIENT_MAX_NORM = "exploitation_actor_gradient_max_norm"
ARGUMENT_EXPLORATION_CRITIC_GRADIENT_MAX_NORM = "exploration_critic_gradient_max_norm"
ARGUMENT_EXPLORATION_ACTOR_GRADIENT_MAX_NORM = "exploration_actor_gradient_max_norm"
ARGUMENT_DEVICE = "device"
ARGUMENT_ADDITIONAL_REPLAY_BUFFER_ARGUMENTS = "additional_replay_buffer_arguments"
ARGUMENT_POST_REPLAY_BUFFER_DATA_PREPROCESSING = "post_replay_buffer_data_preprocessing"
ARGUMENT_POLICY_MIXTURE = "policy_mixture"
ARGUMENT_ACTION_SELECTION_TEMPERATURE = "action_selection_temperature"
ARGUMENT_EXPLORATION_TARGET_CALCULATION_METHOD = "exploration_target_calculation_method"
ARGUMENT_USE_ALTERNATING_TRAINING_POLICY = "use_alternating_training_policy"

DEFAULT_CONFIGURATION = {
    ARGUMENT_EXPLOITATION_DISCOUNT: 0.99,
    ARGUMENT_EXPLORATION_DISCOUNT: 0.99,
    ARGUMENT_EXPLOITATION_CRITIC_1_NETWORK_CLASS: FCContinuousQValueFunction,
    ARGUMENT_EXPLOITATION_CRITIC_1_NETWORK_ARGUMENTS: {
        "hidden_dims": [400, 300],
        "init_state_dict_path": None,
    },
    ARGUMENT_EXPLOITATION_CRITIC_2_NETWORK_CLASS: FCContinuousQValueFunction,
    ARGUMENT_EXPLOITATION_CRITIC_2_NETWORK_ARGUMENTS: {
        constants.GENERAL_ARGUMENT_OBSERVATION_SPACE: constants.VALUE_PLACEHOLDER,
        constants.GENERAL_ARGUMENT_ACTION_SPACE: constants.VALUE_PLACEHOLDER,
        "hidden_dims": [400, 300],
        "init_state_dict_path": None,
    },
    ARGUMENT_EXPLOITATION_ACTOR_NETWORK_CLASS: GaussianActor,
    ARGUMENT_EXPLOITATION_ACTOR_NETWORK_ARGUMENTS: {
        constants.GENERAL_ARGUMENT_OBSERVATION_SPACE: constants.VALUE_PLACEHOLDER,
        constants.GENERAL_ARGUMENT_ACTION_SPACE: constants.VALUE_PLACEHOLDER,
        "hidden_dims": [400, 300],
        "log_std_min": -20.0,
        "log_std_max": 2.0,
        "init_state_dict_path": None,
    },
    ARGUMENT_EXPLORATION_CRITIC_1_NETWORK_CLASS: FingerprintingConditionedContinuousQValueFunction,
    ARGUMENT_EXPLORATION_CRITIC_1_NETWORK_ARGUMENTS: {
        "example_conditioning_q_value_function": constants.VALUE_PLACEHOLDER,
        "hidden_dims": [400, 300],
        "num_probe_values": 16,
        constants.GENERAL_ARGUMENT_OBSERVATION_SPACE: constants.VALUE_PLACEHOLDER,
        constants.GENERAL_ARGUMENT_ACTION_SPACE: constants.VALUE_PLACEHOLDER,
        "normalized_observation": False,
        "preprocessing_pipe": constants.VALUE_PLACEHOLDER,
        "init_state_dict_path": None,
    },
    ARGUMENT_EXPLORATION_CRITIC_2_NETWORK_CLASS: FingerprintingConditionedContinuousQValueFunction,
    ARGUMENT_EXPLORATION_CRITIC_2_NETWORK_ARGUMENTS: {
        "example_conditioning_q_value_function": constants.VALUE_PLACEHOLDER,
        "hidden_dims": [400, 300],
        "num_probe_values": 16,
        constants.GENERAL_ARGUMENT_OBSERVATION_SPACE: constants.VALUE_PLACEHOLDER,
        constants.GENERAL_ARGUMENT_ACTION_SPACE: constants.VALUE_PLACEHOLDER,
        "normalized_observation": False,
        "preprocessing_pipe": constants.VALUE_PLACEHOLDER,
        "init_state_dict_path": None,
    },
    ARGUMENT_EXPLORATION_ACTOR_NETWORK_CLASS: GaussianActor,
    ARGUMENT_EXPLORATION_ACTOR_NETWORK_ARGUMENTS: {
        constants.GENERAL_ARGUMENT_OBSERVATION_SPACE: constants.VALUE_PLACEHOLDER,
        constants.GENERAL_ARGUMENT_ACTION_SPACE: constants.VALUE_PLACEHOLDER,
        "hidden_dims": [400, 300],
        "log_std_min": -20.0,
        "log_std_max": 2.0,
        "init_state_dict_path": None,
    },
    ARGUMENT_EXPLOITATION_CRITIC_OPTIMIZER_CLASS: torch.optim.Adam,
    ARGUMENT_EXPLOITATION_CRITIC_OPTIMIZER_ARGUMENTS: {"lr": 3e-4},
    ARGUMENT_EXPLOITATION_ACTOR_OPTIMIZER_CLASS: torch.optim.Adam,
    ARGUMENT_EXPLOITATION_ACTOR_OPTIMIZER_ARGUMENTS: {"lr": 3e-4},
    ARGUMENT_EXPLORATION_CRITIC_OPTIMIZER_CLASS: torch.optim.Adam,
    ARGUMENT_EXPLORATION_CRITIC_OPTIMIZER_ARGUMENTS: {"lr": 3e-4},
    ARGUMENT_EXPLORATION_ACTOR_OPTIMIZER_CLASS: torch.optim.Adam,
    ARGUMENT_EXPLORATION_ACTOR_OPTIMIZER_ARGUMENTS: {"lr": 3e-4},
    ARGUMENT_EXPLOITATION_TEMPERATURE: SACSEEUpdateRule.SETTING_AUTO,
    ARGUMENT_EXPLOITATION_TARGET_ENTROPY: SACSEEUpdateRule.SETTING_AUTO,
    ARGUMENT_EXPLOITATION_INITIAL_TEMPERATURE: 1.0,
    ARGUMENT_EXPLORATION_TEMPERATURE: SACSEEUpdateRule.SETTING_AUTO,
    ARGUMENT_EXPLORATION_TARGET_ENTROPY: SACSEEUpdateRule.SETTING_AUTO,
    ARGUMENT_EXPLORATION_INITIAL_TEMPERATURE: 1.0,
    ARGUMENT_REPLAY_BUFFER_CAPACITY: 1000000,
    ARGUMENT_REPLAY_BUFFER_MINI_BATCH_SIZE: 256,
    constants.GENERAL_ARGUMENT_WARMUP_STEPS: 10000,
    ARGUMENT_EXPLOITATION_CRITIC_UPDATE_FREQUENCY: 1,
    ARGUMENT_EXPLOITATION_CRITIC_NUMBER_OF_UPDATES: 1,
    ARGUMENT_EXPLOITATION_ACTOR_UPDATE_FREQUENCY: 1,
    ARGUMENT_EXPLOITATION_ACTOR_NUMBER_OF_UPDATES: 1,
    ARGUMENT_EXPLORATION_CRITIC_UPDATE_FREQUENCY: 1,
    ARGUMENT_EXPLORATION_CRITIC_NUMBER_OF_UPDATES: 1,
    ARGUMENT_EXPLORATION_ACTOR_UPDATE_FREQUENCY: 1,
    ARGUMENT_EXPLORATION_ACTOR_NUMBER_OF_UPDATES: 1,
    ARGUMENT_MULTIPLY_NUMBER_OF_UPDATES_BY_ENVIRONMENT_STEPS: False,
    ARGUMENT_EXPLOITATION_TARGET_CRITIC_UPDATE_FREQUENCY: 1,
    ARGUMENT_EXPLOITATION_TARGET_CRITIC_TAU: 0.005,
    ARGUMENT_EXPLORATION_TARGET_CRITIC_UPDATE_FREQUENCY: 1,
    ARGUMENT_EXPLORATION_TARGET_CRITIC_TAU: 0.005,
    ARGUMENT_POLICY_MIXTURE: 0.5,
    ARGUMENT_ACTION_SELECTION_TEMPERATURE: 1.0,
    ARGUMENT_EXPLORATION_TARGET_CALCULATION_METHOD: "maximum",
    ARGUMENT_EXPLOITATION_CRITIC_CRITERIA: torch.nn.MSELoss(),
    ARGUMENT_EXPLORATION_CRITIC_CRITERIA: torch.nn.MSELoss(),
    ARGUMENT_EXPLOITATION_CRITIC_GRADIENT_MAX_NORM: None,
    ARGUMENT_EXPLOITATION_ACTOR_GRADIENT_MAX_NORM: None,
    ARGUMENT_EXPLORATION_CRITIC_GRADIENT_MAX_NORM: None,
    ARGUMENT_EXPLORATION_ACTOR_GRADIENT_MAX_NORM: None,
    ARGUMENT_DEVICE: (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    ),
    ARGUMENT_ADDITIONAL_REPLAY_BUFFER_ARGUMENTS: {},
    ARGUMENT_POST_REPLAY_BUFFER_DATA_PREPROCESSING: None,
    ARGUMENT_USE_ALTERNATING_TRAINING_POLICY: False,
}


def make_sac_see_components(
    observation_space: Space, action_space: Space, configuration: Dict[str, Any]
) -> Tuple[DataCollector, UpdateRule, PolicyBuilder]:
    """Create the components for the SAC+SEE algorithm, given a configuration and environment
    information. The components are the data collector, update rule, and policy builder.
    For more information refer to the documentation of Athlete 0.1.2.

    Args:
        observation_space (Space): Observation space of the environment.
        action_space (Space): Action space of the environment.
        configuration (Dict[str, Any]): Configuration for the algorithm.

    Raises:
        ValueError: If the observation space is not a Box space.
        ValueError: If the action space is not a Box space.

    Returns:
        Tuple[DataCollector, UpdateRule, PolicyBuilder]: The data collector, update rule, and policy builder for the SAC+SEE algorithm.
    """

    if not isinstance(observation_space, Box):
        raise ValueError(
            f"This SAC+SEE implementation only supports Box observation spaces, but got {type(observation_space)}"
        )
    if not isinstance(action_space, Box):
        raise ValueError(
            f"This SAC+SEE implementation only supports Box action spaces, but got {type(action_space)}"
        )

    environment_info = {
        constants.GENERAL_ARGUMENT_OBSERVATION_SPACE: observation_space,
        constants.GENERAL_ARGUMENT_ACTION_SPACE: action_space,
    }

    configuration[ARGUMENT_EXPLOITATION_CRITIC_1_NETWORK_ARGUMENTS].update(
        environment_info
    )
    configuration[ARGUMENT_EXPLOITATION_CRITIC_2_NETWORK_ARGUMENTS].update(
        environment_info
    )

    exploitation_critic_1 = configuration[ARGUMENT_EXPLOITATION_CRITIC_1_NETWORK_CLASS](
        **configuration[ARGUMENT_EXPLOITATION_CRITIC_1_NETWORK_ARGUMENTS]
    )
    exploitation_critic_2 = configuration[ARGUMENT_EXPLOITATION_CRITIC_2_NETWORK_CLASS](
        **configuration[ARGUMENT_EXPLOITATION_CRITIC_2_NETWORK_ARGUMENTS]
    )

    configuration[ARGUMENT_EXPLOITATION_ACTOR_NETWORK_ARGUMENTS].update(
        environment_info
    )

    exploitation_actor = configuration[ARGUMENT_EXPLOITATION_ACTOR_NETWORK_CLASS](
        **configuration[ARGUMENT_EXPLOITATION_ACTOR_NETWORK_ARGUMENTS]
    )

    configuration[ARGUMENT_EXPLORATION_CRITIC_1_NETWORK_ARGUMENTS].update(
        {
            "example_conditioning_q_value_function": exploitation_critic_1,
            **environment_info,
            "preprocessing_pipe": configuration[
                ARGUMENT_POST_REPLAY_BUFFER_DATA_PREPROCESSING
            ],
        }
    )
    configuration[ARGUMENT_EXPLORATION_CRITIC_2_NETWORK_ARGUMENTS].update(
        {
            "example_conditioning_q_value_function": exploitation_critic_1,
            **environment_info,
            "preprocessing_pipe": configuration[
                ARGUMENT_POST_REPLAY_BUFFER_DATA_PREPROCESSING
            ],
        }
    )
    exploration_critic_1 = configuration[ARGUMENT_EXPLORATION_CRITIC_1_NETWORK_CLASS](
        **configuration[ARGUMENT_EXPLORATION_CRITIC_1_NETWORK_ARGUMENTS]
    )
    exploration_critic_2 = configuration[ARGUMENT_EXPLORATION_CRITIC_2_NETWORK_CLASS](
        **configuration[ARGUMENT_EXPLORATION_CRITIC_2_NETWORK_ARGUMENTS]
    )
    configuration[ARGUMENT_EXPLORATION_ACTOR_NETWORK_ARGUMENTS].update(environment_info)
    exploration_actor = configuration[ARGUMENT_EXPLORATION_ACTOR_NETWORK_CLASS](
        **configuration[ARGUMENT_EXPLORATION_ACTOR_NETWORK_ARGUMENTS]
    )

    update_data_provider = UpdateDataProvider()

    # DATA RECEIVER
    # This data receiver uses the unscaled action from the policy to replace the action in the transition data.
    data_collector = ActionReplacementGymnasiumTransitionDataCollector(
        policy_info_replacement_key=INFO_KEY_UNSCALED_ACTION,
        update_data_provider=update_data_provider,
    )

    # UPDATE RULE
    update_rule = SACSEEUpdateRule(
        exploitation_critic_1=exploitation_critic_1,
        exploitation_critic_2=exploitation_critic_2,
        exploitation_actor=exploitation_actor,
        exploration_critic_1=exploration_critic_1,
        exploration_critic_2=exploration_critic_2,
        exploration_actor=exploration_actor,
        exploitation_discount=configuration[ARGUMENT_EXPLOITATION_DISCOUNT],
        exploration_discount=configuration[ARGUMENT_EXPLORATION_DISCOUNT],
        observation_space=observation_space,
        action_space=action_space,
        update_data_provider=update_data_provider,
        exploitation_critic_optimizer_class=configuration[
            ARGUMENT_EXPLOITATION_CRITIC_OPTIMIZER_CLASS
        ],
        exploitation_actor_optimizer_class=configuration[
            ARGUMENT_EXPLOITATION_ACTOR_OPTIMIZER_CLASS
        ],
        exploration_critic_optimizer_class=configuration[
            ARGUMENT_EXPLORATION_CRITIC_OPTIMIZER_CLASS
        ],
        exploration_actor_optimizer_class=configuration[
            ARGUMENT_EXPLORATION_ACTOR_OPTIMIZER_CLASS
        ],
        exploitation_critic_optimizer_arguments=configuration[
            ARGUMENT_EXPLOITATION_CRITIC_OPTIMIZER_ARGUMENTS
        ],
        exploitation_actor_optimizer_arguments=configuration[
            ARGUMENT_EXPLOITATION_ACTOR_OPTIMIZER_ARGUMENTS
        ],
        exploration_critic_optimizer_arguments=configuration[
            ARGUMENT_EXPLORATION_CRITIC_OPTIMIZER_ARGUMENTS
        ],
        exploration_actor_optimizer_arguments=configuration[
            ARGUMENT_EXPLORATION_ACTOR_OPTIMIZER_ARGUMENTS
        ],
        exploitation_critic_criteria=configuration[
            ARGUMENT_EXPLOITATION_CRITIC_CRITERIA
        ],
        exploration_critic_criteria=configuration[ARGUMENT_EXPLORATION_CRITIC_CRITERIA],
        exploitation_temperature=configuration[ARGUMENT_EXPLOITATION_TEMPERATURE],
        exploitation_target_entropy=configuration[ARGUMENT_EXPLOITATION_TARGET_ENTROPY],
        exploitation_initial_temperature=configuration[
            ARGUMENT_EXPLOITATION_INITIAL_TEMPERATURE
        ],
        exploration_temperature=configuration[ARGUMENT_EXPLORATION_TEMPERATURE],
        exploration_target_entropy=configuration[ARGUMENT_EXPLORATION_TARGET_ENTROPY],
        exploration_initial_temperature=configuration[
            ARGUMENT_EXPLORATION_INITIAL_TEMPERATURE
        ],
        exploration_target_calculation_method=configuration[
            ARGUMENT_EXPLORATION_TARGET_CALCULATION_METHOD
        ],
        exploitation_critic_update_frequency=configuration[
            ARGUMENT_EXPLOITATION_CRITIC_UPDATE_FREQUENCY
        ],
        exploitation_critic_number_of_updates=configuration[
            ARGUMENT_EXPLOITATION_CRITIC_NUMBER_OF_UPDATES
        ],
        exploitation_actor_update_frequency=configuration[
            ARGUMENT_EXPLOITATION_ACTOR_UPDATE_FREQUENCY
        ],
        exploitation_actor_number_of_updates=configuration[
            ARGUMENT_EXPLOITATION_ACTOR_NUMBER_OF_UPDATES
        ],
        exploration_critic_update_frequency=configuration[
            ARGUMENT_EXPLORATION_CRITIC_UPDATE_FREQUENCY
        ],
        exploration_critic_number_of_updates=configuration[
            ARGUMENT_EXPLORATION_CRITIC_NUMBER_OF_UPDATES
        ],
        exploration_actor_update_frequency=configuration[
            ARGUMENT_EXPLORATION_ACTOR_UPDATE_FREQUENCY
        ],
        exploration_actor_number_of_updates=configuration[
            ARGUMENT_EXPLORATION_ACTOR_NUMBER_OF_UPDATES
        ],
        multiply_number_of_updates_by_environment_steps=configuration[
            ARGUMENT_MULTIPLY_NUMBER_OF_UPDATES_BY_ENVIRONMENT_STEPS
        ],
        exploitation_target_critic_update_frequency=configuration[
            ARGUMENT_EXPLOITATION_TARGET_CRITIC_UPDATE_FREQUENCY
        ],
        exploration_target_critic_update_frequency=configuration[
            ARGUMENT_EXPLORATION_TARGET_CRITIC_UPDATE_FREQUENCY
        ],
        exploitation_target_critic_tau=configuration[
            ARGUMENT_EXPLOITATION_TARGET_CRITIC_TAU
        ],
        exploration_target_critic_tau=configuration[
            ARGUMENT_EXPLORATION_TARGET_CRITIC_TAU
        ],
        exploitation_critic_gradient_max_norm=configuration[
            ARGUMENT_EXPLOITATION_CRITIC_GRADIENT_MAX_NORM
        ],
        exploitation_actor_gradient_max_norm=configuration[
            ARGUMENT_EXPLOITATION_ACTOR_GRADIENT_MAX_NORM
        ],
        exploration_critic_gradient_max_norm=configuration[
            ARGUMENT_EXPLORATION_CRITIC_GRADIENT_MAX_NORM
        ],
        exploration_actor_gradient_max_norm=configuration[
            ARGUMENT_EXPLORATION_ACTOR_GRADIENT_MAX_NORM
        ],
        replay_buffer_capacity=configuration[ARGUMENT_REPLAY_BUFFER_CAPACITY],
        replay_buffer_mini_batch_size=configuration[
            ARGUMENT_REPLAY_BUFFER_MINI_BATCH_SIZE
        ],
        additional_replay_buffer_arguments=configuration[
            ARGUMENT_ADDITIONAL_REPLAY_BUFFER_ARGUMENTS
        ],
        post_replay_buffer_data_preprocessing=configuration[
            ARGUMENT_POST_REPLAY_BUFFER_DATA_PREPROCESSING
        ],
        device=configuration[ARGUMENT_DEVICE],
    )

    # POLICY BUILDER
    policy_builder = SACSEEPolicyBuilder(
        exploration_actor=exploration_actor,
        exploration_critic_1=exploration_critic_1,
        exploration_critic_2=exploration_critic_2,
        exploitation_actor=exploitation_actor,
        exploitation_critic_1=exploitation_critic_1,
        exploitation_critic_2=exploitation_critic_2,
        exploration_advantage_mixture=configuration[ARGUMENT_POLICY_MIXTURE],
        action_selection_temperature=configuration[
            ARGUMENT_ACTION_SELECTION_TEMPERATURE
        ],
        action_space=action_space,
        post_replay_buffer_preprocessing=configuration[
            ARGUMENT_POST_REPLAY_BUFFER_DATA_PREPROCESSING
        ],
        use_alternating_training_policy=configuration[
            ARGUMENT_USE_ALTERNATING_TRAINING_POLICY
        ],
    )

    return data_collector, update_rule, policy_builder


athlete.register(
    id="sac+see",
    component_factory=make_sac_see_components,
    default_configuration=DEFAULT_CONFIGURATION,
)
