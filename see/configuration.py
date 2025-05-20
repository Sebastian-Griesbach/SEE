from module import MockFingerprintingConditionedContinuousQValueFunction

"""This File holds predefined configurations identical to the ones used in the experiments of the paper.
    No specific random seeds are defined.
"""

# Choose one of the environments:
environments = [
    "Pendulum-v1",
    "LocalOptimumCar-v0",
    "HalfCheetah-v5",
    "Ant-v5",
    "Hopper-v5",
    "Swimmer-v5",
    "FetchPickAndPlace-v4",
    "LargePointMaze-v3",
]

# Choose one of the reward settings:
reward_settings = ["dense", "sparse", "adverse"]

# Choose one of the methods:
methods = [
    "SAC",
    "SAC+SEE",
    "TD3",
    "TD3+SEE",
    "SAC+SEE w/o conditioning",  # ablations
    "SAC+SEE w/o maximum update",
    "SAC+SEE w/o mixing",
    "TD3+SEE w/o conditioning",
    "TD3+SEE w/o maximum update",
    "TD3+SEE w/o mixing",
]

# Mapping of environment names and reward settings to actual environment IDs
environment_map = {
    ("Pendulum-v1", "dense"): "Pendulum-v1",
    ("Pendulum-v1", "sparse"): "SparsePendulum-v1",
    ("Pendulum-v1", "adverse"): "AdverseSparsePendulum-v1",
    ("LocalOptimumCar-v0", "dense"): "DenseLocalOptimaMountainCarContinuous-v0",
    ("LocalOptimumCar-v0", "sparse"): "SparseLocalOptimaMountainCarContinuous-v0",
    (
        "LocalOptimumCar-v0",
        "adverse",
    ): "AdverseSparseLocalOptimaMountainCarContinuous-v0",
    ("HalfCheetah-v5", "dense"): "HalfCheetah-v5",
    ("HalfCheetah-v5", "sparse"): "SparseHalfCheetah-v5",
    ("HalfCheetah-v5", "adverse"): "AdverseSparseHalfCheetah-v5",
    ("Ant-v5", "dense"): "Ant-v5",
    ("Ant-v5", "sparse"): "SparseAnt-v5",
    ("Ant-v5", "adverse"): "AdverseSparseAnt-v5",
    ("Hopper-v5", "dense"): "Hopper-v5",
    ("Hopper-v5", "sparse"): "SparseHopper-v5",
    ("Hopper-v5", "adverse"): "AdverseSparseHopper-v5",
    ("Swimmer-v5", "dense"): "Swimmer-v5",
    ("Swimmer-v5", "sparse"): "SparseSwimmer-v5",
    ("Swimmer-v5", "adverse"): "AdverseSparseSwimmer-v5",
    (
        "FetchPickAndPlace-v4",
        "dense",
    ): "GoalConverterWrappedDenseFetchPickAndPlace-v4",
    ("FetchPickAndPlace-v4", "sparse"): "GoalConverterWrappedFetchPickAndPlace-v4",
    (
        "FetchPickAndPlace-v4",
        "adverse",
    ): "GoalConverterWrappedAdverseFetchPickAndPlace-v4",
    ("LargePointMaze-v3", "dense"): "GoalConverterWrappedDenseLargePointMaze-v3",
    ("LargePointMaze-v3", "sparse"): "GoalConverterWrappedLargePointMaze-v3",
    (
        "LargePointMaze-v3",
        "adverse",
    ): "GoalConverterWrappedAdverseLargePointMaze-v3",
}

# Hyperparameter sets, only includes the ones that are different from the default in Athlete 0.1.1
# the default can be found here:
# https://github.com/Sebastian-Griesbach/Athlete/blob/main/athlete/algorithms/sac/__init__.py
# https://github.com/Sebastian-Griesbach/Athlete/blob/main/athlete/algorithms/td3/__init__.py

mujoco_sac = {
    "warmup_steps": 10000,
}
classic_sac = {
    "critic_optimizer_arguments": {"lr": 1e-3},
    "actor_optimizer_arguments": {"lr": 1e-3},
    "replay_buffer_capacity": 200000,
}

mujoco_td3 = {
    "warmup_steps": 10000,
}
classic_td3 = {}
fetch_pick_and_place_sac = {
    **mujoco_sac,
    "critic_network_arguments": {"hidden_dims": [400, 300, 200]},
    "critic_2_network_arguments": {"hidden_dims": [400, 300, 200]},
    "actor_network_arguments": {"hidden_dims": [400, 300, 200]},
}
fetch_pick_and_place_td3 = {
    **mujoco_td3,
    "critic_network_arguments": {"hidden_dims": [400, 300, 200]},
    "critic_2_network_arguments": {"hidden_dims": [400, 300, 200]},
    "actor_network_arguments": {"hidden_dims": [400, 300, 200]},
}

# only includes the ones that are different from the default
# default cen be found in sac_see.__init__.py and td3_see.__init__.py respectively
mujoco_sac_see = {}
classic_sac_see = {
    "exploitation_critic_optimizer_arguments": {"lr": 1e-3},
    "exploitation_actor_optimizer_arguments": {"lr": 1e-3},
    "exploration_critic_optimizer_arguments": {"lr": 1e-3},
    "exploration_actor_optimizer_arguments": {"lr": 1e-3},
    "replay_buffer_capacity": 200000,
    "warmup_steps": 1000,
}
fetch_pick_and_place_sac_see = {
    **mujoco_sac_see,
    "exploitation_critic_network_arguments": {"hidden_dims": [400, 300, 200]},
    "exploitation_critic_2_network_arguments": {"hidden_dims": [400, 300, 200]},
    "exploitation_actor_network_arguments": {"hidden_dims": [400, 300, 200]},
    "exploration_critic_1_network_arguments": {
        "hidden_dims": [400, 300, 200],
        "num_probe_values": 16,
    },
    "exploration_critic_2_network_arguments": {
        "hidden_dims": [400, 300, 200],
        "num_probe_values": 16,
    },
    "exploration_actor_network_arguments": {"hidden_dims": [400, 300, 200]},
}
mujoco_td3_see = {}
classic_td3_see = {"warmup_steps": 1000, "replay_buffer_capacity": 200000}
fetch_pick_and_place_td3_see = {
    **mujoco_td3_see,
    "exploitation_critic_network_arguments": {
        "hidden_dims": [400, 300, 200],
    },
    "exploitation_critic_2_network_arguments": {
        "hidden_dims": [400, 300, 200],
    },
    "exploitation_actor_network_arguments": {
        "hidden_dims": [400, 300, 200],
    },
    "exploration_critic_1_network_arguments": {
        "hidden_dims": [400, 300, 200],
        "num_probe_values": 16,
    },
    "exploration_critic_2_network_arguments": {
        "hidden_dims": [400, 300, 200],
        "num_probe_values": 16,
    },
    "exploration_actor_network_arguments": {
        "hidden_dims": [400, 300, 200],
    },
}
# configuration additions for ablation studies
see_wo_maximum = {"exploration_target_calculation_method": "bellmann"}
see_wo_conditioning = {
    "exploration_critic_1_network_class": MockFingerprintingConditionedContinuousQValueFunction,
    "exploration_critic_2_network_class": MockFingerprintingConditionedContinuousQValueFunction,
}
see_wo_mixing = {"use_alternating_training_policy": True}

# Environment and method mapping to hyperparameter sets and algorithm id
algorithm_map = {
    ("SAC", "Pendulum-v1"): ("sac", classic_sac),
    ("SAC", "LocalOptimumCar-v0"): ("sac", classic_sac),
    ("SAC", "HalfCheetah-v5"): ("sac", mujoco_sac),
    ("SAC", "Ant-v5"): ("sac", mujoco_sac),
    ("SAC", "Hopper-v5"): ("sac", mujoco_sac),
    ("SAC", "Swimmer-v5"): ("sac", mujoco_sac),
    ("SAC", "FetchPickAndPlace-v4"): ("sac", fetch_pick_and_place_sac),
    ("SAC", "LargePointMaze-v3"): ("sac", mujoco_sac),
    ("TD3", "Pendulum-v1"): ("td3", classic_td3),
    ("TD3", "LocalOptimumCar-v0"): ("td3", classic_td3),
    ("TD3", "HalfCheetah-v5"): ("td3", mujoco_td3),
    ("TD3", "Ant-v5"): ("td3", mujoco_td3),
    ("TD3", "Hopper-v5"): ("td3", mujoco_td3),
    ("TD3", "Swimmer-v5"): ("td3", mujoco_td3),
    ("TD3", "FetchPickAndPlace-v4"): ("td3", fetch_pick_and_place_td3),
    ("TD3", "LargePointMaze-v3"): ("td3", mujoco_td3),
    ("SAC+SEE", "Pendulum-v1"): ("sac+see", classic_sac_see),
    ("SAC+SEE", "LocalOptimumCar-v0"): ("sac+see", classic_sac_see),
    ("SAC+SEE", "HalfCheetah-v5"): ("sac+see", mujoco_sac_see),
    ("SAC+SEE", "Ant-v5"): ("sac+see", mujoco_sac_see),
    ("SAC+SEE", "Hopper-v5"): ("sac+see", mujoco_sac_see),
    ("SAC+SEE", "Swimmer-v5"): ("sac+see", mujoco_sac_see),
    ("SAC+SEE", "FetchPickAndPlace-v4"): ("sac+see", fetch_pick_and_place_sac_see),
    ("SAC+SEE", "LargePointMaze-v3"): ("sac+see", mujoco_sac_see),
    ("TD3+SEE", "Pendulum-v1"): ("td3+see", classic_td3_see),
    ("TD3+SEE", "LocalOptimumCar-v0"): ("td3+see", classic_td3_see),
    ("TD3+SEE", "HalfCheetah-v5"): ("td3+see", mujoco_td3_see),
    ("TD3+SEE", "Ant-v5"): ("td3+see", mujoco_td3_see),
    ("TD3+SEE", "Hopper-v5"): ("td3+see", mujoco_td3_see),
    ("TD3+SEE", "Swimmer-v5"): ("td3+see", mujoco_td3_see),
    ("TD3+SEE", "FetchPickAndPlace-v4"): ("td3+see", fetch_pick_and_place_td3),
    ("TD3+SEE", "LargePointMaze-v3"): ("td3+see", mujoco_td3_see),
    # TODO add ablations here
}
# Adding ablation mapping
ablation_map = {}
for key, value in algorithm_map.items():
    algorithm_name, environment_name = key
    algorithm_id, hyperparameter_set = value
    if algorithm_name in ["SAC+SEE", "TD3+SEE"]:
        ablation_map[(f"{algorithm_name} w/o conditioning", environment_name)] = (
            algorithm_id,
            hyperparameter_set | see_wo_conditioning,
        )
        ablation_map[(f"{algorithm_name} w/o maximum update", environment_name)] = (
            algorithm_id,
            hyperparameter_set | see_wo_maximum,
        )
        ablation_map[(f"{algorithm_name} w/o mixing", environment_name)] = (
            algorithm_id,
            hyperparameter_set | see_wo_mixing,
        )

algorithm_map.update(ablation_map)


# experiment settings mapping by environment, tuple contains training_duration, eval_freq, num_eval_episodes
classic_experiment_settings = (100000, 1000, 10)
mujoco_experiment_settings = (3000000, 10000, 10)
long_experiment_settings = (10000000, 10000, 10)
experiment_settings_map = {
    "Pendulum-v1": classic_experiment_settings,
    "LocalOptimumCar-v0": classic_experiment_settings,
    "HalfCheetah-v5": mujoco_experiment_settings,
    "Ant-v5": mujoco_experiment_settings,
    "Hopper-v5": mujoco_experiment_settings,
    "Swimmer-v5": mujoco_experiment_settings,
    "FetchPickAndPlace-v4": long_experiment_settings,
    "LargePointMaze-v3": mujoco_experiment_settings,
}
