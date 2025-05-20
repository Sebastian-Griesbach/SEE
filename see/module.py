import torch
from torch import nn
from typing import List, Optional, Callable

import numpy as np
from gymnasium.spaces import Box

from athlete.module.torch.fully_connected import NonLinearFullyConnectedNet


from athlete.function import chain_functions


class CriticFingerprintEmbedding(nn.Module):
    """An embedding module that generates a fingerprint embedding for a given
    q-value function.
    """

    def __init__(
        self,
        network_example: nn.Module,
        num_probe_values: int,
        observation_space: Box,
        action_space: Box,
        normalized_observation: bool = False,
        preprocessing_pipe: Optional[List[Callable]] = None,
    ):
        """Initializes the fingerprint embedding module.

        Args:
            network_example (nn.Module): An example of the q-value function to embed.
                All future modules need to have the same input and out dimensions.
            num_probe_values (int): Number of probe observations and actions to create the embedding.
            observation_space (Box): The observation space of the environment. Used to generate the initial probe observations.
            action_space (Box): The action space of the environment. Used to generate the initial probe actions.
            normalized_observation (bool, optional): Whether the observations are being normalized for the agent.
                In that case also normalized initial probe values are generated. Defaults to False.
            preprocessing_pipe (Optional[List[Callable]], optional): A list of functions to preprocess the observations.
                Used to generate the initial probe observations. Defaults to None.
        """
        super(CriticFingerprintEmbedding, self).__init__()
        # initialize input states from observation space if the observation space is reasonable otherwise use random normal distribution
        probe_observations = self._initialize_probe_observations(
            num_probe_values,
            observation_space,
            normalized_observation=normalized_observation,
            preprocessing_pipe=preprocessing_pipe,
        )
        # We assume actions have a normalized range of  [-1, 1] for all internal matters of the agent
        # (The policy can still scale the actions to the action space)
        probe_actions = np.random.uniform(
            low=-1.0, high=1.0, size=(num_probe_values, *action_space.shape)
        )
        probe_actions = torch.tensor(probe_actions, dtype=torch.float32)
        probe_actions = self._initialize_probe_observations(
            num_probe_values, action_space
        )

        # Turn the probe values into parameters suc that they can be tuned via regular optimizers
        self.probe_actions = nn.Parameter(data=probe_actions, requires_grad=True)
        self.probe_observations = nn.Parameter(
            data=probe_observations, requires_grad=True
        )

        # determine output size
        with torch.no_grad():
            module_output = network_example(self.probe_observations, self.probe_actions)
            self._embedding_size = np.prod(module_output.shape)

    def _initialize_probe_observations(
        self,
        num_probe_states: int,
        observation_space: Box,
        normalized_observation: bool = False,
        preprocessing_pipe: Optional[List[Callable]] = None,
    ) -> torch.Tensor:
        """Initializes probe observations based on observation space and arguments.

        Args:
            num_probe_states (int): How many probe observations to initialize.
            observation_space (Box): The observation space of the critic.
            normalized_observation (bool, optional): Whether to use normally distributed initialization. Defaults to False.
            preprocessing_pipe (Optional[List[Callable]], optional): List of preprocessing functions to use
                on the sampled initial probe observations.

        Returns:
            torch.Tensor: Tensor of initialized probe observations.
        """
        random = np.random.randn(num_probe_states, *observation_space.shape)
        if not normalized_observation:
            with np.errstate(over="ignore"):
                sample_dimensions = (
                    observation_space.high - observation_space.low < np.inf
                )
            samples = np.array(
                [observation_space.sample() for _ in range(num_probe_states)]
            ).reshape(-1, *observation_space.shape)
            sample_dimensions = np.expand_dims(sample_dimensions, axis=0)
            sample_dimensions = np.repeat(sample_dimensions, num_probe_states, axis=0)
            probe_observations = np.where(sample_dimensions, samples, random)
            probe_observations = torch.tensor(probe_observations, dtype=torch.float32)
        else:
            probe_observations = torch.tensor(random, dtype=torch.float32)

            if preprocessing_pipe is not None:
                probe_observations = chain_functions(
                    function_list=preprocessing_pipe, input_value=probe_observations
                )

        return probe_observations

    def forward(self, module: torch.nn.Module) -> torch.Tensor:
        """Embed the given module.

        Args:
            module (torch.nn.Module): Module to embed with Fingerprinting.

        Returns:
            torch.Tensor: The Fingerprinting embedding of the module.
        """
        module_output = module(self.probe_observations, self.probe_actions)
        # Flatten the output to get a single embedding vector
        fingerprint_embedding = module_output.flatten()

        return fingerprint_embedding

    @property
    def embedding_size(self) -> int:
        """The size of the embedding."""
        return self._embedding_size


class FingerprintingConditionedContinuousQValueFunction(torch.nn.Module):
    """This class implements a critic that is conditioned by another critic."""

    def __init__(
        self,
        example_conditioning_q_value_function: torch.nn.Module,
        hidden_dims: List[int],
        num_probe_values: int,
        observation_space: Box,
        action_space: Box,
        normalized_observation: bool = False,
        preprocessing_pipe: Optional[List[Callable]] = None,
        init_state_dict_path: Optional[str] = None,
    ):
        """Initialized the critic and the fingerprinting embedding method.

        Args:
            example_conditioning_q_value_function (torch.nn.Module): Example network of the other critic
                that will be used to condition this one.
            hidden_dims (List[int]): Dimension of the hidden states of the network.
            num_probe_values (int): The number of probe values to use for the fingerprinting.
            observation_space (Box): The observation space of the environment.
            action_space (Box): The action space of the environment.
            normalized_observation (bool, optional): Whether to use a normal distribution to
                initialize the probe observations. Defaults to False.
            preprocessing_pipe (Optional[List[Callable]], optional): List of preprocess functions to use during probe observations initialization. Defaults to None.
            init_state_dict_path (Optional[str], optional): Path to a state dictionary to load initial parameters from. Defaults to None.
        """
        super().__init__()

        self.fingerprint_embedding = CriticFingerprintEmbedding(
            network_example=example_conditioning_q_value_function,
            num_probe_values=num_probe_values,
            observation_space=observation_space,
            action_space=action_space,
            normalized_observation=normalized_observation,
            preprocessing_pipe=preprocessing_pipe,
        )

        critic_network_input_size = (
            np.prod(observation_space.shape)
            + np.prod(action_space.shape)
            + self.fingerprint_embedding.embedding_size
        )

        self.critic_network = NonLinearFullyConnectedNet(
            layer_dims=[
                critic_network_input_size,
                *hidden_dims,
                1,
            ]
        )

        if init_state_dict_path:
            self.load_state_dict(torch.load(init_state_dict_path))

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        conditioning_module: torch.nn.Module,
    ) -> torch.Tensor:
        """Estimate the state value given a conditioning module.

        Args:
            observations (torch.Tensor): Batch of observations of shape (batch_size, observations_size)
            actions (torch.Tensor): Batch of actions of shape (batch_size, action_size)
            conditioning_module (torch.nn.Module): The critic network to condition this estimation on.

        Returns:
            torch.Tensor: Q-value estimate of shape (batch_size, 1)
        """
        # Get the fingerprint embedding
        fingerprint_embedding = self.fingerprint_embedding(conditioning_module)

        batch_size = observations.shape[0]
        observations = observations.reshape(batch_size, -1)
        actions = actions.reshape(batch_size, -1)
        # repeat embedding for batch size
        repeated_fingerprint_embedding = fingerprint_embedding.repeat(batch_size, 1)

        # Concatenate the observation, action, and fingerprint embedding
        concatenated_inputs = torch.cat(
            (observations, actions, repeated_fingerprint_embedding), dim=1
        )

        # Pass through the critic network
        q_values = self.critic_network(concatenated_inputs)

        return q_values


# Used for ablation
class MockFingerprintingConditionedContinuousQValueFunction(torch.nn.Module):
    """A mock class used for an ablation study that has the same signature
    as the original class but does not implement any conditioning.
    """

    def __init__(
        self,
        example_conditioning_q_value_function: torch.nn.Module,
        hidden_dims: List[int],
        num_probe_values: int,
        observation_space: Box,
        action_space: Box,
        normalized_observation: bool = False,
        preprocessing_pipe: Optional[List[Callable]] = None,
        init_state_dict_path: Optional[str] = None,
    ):
        """Initialize Mock class.

        Args:
            example_conditioning_q_value_function (torch.nn.Module): This is not used.
            hidden_dims (List[int]): List of hidden dimension of the fully connected network.
            num_probe_values (int): This is not used.
            observation_space (Box): Observation space of the environment.
            action_space (Box): Action space of the environment.
            normalized_observation (bool, optional): This is not used. Defaults to False.
            preprocessing_pipe (Optional[List[Callable]], optional): This is not used. Defaults to None.
            init_state_dict_path (Optional[str], optional): Path to a state dictionary to load initial parameters from. Defaults to None.
        """

        super().__init__()

        critic_network_input_size = np.prod(observation_space.shape) + np.prod(
            action_space.shape
        )

        self.critic_network = NonLinearFullyConnectedNet(
            layer_dims=[
                critic_network_input_size,
                *hidden_dims,
                1,
            ]
        )

        if init_state_dict_path:
            self.load_state_dict(torch.load(init_state_dict_path))

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        conditioning_module: torch.nn.Module,
    ) -> torch.Tensor:
        """Mock forwarding, performs regular estimation without conditioning.

        Args:
            observations (torch.Tensor): Batch of observation of shape (batch_size, observation_size)
            actions (torch.Tensor): Batch of actions of shape (batch_size, action_size)
            conditioning_module (torch.nn.Module): This is not used.

        Returns:
            torch.Tensor: Q-value estimation of size (batch_size, 1), without conditioning.
        """

        batch_size = observations.shape[0]
        observations = observations.reshape(batch_size, -1)
        actions = actions.reshape(batch_size, -1)

        # Concatenate the observation, action, and fingerprint embedding
        concatenated_inputs = torch.cat((observations, actions), dim=1)

        # Pass through the critic network
        q_values = self.critic_network(concatenated_inputs)

        return q_values
