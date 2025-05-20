import torch


def calculate_bellmann_target(
    rewards: torch.tensor,
    not_terminateds: torch.tensor,
    next_predictions: torch.nn.Module,
    discount: float,
) -> torch.Tensor:
    """Calculate a target using the regular Bellmann update

    Args:
        rewards (torch.tensor): Rewards to use for the target.
        not_terminateds (torch.tensor): Whether these transitions are terminal.
        next_predictions (torch.nn.Module): Next estimate used for the target.
            We assume that there are only next predictions for non terminal transitions.
        discount (float): Discount factor used for the target.

    Returns:
        torch.Tensor: The targets of the regular Bellmann update.
    """
    target = torch.clone(rewards)
    target[not_terminateds] += discount * next_predictions
    return target


def calculate_maximum_target(
    rewards: torch.tensor,
    not_terminateds: torch.tensor,
    next_predictions: torch.nn.Module,
    discount: float,
) -> torch.Tensor:
    """Calculates a target using the maximum update.

    Args:
        rewards (torch.tensor): Rewards to use for the target.
        not_terminateds (torch.tensor): Whether these transitions are terminal.
        next_predictions (torch.nn.Module): Next estimate used for the target.
            We assume that there are only next predictions for non terminal transitions.
        discount (float): Discount factor used for the target.

    Returns:
        torch.Tensor: The target according to the maximum update.
    """
    target = torch.clone(rewards)
    stacked = torch.stack([target[not_terminateds], discount * next_predictions])
    target[not_terminateds] = torch.max(stacked, dim=0)[0]
    return target
