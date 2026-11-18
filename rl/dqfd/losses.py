"""
Loss functions for Guided Double DQN with Supervised Large Margin Classification Loss
(Inspired by Hester et al., 2017: Deep Q-learning from Demonstrations).
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedLargeMarginLoss(nn.Module):
    """
    Supervised Large Margin Classification Loss for Q-learning.

    Reference: Hester et al. (2017) 'Deep Q-learning from Demonstrations' (DQfD).
    Formula:
        J_E(Q) = max_{a in A} [ Q(s, a; theta) + l(a_E, a) ] - Q(s, a_E; theta)
    where:
        l(a_E, a) = 0 if a == a_E else margin (default 0.2)

    Properties:
    - Non-negative: always >= 0.
    - Zero iff Q(s, a_E) >= Q(s, a) + margin for all a != a_E.
    - Forces the Q-value of the expert action to exceed all non-expert actions by at least the margin.
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = float(margin)

    def forward(
        self,
        q_values: torch.Tensor,
        expert_actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        q_values : (batch_size, num_actions) tensor of predicted Q-values.
        expert_actions : (batch_size,) tensor of integer expert action indices.
        mask : optional (batch_size,) boolean or float tensor indicating which transitions
               are expert demonstrations (1.0) vs online replays (0.0).

        Returns
        -------
        torch.Tensor : scalar mean supervised margin loss.
        """
        batch_size, num_actions = q_values.shape

        # Construct margin matrix l(a_E, a)
        # Initialize with uniform margin value
        margin_tensor = torch.full_like(q_values, self.margin)

        # Set margin to 0 for expert action: l(a_E, a_E) = 0
        expert_actions_2d = expert_actions.view(-1, 1)
        margin_tensor.scatter_(dim=1, index=expert_actions_2d, src=torch.zeros_like(expert_actions_2d, dtype=q_values.dtype))

        # Augmented Q-values: Q(s, a) + l(a_E, a)
        augmented_q = q_values + margin_tensor

        # Max over actions
        max_augmented_q, _ = torch.max(augmented_q, dim=1)  # (batch_size,)

        # Q-value of expert action: Q(s, a_E)
        q_expert = q_values.gather(dim=1, index=expert_actions_2d).squeeze(1)  # (batch_size,)

        # Per-sample loss: max_a [Q(s, a) + l(a_E, a)] - Q(s, a_E)
        sample_loss = max_augmented_q - q_expert

        if mask is not None:
            mask_float = mask.float()
            denom = mask_float.sum()
            if denom > 0:
                return (sample_loss * mask_float).sum() / denom
            return torch.tensor(0.0, device=q_values.device, dtype=q_values.dtype)

        return sample_loss.mean()


def compute_l2_regularization(model: nn.Module) -> torch.Tensor:
    """
    Compute L2 weight decay penalty on network weights (excluding biases).
    J_L2(theta) = 0.5 * sum_w ||w||_2^2
    """
    l2_reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if param.requires_grad and "weight" in name:
            l2_reg = l2_reg + torch.sum(param ** 2)
    return 0.5 * l2_reg


def compute_double_dqn_td_loss(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    target_net: nn.Module,
    online_net: nn.Module,
    gamma: float = 0.995,
    loss_type: str = "huber",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Double DQN 1-step Bellman TD loss and TD target.

    Target formulation:
        a* = argmax_a Q(s', a; theta_online)
        y = r + gamma * (1 - done) * Q(s', a*; theta_target)
    """
    # 1. Q(s, a; theta_online)
    q_sa = q_values.gather(1, actions.view(-1, 1)).squeeze(1)

    # 2. Decoupled Double DQN target
    with torch.no_grad():
        q_next_online = online_net(next_states)
        best_actions = q_next_online.argmax(dim=1, keepdim=True)

        q_next_target = target_net(next_states)
        q_target_selected = q_next_target.gather(1, best_actions).squeeze(1)

        td_target = rewards + gamma * (1.0 - dones.float()) * q_target_selected

    # 3. Loss
    if loss_type.lower() == "huber":
        td_loss = F.smooth_l1_loss(q_sa, td_target)
    else:
        td_loss = F.mse_loss(q_sa, td_target)

    return td_loss, td_target
