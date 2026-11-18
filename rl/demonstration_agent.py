"""
Demonstration-Assisted Deep Q-Network Agent (Guided Double DQN with Large Margin Loss).
Inspired by Hester et al. (2017) "Deep Q-learning from Demonstrations" (DQfD).

ACADEMIC INTEGRITY NOTICE:
This agent combines Double DQN 1-step TD loss with Supervised Large Margin Loss and L2
weight regularization. It does NOT implement n-step TD returns or Prioritized Experience
Replay (SumTree), and therefore is strictly designated as "Demonstration-Assisted Double DQN"
rather than "Full DQfD".
"""

from __future__ import annotations
import math
import os
import json
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl.dqn.network import QNetwork
from rl.dqn.agent import DQNConfig
from rl.double_dqn.agent import DoubleDQNAgent
from environment.action import NUM_ACTIONS


@dataclass
class DemonstrationAgentConfig(DQNConfig):
    """
    Configuration parameters for Demonstration-Assisted DQN Agent.
    """
    lambda_margin: float = 5.0          # Weight lambda_E for Supervised Large Margin Loss
    margin_value: float = 0.2           # Margin parameter m: separation required between expert and non-expert Q
    lambda_l2: float = 1e-5             # Weight lambda_2 for L2 parameter regularization
    demo_ratio: float = 0.25            # Fraction of expert demonstrations in mixed mini-batches (25% demo, 75% online)
    pretrain_steps: int = 30000         # Number of pre-training gradient updates on demonstrations
    loss_type: str = "huber"            # "huber" (smooth L1) or "mse"


class SupervisedMarginLoss(nn.Module):
    """
    Supervised Large Margin Classification Loss for Deep Q-learning.

    Reference:
        Hester et al. (2017) 'Deep Q-learning from Demonstrations', Equation 2:
        J_E(Q) = max_{a in A} [ Q(s, a; theta) + l(a_E, a) ] - Q(s, a_E; theta)

    Where:
        l(a_E, a) = 0 if a == a_E else margin (e.g. 0.2)

    Properties:
        1. J_E(Q) >= 0 always.
        2. J_E(Q) = 0 iff Q(s, a_E) >= Q(s, a) + margin for all a != a_E.
        3. Forces the Q-value of the expert demonstration action to exceed
           all other actions by at least the margin m.
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
        q_values : (batch_size, num_actions) predicted Q-values from online network
        expert_actions : (batch_size,) ground truth expert action indices
        mask : optional (batch_size,) boolean or float mask (1.0 = demo, 0.0 = online)

        Returns
        -------
        torch.Tensor : scalar mean margin loss
        """
        batch_size, num_actions = q_values.shape

        # Margin matrix l(a_E, a): initialize to margin value everywhere
        margin_matrix = torch.full_like(q_values, self.margin)

        # Set l(a_E, a_E) = 0 for the expert action
        expert_2d = expert_actions.view(-1, 1)
        margin_matrix.scatter_(
            dim=1,
            index=expert_2d,
            src=torch.zeros_like(expert_2d, dtype=q_values.dtype),
        )

        # Q(s, a) + l(a_E, a)
        augmented_q = q_values + margin_matrix

        # max_{a in A} [ Q(s, a) + l(a_E, a) ]
        max_aug_q, _ = torch.max(augmented_q, dim=1)

        # Q(s, a_E)
        q_expert = q_values.gather(dim=1, index=expert_2d).squeeze(1)

        # Per-transition loss
        loss_per_sample = max_aug_q - q_expert

        # If mask is provided, only average over expert demonstration samples
        if mask is not None:
            mask_float = mask.float()
            denom = mask_float.sum()
            if denom > 0:
                return (loss_per_sample * mask_float).sum() / denom
            return torch.tensor(0.0, device=q_values.device, dtype=q_values.dtype)

        return loss_per_sample.mean()


class DemonstrationDQNAgent(DoubleDQNAgent):
    """
    Demonstration-Assisted DQN Agent.
    Integrates Double DQN Bellman optimality with Supervised Large Margin Classification Loss
    and L2 regularization.

    Total Loss:
        L_total(theta) = L_TD(theta) + lambda_margin * L_margin(theta) + lambda_l2 * L_l2(theta)
    """

    def __init__(self, config: Optional[DemonstrationAgentConfig] = None):
        super().__init__(config=config or DemonstrationAgentConfig())
        self.config: DemonstrationAgentConfig = self.config

        self.margin_loss_fn = SupervisedMarginLoss(margin=self.config.margin_value)
        self.pretrain_updates: int = 0
        self.mixed_updates: int = 0

    def compute_l2_loss(self) -> torch.Tensor:
        """Compute L2 parameter regularization penalty on linear layer weights."""
        l2_reg = torch.tensor(0.0, device=self.device)
        for name, param in self.online_net.named_parameters():
            if param.requires_grad and "weight" in name:
                l2_reg = l2_reg + torch.sum(param ** 2)
        return 0.5 * l2_reg

    def compute_double_td_loss(
        self,
        q_values: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Double DQN TD Loss:
            a* = argmax_a Q(s', a; theta_online)
            y = r + gamma * (1 - done) * Q(s', a*; theta_target)
            L_TD = SmoothL1(Q(s, a; theta_online) - y)
        """
        # Predicted Q(s, a)
        q_sa = q_values.gather(1, actions.view(-1, 1)).squeeze(1)

        # Target network evaluation decoupled from online network selection
        with torch.no_grad():
            q_next_online = self.online_net(next_states)
            best_actions = q_next_online.argmax(dim=1, keepdim=True)

            q_next_target = self.target_net(next_states)
            q_target_selected = q_next_target.gather(1, best_actions).squeeze(1)

            td_target = rewards + self.config.gamma * (1.0 - dones.float()) * q_target_selected

        if self.config.loss_type.lower() == "huber":
            td_loss = F.smooth_l1_loss(q_sa, td_target)
        else:
            td_loss = F.mse_loss(q_sa, td_target)

        return td_loss, td_target

    def pretrain_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute one gradient descent step on a pure demonstration batch.

        Parameters
        ----------
        batch : dict containing 'states', 'actions', 'rewards', 'next_states', 'dones'

        Returns
        -------
        dict with loss breakdown
        """
        self.online_net.train()

        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # 1. Forward pass
        q_values = self.online_net(states)

        # 2. Double DQN TD loss
        td_loss, td_target = self.compute_double_td_loss(
            q_values, actions, rewards, next_states, dones
        )

        # 3. Supervised Large Margin Loss (all samples are expert demonstrations)
        margin_loss = self.margin_loss_fn(q_values, actions)

        # 4. L2 parameter regularization
        l2_loss = self.compute_l2_loss()

        # 5. Combined Loss
        total_loss = (
            td_loss
            + self.config.lambda_margin * margin_loss
            + self.config.lambda_l2 * l2_loss
        )

        # 6. Backward and step
        self.optimizer.zero_grad()
        total_loss.backward()

        if self.config.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.config.grad_clip)

        self.optimizer.step()

        self.pretrain_updates += 1
        self.total_updates += 1

        if self.total_updates % self.config.target_update_frequency == 0:
            self.update_target_network()

        with torch.no_grad():
            q_sa = q_values.gather(1, actions.view(-1, 1)).squeeze(1)
            mean_q = float(q_sa.mean().item())
            mean_target = float(td_target.mean().item())

        return {
            "loss_total": float(total_loss.item()),
            "loss_td": float(td_loss.item()),
            "loss_margin": float(margin_loss.item()),
            "loss_l2": float(l2_loss.item()),
            "q_mean": mean_q,
            "td_target_mean": mean_target,
        }

    def mixed_train_step(
        self,
        demo_batch: Dict[str, torch.Tensor],
        online_batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Execute one mixed mini-batch training step combining expert demonstrations
        and online exploration experience.

        Margin loss is computed ONLY on the demonstration portion of the mini-batch.
        TD loss is computed on ALL transitions.
        """
        self.online_net.train()

        # Concatenate transitions into single tensor batch
        states = torch.cat([demo_batch["states"], online_batch["states"]], dim=0)
        actions = torch.cat([demo_batch["actions"], online_batch["actions"]], dim=0)
        rewards = torch.cat([demo_batch["rewards"], online_batch["rewards"]], dim=0)
        next_states = torch.cat([demo_batch["next_states"], online_batch["next_states"]], dim=0)
        dones = torch.cat([demo_batch["dones"], online_batch["dones"]], dim=0)

        n_demo = len(demo_batch["states"])
        n_online = len(online_batch["states"])
        is_demo_mask = torch.cat([
            torch.ones(n_demo, dtype=torch.bool, device=states.device),
            torch.zeros(n_online, dtype=torch.bool, device=states.device),
        ], dim=0)

        # 1. Forward pass
        q_values = self.online_net(states)

        # 2. Double DQN TD loss on entire combined batch
        td_loss, td_target = self.compute_double_td_loss(
            q_values, actions, rewards, next_states, dones
        )

        # 3. Supervised margin loss ONLY on demonstration transitions
        margin_loss = self.margin_loss_fn(
            q_values=q_values,
            expert_actions=actions,
            mask=is_demo_mask,
        )

        # 4. L2 regularization
        l2_loss = self.compute_l2_loss()

        # 5. Combined loss
        total_loss = (
            td_loss
            + self.config.lambda_margin * margin_loss
            + self.config.lambda_l2 * l2_loss
        )

        # 6. Backward and step
        self.optimizer.zero_grad()
        total_loss.backward()

        if self.config.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.config.grad_clip)

        self.optimizer.step()

        self.mixed_updates += 1
        self.total_updates += 1

        if self.total_updates % self.config.target_update_frequency == 0:
            self.update_target_network()

        with torch.no_grad():
            q_sa = q_values.gather(1, actions.view(-1, 1)).squeeze(1)
            mean_q = float(q_sa.mean().item())
            mean_target = float(td_target.mean().item())

        return {
            "loss_total": float(total_loss.item()),
            "loss_td": float(td_loss.item()),
            "loss_margin": float(margin_loss.item()),
            "loss_l2": float(l2_loss.item()),
            "q_mean": mean_q,
            "td_target_mean": mean_target,
        }
