"""
Guided Double DQN with Supervised Large Margin Classification Loss (DQfD-inspired).
Inherits from DoubleDQNAgent, providing pretraining from demonstration datasets
and mixed-sampling fine-tuning.
"""

from __future__ import annotations
import math
import os
import json
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl.dqn.agent import DQNConfig
from rl.double_dqn.agent import DoubleDQNAgent
from rl.dqfd.losses import (
    SupervisedLargeMarginLoss,
    compute_l2_regularization,
    compute_double_dqn_td_loss,
)


@dataclass
class DQfDConfig(DQNConfig):
    """Configuration for DQfD Agent."""
    lambda_margin: float = 5.0          # Weight for supervised margin loss (lambda_E)
    margin_value: float = 0.2           # Minimum separation margin (m)
    lambda_l2: float = 1e-5             # Weight for L2 parameter regularization
    demo_batch_ratio: float = 0.25      # Fraction of demonstration samples in mixed mini-batches
    loss_type: str = "huber"            # "huber" or "mse"


class DQfDAgent(DoubleDQNAgent):
    """
    DQfD Agent: Double DQN integrated with Supervised Large Margin Loss
    and L2 regularization.

    Total Loss Formulation:
        L_total(theta) = L_TD(theta) + lambda_E * L_margin(theta) + lambda_2 * L_reg(theta)
    """

    def __init__(self, config: Optional[DQfDConfig] = None):
        super().__init__(config=config or DQfDConfig())
        self.config: DQfDConfig = self.config  # type hint refinement

        self.margin_loss_fn = SupervisedLargeMarginLoss(margin=self.config.margin_value)
        self.pretrain_updates: int = 0

    def pretrain_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute a single gradient update purely on expert demonstration data.

        Parameters
        ----------
        batch : dict containing keys 'states', 'actions', 'rewards', 'next_states', 'dones'
                as PyTorch tensors on the appropriate device.

        Returns
        -------
        dict with loss components
        """
        self.online_net.train()

        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # 1. Forward pass on online network
        q_values = self.online_net(states)  # (batch_size, 16)

        # 2. Compute Double DQN TD loss
        td_loss, td_target = compute_double_dqn_td_loss(
            q_values=q_values,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            target_net=self.target_net,
            online_net=self.online_net,
            gamma=self.config.gamma,
            loss_type=self.config.loss_type,
        )

        # 3. Compute Supervised Large Margin Loss
        margin_loss = self.margin_loss_fn(q_values, actions)

        # 4. Compute L2 Regularization Loss
        l2_loss = compute_l2_regularization(self.online_net)

        # 5. Combined DQfD Loss
        total_loss = td_loss + self.config.lambda_margin * margin_loss + self.config.lambda_l2 * l2_loss

        # 6. Optimization Step
        self.optimizer.zero_grad()
        total_loss.backward()

        if self.config.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.config.grad_clip)

        self.optimizer.step()

        self.pretrain_updates += 1
        self.total_updates += 1

        # Periodic hard target update
        if self.total_updates % self.config.target_update_frequency == 0:
            self.update_target_network()

        # Gather metrics
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
        Execute a mixed mini-batch training step combining expert demonstrations and online experience.

        Parameters
        ----------
        demo_batch : batch of transitions sampled from the demonstration buffer
        online_batch : batch of transitions sampled from online exploration buffer

        Returns
        -------
        dict with loss components
        """
        self.online_net.train()

        # Concatenate tensors
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

        # 2. Double DQN TD Loss on all transitions
        td_loss, td_target = compute_double_dqn_td_loss(
            q_values=q_values,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            target_net=self.target_net,
            online_net=self.online_net,
            gamma=self.config.gamma,
            loss_type=self.config.loss_type,
        )

        # 3. Supervised Margin Loss applied ONLY to expert demonstration transitions
        margin_loss = self.margin_loss_fn(
            q_values=q_values,
            expert_actions=actions,
            mask=is_demo_mask,
        )

        # 4. L2 Regularization
        l2_loss = compute_l2_regularization(self.online_net)

        # 5. Combined Loss
        total_loss = td_loss + self.config.lambda_margin * margin_loss + self.config.lambda_l2 * l2_loss

        # 6. Backpropagation & step
        self.optimizer.zero_grad()
        total_loss.backward()

        if self.config.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.config.grad_clip)

        self.optimizer.step()

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
