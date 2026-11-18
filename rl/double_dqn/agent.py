"""
Double Deep Q-Network (Double DQN) Agent.
Reference: Hado van Hasselt, Arthur Guez, David Silver (AAAI 2016).
Decouples action selection from action evaluation to eliminate overestimation bias.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.dqn.agent import DQNAgent, DQNConfig


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN Agent.

    Decoupled Target Formulation:
        a* = argmax_a Q(s_{t+1}, a; theta_online)
        y_t = r_t + gamma * (1 - done_t) * Q(s_{t+1}, a*; theta_target)

    Compared to Standard DQN:
        y_t^DQN = r_t + gamma * (1 - done_t) * max_a' Q(s_{t+1}, a'; theta_target)

    In standard DQN, the same target parameters theta_target are used to BOTH select
    and evaluate the action, creating Jensen's inequality maximization bias:
        E[max_a X_a] >= max_a E[X_a].
    Double DQN uses online parameters to select, and target parameters to evaluate,
    effectively eliminating upward estimation bias.
    """

    def __init__(self, config: Optional[DQNConfig] = None):
        super().__init__(config=config)
        self.last_td_targets_mean: float = 0.0
        self.last_q_predicted_mean: float = 0.0

    def compute_td_target(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Double DQN TD target.

        Step 1: Action Selection via Online Network:
            a* = argmax_a Q(s', a; theta_online)
        Step 2: Action Evaluation via Target Network:
            Q(s', a*; theta_target)
        Step 3: Target:
            y = r + gamma * (1 - done) * Q(s', a*; theta_target)
        """
        with torch.no_grad():
            # 1. Action selection using online network
            q_next_online = self.online_net(next_states)  # (batch_size, 16)
            best_actions = q_next_online.argmax(dim=1, keepdim=True)  # (batch_size, 1)

            # 2. Action evaluation using target network
            q_next_target = self.target_net(next_states)  # (batch_size, 16)
            q_target_selected = q_next_target.gather(1, best_actions).squeeze(1)  # (batch_size,)

            # 3. Bellman target
            td_target = rewards + self.config.gamma * (1.0 - dones) * q_target_selected
            self.last_td_targets_mean = float(td_target.mean().item())
            return td_target

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> float:
        """
        Execute Double DQN gradient step.
        """
        states, actions, rewards, next_states, dones = batch

        # 1. Predicted Q(s, a; theta_online)
        self.online_net.train()
        q_values = self.online_net(states)  # (batch_size, 16)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch_size,)
        self.last_q_predicted_mean = float(q_sa.mean().item())

        # 2. Double DQN target
        td_target = self.compute_td_target(rewards, next_states, dones)

        # 3. Loss (MSE or Huber)
        if self.config.loss_type.lower() == "huber":
            loss = F.smooth_l1_loss(q_sa, td_target)
        else:
            loss = F.mse_loss(q_sa, td_target)

        # 4. Optimization step with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()

        if self.config.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.config.grad_clip)

        self.optimizer.step()

        # 5. Periodic target sync
        self.total_updates += 1
        if self.total_updates % self.config.target_update_frequency == 0:
            self.update_target_network()

        return float(loss.item())
