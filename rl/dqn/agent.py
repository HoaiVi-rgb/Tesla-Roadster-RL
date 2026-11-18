"""
Standard Deep Q-Network (DQN) Agent.
Implements epsilon-greedy exploration, Bellman TD target calculation,
periodic hard target update, gradient clipping, and checkpoint persistence.
STRICTLY NO TEACHER DEMONSTRATIONS IN BASELINE DQN.
"""

from __future__ import annotations
import math
import os
import json
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl.dqn.network import QNetwork
from environment.action import NUM_ACTIONS


@dataclass
class DQNConfig:
    """Hyperparameters and configuration for DQN Agent."""
    state_dim: int = 9
    action_dim: int = NUM_ACTIONS  # 16
    hidden_dim: int = 64
    learning_rate: float = 0.0005
    gamma: float = 0.995
    batch_size: int = 64
    buffer_size: int = 50000
    warmup_steps: int = 512
    target_update_frequency: int = 500  # steps/updates per target sync
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 100000  # environment steps to decay epsilon
    grad_clip: float = 1.0
    loss_type: str = "mse"  # "mse" or "huber"
    optimizer_type: str = "rmsprop"  # "rmsprop" or "adam"
    device: str = "cpu"


class DQNAgent:
    """
    DQN Agent implementing Mnih et al. (2015) Human-level control through deep reinforcement learning.
    """

    def __init__(self, config: Optional[DQNConfig] = None):
        self.config = config or DQNConfig()
        self.device = torch.device(self.config.device)

        # 1. Online Network & Target Network
        self.online_net = QNetwork(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)

        self.target_net = QNetwork(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)

        # Initialize target network with exact same weights
        self.update_target_network()
        self.target_net.eval()

        # 2. Optimizer
        if self.config.optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.config.learning_rate)
        else:
            # RMSprop with decay=0.95, eps=1e-5 matching Roadster_4_Voi_DQN specification
            self.optimizer = optim.RMSprop(
                self.online_net.parameters(),
                lr=self.config.learning_rate,
                alpha=0.95,
                eps=1e-5,
            )

        # 3. Counters
        self.total_env_steps: int = 0
        self.total_updates: int = 0
        self.rng = np.random.default_rng(42)

    def get_epsilon(self) -> float:
        """
        Compute current exploration rate epsilon using linear decay schedule.
        """
        if self.total_env_steps >= self.config.epsilon_decay:
            return self.config.epsilon_end
        frac = self.total_env_steps / float(self.config.epsilon_decay)
        return self.config.epsilon_start - frac * (self.config.epsilon_start - self.config.epsilon_end)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.

        Parameters
        ----------
        state : 9D normalized state vector
        evaluate : if True, act greedily (epsilon = 0.0)

        Returns
        -------
        int : action in [0, 15]
        """
        eps = 0.0 if evaluate else self.get_epsilon()

        # Epsilon exploration: pick uniformly random action
        if self.rng.random() < eps:
            return int(self.rng.integers(0, self.config.action_dim))

        # Greedy exploitation: pick action with maximum Q-value
        q_vals = self.get_q_values(state)
        return int(np.argmax(q_vals))

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Return 16 Q-values predicted by the online network."""
        return self.online_net.get_q_values_numpy(state, device=str(self.device))

    def update_target_network(self) -> None:
        """Hard copy online network weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> float:
        """
        Execute a single gradient descent update step on a batch of transitions.

        Parameters
        ----------
        batch : (states, actions, rewards, next_states, dones)

        Returns
        -------
        float : loss value
        """
        states, actions, rewards, next_states, dones = batch

        # 1. Compute Q(s, a; theta) using online network
        self.online_net.train()
        q_values = self.online_net(states)  # (batch_size, 16)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch_size,)

        # 2. Compute Bellman TD target: y = r + gamma * (1 - done) * max_a' Q_target(s', a')
        with torch.no_grad():
            q_next_target = self.target_net(next_states)  # (batch_size, 16)
            max_q_next = q_next_target.max(dim=1)[0]      # (batch_size,)
            td_target = rewards + self.config.gamma * (1.0 - dones) * max_q_next

        # 3. Compute loss (MSE or Huber)
        if self.config.loss_type.lower() == "huber":
            loss = F.smooth_l1_loss(q_sa, td_target)
        else:
            loss = F.mse_loss(q_sa, td_target)

        # 4. Backward pass and optimizer step
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.config.grad_clip)

        self.optimizer.step()

        # 5. Periodic target network hard update
        self.total_updates += 1
        if self.total_updates % self.config.target_update_frequency == 0:
            self.update_target_network()

        return float(loss.item())

    def save_checkpoint(self, filepath: str) -> None:
        """Save agent checkpoint to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "online_net_state_dict": self.online_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "total_env_steps": self.total_env_steps,
            "total_updates": self.total_updates,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load agent checkpoint from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_env_steps = checkpoint.get("total_env_steps", 0)
        self.total_updates = checkpoint.get("total_updates", 0)

    def export_weights_json(self, filepath: str) -> None:
        """Export online network weights to JSON format for browser execution."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        weights_dict = self.online_net.export_weights_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(weights_dict, f, indent=2)
