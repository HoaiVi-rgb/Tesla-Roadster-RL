"""
Experience Replay Buffer for DQN.
Uses pre-allocated circular NumPy arrays for maximum memory and speed efficiency.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer storing (s, a, r, s', done).
    """

    def __init__(self, capacity: int = 50000, state_dim: int = 9):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)

        # Pre-allocated arrays
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.bool_)

        self.ptr = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a new transition to the circular buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = bool(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: str = "cpu",
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random batch of transitions as PyTorch Tensors.
        """
        if self.size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} transitions from buffer with size {self.size}")

        rng = rng or np.random.default_rng()
        indices = rng.integers(0, self.size, size=batch_size)

        b_states = torch.as_tensor(self.states[indices], dtype=torch.float32, device=device)
        b_actions = torch.as_tensor(self.actions[indices], dtype=torch.int64, device=device)
        b_rewards = torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device)
        b_next_states = torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=device)
        b_dones = torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device)

        return b_states, b_actions, b_rewards, b_next_states, b_dones

    def __len__(self) -> int:
        return self.size

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def clear(self) -> None:
        self.ptr = 0
        self.size = 0
