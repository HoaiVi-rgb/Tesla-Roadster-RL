"""
PyTorch Neural Network Architecture for the Deep Q-Network (DQN).
Architecture: 9 (State) -> 64 -> 64 -> 16 (Q-values).
Activation: ReLU.
"""

from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Standard Deep Q-Network (MLP).
    Input:  9-dimensional state vector:
            [x/10, h/15, vx/10, vy/10, sin(theta), cos(theta), omega/5, fuel/100, t/20]
    Output: 16 Q-values corresponding to the 16 discrete thruster combinations.
    """

    def __init__(self, state_dim: int = 9, action_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using He/Kaiming uniform initialization with bias zero."""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for all 16 actions.

        Parameters
        ----------
        state : torch.Tensor of shape (batch_size, 9) or (9,)

        Returns
        -------
        torch.Tensor of shape (batch_size, 16) or (16,)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

    def get_q_values_numpy(self, state_np: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Helper to compute Q-values directly from numpy state input."""
        self.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(state_np, dtype=torch.float32, device=device)
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
            q_vals = self.forward(state_tensor)
            return q_vals.squeeze(0).cpu().numpy()

    def export_weights_dict(self) -> Dict[str, Any]:
        """
        Export network weights as raw nested lists/floats for JSON export to Web/JavaScript.
        """
        return {
            "fc1_weight": self.fc1.weight.detach().cpu().numpy().tolist(),
            "fc1_bias": self.fc1.bias.detach().cpu().numpy().tolist(),
            "fc2_weight": self.fc2.weight.detach().cpu().numpy().tolist(),
            "fc2_bias": self.fc2.bias.detach().cpu().numpy().tolist(),
            "fc3_weight": self.fc3.weight.detach().cpu().numpy().tolist(),
            "fc3_bias": self.fc3.bias.detach().cpu().numpy().tolist(),
        }
