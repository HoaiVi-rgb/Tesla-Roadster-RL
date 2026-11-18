"""
DQfD Module: Deep Q-learning from Demonstrations (Double DQN + Supervised Margin Loss).
"""

from rl.dqfd.losses import SupervisedLargeMarginLoss, compute_l2_regularization
from rl.dqfd.agent import DQfDAgent, DQfDConfig

__all__ = [
    "SupervisedLargeMarginLoss",
    "compute_l2_regularization",
    "DQfDAgent",
    "DQfDConfig",
]
