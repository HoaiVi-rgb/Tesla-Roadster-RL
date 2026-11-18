"""
Unit tests for Guided Double DQN / DQfD components:
- Supervised Large Margin Classification Loss
- L2 Regularization Loss
- DQfDAgent pre-training and mixed training steps
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from rl.dqfd.losses import (
    SupervisedLargeMarginLoss,
    compute_l2_regularization,
    compute_double_dqn_td_loss,
)
from rl.dqfd.agent import DQfDAgent, DQfDConfig
from rl.dqn.network import QNetwork


def test_margin_loss_zero_when_expert_dominates():
    """
    When Q(s, a_E) >= Q(s, a) + margin for all a != a_E,
    the margin loss must be exactly 0.0.
    """
    margin = 0.5
    loss_fn = SupervisedLargeMarginLoss(margin=margin)

    # 2 samples, 4 actions
    # Sample 0: expert = 2, Q values: [1.0, 1.0, 2.0, 1.0] -> Q(2) - Q(others) = 1.0 >= 0.5 -> loss = 0
    # Sample 1: expert = 0, Q values: [5.0, 3.0, 2.0, 4.0] -> Q(0) - Q(others) = 1.0 >= 0.5 -> loss = 0
    q_values = torch.tensor([
        [1.0, 1.0, 2.0, 1.0],
        [5.0, 3.0, 2.0, 4.0],
    ])
    expert_actions = torch.tensor([2, 0], dtype=torch.long)

    loss = loss_fn(q_values, expert_actions)
    assert pytest.approx(loss.item(), abs=1e-6) == 0.0


def test_margin_loss_positive_when_expert_underestimated():
    """
    When Q(s, a_E) < Q(s, a) + margin, the loss must be strictly positive.
    """
    margin = 0.5
    loss_fn = SupervisedLargeMarginLoss(margin=margin)

    # Expert is 0, but Q(0) = 1.0 and Q(1) = 2.0.
    # Q(0) + l(0, 0) = 1.0 + 0 = 1.0
    # Q(1) + l(0, 1) = 2.0 + 0.5 = 2.5
    # Max is 2.5. Loss = 2.5 - 1.0 = 1.5.
    q_values = torch.tensor([[1.0, 2.0, 0.0]])
    expert_actions = torch.tensor([0], dtype=torch.long)

    loss = loss_fn(q_values, expert_actions)
    assert pytest.approx(loss.item(), abs=1e-5) == 1.5


def test_margin_loss_masking():
    """
    Ensure mask correctly ignores online transitions where mask == False.
    """
    margin = 0.5
    loss_fn = SupervisedLargeMarginLoss(margin=margin)

    q_values = torch.tensor([
        [1.0, 2.0],  # sample 0 (expert=0): loss would be (2.0 + 0.5) - 1.0 = 1.5
        [0.0, 10.0], # sample 1 (expert=0): masked out
    ])
    expert_actions = torch.tensor([0, 0], dtype=torch.long)
    mask = torch.tensor([True, False], dtype=torch.bool)

    loss = loss_fn(q_values, expert_actions, mask=mask)
    assert pytest.approx(loss.item(), abs=1e-5) == 1.5


def test_l2_regularization():
    """
    Test L2 weight penalty computation on a toy module.
    """
    linear = nn.Linear(3, 2, bias=True)
    # Set weights to 2.0, bias to 10.0
    with torch.no_grad():
        linear.weight.fill_(2.0)
        linear.bias.fill_(10.0)

    # 3x2 = 6 weights, each 2.0^2 = 4.0. Total = 24.0.
    # 0.5 * 24.0 = 12.0
    l2_loss = compute_l2_regularization(linear)
    assert pytest.approx(l2_loss.item(), abs=1e-5) == 12.0


def test_dqfd_pretrain_step():
    """
    Test that a single DQfD pretrain step updates network weights and decreases/calculates losses.
    """
    torch.manual_seed(42)
    config = DQfDConfig(
        state_dim=9,
        action_dim=16,
        hidden_dim=64,
        learning_rate=0.001,
        lambda_margin=5.0,
        margin_value=0.2,
        lambda_l2=1e-4,
    )
    agent = DQfDAgent(config=config)

    batch = {
        "states": torch.randn(16, 9),
        "actions": torch.randint(0, 16, (16,), dtype=torch.long),
        "rewards": torch.randn(16),
        "next_states": torch.randn(16, 9),
        "dones": torch.zeros(16, dtype=torch.bool),
    }

    metrics = agent.pretrain_step(batch)

    assert "loss_total" in metrics
    assert "loss_td" in metrics
    assert "loss_margin" in metrics
    assert "loss_l2" in metrics
    assert metrics["loss_total"] > 0.0
    assert agent.pretrain_updates == 1
    assert agent.total_updates == 1


def test_dqfd_mixed_train_step():
    """
    Test mixed mini-batch training step combining demo and online batches.
    """
    torch.manual_seed(42)
    config = DQfDConfig(
        state_dim=9,
        action_dim=16,
        hidden_dim=64,
        learning_rate=0.001,
        lambda_margin=5.0,
    )
    agent = DQfDAgent(config=config)

    demo_batch = {
        "states": torch.randn(8, 9),
        "actions": torch.randint(0, 16, (8,), dtype=torch.long),
        "rewards": torch.randn(8),
        "next_states": torch.randn(8, 9),
        "dones": torch.zeros(8, dtype=torch.bool),
    }
    online_batch = {
        "states": torch.randn(24, 9),
        "actions": torch.randint(0, 16, (24,), dtype=torch.long),
        "rewards": torch.randn(24),
        "next_states": torch.randn(24, 9),
        "dones": torch.zeros(24, dtype=torch.bool),
    }

    metrics = agent.mixed_train_step(demo_batch, online_batch)
    assert "loss_total" in metrics
    assert "loss_td" in metrics
    assert "loss_margin" in metrics
    assert agent.total_updates == 1
