"""
Unit tests for rl/demonstration_agent.py.
Validates:
- SupervisedMarginLoss non-negativity and margin separation
- DemonstrationDQNAgent pretrain step
- DemonstrationDQNAgent mixed train step
- L2 regularization
"""

import pytest
import torch
import numpy as np

from rl.demonstration_agent import (
    DemonstrationAgentConfig,
    SupervisedMarginLoss,
    DemonstrationDQNAgent,
)


def test_supervised_margin_loss_mathematical_properties():
    loss_fn = SupervisedMarginLoss(margin=0.2)

    # Case 1: Expert action has highest Q by at least margin -> Loss = 0.0
    # Q(s, a_E) = 2.0, non-expert max = 1.7 -> diff = 0.3 > 0.2
    q_values = torch.tensor([[1.0, 2.0, 1.7, 0.5]])
    expert_action = torch.tensor([1], dtype=torch.long)
    loss = loss_fn(q_values, expert_action)
    assert pytest.approx(loss.item(), abs=1e-6) == 0.0

    # Case 2: Expert action Q is equal to non-expert Q -> Loss = margin (0.2)
    q_values = torch.tensor([[1.0, 2.0, 2.0, 0.5]])
    loss = loss_fn(q_values, expert_action)
    assert pytest.approx(loss.item(), abs=1e-5) == 0.2

    # Case 3: Non-expert action is larger by 1.0 -> Loss = 1.0 + margin = 1.2
    q_values = torch.tensor([[1.0, 2.0, 3.0, 0.5]])
    loss = loss_fn(q_values, expert_action)
    assert pytest.approx(loss.item(), abs=1e-5) == 1.2


def test_supervised_margin_loss_masking():
    loss_fn = SupervisedMarginLoss(margin=0.2)

    # Sample 0 is demo, Sample 1 is online (masked)
    q_values = torch.tensor([
        [1.0, 2.0, 3.0, 0.5],   # sample 0: expert=1, max_aug = 3.2, loss = 1.2
        [0.0, 0.0, 10.0, 0.0],  # sample 1: masked out
    ])
    expert_actions = torch.tensor([1, 1], dtype=torch.long)
    mask = torch.tensor([True, False], dtype=torch.bool)

    loss = loss_fn(q_values, expert_actions, mask=mask)
    assert pytest.approx(loss.item(), abs=1e-5) == 1.2


def test_demonstration_agent_pretrain_step():
    agent = DemonstrationDQNAgent()
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


def test_demonstration_agent_mixed_step():
    agent = DemonstrationDQNAgent()
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
    assert agent.mixed_updates == 1
