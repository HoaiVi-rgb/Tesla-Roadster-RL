"""
Unit tests for the Double DQN implementation and target calculation.
"""

import pytest
import numpy as np
import torch

from rl.double_dqn.agent import DoubleDQNAgent
from rl.dqn.agent import DQNConfig


def test_double_dqn_target_decoupling():
    """
    Verify exact Double DQN Target formulation:
    a* = argmax_a Q_online(s', a)
    y = r + gamma * (1 - done) * Q_target(s', a*)
    """
    config = DQNConfig(gamma=0.995, batch_size=2)
    agent = DoubleDQNAgent(config)

    s = torch.zeros((2, 9), dtype=torch.float32)
    a = torch.tensor([1, 4], dtype=torch.int64)
    r = torch.tensor([0.5, 2.0], dtype=torch.float32)
    s_next = torch.ones((2, 9), dtype=torch.float32)
    dones = torch.tensor([0.0, 0.0], dtype=torch.float32)

    # 1. Manually find best action according to ONLINE network
    with torch.no_grad():
        q_next_online = agent.online_net(s_next)
        best_actions_online = q_next_online.argmax(dim=1)

        # 2. Evaluate that specific action using TARGET network
        q_next_target = agent.target_net(s_next)
        q_target_evaluated = q_next_target.gather(1, best_actions_online.unsqueeze(1)).squeeze(1)

        expected_y = r + 0.995 * (1.0 - dones) * q_target_evaluated

    # Call agent's method
    actual_y = agent.compute_td_target(r, s_next, dones)

    torch.testing.assert_close(actual_y, expected_y)


def test_double_dqn_gradient_step():
    """Verify Double DQN executes training step and decreases/changes weights."""
    config = DQNConfig(batch_size=4)
    agent = DoubleDQNAgent(config)

    s = torch.randn((4, 9), dtype=torch.float32)
    a = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    r = torch.randn(4, dtype=torch.float32)
    s_next = torch.randn((4, 9), dtype=torch.float32)
    dones = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32)

    initial_weights = agent.online_net.fc1.weight.clone().detach()
    loss = agent.train_step((s, a, r, s_next, dones))

    assert isinstance(loss, float)
    assert loss >= 0.0
    updated_weights = agent.online_net.fc1.weight.detach()
    diff = torch.sum(torch.abs(initial_weights - updated_weights)).item()
    assert diff > 0.0
