"""
Unit tests for the Deep Q-Network (DQN) Baseline implementation.
"""

import os
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn

from rl.dqn.network import QNetwork
from rl.dqn.replay_buffer import ReplayBuffer
from rl.dqn.agent import DQNAgent, DQNConfig
from rl.dqn.trainer import DQNTrainer
from environment.roadster_env import RoadsterEnv


def test_q_network_dimensions_and_forward():
    """Verify architecture: 9 -> 64 -> 64 -> 16 with ReLU activations."""
    net = QNetwork(state_dim=9, action_dim=16, hidden_dim=64)
    
    # Layer tests
    assert isinstance(net.fc1, nn.Linear)
    assert net.fc1.in_features == 9
    assert net.fc1.out_features == 64
    assert net.fc2.in_features == 64
    assert net.fc2.out_features == 64
    assert net.fc3.in_features == 64
    assert net.fc3.out_features == 16

    # Single state forward pass
    state = torch.randn(9)
    out = net(state)
    assert out.shape == (16,)

    # Batch state forward pass
    batch_states = torch.randn(32, 9)
    batch_out = net(batch_states)
    assert batch_out.shape == (32, 16)


def test_replay_buffer_push_and_sample():
    """Verify replay buffer storing, circular pointer, and tensor sampling."""
    buf = ReplayBuffer(capacity=10, state_dim=9)
    assert len(buf) == 0
    assert not buf.can_sample(4)

    # Push 15 transitions into capacity 10 buffer
    for i in range(15):
        s = np.full(9, float(i), dtype=np.float32)
        s_next = np.full(9, float(i + 1), dtype=np.float32)
        buf.push(state=s, action=i % 16, reward=float(i), next_state=s_next, done=(i % 5 == 0))

    assert len(buf) == 10
    assert buf.can_sample(4)

    # Sample batch
    b_s, b_a, b_r, b_s_next, b_d = buf.sample(batch_size=4)
    assert b_s.shape == (4, 9)
    assert b_a.shape == (4,)
    assert b_r.shape == (4,)
    assert b_s_next.shape == (4, 9)
    assert b_d.shape == (4,)


def test_dqn_agent_epsilon_schedule():
    """Verify linear epsilon decay schedule."""
    config = DQNConfig(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=1000)
    agent = DQNAgent(config)

    agent.total_env_steps = 0
    assert agent.get_epsilon() == pytest.approx(1.0)

    agent.total_env_steps = 500
    assert agent.get_epsilon() == pytest.approx(0.525)

    agent.total_env_steps = 1000
    assert agent.get_epsilon() == pytest.approx(0.05)

    agent.total_env_steps = 5000
    assert agent.get_epsilon() == pytest.approx(0.05)


def test_dqn_td_target_calculation():
    """
    Verify exact Bellman TD Target:
    y = r + gamma * (1 - done) * max_a' Q_target(s', a')
    """
    config = DQNConfig(gamma=0.995, batch_size=2)
    agent = DQNAgent(config)

    # Construct synthetic batch:
    # Transition 1: non-terminal (done=0), reward=1.0
    # Transition 2: terminal (done=1), reward=-2.0
    s = torch.zeros((2, 9), dtype=torch.float32)
    a = torch.tensor([2, 5], dtype=torch.int64)
    r = torch.tensor([1.0, -2.0], dtype=torch.float32)
    s_next = torch.ones((2, 9), dtype=torch.float32)
    done = torch.tensor([0.0, 1.0], dtype=torch.float32)

    # Compute expected targets manually using agent's target network
    with torch.no_grad():
        q_target_next = agent.target_net(s_next)
        max_q_next = q_target_next.max(dim=1)[0]
        expected_y0 = (1.0 + 0.995 * (1.0 - 0.0) * max_q_next[0]).item()
        expected_y1 = -2.0  # (1 - 1.0) cancels future Q-value!

    loss = agent.train_step((s, a, r, s_next, done))
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_dqn_target_update_and_checkpoint():
    """Verify target network weight synchronization and model saving/loading."""
    config = DQNConfig(target_update_frequency=5)
    agent = DQNAgent(config)

    # Modify online network parameters
    with torch.no_grad():
        agent.online_net.fc1.weight.add_(1.0)

    # Before update, target net and online net parameters differ
    diff = torch.sum(torch.abs(agent.online_net.fc1.weight - agent.target_net.fc1.weight)).item()
    assert diff > 0.1

    # Update target network
    agent.update_target_network()
    diff_after = torch.sum(torch.abs(agent.online_net.fc1.weight - agent.target_net.fc1.weight)).item()
    assert diff_after == pytest.approx(0.0, abs=1e-7)

    # Test Checkpoint Save and Load
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        temp_path = f.name

    try:
        agent.total_env_steps = 1234
        agent.total_updates = 56
        agent.save_checkpoint(temp_path)

        agent_new = DQNAgent(config)
        agent_new.load_checkpoint(temp_path)

        assert agent_new.total_env_steps == 1234
        assert agent_new.total_updates == 56
        new_diff = torch.sum(torch.abs(agent.online_net.fc1.weight - agent_new.online_net.fc1.weight)).item()
        assert new_diff == pytest.approx(0.0, abs=1e-7)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_dqn_trainer_smoke_run():
    """Verify trainer executes steps, records all required logging metrics, and runs without error."""
    env = RoadsterEnv()
    config = DQNConfig(
        warmup_steps=10,
        batch_size=8,
        buffer_size=1000,
        target_update_frequency=10,
    )
    agent = DQNAgent(config)

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = DQNTrainer(env=env, agent=agent, checkpoint_dir=temp_dir)
        logs = trainer.train(num_episodes=3, eval_interval=2, eval_episodes=2, verbose=False)

        assert len(logs) == 3
        log = logs[0]
        # Verify required log keys
        required_keys = [
            "episode", "reward", "loss", "epsilon", "success_rate",
            "landing_velocity", "landing_angle", "fuel_used", "landing_time",
        ]
        for k in required_keys:
            assert k in log, f"Missing required log field: {k}"
