"""
Unit tests for the Demonstration module (Teacher, Dataset, Generator).
"""

import os
import tempfile
import pytest
import numpy as np

from environment.roadster_env import RoadsterEnv
from demonstration.teacher import HeuristicTeacherController
from demonstration.dataset import DemonstrationTransition, DemonstrationDataset
from demonstration.generator import DemonstrationGenerator


def test_teacher_action_validity():
    """Verify teacher produces actions strictly in [0, 15]."""
    teacher = HeuristicTeacherController()
    env = RoadsterEnv()
    
    obs, info = env.reset(seed=10)
    for _ in range(20):
        action = teacher.select_action(obs=obs, info=info)
        assert isinstance(action, int)
        assert 0 <= action <= 15
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break


def test_dataset_serialization_jsonl():
    """Verify DemonstrationDataset serialization and roundtrip loading via JSONL."""
    t1 = DemonstrationTransition(
        state=[0.1, 0.5, 0.0, -0.2, 0.0, 1.0, 0.0, 1.0, 0.1],
        action=3,
        reward=1.5,
        raw_reward=150.0,
        next_state=[0.1, 0.48, 0.0, -0.15, 0.0, 1.0, 0.0, 0.99, 0.15],
        done=False,
        episode_id=0,
        step_idx=0,
        info={"h": 7.2},
    )
    t2 = DemonstrationTransition(
        state=[0.1, 0.48, 0.0, -0.15, 0.0, 1.0, 0.0, 0.99, 0.15],
        action=3,
        reward=2.3,
        raw_reward=230.0,
        next_state=[0.1, 0.48, 0.0, -0.1, 0.0, 1.0, 0.0, 0.98, 0.2],
        done=True,
        episode_id=0,
        step_idx=1,
        info={"safe_landing": True},
    )

    dataset = DemonstrationDataset(transitions=[t1, t2])
    assert dataset.num_transitions == 2
    assert dataset.num_episodes == 1

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        temp_path = f.name

    try:
        dataset.save_jsonl(temp_path)
        loaded = DemonstrationDataset.load_jsonl(temp_path)

        assert loaded.num_transitions == 2
        assert loaded.num_episodes == 1
        assert loaded.transitions[0].action == 3
        assert loaded.transitions[1].done is True
        assert loaded.transitions[1].raw_reward == 230.0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_dataset_sample_batch():
    """Verify batch sampling returns expected numpy array shapes and dtypes."""
    transitions = []
    for i in range(10):
        t = DemonstrationTransition(
            state=[float(j) for j in range(9)],
            action=i % 16,
            reward=float(i),
            raw_reward=float(i * 10),
            next_state=[float(j + 1) for j in range(9)],
            done=(i == 9),
            episode_id=0,
            step_idx=i,
        )
        transitions.append(t)

    dataset = DemonstrationDataset(transitions=transitions)
    batch = dataset.sample_batch(batch_size=4, rng=np.random.default_rng(42))

    assert batch["states"].shape == (4, 9)
    assert batch["actions"].shape == (4,)
    assert batch["rewards"].shape == (4,)
    assert batch["raw_rewards"].shape == (4,)
    assert batch["next_states"].shape == (4, 9)
    assert batch["dones"].shape == (4,)
    assert batch["states"].dtype == np.float32
    assert batch["actions"].dtype == np.int64


def test_generator_filter_safe_landings():
    """Verify generator collects requested number of safe episodes and filters out failures."""
    generator = DemonstrationGenerator()
    
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        temp_path = f.name

    try:
        target_safe = 5
        dataset, stats = generator.generate(
            target_safe_episodes=target_safe,
            output_filepath=temp_path,
            seed_start=100,
            verbose=False,
        )

        assert dataset.num_episodes == target_safe
        assert stats["episodes_collected"] == target_safe
        assert stats["num_transitions"] > 0
        assert stats["avg_landing_time_s"] < 5.0  # Must be fast
        assert stats["avg_landing_speed_vy_ms"] < 1.0  # Must be safe landing
        assert stats["avg_landing_angle_deg"] < 5.0  # Must be horizontal
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
