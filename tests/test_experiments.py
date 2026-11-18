"""
Unit tests for the reproducible experiment framework.
Validates config generation, evaluator metrics, seeding, and execution pipeline.
"""

from __future__ import annotations
import os
import shutil
import pytest
import numpy as np
import torch

from experiments.config import ExperimentConfig, build_experiment_suite, get_code_version_hash
from experiments.evaluator import evaluate_policy_on_test_domain
from rl.dqn.agent import DQNAgent, DQNConfig


def test_code_version_hash():
    """Verify code version hash is deterministic and non-empty."""
    h1 = get_code_version_hash()
    h2 = get_code_version_hash()
    assert len(h1) == 16
    assert h1 == h2


def test_build_experiment_suite():
    """Verify suite generates 15 configurations for 3 seeds across 5 algorithms."""
    suite = build_experiment_suite(seeds=[42, 123, 999])
    assert len(suite) == 15
    algo_names = set(c.algorithm for c in suite)
    assert len(algo_names) == 5
    assert "Standard DQN" in algo_names
    assert "Double DQN" in algo_names
    assert "Double DQN + Shaping" in algo_names
    assert "Double DQN + Curriculum" in algo_names
    assert "Demonstration-Assisted DDQN" in algo_names


def test_evaluator_deterministic():
    """Verify evaluator produces consistent metrics on fixed seeds."""
    agent = DQNAgent()
    res1 = evaluate_policy_on_test_domain(agent, num_episodes=5, seed_base=20000, level=3)
    res2 = evaluate_policy_on_test_domain(agent, num_episodes=5, seed_base=20000, level=3)

    assert res1["num_episodes"] == 5
    assert res1["success_rate"] == res2["success_rate"]
    assert res1["collision_rate"] == res2["collision_rate"]
    assert res1["mean_reward"] == res2["mean_reward"]
    assert "mean_landing_velocity" in res1
    assert "mean_abs_landing_angle" in res1
    assert "mean_fuel_used" in res1
