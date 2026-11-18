"""
Unit tests for the independent evaluation engine and failure classification taxonomy.
"""

from __future__ import annotations
import math
import pytest
import numpy as np

from evaluation.test_cases import build_curated_test_cases, get_stochastic_unseen_seeds
from evaluation.evaluator import StandaloneEvaluator, EpisodeTelemetry
from rl.demonstration_agent import DemonstrationDQNAgent
from environment.roadster_env import RoadsterEnv


def test_test_cases_generation():
    """Verify curated corner test cases cover 60 cases across 6 distinct categories."""
    cases = build_curated_test_cases()
    assert len(cases) == 60
    categories = set(c.category for c in cases)
    assert len(categories) == 6
    assert "Nominal Approach" in categories
    assert "High Altitude Drop" in categories
    assert "Severe Attitude Tilt" in categories
    assert "High Horizontal Drift" in categories
    assert "Low Fuel / Rapid Descent" in categories
    assert "Compound Extreme" in categories


def test_stochastic_unseen_seeds():
    """Verify stochastic seeds are strictly >= 30000 (disjoint from training and prior evals)."""
    seeds = get_stochastic_unseen_seeds(count=100, seed_start=30000)
    assert len(seeds) == 100
    assert seeds[0] == 30000
    assert seeds[-1] == 30099
    # Zero overlap with training seeds (1..1000) or previous test seeds (20000..20099)
    assert all(s >= 30000 for s in seeds)


def test_failure_classification_taxonomy():
    """Verify correct classification of failure modes."""
    evaluator = StandaloneEvaluator()

    # Safe landing
    f_up, f_over, f_under, f_fast, f_rot, prim = evaluator._classify_failure(
        is_safe=True, is_collision=False, is_timeout=False, is_oob=False,
        final_x=0.2, final_y=0.0, final_vx=0.1, final_vy=-0.4, final_theta_deg=1.5, final_omega=0.05
    )
    assert prim == "SAFE_LANDING"
    assert not f_up and not f_fast and not f_rot

    # Crash: too fast
    f_up, f_over, f_under, f_fast, f_rot, prim = evaluator._classify_failure(
        is_safe=False, is_collision=True, is_timeout=False, is_oob=False,
        final_x=0.1, final_y=0.0, final_vx=0.1, final_vy=-2.5, final_theta_deg=1.0, final_omega=0.05
    )
    assert f_fast
    assert not f_rot
    assert prim == "CRASH_TOO_FAST"

    # Crash: too much rotation
    f_up, f_over, f_under, f_fast, f_rot, prim = evaluator._classify_failure(
        is_safe=False, is_collision=True, is_timeout=False, is_oob=False,
        final_x=0.1, final_y=0.0, final_vx=0.1, final_vy=-0.4, final_theta_deg=12.0, final_omega=0.3
    )
    assert f_rot
    assert not f_fast
    assert prim == "CRASH_TOO_MUCH_ROTATION"

    # Crash: upside down
    f_up, f_over, f_under, f_fast, f_rot, prim = evaluator._classify_failure(
        is_safe=False, is_collision=True, is_timeout=False, is_oob=False,
        final_x=0.1, final_y=0.0, final_vx=0.1, final_vy=-1.5, final_theta_deg=120.0, final_omega=0.5
    )
    assert f_up
    assert prim == "CRASH_UPSIDE_DOWN"

    # Timeout
    f_up, f_over, f_under, f_fast, f_rot, prim = evaluator._classify_failure(
        is_safe=False, is_collision=False, is_timeout=True, is_oob=False,
        final_x=0.0, final_y=5.0, final_vx=0.0, final_vy=0.0, final_theta_deg=0.0, final_omega=0.0
    )
    assert prim == "TIMEOUT"
