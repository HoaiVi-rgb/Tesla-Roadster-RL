"""
Unit tests for environment reproducibility with fixed seeds and curriculum level ranges.
"""

import math
import pytest
import numpy as np

from environment.roadster_env import RoadsterEnv


def test_seed_reproducibility():
    """Verify that resetting with the same seed generates exact identical trajectories."""
    env1 = RoadsterEnv()
    obs1, info1 = env1.reset(seed=42)
    
    env2 = RoadsterEnv()
    obs2, info2 = env2.reset(seed=42)
    
    np.testing.assert_array_almost_equal(obs1, obs2)
    assert info1["x"] == info2["x"]
    assert info1["h"] == info2["h"]
    assert info1["theta"] == info2["theta"]
    
    # Run 5 steps with identical actions
    actions = [0, 3, 3, 1, 4]
    for act in actions:
        next_obs1, r1, term1, trunc1, inf1 = env1.step(act)
        next_obs2, r2, term2, trunc2, inf2 = env2.step(act)
        
        np.testing.assert_array_almost_equal(next_obs1, next_obs2)
        assert r1 == pytest.approx(r2, abs=1e-6)
        assert term1 == term2
        assert trunc1 == trunc2
        assert inf1["fuel"] == inf2["fuel"]


def test_curriculum_level_bounds():
    """Verify that resetting with different curriculum levels respects their specified bounds."""
    env = RoadsterEnv()
    
    # Level 1: Low altitude (1.2 - 3.0m), small angle (+-3 deg), near center (+-0.2m)
    for seed in range(10):
        _, info_l1 = env.reset(seed=seed, options={"level": 1})
        assert 1.2 <= info_l1["h"] <= 3.0
        assert -0.2 <= info_l1["x"] <= 0.2
        assert abs(info_l1["landing_angle_deg"]) <= 3.01

    # Level 2: Medium altitude (2.0 - 5.0m), medium angle (+-8 deg), (+-0.5m)
    for seed in range(10):
        _, info_l2 = env.reset(seed=seed + 100, options={"level": 2})
        assert 2.0 <= info_l2["h"] <= 5.0
        assert -0.5 <= info_l2["x"] <= 0.5
        assert abs(info_l2["landing_angle_deg"]) <= 8.01

    # Level 3: Full challenge (4.0 - 9.0m), (+-15 deg), (+-2.0m)
    for seed in range(10):
        _, info_l3 = env.reset(seed=seed + 200, options={"level": 3})
        assert 4.0 <= info_l3["h"] <= 9.0
        assert -2.0 <= info_l3["x"] <= 2.0
        assert abs(info_l3["landing_angle_deg"]) <= 15.01
