"""
Unit tests for Environment Step cycle, 0.1s decision holding, 12 substeps, and info telemetry.
"""

import math
import pytest
import numpy as np

from environment.roadster_env import RoadsterEnv
from environment.state import validate_state
from physics.constants import (
    DECISION_DT,
    GRAVITY_ACCEL,
)


def test_step_timing_and_substeps():
    """Verify each step holds action for 0.1s across 12 physics substeps at 120Hz."""
    env = RoadsterEnv()
    obs, info = env.reset(seed=123, options={"initial_y": 15.0, "initial_x": 0.0})
    
    assert info["time"] == 0.0
    assert info["step_count"] == 0
    
    # Take 1 step with action 0 (free fall)
    next_obs, reward, terminated, truncated, next_info = env.step(0)
    
    assert validate_state(next_obs) is True
    assert next_info["time"] == pytest.approx(0.1, abs=1e-5)
    assert next_info["step_count"] == 1
    assert terminated is False
    assert truncated is False
    assert next_info["valves"] == (0, 0, 0, 0)
    assert next_info["action"] == 0


def test_step_info_keys():
    """Verify all required telemetry keys are present in info dictionary."""
    env = RoadsterEnv()
    obs, info = env.reset(seed=42)
    next_obs, reward, terminated, truncated, next_info = env.step(3)
    
    required_keys = [
        "x", "h", "vx", "vy", "theta", "omega", "fuel", "time",
        "landing_speed", "landing_angle", "landing_position",
        "termination_reason", "safe_landing", "crash", "out_of_bounds", "timeout",
        "done", "in_target_zone", "action", "valves", "bin_action",
        "flips_completed", "step_count", "fuel_consumed_step",
        "raw_reward", "learning_reward", "shaping_reward",
    ]
    for key in required_keys:
        assert key in next_info, f"Missing required info key: {key}"


def test_vertical_acceleration_step():
    """Verify vertical velocity change after 0.1s step under pure gravity vs full lift."""
    # Free fall test: delta_vy should be approx -g * 0.1 = -0.981 m/s
    env_fall = RoadsterEnv()
    env_fall.reset(options={"initial_y": 15.0, "initial_x": 0.0, "initial_vy": 0.0})
    _, _, _, _, info_fall = env_fall.step(0)
    expected_vy_fall = -GRAVITY_ACCEL * 0.1
    assert info_fall["vy"] == pytest.approx(expected_vy_fall, abs=1e-3)

    # Full lift test (Action 3): upward acceleration a_y = +5.19 m/s^2
    # delta_vy should be approx +5.19 * 0.1 = +0.519 m/s
    env_lift = RoadsterEnv()
    env_lift.reset(options={"initial_y": 15.0, "initial_x": 0.0, "initial_vy": 0.0})
    _, _, _, _, info_lift = env_lift.step(3)
    expected_vy_lift = 5.19 * 0.1
    assert info_lift["vy"] == pytest.approx(expected_vy_lift, abs=1e-2)


def test_step_done_backward_compatibility():
    """Verify step_done helper matches Gym API (obs, reward, done, info)."""
    env = RoadsterEnv()
    env.reset(seed=100)
    obs, reward, done, info = env.step_done(0)
    
    assert validate_state(obs) is True
    assert isinstance(done, bool)
    assert done == info["done"]
