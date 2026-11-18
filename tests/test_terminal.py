"""
Unit tests for Terminal conditions: Safe Landing, Crash, Out of Bounds, Timeout.
"""

import math
import pytest
from environment.roadster_env import RoadsterEnv
from physics.constants import REST_HEIGHT


def test_terminal_safe_landing():
    """Verify episode terminates with safe_landing=True when landing gently at x=0."""
    env = RoadsterEnv()
    # Place car gently right above ground contact
    env.reset(
        options={
            "initial_x": 0.0,
            "initial_y": REST_HEIGHT + 0.01,
            "initial_vx": 0.1,
            "initial_vy": -0.4,
            "initial_theta": math.radians(1.5),
            "initial_omega": 0.05,
        }
    )
    obs, reward, terminated, truncated, info = env.step(0)
    
    assert terminated is True
    assert truncated is False
    assert info["safe_landing"] is True
    assert info["crash"] is False
    assert info["in_target_zone"] is True
    assert info["termination_reason"] == "SAFE_LANDING"
    assert info["raw_reward"] >= 200.0  # +200 safe landing bonus included


def test_terminal_crash_high_speed():
    """Verify episode terminates with crash=True when hitting ground at excessive speed."""
    env = RoadsterEnv()
    env.reset(
        options={
            "initial_x": 0.0,
            "initial_y": REST_HEIGHT + 0.01,
            "initial_vx": 0.0,
            "initial_vy": -2.5,  # Speed > 1.0 m/s safe threshold
            "initial_theta": 0.0,
            "initial_omega": 0.0,
        }
    )
    obs, reward, terminated, truncated, info = env.step(0)
    
    assert terminated is True
    assert truncated is False
    assert info["safe_landing"] is False
    assert info["crash"] is True
    assert "CRASH" in info["termination_reason"]
    assert info["reward_terminal"] == -100.0  # Crash penalty


def test_terminal_out_of_bounds_x():
    """Verify episode terminates with out_of_bounds=True when x exceeds boundary."""
    env = RoadsterEnv()
    env.reset(
        options={
            "initial_x": 20.5,  # Exceeds WORLD_X_MAX (20.0m)
            "initial_y": 10.0,
            "initial_vx": 1.0,
            "initial_vy": 0.0,
        }
    )
    obs, reward, terminated, truncated, info = env.step(0)
    
    assert terminated is True
    assert info["out_of_bounds"] is True
    assert info["termination_reason"] == "OUT_OF_BOUNDS"
    assert info["reward_terminal"] == -250.0  # OOB penalty


def test_terminal_timeout():
    """Verify episode truncates with timeout=True when reaching 20.0s."""
    env = RoadsterEnv()
    # Start at high altitude with time already close to 20s
    env.reset(options={"initial_y": 20.0, "initial_x": 0.0})
    
    # Run steps until 20.0s
    for step_idx in range(200):
        # Keep airborne by oscillating lift
        action = 3 if env.world.body.vy < 0 else 0
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
            
    assert info["done"] is True
    assert info["timeout"] is True
    assert truncated is True
    assert info["termination_reason"] == "TIMEOUT"
    assert info["reward_terminal"] == -500.0  # Timeout penalty
