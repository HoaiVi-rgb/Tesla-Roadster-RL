"""
Unit tests for training/curriculum.py.
Validates:
- Baseline (curriculum disabled) behavior
- Level specifications and physical ranges
- Threshold-gated promotion mechanics
- Per-level telemetry logging (episodes, success rate, reward, velocity, angle)
"""

import pytest
import numpy as np

from training.curriculum import (
    RoadsterCurriculum,
    LevelSpec,
    LevelStatistics,
    STANDARD_LEVELS,
)


def test_level_specs():
    specs = {s.level: s for s in STANDARD_LEVELS}
    # Level 1
    assert specs[1].height_range == (1.2, 3.0)
    assert specs[1].angle_range_deg == (-3.0, 3.0)
    assert specs[1].x_range == (-0.2, 0.2)
    # Level 2
    assert specs[2].height_range == (2.0, 5.0)
    assert specs[2].angle_range_deg == (-8.0, 8.0)
    assert specs[2].x_range == (-0.5, 0.5)
    # Level 3
    assert specs[3].height_range == (4.0, 9.0)
    assert specs[3].angle_range_deg == (-15.0, 15.0)
    assert specs[3].x_range == (-2.0, 2.0)


def test_curriculum_disabled_baseline():
    curriculum = RoadsterCurriculum(enabled=False, baseline_level=3)
    assert curriculum.current_level == 3
    assert curriculum.get_env_options() == {"level": 3}

    # Record 100 successful episodes
    info = {"safe_landing": True, "fuel": 95.0, "time": 3.5}
    for _ in range(100):
        promoted, new_level = curriculum.record_episode(info, raw_reward=2.0)
        assert not promoted
        assert new_level is None
        assert curriculum.current_level == 3


def test_threshold_advancement():
    # Create test curriculum with small window and min episodes for fast testing
    test_levels = [
        LevelSpec(1, "L1", (1.2, 3.0), (-3.0, 3.0), (-0.2, 0.2), min_episodes=5, success_threshold=0.8),
        LevelSpec(2, "L2", (2.0, 5.0), (-8.0, 8.0), (-0.5, 0.5), min_episodes=5, success_threshold=0.8),
        LevelSpec(3, "L3", (4.0, 9.0), (-15.0, 15.0), (-2.0, 2.0), min_episodes=5, success_threshold=0.8),
    ]
    curriculum = RoadsterCurriculum(enabled=True, levels=test_levels, window_size=5)

    assert curriculum.current_level == 1
    assert curriculum.get_env_options() == {"level": 1}

    # Case A: 5 failures -> success rate 0% -> No promotion
    fail_info = {"safe_landing": False, "crash": True, "fuel": 95.0}
    for _ in range(5):
        promoted, _ = curriculum.record_episode(fail_info, raw_reward=-1.0)
        assert not promoted
    assert curriculum.current_level == 1

    # Case B: successes to raise rolling rate
    success_info = {
        "safe_landing": True,
        "landing_speed": 0.5,
        "landing_speed_vy": 0.45,
        "landing_speed_vx": 0.1,
        "landing_angle_deg": 1.5,
        "time": 3.8,
        "fuel": 94.0,
    }
    # First 3 successes: rolling queue has 1, 2, 3 successes out of 5 -> rates 20%, 40%, 60% < 80%
    for _ in range(3):
        promoted, _ = curriculum.record_episode(success_info, raw_reward=2.1)
        assert not promoted

    # 4th success: rolling queue has 4 successes out of 5 -> rate = 80% >= 80% threshold -> Promotes!
    promoted, new_level = curriculum.record_episode(success_info, raw_reward=2.1)
    assert promoted
    assert new_level == 2
    assert curriculum.current_level == 2
    assert curriculum.get_env_options() == {"level": 2}


def test_per_level_telemetry_logging():
    stats = LevelStatistics(level=1, name="Test Level")
    assert stats.episodes == 0

    # Record 1 safe landing
    stats.record({
        "safe_landing": True,
        "landing_speed": 0.6,
        "landing_speed_vy": 0.55,
        "landing_speed_vx": 0.12,
        "landing_angle_deg": 2.0,
        "time": 4.2,
        "fuel": 92.0,
    }, raw_reward=2.2)

    # Record 1 crash
    stats.record({
        "safe_landing": False,
        "crash": True,
        "fuel": 96.0,
    }, raw_reward=-1.0)

    summary = stats.to_dict()
    assert summary["episodes"] == 2
    assert summary["safe_landings"] == 1
    assert summary["success_rate_percent"] == 50.0
    assert summary["crash_rate_percent"] == 50.0
    assert summary["mean_reward"] == 0.6  # (2.2 + -1.0) / 2
    assert summary["mean_landing_speed_mps"] == 0.6
    assert summary["mean_landing_angle_deg"] == 2.0
    assert summary["mean_landing_time_s"] == 4.2
