"""
Curriculum Learning Architecture for Tesla Roadster Landing.
Orchestrates progressive training across Level 1, Level 2, and Level 3
based on performance threshold validation, with full per-level telemetry logging.

ACADEMIC INTEGRITY & LEAKAGE PREVENTION:
Training environments use training seeds (e.g. 1..10000).
Test evaluation uses disjoint independent seeds (e.g. 20000..20099).
Curriculum transitions are strictly based on training episodes and never touch test seeds.
"""

from __future__ import annotations
import math
import os
import json
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


@dataclass
class LevelSpec:
    """Specification of a curriculum stage."""
    level: int
    name: str
    height_range: Tuple[float, float]
    angle_range_deg: Tuple[float, float]
    x_range: Tuple[float, float]
    min_episodes: int
    success_threshold: float  # Rolling success rate required to promote (0.0 to 1.0)


# Standard 3-stage curriculum specs matching project specifications
STANDARD_LEVELS: List[LevelSpec] = [
    LevelSpec(
        level=1,
        name="Level 1: Low Altitude, Upright Stance",
        height_range=(1.2, 3.0),
        angle_range_deg=(-3.0, 3.0),
        x_range=(-0.2, 0.2),
        min_episodes=150,
        success_threshold=0.80,
    ),
    LevelSpec(
        level=2,
        name="Level 2: Moderate Altitude & Tilt",
        height_range=(2.0, 5.0),
        angle_range_deg=(-8.0, 8.0),
        x_range=(-0.5, 0.5),
        min_episodes=200,
        success_threshold=0.80,
    ),
    LevelSpec(
        level=3,
        name="Level 3: Full Challenge Domain",
        height_range=(4.0, 9.0),
        angle_range_deg=(-15.0, 15.0),
        x_range=(-2.0, 2.0),
        min_episodes=300,
        success_threshold=0.85,
    ),
]


class LevelStatistics:
    """Per-level telemetry accumulator."""

    def __init__(self, level: int, name: str):
        self.level = level
        self.name = name
        self.episodes: int = 0
        self.safe_landings: int = 0
        self.crashes: int = 0
        self.timeouts: int = 0
        self.oobs: int = 0
        self.raw_rewards: List[float] = []
        self.landing_speeds: List[float] = []
        self.landing_vys: List[float] = []
        self.landing_vxs: List[float] = []
        self.landing_angles: List[float] = []
        self.landing_times: List[float] = []
        self.fuel_used: List[float] = []

    def record(self, info: Dict[str, Any], raw_reward: float) -> None:
        self.episodes += 1
        self.raw_rewards.append(float(raw_reward))
        self.fuel_used.append(float(info.get("fuel_used", 100.0 - info.get("fuel", 100.0))))

        is_safe = bool(info.get("safe_landing", False))
        is_crash = bool(info.get("crash", False))
        is_timeout = bool(info.get("timeout", False))
        is_oob = bool(info.get("out_of_bounds", False))

        if is_safe:
            self.safe_landings += 1
            self.landing_speeds.append(float(info.get("landing_speed", 0.0)))
            self.landing_vys.append(float(info.get("landing_speed_vy", 0.0)))
            self.landing_vxs.append(float(info.get("landing_speed_vx", 0.0)))
            self.landing_angles.append(abs(float(info.get("landing_angle_deg", 0.0))))
            self.landing_times.append(float(info.get("time", 0.0)))
        elif is_crash:
            self.crashes += 1
        elif is_timeout:
            self.timeouts += 1
        elif is_oob:
            self.oobs += 1

    @property
    def success_rate(self) -> float:
        return (self.safe_landings / self.episodes) if self.episodes > 0 else 0.0

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.raw_rewards)) if self.raw_rewards else 0.0

    @property
    def mean_landing_speed(self) -> float:
        return float(np.mean(self.landing_speeds)) if self.landing_speeds else 0.0

    @property
    def mean_landing_vy(self) -> float:
        return float(np.mean(self.landing_vys)) if self.landing_vys else 0.0

    @property
    def mean_landing_angle(self) -> float:
        return float(np.mean(self.landing_angles)) if self.landing_angles else 0.0

    @property
    def mean_landing_time(self) -> float:
        return float(np.mean(self.landing_times)) if self.landing_times else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "name": self.name,
            "episodes": self.episodes,
            "safe_landings": self.safe_landings,
            "success_rate_percent": round(self.success_rate * 100.0, 2),
            "crash_rate_percent": round((self.crashes / self.episodes) * 100.0, 2) if self.episodes else 0.0,
            "timeout_rate_percent": round((self.timeouts / self.episodes) * 100.0, 2) if self.episodes else 0.0,
            "oob_rate_percent": round((self.oobs / self.episodes) * 100.0, 2) if self.episodes else 0.0,
            "mean_reward": round(self.mean_reward, 3),
            "mean_landing_speed_mps": round(self.mean_landing_speed, 3),
            "mean_landing_vy_mps": round(self.mean_landing_vy, 3),
            "mean_landing_angle_deg": round(self.mean_landing_angle, 3),
            "mean_landing_time_s": round(self.mean_landing_time, 2),
            "mean_fuel_used_pct": round(float(np.mean(self.fuel_used)), 2) if self.fuel_used else 0.0,
        }


class RoadsterCurriculum:
    """
    Roadster Curriculum Controller.
    Can be enabled (progressive Level 1 -> 2 -> 3) or disabled (baseline: straight Level 3).
    """

    def __init__(
        self,
        enabled: bool = True,
        levels: Optional[List[LevelSpec]] = None,
        window_size: int = 50,
        baseline_level: int = 3,
    ):
        self.enabled = enabled
        self.levels = levels or STANDARD_LEVELS
        self.level_map = {spec.level: spec for spec in self.levels}
        self.window_size = window_size
        self.baseline_level = baseline_level

        # Active state
        self.current_level = self.levels[0].level if self.enabled else self.baseline_level
        self.rolling_outcomes = deque(maxlen=window_size)
        self.total_episodes: int = 0
        self.level_episodes: int = 0

        # Per-level telemetry loggers
        self.level_stats: Dict[int, LevelStatistics] = {
            spec.level: LevelStatistics(spec.level, spec.name) for spec in self.levels
        }
        self.transition_history: List[Dict[str, Any]] = []

    @property
    def current_spec(self) -> LevelSpec:
        return self.level_map.get(
            self.current_level,
            LevelSpec(
                level=self.current_level,
                name=f"Level {self.current_level}",
                height_range=(4.0, 9.0),
                angle_range_deg=(-15.0, 15.0),
                x_range=(-2.0, 2.0),
                min_episodes=100,
                success_threshold=0.85,
            ),
        )

    @property
    def rolling_success_rate(self) -> float:
        if not self.rolling_outcomes:
            return 0.0
        return sum(self.rolling_outcomes) / float(len(self.rolling_outcomes))

    def get_env_options(self) -> Dict[str, Any]:
        """Generate options dictionary for env.reset(options=...)."""
        return {"level": self.current_level}

    def record_episode(self, info: Dict[str, Any], raw_reward: float) -> Tuple[bool, Optional[int]]:
        """
        Record episode result, update per-level stats, and evaluate promotion criteria.

        Parameters
        ----------
        info : environment step info dictionary
        raw_reward : unshaped episode raw reward

        Returns
        -------
        (promoted, new_level):
            promoted is True if agent advanced to a new curriculum level.
        """
        self.total_episodes += 1
        self.level_episodes += 1

        is_safe = bool(info.get("safe_landing", False))
        self.rolling_outcomes.append(1.0 if is_safe else 0.0)

        # Log into per-level stats
        if self.current_level in self.level_stats:
            self.level_stats[self.current_level].record(info, raw_reward)

        # If curriculum is disabled (baseline mode), never promote
        if not self.enabled:
            return False, None

        # Check threshold promotion criteria
        spec = self.current_spec
        has_min_episodes = self.level_episodes >= spec.min_episodes
        has_full_window = len(self.rolling_outcomes) >= self.window_size
        meets_threshold = self.rolling_success_rate >= spec.success_threshold

        if has_min_episodes and has_full_window and meets_threshold:
            return self._promote()

        return False, None

    def _promote(self) -> Tuple[bool, Optional[int]]:
        """Advance to next level."""
        current_idx = [i for i, s in enumerate(self.levels) if s.level == self.current_level][0]
        if current_idx + 1 < len(self.levels):
            old_level = self.current_level
            new_spec = self.levels[current_idx + 1]
            self.current_level = new_spec.level

            transition = {
                "from_level": old_level,
                "to_level": self.current_level,
                "episodes_at_transition": self.level_episodes,
                "total_episodes": self.total_episodes,
                "final_rolling_success_rate": round(self.rolling_success_rate * 100.0, 2),
                "promotion_threshold": round(self.level_map[old_level].success_threshold * 100.0, 2),
            }
            self.transition_history.append(transition)

            # Reset rolling window and level counters for new level
            self.level_episodes = 0
            self.rolling_outcomes.clear()
            return True, self.current_level

        return False, None

    def get_summary(self) -> Dict[str, Any]:
        """Export comprehensive curriculum report dictionary."""
        return {
            "curriculum_enabled": self.enabled,
            "current_level": self.current_level,
            "total_episodes": self.total_episodes,
            "rolling_success_rate_percent": round(self.rolling_success_rate * 100.0, 2),
            "level_stats": {lvl: stats.to_dict() for lvl, stats in self.level_stats.items()},
            "transition_history": self.transition_history,
        }

    def save_summary_json(self, filepath: str) -> None:
        """Persist curriculum summary to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.get_summary(), f, indent=2)
