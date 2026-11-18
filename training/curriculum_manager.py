"""
Curriculum Learning Manager for 2D Roadster Landing.
Orchestrates progressive task difficulty across Level 1, Level 2, Level 3, and Level Flip.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class CurriculumLevelSpec:
    level: int
    name: str
    altitude_range: tuple
    angle_range_deg: tuple
    x_range: tuple
    min_episodes: int
    success_threshold: float  # Required rolling success rate to promote


DEFAULT_LEVELS = [
    CurriculumLevelSpec(
        level=1,
        name="Level 1: Low Altitude, Upright Stance",
        altitude_range=(1.2, 3.0),
        angle_range_deg=(-3.0, 3.0),
        x_range=(-0.2, 0.2),
        min_episodes=200,
        success_threshold=0.80,
    ),
    CurriculumLevelSpec(
        level=2,
        name="Level 2: Moderate Altitude & Tilt",
        altitude_range=(2.0, 5.0),
        angle_range_deg=(-8.0, 8.0),
        x_range=(-0.5, 0.5),
        min_episodes=300,
        success_threshold=0.80,
    ),
    CurriculumLevelSpec(
        level=3,
        name="Level 3: Full Challenge Domain",
        altitude_range=(4.0, 9.0),
        angle_range_deg=(-15.0, 15.0),
        x_range=(-2.0, 2.0),
        min_episodes=500,
        success_threshold=0.85,
    ),
    CurriculumLevelSpec(
        level=4,
        name="Level 4: Flip / Acrobatic Inverted Stance",
        altitude_range=(7.0, 9.5),
        angle_range_deg=(90.0, 180.0),
        x_range=(-1.0, 1.0),
        min_episodes=200,
        success_threshold=0.70,
    ),
]


class CurriculumManager:
    """
    Manages curriculum stages and adaptive promotion based on agent performance.
    """

    def __init__(
        self,
        starting_level: int = 1,
        levels: Optional[List[CurriculumLevelSpec]] = None,
        window_size: int = 50,
        auto_advance: bool = True,
    ):
        self.levels = levels or DEFAULT_LEVELS
        self.level_map = {spec.level: spec for spec in self.levels}
        if starting_level not in self.level_map:
            raise ValueError(f"Starting level {starting_level} not found in configured levels")

        self.current_level = starting_level
        self.window_size = window_size
        self.auto_advance = auto_advance

        self.level_episodes: int = 0
        self.total_episodes: int = 0
        self.recent_outcomes = deque(maxlen=window_size)
        self.level_history: List[Dict[str, Any]] = []

    @property
    def current_spec(self) -> CurriculumLevelSpec:
        return self.level_map[self.current_level]

    @property
    def success_rate(self) -> float:
        if not self.recent_outcomes:
            return 0.0
        return sum(self.recent_outcomes) / float(len(self.recent_outcomes))

    def get_reset_options(self) -> Dict[str, Any]:
        """Returns the dictionary to pass to env.reset(options=...)."""
        return {"level": self.current_level}

    def record_episode(self, success: bool, raw_reward: Optional[float] = None) -> bool:
        """
        Record the outcome of an episode and check for promotion.

        Parameters
        ----------
        success : bool indicating if episode resulted in safe landing
        raw_reward : optional scalar raw reward

        Returns
        -------
        bool : True if the agent was just promoted to a new level
        """
        self.level_episodes += 1
        self.total_episodes += 1
        self.recent_outcomes.append(1.0 if success else 0.0)

        if not self.auto_advance:
            return False

        spec = self.current_spec
        # Check promotion conditions
        if (
            self.level_episodes >= spec.min_episodes
            and len(self.recent_outcomes) >= self.window_size
            and self.success_rate >= spec.success_threshold
        ):
            return self.promote()

        return False

    def promote(self) -> bool:
        """Promote to the next curriculum level if available."""
        current_idx = [i for i, spec in enumerate(self.levels) if spec.level == self.current_level][0]
        if current_idx + 1 < len(self.levels):
            old_level = self.current_level
            new_spec = self.levels[current_idx + 1]
            self.current_level = new_spec.level

            self.level_history.append({
                "level": old_level,
                "episodes_spent": self.level_episodes,
                "final_success_rate": self.success_rate,
            })

            self.level_episodes = 0
            self.recent_outcomes.clear()
            return True
        return False

    def set_level(self, level: int) -> None:
        """Manually override current curriculum level."""
        if level not in self.level_map:
            raise ValueError(f"Invalid level {level}. Available: {list(self.level_map.keys())}")
        self.current_level = level
        self.level_episodes = 0
        self.recent_outcomes.clear()

    def get_status(self) -> Dict[str, Any]:
        spec = self.current_spec
        return {
            "current_level": self.current_level,
            "level_name": spec.name,
            "level_episodes": self.level_episodes,
            "min_episodes": spec.min_episodes,
            "total_episodes": self.total_episodes,
            "rolling_success_rate": self.success_rate,
            "promotion_threshold": spec.success_threshold,
            "history": list(self.level_history),
        }
