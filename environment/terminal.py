"""
Terminal condition detection and classification for the Roadster MDP Environment.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from physics.constants import (
    MAX_EPISODE_TIME,
)
from physics.collision import CollisionInfo


class TerminationType(str, Enum):
    IN_FLIGHT = "IN_FLIGHT"
    SAFE_LANDING = "SAFE_LANDING"
    CRASH = "CRASH"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class TerminalResult:
    """Structured terminal evaluation."""
    done: bool
    terminated: bool
    truncated: bool
    termination_reason: str
    is_safe_landing: bool
    is_crash: bool
    is_out_of_bounds: bool
    is_timeout: bool


class TerminalDetector:
    """
    Evaluates whether an episode has reached a terminal state:
    1. Ground contact -> SAFE_LANDING or CRASH (terminated=True)
    2. Boundary violation -> OUT_OF_BOUNDS (terminated=True)
    3. Maximum time horizon (20.0s) -> TIMEOUT (truncated=True)
    """

    def __init__(self, max_time: float = MAX_EPISODE_TIME):
        self.max_time = float(max_time)

    def evaluate(self, col_info: CollisionInfo, current_time: float) -> TerminalResult:
        """
        Evaluate terminal condition from collision telemetry and elapsed time.
        """
        is_safe_landing = False
        is_crash = False
        is_oob = False
        is_timeout = False
        terminated = False
        truncated = False
        reason = TerminationType.IN_FLIGHT.value

        # Check out of bounds
        if col_info.is_out_of_bounds:
            is_oob = True
            terminated = True
            reason = TerminationType.OUT_OF_BOUNDS.value

        # Check ground contact
        elif col_info.is_contact:
            if col_info.is_safe_landing:
                is_safe_landing = True
                terminated = True
                reason = TerminationType.SAFE_LANDING.value
            else:
                is_crash = True
                terminated = True
                reason = col_info.contact_reason  # includes specific violation details

        # Check timeout (with floating point epsilon tolerance)
        elif current_time >= self.max_time - 1e-5:
            is_timeout = True
            truncated = True
            reason = TerminationType.TIMEOUT.value

        done = terminated or truncated

        return TerminalResult(
            done=done,
            terminated=terminated,
            truncated=truncated,
            termination_reason=reason,
            is_safe_landing=is_safe_landing,
            is_crash=is_crash,
            is_out_of_bounds=is_oob,
            is_timeout=is_timeout,
        )
