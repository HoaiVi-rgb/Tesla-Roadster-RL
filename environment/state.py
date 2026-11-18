"""
State representation, normalization, and validation for the Roadster MDP Environment.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Union
import numpy as np

from physics.constants import (
    REST_HEIGHT,
    MAX_EPISODE_TIME,
)

# Reference Normalization Constants matching Roadster_4_Voi_DQN specification:
# s = [x/10, h/15, vx/10, vy/10, sin(theta), cos(theta), omega/5, fuel/100, t/20]
SCALE_X: float = 10.0
SCALE_H: float = 15.0
SCALE_VX: float = 10.0
SCALE_VY: float = 10.0
SCALE_OMEGA: float = 5.0
SCALE_FUEL: float = 100.0
SCALE_TIME: float = MAX_EPISODE_TIME  # 20.0

STATE_DIM: int = 9


@dataclass(frozen=True)
class RawState:
    """Unnormalized physical telemetry state."""
    x: float
    h: float
    vx: float
    vy: float
    theta: float
    omega: float
    fuel: float
    time: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "x": self.x,
            "h": self.h,
            "vx": self.vx,
            "vy": self.vy,
            "theta": self.theta,
            "theta_deg": math.degrees(self.theta),
            "omega": self.omega,
            "fuel": self.fuel,
            "time": self.time,
        }


def raw_to_normalized(raw: RawState) -> np.ndarray:
    """
    Deterministic mapping from raw physical state to normalized 9D observation vector:
    s = [x/10, h/15, vx/10, vy/10, sin(theta), cos(theta), omega/5, fuel/100, t/20]
    """
    sin_theta = math.sin(raw.theta)
    cos_theta = math.cos(raw.theta)

    obs = np.array(
        [
            raw.x / SCALE_X,
            raw.h / SCALE_H,
            raw.vx / SCALE_VX,
            raw.vy / SCALE_VY,
            sin_theta,
            cos_theta,
            raw.omega / SCALE_OMEGA,
            raw.fuel / SCALE_FUEL,
            raw.time / SCALE_TIME,
        ],
        dtype=np.float32,
    )
    validate_state(obs)
    return obs


def normalized_to_raw(obs: np.ndarray) -> RawState:
    """
    Reconstruct raw state from normalized observation vector (theta derived from atan2(sin, cos)).
    """
    validate_state(obs)
    x = float(obs[0] * SCALE_X)
    h = float(obs[1] * SCALE_H)
    vx = float(obs[2] * SCALE_VX)
    vy = float(obs[3] * SCALE_VY)
    theta = float(math.atan2(float(obs[4]), float(obs[5])))
    omega = float(obs[6] * SCALE_OMEGA)
    fuel = float(obs[7] * SCALE_FUEL)
    time = float(obs[8] * SCALE_TIME)
    return RawState(x=x, h=h, vx=vx, vy=vy, theta=theta, omega=omega, fuel=fuel, time=time)


def validate_state(obs: Union[np.ndarray, Sequence[float]]) -> bool:
    """
    Strict validation of the observation vector:
    1. Must be a numpy array or convertible sequence.
    2. Dimension must be exactly 9.
    3. No NaN values allowed.
    4. No +Infinity or -Infinity values allowed.
    5. Trigonometric constraints: sin^2 + cos^2 ~ 1.0 (within tolerance).

    Raises ValueError if validation fails.
    """
    if not isinstance(obs, np.ndarray):
        try:
            obs = np.asarray(obs, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"State observation must be convertible to numpy array: {e}")

    if obs.shape != (STATE_DIM,):
        raise ValueError(f"State dimension must be exactly ({STATE_DIM},), got shape {obs.shape}")

    if np.isnan(obs).any():
        raise ValueError(f"State contains NaN values: {obs}")

    if np.isinf(obs).any():
        raise ValueError(f"State contains Infinite values: {obs}")

    # Check trig consistency
    sin_val = float(obs[4])
    cos_val = float(obs[5])
    trig_sq_sum = sin_val**2 + cos_val**2
    if abs(trig_sq_sum - 1.0) > 0.05:  # Tolerance for floating point precision
        raise ValueError(
            f"Trigonometric values sin(theta)={sin_val}, cos(theta)={cos_val} violate sin^2+cos^2=1 (got {trig_sq_sum})"
        )

    return True
