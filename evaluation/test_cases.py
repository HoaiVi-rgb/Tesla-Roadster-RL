"""
Evaluation Test Cases Definition for the Roadster 2D Flying Car.
Strictly decoupled from training and demonstration datasets.
Provides:
  1. Curated Deterministic Benchmark Suite (60 structured corner/stress test cases across 6 categories)
  2. Independent Stochastic Test Set (100 novel seeds 30000..30099, Level 3 Full Domain)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass(frozen=True)
class TestCase:
    """Explicit initial condition for deterministic stress testing."""
    __test__ = False  # Prevent pytest from treating this as a test class
    test_id: str
    category: str
    description: str
    initial_x: float            # Horizontal offset (m)
    initial_y: float            # Altitude (m)
    initial_vx: float = 0.0     # Horizontal velocity (m/s)
    initial_vy: float = 0.0     # Vertical velocity (m/s)
    initial_theta_deg: float = 0.0  # Pitch angle (deg)
    initial_omega: float = 0.0  # Angular velocity (rad/s)
    initial_fuel: float = 100.0 # Propellant (%)

    @property
    def initial_theta_rad(self) -> float:
        return math.radians(self.initial_theta_deg)

    def to_env_options(self) -> Dict[str, Any]:
        return {
            "initial_x": self.initial_x,
            "initial_y": self.initial_y,
            "initial_vx": self.initial_vx,
            "initial_vy": self.initial_vy,
            "initial_theta": self.initial_theta_rad,
            "initial_omega": self.initial_omega,
            "initial_fuel": self.initial_fuel,
        }


def build_curated_test_cases() -> List[TestCase]:
    """
    Construct 60 rigorous, categorized corner cases to evaluate
    dynamic stability, flare authority, attitude recovery, and target accuracy.
    """
    cases: List[TestCase] = []

    # =========================================================================
    # CATEGORY 1: NOMINAL APPROACH (10 Cases)
    # Moderate altitudes, slight offset, small tilt. Standard landing scenarios.
    # =========================================================================
    nominal_params = [
        ("NOM_01", 0.0, 5.0, 0.0, -0.2, 0.0, 0.0, "Dead center, moderate descent"),
        ("NOM_02", 0.3, 6.0, 0.1, -0.3, 2.0, 0.01, "Slight right offset, minor tilt"),
        ("NOM_03", -0.3, 5.5, -0.1, -0.2, -2.0, -0.01, "Slight left offset, minor tilt"),
        ("NOM_04", 0.5, 4.5, 0.2, -0.4, 3.0, 0.02, "Right bias, small rightward drift"),
        ("NOM_05", -0.5, 6.5, -0.2, -0.3, -3.0, -0.02, "Left bias, small leftward drift"),
        ("NOM_06", 0.1, 7.0, 0.0, -0.5, 1.0, 0.0, "High nominal, center line"),
        ("NOM_07", -0.2, 4.0, 0.0, -0.1, -1.5, 0.0, "Low nominal descent"),
        ("NOM_08", 0.4, 5.8, -0.1, -0.3, -2.0, 0.01, "Opposing offset and tilt"),
        ("NOM_09", -0.4, 6.2, 0.1, -0.4, 2.5, -0.01, "Opposing left offset and right tilt"),
        ("NOM_10", 0.0, 4.8, 0.0, -0.3, 0.5, 0.0, "Pure vertical drop at 4.8m"),
    ]
    for tid, x, y, vx, vy, th, om, desc in nominal_params:
        cases.append(TestCase(tid, "Nominal Approach", desc, x, y, vx, vy, th, om))

    # =========================================================================
    # CATEGORY 2: HIGH ALTITUDE DROP (10 Cases)
    # Altitudes 8.0 - 9.5m, requiring extended descent management and fuel pacing.
    # =========================================================================
    high_alt_params = [
        ("HALT_01", 0.0, 8.5, 0.0, -0.5, 0.0, 0.0, "High altitude center drop (8.5m)"),
        ("HALT_02", 0.8, 9.0, 0.2, -0.6, 4.0, 0.02, "High drop (9.0m) with right drift"),
        ("HALT_03", -0.8, 9.0, -0.2, -0.6, -4.0, -0.02, "High drop (9.0m) with left drift"),
        ("HALT_04", 1.2, 8.2, 0.3, -0.8, 5.0, 0.03, "High drop near right edge of Level 3"),
        ("HALT_05", -1.2, 8.2, -0.3, -0.8, -5.0, -0.03, "High drop near left edge of Level 3"),
        ("HALT_06", 0.0, 9.5, 0.0, -1.0, 0.0, 0.0, "Max ceiling drop (9.5m) high descent rate"),
        ("HALT_07", 0.5, 8.8, -0.2, -0.7, -3.0, -0.02, "High altitude cross-drift"),
        ("HALT_08", -0.5, 8.8, 0.2, -0.7, 3.0, 0.02, "High altitude cross-drift reverse"),
        ("HALT_09", 1.5, 9.2, 0.1, -0.5, 2.0, 0.01, "Extreme right start at 9.2m"),
        ("HALT_10", -1.5, 9.2, -0.1, -0.5, -2.0, -0.01, "Extreme left start at 9.2m"),
    ]
    for tid, x, y, vx, vy, th, om, desc in high_alt_params:
        cases.append(TestCase(tid, "High Altitude Drop", desc, x, y, vx, vy, th, om))

    # =========================================================================
    # CATEGORY 3: SEVERE ATTITUDE TILT (10 Cases)
    # Severe initial angles (+-10 to +-15 deg) and angular rates (+-0.1 to +-0.15 rad/s).
    # =========================================================================
    tilt_params = [
        ("TILT_01", 0.0, 6.0, 0.0, -0.3, 12.0, 0.10, "Right tilt +12 deg with clockwise spin"),
        ("TILT_02", 0.0, 6.0, 0.0, -0.3, -12.0, -0.10, "Left tilt -12 deg with CCW spin"),
        ("TILT_03", 0.5, 7.0, 0.1, -0.4, 15.0, 0.05, "Max domain tilt +15 deg"),
        ("TILT_04", -0.5, 7.0, -0.1, -0.4, -15.0, -0.05, "Max domain tilt -15 deg"),
        ("TILT_05", 0.2, 5.0, 0.0, -0.2, 14.0, 0.15, "Low altitude +14 deg fast spin (Critical)"),
        ("TILT_06", -0.2, 5.0, 0.0, -0.2, -14.0, -0.15, "Low altitude -14 deg fast spin (Critical)"),
        ("TILT_07", 1.0, 6.5, 0.2, -0.5, 10.0, -0.10, "Right tilt with counter-torque spin"),
        ("TILT_08", -1.0, 6.5, -0.2, -0.5, -10.0, 0.10, "Left tilt with counter-torque spin"),
        ("TILT_09", 0.0, 8.0, 0.0, -0.6, 13.5, 0.08, "High altitude +13.5 deg restoration"),
        ("TILT_10", 0.0, 8.0, 0.0, -0.6, -13.5, -0.08, "High altitude -13.5 deg restoration"),
    ]
    for tid, x, y, vx, vy, th, om, desc in tilt_params:
        cases.append(TestCase(tid, "Severe Attitude Tilt", desc, x, y, vx, vy, th, om))

    # =========================================================================
    # CATEGORY 4: HIGH HORIZONTAL DRIFT (10 Cases)
    # High horizontal velocity (|vx| in [0.4, 0.8] m/s), testing lateral thrusters V3/V4.
    # =========================================================================
    drift_params = [
        ("DRFT_01", -1.8, 6.0, 0.6, -0.3, 0.0, 0.0, "Left boundary drifting right at 0.6 m/s"),
        ("DRFT_02", 1.8, 6.0, -0.6, -0.3, 0.0, 0.0, "Right boundary drifting left at -0.6 m/s"),
        ("DRFT_03", 0.0, 7.0, 0.7, -0.4, 4.0, 0.02, "Center with strong right drift 0.7 m/s"),
        ("DRFT_04", 0.0, 7.0, -0.7, -0.4, -4.0, -0.02, "Center with strong left drift -0.7 m/s"),
        ("DRFT_05", -1.5, 5.5, 0.5, -0.5, -5.0, 0.0, "Compounded left offset and right drift"),
        ("DRFT_06", 1.5, 5.5, -0.5, -0.5, 5.0, 0.0, "Compounded right offset and left drift"),
        ("DRFT_07", 1.0, 8.0, 0.8, -0.6, 2.0, 0.01, "High altitude runaway drift right 0.8 m/s"),
        ("DRFT_08", -1.0, 8.0, -0.8, -0.6, -2.0, -0.01, "High altitude runaway drift left -0.8 m/s"),
        ("DRFT_09", -0.5, 4.5, 0.5, -0.2, 3.0, 0.02, "Low altitude fast rightward drift"),
        ("DRFT_10", 0.5, 4.5, -0.5, -0.2, -3.0, -0.02, "Low altitude fast leftward drift"),
    ]
    for tid, x, y, vx, vy, th, om, desc in drift_params:
        cases.append(TestCase(tid, "High Horizontal Drift", desc, x, y, vx, vy, th, om))

    # =========================================================================
    # CATEGORY 5: LOW PROPELLANT / RAPID DESCENT (10 Cases)
    # Reduced starting fuel (30%-50%) and high initial vertical sink rate (vy <= -0.8 m/s).
    # =========================================================================
    fuel_sink_params = [
        ("FAST_01", 0.0, 5.0, 0.0, -1.0, 0.0, 0.0, 40.0, "Fuel 40%, plunge vy = -1.0 m/s"),
        ("FAST_02", 0.4, 6.0, 0.2, -1.2, 3.0, 0.02, 50.0, "Fuel 50%, plunge vy = -1.2 m/s right"),
        ("FAST_03", -0.4, 6.0, -0.2, -1.2, -3.0, -0.02, 50.0, "Fuel 50%, plunge vy = -1.2 m/s left"),
        ("FAST_04", 0.0, 7.5, 0.0, -1.4, 0.0, 0.0, 45.0, "Fuel 45%, severe plunge vy = -1.4 m/s"),
        ("FAST_05", 0.6, 4.5, 0.1, -0.9, 2.0, 0.01, 30.0, "Low fuel 30%, sink rate -0.9 m/s"),
        ("FAST_06", -0.6, 4.5, -0.1, -0.9, -2.0, -0.01, 30.0, "Low fuel 30%, sink rate -0.9 m/s left"),
        ("FAST_07", 0.2, 8.0, 0.0, -1.3, 4.0, 0.03, 35.0, "Low fuel 35%, high plunge 8.0m"),
        ("FAST_08", -0.2, 8.0, 0.0, -1.3, -4.0, -0.03, 35.0, "Low fuel 35%, high plunge 8.0m left"),
        ("FAST_09", 0.0, 4.0, 0.0, -0.8, 1.0, 0.0, 25.0, "Critical fuel 25%, 4m drop"),
        ("FAST_10", 0.3, 5.2, -0.1, -1.1, -2.0, 0.0, 40.0, "Fuel 40%, sink -1.1 m/s opposing tilt"),
    ]
    for tid, x, y, vx, vy, th, om, f_pct, desc in fuel_sink_params:
        cases.append(TestCase(tid, "Low Fuel / Rapid Descent", desc, x, y, vx, vy, th, om, f_pct))

    # =========================================================================
    # CATEGORY 6: COMPOUND EXTREME EDGE CASES (10 Cases)
    # Simultaneous extreme displacement (x=+-2.0m), max angle (+-15 deg), and drift.
    # =========================================================================
    compound_params = [
        ("EDGE_01", 2.0, 9.0, -0.5, -0.8, 15.0, 0.12, "Max envelope: x=2m, y=9m, th=15 deg, om=0.12"),
        ("EDGE_02", -2.0, 9.0, 0.5, -0.8, -15.0, -0.12, "Max envelope: x=-2m, y=9m, th=-15 deg, om=-0.12"),
        ("EDGE_03", 2.0, 5.0, 0.4, -0.5, 14.0, 0.10, "Low edge: x=2m, drifting outward with right tilt"),
        ("EDGE_04", -2.0, 5.0, -0.4, -0.5, -14.0, -0.10, "Low edge: x=-2m, drifting outward with left tilt"),
        ("EDGE_05", 1.8, 7.5, -0.6, -1.0, -12.0, -0.08, "Opposing angular velocity and inward drift"),
        ("EDGE_06", -1.8, 7.5, 0.6, -1.0, 12.0, 0.08, "Opposing angular velocity and inward drift left"),
        ("EDGE_07", 0.0, 9.2, 0.6, -1.2, 15.0, 0.15, "Center high plunge with max tilt and spin"),
        ("EDGE_08", 0.0, 9.2, -0.6, -1.2, -15.0, -0.15, "Center high plunge with max left tilt/spin"),
        ("EDGE_09", 1.9, 4.2, -0.3, -0.4, 8.0, 0.05, "Low corner approach right"),
        ("EDGE_10", -1.9, 4.2, 0.3, -0.4, -8.0, -0.05, "Low corner approach left"),
    ]
    for tid, x, y, vx, vy, th, om, desc in compound_params:
        cases.append(TestCase(tid, "Compound Extreme", desc, x, y, vx, vy, th, om))

    return cases


def get_stochastic_unseen_seeds(count: int = 100, seed_start: int = 30000) -> List[int]:
    """
    Return 100 random seeds (30000..30099) strictly disjoint from:
      - Training seeds (1..1000)
      - Previous benchmark seeds (20000..20049)
    """
    return list(range(seed_start, seed_start + count))
