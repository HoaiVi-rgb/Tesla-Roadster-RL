"""
Collision Detection and Landing Criteria Validation for the 2D Flying Roadster.
Evaluates ground contact, safe landing tolerances, target zone accuracy, and world bounds.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np

from physics.constants import (
    WORLD_X_MIN,
    WORLD_X_MAX,
    WORLD_Y_MIN,
    WORLD_Y_MAX,
    TARGET_ZONE_X_MIN,
    TARGET_ZONE_X_MAX,
    MAX_SAFE_LANDING_VY,
    MAX_SAFE_LANDING_VX,
    MAX_SAFE_LANDING_ANGLE_RAD,
    MAX_SAFE_LANDING_OMEGA,
    CONTACT_POINTS_LOCAL,
    WHEEL_REAR_LOCAL,
    WHEEL_FRONT_LOCAL,
)
from physics.rigid_body import RigidBody2D


def normalize_angle_symmetric(angle_rad: float) -> float:
    """Normalize any angle in radians to [-pi, +pi]."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class CollisionInfo:
    """Structured collision and boundary telemetry."""
    is_contact: bool
    is_safe_landing: bool
    is_crash: bool
    is_out_of_bounds: bool
    is_in_target_zone: bool
    lowest_y: float
    penetration_depth: float
    contact_point_world: Tuple[float, float]
    contact_reason: str
    impact_vy: float
    impact_vx: float
    impact_angle_rad: float
    impact_omega: float
    rear_wheel_world: Tuple[float, float]
    front_wheel_world: Tuple[float, float]


class CollisionDetector:
    """
    Evaluates collisions between the vehicle's geometry and the environment:
    1. Ground plane at y = 0.
    2. Spatial bounds [-20, 20]m x [0, 25]m.
    3. Safe landing criteria check (|vy| < 1, |vx| < 0.5, |theta| < 5 deg, |omega| < 0.2).
    4. Target zone check (both wheels inside [-1.5, +1.5]m).

    STRICT: Evaluates physical state directly without modifying position or velocity.
    """

    def __init__(self, contact_points_local: Optional[List[Tuple[float, float]]] = None):
        self.contact_points_local = (
            contact_points_local if contact_points_local is not None else CONTACT_POINTS_LOCAL
        )

    def check_collision(self, body: RigidBody2D) -> CollisionInfo:
        """
        Evaluate full collision status for the given rigid body.

        Parameters
        ----------
        body : RigidBody2D

        Returns
        -------
        CollisionInfo
        """
        # 1. Transform all local contact vertices to world coordinates
        contact_points_world = [
            body.body_to_world_point(pt) for pt in self.contact_points_local
        ]

        # Find the lowest vertex
        lowest_pt = min(contact_points_world, key=lambda p: p[1])
        lowest_y = float(lowest_pt[1])
        penetration = max(0.0, -lowest_y)

        # Transform wheel positions to world coordinates
        rear_wheel_world_pt = body.body_to_world_point(WHEEL_REAR_LOCAL)
        front_wheel_world_pt = body.body_to_world_point(WHEEL_FRONT_LOCAL)
        rear_wheel_world = (float(rear_wheel_world_pt[0]), float(rear_wheel_world_pt[1]))
        front_wheel_world = (float(front_wheel_world_pt[0]), float(front_wheel_world_pt[1]))

        # Check target zone: both wheels must land within [TARGET_ZONE_X_MIN, TARGET_ZONE_X_MAX]
        in_target_zone = (
            TARGET_ZONE_X_MIN <= rear_wheel_world[0] <= TARGET_ZONE_X_MAX
            and TARGET_ZONE_X_MIN <= front_wheel_world[0] <= TARGET_ZONE_X_MAX
        )

        # 2. Check out of bounds (center of mass outside spatial box)
        is_oob = (
            body.x < WORLD_X_MIN
            or body.x > WORLD_X_MAX
            or body.y > WORLD_Y_MAX
        )

        # 3. Check ground contact (lowest vertex hits y <= 0.0)
        is_contact = (lowest_y <= 0.0)

        is_safe_landing = False
        is_crash = False
        contact_reason = "AIRBORNE"

        if is_oob:
            contact_reason = "OUT_OF_BOUNDS"
        elif is_contact:
            # Normalize angle to [-pi, pi] to check deviation from horizontal
            norm_angle = abs(normalize_angle_symmetric(body.theta))
            abs_vy = abs(body.vy)
            abs_vx = abs(body.vx)
            abs_omega = abs(body.omega)

            # Strict safe landing condition: ALL 4 criteria must be satisfied
            vy_ok = abs_vy <= MAX_SAFE_LANDING_VY
            vx_ok = abs_vx <= MAX_SAFE_LANDING_VX
            angle_ok = norm_angle <= MAX_SAFE_LANDING_ANGLE_RAD
            omega_ok = abs_omega <= MAX_SAFE_LANDING_OMEGA

            if vy_ok and vx_ok and angle_ok and omega_ok:
                is_safe_landing = True
                is_crash = False
                contact_reason = "SAFE_LANDING"
            else:
                is_safe_landing = False
                is_crash = True
                failed_reasons = []
                if not vy_ok:
                    failed_reasons.append(f"vy={abs_vy:.2f}>{MAX_SAFE_LANDING_VY}")
                if not vx_ok:
                    failed_reasons.append(f"vx={abs_vx:.2f}>{MAX_SAFE_LANDING_VX}")
                if not angle_ok:
                    failed_reasons.append(f"angle={math.degrees(norm_angle):.1f}>{math.degrees(MAX_SAFE_LANDING_ANGLE_RAD):.1f}deg")
                if not omega_ok:
                    failed_reasons.append(f"omega={abs_omega:.2f}>{MAX_SAFE_LANDING_OMEGA}")
                contact_reason = f"CRASH: {', '.join(failed_reasons)}"

        return CollisionInfo(
            is_contact=is_contact,
            is_safe_landing=is_safe_landing,
            is_crash=is_crash,
            is_out_of_bounds=is_oob,
            is_in_target_zone=in_target_zone,
            lowest_y=lowest_y,
            penetration_depth=penetration,
            contact_point_world=(float(lowest_pt[0]), float(lowest_pt[1])),
            contact_reason=contact_reason,
            impact_vy=float(body.vy),
            impact_vx=float(body.vx),
            impact_angle_rad=float(body.theta),
            impact_omega=float(body.omega),
            rear_wheel_world=rear_wheel_world,
            front_wheel_world=front_wheel_world,
        )
