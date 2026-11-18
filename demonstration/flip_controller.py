"""
Physics-accurate Acrobatic Flip Controller with Newton Stopping Distance and Controlled Descent.
"""

import math
from typing import Dict
from environment.action import encode_action

ALPHA_BRAKE = 16200.0 / 2017.0
TWO_PI = 2.0 * math.pi


def wrap_angle_rad(angle: float) -> float:
    return (angle + math.pi) % TWO_PI - math.pi


class PhysicsFlipController:
    def __init__(self):
        self.step_idx = 0
        self.flip_completed = False

    def reset(self):
        self.step_idx = 0
        self.flip_completed = False

    def compute_action(self, raw: Dict[str, float]) -> int:
        self.step_idx += 1
        h = raw["y"]
        vy = raw["vy"]
        vx = raw["vx"]
        theta = raw["theta"]
        omega = raw["omega"]
        accum_rot = abs(raw.get("accumulated_rotation", theta))
        wrapped_th = wrap_angle_rad(theta)

        # Trigger completion when near 360 and low spin
        if accum_rot >= 5.8 and abs(wrapped_th) < 0.15 and abs(omega) < 0.6:
            self.flip_completed = True

        if not self.flip_completed:
            # 1. Loft phase (first 8 steps = 0.8s)
            if self.step_idx <= 8:
                return encode_action(1, 1, 0, 0)

            # 2. Kick spin (V2 only: CCW torque)
            if accum_rot < 1.6:
                return encode_action(0, 1, 0, 0)

            # 3. Dynamic stopping distance check
            if omega > 0.0:
                stop_dist = (omega * omega) / (2.0 * ALPHA_BRAKE)
                rem_dist = TWO_PI - accum_rot
                if rem_dist <= stop_dist + 0.05:
                    return encode_action(1, 0, 0, 0)  # V1 counter-torque!

            # Spin phase under momentum
            if accum_rot < 3.8 and omega < 5.5:
                return encode_action(0, 1, 0, 0)

            return encode_action(0, 0, 0, 0)

        # Flip completed! Both thrusters brake down to target vy, then touch down
        u3, u4 = 0, 0
        if vx > 0.15:
            u4 = 1  # push left
        elif vx < -0.15:
            u3 = 1  # push right

        # Flare vertical descent control:
        # Near ground (h < 1.5m), target vy = -0.45 m/s. High up, target vy = -1.5 m/s.
        target_vy = -0.45 if h < 1.5 else -1.5
        if vy < target_vy:
            return encode_action(1, 1, u3, u4)
        else:
            return encode_action(0, 0, u3, u4)
