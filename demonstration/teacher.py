"""
Heuristic Feedback Teacher Controller for generating expert demonstrations.
WARNING: STRICTLY FOR OFFLINE DEMONSTRATION GENERATION ONLY.
MUST NEVER BE USED AS A POLICY REPLACEMENT DURING DQN INFERENCE OR EVALUATION.
"""

from __future__ import annotations
import math
from typing import Dict, Any, Optional
import numpy as np

from environment.action import encode_action, validate_action


class HeuristicTeacherController:
    """
    Expert Teacher Controller for the 2D Flying Roadster.

    Architecture:
    1. Lateral Guidance: Proportional horizontal velocity targeting x -> 0 using V3 and V4.
    2. Vertical Braking: Altitude-dependent velocity profile ensuring smooth touchdown (|vy| < 0.6 m/s).
    3. Attitude Stabilization: Predictive phase-plane rate damping aligning theta -> 0, omega -> 0.
    4. Terminal Deadband Lock: Avoids thruster chattering during final touchdown phase.

    This controller is decoupled from RL agents and serves exclusively to bootstrap
    demonstration datasets for imitation learning and DQfD pre-training.
    """

    def __init__(
        self,
        kp_x: float = 1.0,
        max_vx: float = 1.0,
        deadband_angle_deg: float = 3.5,
        deadband_omega: float = 0.25,
    ):
        self.kp_x = float(kp_x)
        self.max_vx = float(max_vx)
        self.deadband_angle = math.radians(deadband_angle_deg)
        self.deadband_omega = float(deadband_omega)

    def select_action(
        self,
        obs: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Select discrete action in [0, 15] based on current physical telemetry.
        """
        if info is None:
            raise ValueError("Teacher controller requires physical state telemetry in info dict")

        x = float(info["x"])
        y = float(info["h"])
        vx = float(info["vx"])
        vy = float(info["vy"])
        theta = float(info["theta"])
        omega = float(info["omega"])

        # ----------------------------------------------------------------------
        # 1. Lateral Control (V3: push right / V4: push left)
        # Thrusters V3 and V4 pass through the center of mass -> zero torque!
        # ----------------------------------------------------------------------
        target_vx = -self.kp_x * x
        target_vx = max(-self.max_vx, min(self.max_vx, target_vx))
        vx_err = vx - target_vx

        u3 = 1 if vx_err < -0.1 else 0
        u4 = 1 if vx_err > 0.1 else 0

        # ----------------------------------------------------------------------
        # 2. Vertical Descent & Flare Profile
        # ----------------------------------------------------------------------
        h_above = max(0.0, y - 0.72)

        # Smooth velocity profile tapering to gentle touchdown (-0.2 to -0.4 m/s)
        if h_above > 4.5:
            target_vy = -3.0
        elif h_above > 2.5:
            target_vy = -2.0
        elif h_above > 1.2:
            target_vy = -1.0
        elif h_above > 0.4:
            target_vy = -0.5
        else:
            target_vy = -0.2

        need_lift = (vy < target_vy)

        # Terminal cushion: prevent sub-step free fall into the ground
        if h_above < 0.35 and vy < -0.1:
            need_lift = True
        elif h_above < 0.8 and vy < -0.5:
            need_lift = True

        # ----------------------------------------------------------------------
        # 3. Attitude & Angular Rate Damping Control
        # ----------------------------------------------------------------------
        u1 = 0
        u2 = 0

        # In low-altitude terminal phase, lock attitude if already within tolerances
        # to prevent 1-step discrete torque quantum (+-0.8 rad/s) from exciting oscillations
        if h_above < 1.5 and abs(theta) < self.deadband_angle and abs(omega) < self.deadband_omega:
            if need_lift:
                u1 = 1
                u2 = 1
            else:
                u1 = 0
                u2 = 0
        else:
            theta_next = theta + omega * 0.1
            if omega > 0.25:
                u1 = 1  # CW torque (-16,200 Nm) to brake CCW spin
                u2 = 0
            elif omega < -0.25:
                u2 = 1  # CCW torque (+16,200 Nm) to brake CW spin
                u1 = 0
            elif theta_next > math.radians(2.5):
                u1 = 1  # Pitch nose down
                u2 = 0
            elif theta_next < -math.radians(2.5):
                u2 = 1  # Pitch nose up
                u1 = 0
            else:
                # Balanced attitude
                if need_lift:
                    u1 = 1
                    u2 = 1

        action = encode_action(u1, u2, u3, u4)
        return validate_action(action)
