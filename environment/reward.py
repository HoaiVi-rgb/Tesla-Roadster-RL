"""
Reward Engine for the Roadster MDP Environment.
Implements raw physical reward and Ng et al. (1999) Potential-Based Reward Shaping (PBRS).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from physics.constants import (
    REST_HEIGHT,
)
from physics.collision import CollisionInfo
from environment.terminal import TerminalResult


@dataclass(frozen=True)
class RewardBreakdown:
    """Detailed components of the step reward."""
    r_learn: float          # Scaled learning reward used for RL updates: 0.01 * (r_base + F)
    r_base: float           # Raw unscaled, unshaped physical reward
    r_terminal: float       # Safe landing (+200) / Crash (-100) / Timeout (-500) / OOB (-250)
    r_target_zone: float    # Green zone landing bonus (+30)
    r_impact_vel: float     # Ground velocity penalty: -2*(vx^2 + vy^2)
    r_impact_angle: float   # Ground angle penalty: -10*(1 - cos(theta))
    r_gas: float            # Fuel cost: -0.2 * delta_fuel
    r_time: float           # Time cost: -1.0 * delta_time
    r_shaping: float        # Potential shaping: gamma * phi(s') - phi(s)
    phi_s: float            # Potential of previous state
    phi_s_next: float       # Potential of next state


class RewardEngine:
    """
    Reward Calculator following Roadster_4_Voi_DQN specification.

    Formula:
    r_goc = thuong_ket_thuc + thuong_vung - phat_vuot_bien/het_gio
            - w_dap*(vx^2 + vy^2) - w_goc*(1 - cos(theta_cham))
            - w_khi*delta_khi - w_thoi_gian*delta_t

    F = gamma * Phi(s') - Phi(s)
    Phi = -10 * sqrt(x^2 + max(0, h - 0.72)^2) - 4 * sqrt(vx^2 + vy^2)
    (Phi = 0 at any terminal state)

    r_hoc = 0.01 * (r_goc + F)
    """

    def __init__(
        self,
        r_safe_landing: float = 200.0,
        r_target_zone: float = 30.0,
        w_crash: float = 100.0,
        w_timeout: float = 500.0,
        w_oob: float = 250.0,
        w_impact_vel: float = 2.0,
        w_impact_angle: float = 10.0,
        w_gas: float = 0.2,
        w_time: float = 1.0,
        gamma: float = 0.995,
        use_shaping: bool = True,
        reward_scale: float = 0.01,
    ):
        self.r_safe_landing = float(r_safe_landing)
        self.r_target_zone = float(r_target_zone)
        # Crash penalty is always 1/2 of safe landing bonus by specification
        self.w_crash = float(w_crash)
        self.w_timeout = float(w_timeout)
        self.w_oob = float(w_oob)
        self.w_impact_vel = float(w_impact_vel)
        self.w_impact_angle = float(w_impact_angle)
        self.w_gas = float(w_gas)
        self.w_time = float(w_time)
        self.gamma = float(gamma)
        self.use_shaping = bool(use_shaping)
        self.reward_scale = float(reward_scale)

    def compute_potential(
        self,
        x: float,
        h: float,
        vx: float,
        vy: float,
        is_terminal: bool,
    ) -> float:
        """
        Compute potential function Phi(s):
        Phi = -10 * sqrt(x^2 + max(0, h - 0.72)^2) - 4 * sqrt(vx^2 + vy^2)

        Crucial PBRS condition: Phi(s_terminal) = 0.0!
        """
        if is_terminal:
            return 0.0

        dist_h = max(0.0, h - REST_HEIGHT)
        dist_sq = x**2 + dist_h**2
        speed_sq = vx**2 + vy**2

        dist = math.sqrt(dist_sq)
        speed = math.sqrt(speed_sq)

        phi = -10.0 * dist - 4.0 * speed
        return float(phi)

    def compute_raw_reward(
        self,
        col_info: CollisionInfo,
        term_result: TerminalResult,
        state_next: Dict[str, float],
        delta_fuel: float,
        delta_time: float,
    ) -> Dict[str, float]:
        """
        Compute the ORIGINAL / RAW physical reward (unscaled, unshaped).
        This is the true task performance metric used for final evaluation.
        """
        r_terminal = 0.0
        r_target_zone = 0.0
        r_impact_vel = 0.0
        r_impact_angle = 0.0

        # 1. Terminal rewards / penalties
        if term_result.is_safe_landing:
            r_terminal += self.r_safe_landing
            if col_info.is_in_target_zone:
                r_target_zone += self.r_target_zone
        elif term_result.is_crash:
            r_terminal -= self.w_crash
        elif term_result.is_out_of_bounds:
            r_terminal -= self.w_oob
        elif term_result.is_timeout:
            r_terminal -= self.w_timeout

        # 2. Contact penalties (only assessed upon ground contact)
        if col_info.is_contact:
            vx = state_next["vx"]
            vy = state_next["vy"]
            theta = state_next["theta"]
            r_impact_vel = -self.w_impact_vel * (vx**2 + vy**2)
            r_impact_angle = -self.w_impact_angle * (1.0 - math.cos(theta))

        # 3. Continuous step costs
        r_gas = -self.w_gas * delta_fuel
        r_time = -self.w_time * delta_time

        r_base = (
            r_terminal
            + r_target_zone
            + r_impact_vel
            + r_impact_angle
            + r_gas
            + r_time
        )

        return {
            "r_base": float(r_base),
            "r_terminal": float(r_terminal),
            "r_target_zone": float(r_target_zone),
            "r_impact_vel": float(r_impact_vel),
            "r_impact_angle": float(r_impact_angle),
            "r_gas": float(r_gas),
            "r_time": float(r_time),
        }

    def compute_shaping(
        self,
        state_curr: Dict[str, float],
        state_next: Dict[str, float],
        is_terminal: bool,
    ) -> Tuple[float, float, float]:
        """
        Compute PBRS shaping reward: F(s, s') = gamma * Phi(s') - Phi(s).
        Returns (F, phi_s, phi_s_next).
        """
        phi_s = self.compute_potential(
            x=state_curr["x"],
            h=state_curr["y"],
            vx=state_curr["vx"],
            vy=state_curr["vy"],
            is_terminal=False,
        )

        phi_s_next = self.compute_potential(
            x=state_next["x"],
            h=state_next["y"],
            vx=state_next["vx"],
            vy=state_next["vy"],
            is_terminal=is_terminal,
        )

        if self.use_shaping:
            f_shaping = self.gamma * phi_s_next - phi_s
        else:
            f_shaping = 0.0

        return float(f_shaping), float(phi_s), float(phi_s_next)

    def calculate_reward(
        self,
        state_curr: Dict[str, float],
        state_next: Dict[str, float],
        col_info: CollisionInfo,
        term_result: TerminalResult,
        delta_fuel: float,
        delta_time: float,
    ) -> RewardBreakdown:
        """
        Compute full reward breakdown separating Original Reward and Shaped Reward.
        """
        raw_dict = self.compute_raw_reward(
            col_info=col_info,
            term_result=term_result,
            state_next=state_next,
            delta_fuel=delta_fuel,
            delta_time=delta_time,
        )

        f_shaping, phi_s, phi_s_next = self.compute_shaping(
            state_curr=state_curr,
            state_next=state_next,
            is_terminal=term_result.done,
        )

        r_base = raw_dict["r_base"]
        r_learn = self.reward_scale * (r_base + f_shaping)

        return RewardBreakdown(
            r_learn=float(r_learn),
            r_base=float(r_base),
            r_terminal=float(raw_dict["r_terminal"]),
            r_target_zone=float(raw_dict["r_target_zone"]),
            r_impact_vel=float(raw_dict["r_impact_vel"]),
            r_impact_angle=float(raw_dict["r_impact_angle"]),
            r_gas=float(raw_dict["r_gas"]),
            r_time=float(raw_dict["r_time"]),
            r_shaping=float(f_shaping),
            phi_s=float(phi_s),
            phi_s_next=float(phi_s_next),
        )
