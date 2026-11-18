"""
Independent Evaluation Engine for the 2D Roadster Flying Car.
Strictly decoupled from training, replay buffers, and gradient optimizers.

Operational Protocol:
  - Policy Mode: Deterministic Greedy (epsilon = 0.0)
  - Gradient Tracking: Disabled (torch.no_grad())
  - Network Mode: eval()
  - Replay Buffer: Zero interaction
  - Test States: Exclusively novel, unseen initial states
  - Benchmark Domains:
      1. Curated Deterministic Test Cases (60 stress-testing corner cases)
      2. Stochastic Independent Test Domain (100 unseen seeds: 30000..30099)

Comprehensive Evaluation Criteria:
  1. Safe Landing (|vy| < 1 m/s, |vx| < 0.5 m/s, |theta| < 5 deg, |omega| < 0.2 rad/s)
  2. Correct Landing Position (|x| <= 0.8 m target zone)
  3. Landing Time (s)
  4. Landing Orientation (|theta| deg)
  5. Landing Velocity (|vy| and |vx| m/s)
  6. Collision Rate (%)
  7. Timeout Rate (%)
  8. Out-of-bounds Rate (%)
  9. Fuel Consumption (%)

Detailed Failure & Confusion Analysis:
  - collision
  - upside down
  - overshoot
  - undershoot
  - too fast
  - too much rotation
  - timeout
"""

from __future__ import annotations
import os
import sys
import math
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch

# Ensure workspace root is available
sys.path.insert(0, os.path.abspath("."))

from environment.roadster_env import RoadsterEnv
from physics.constants import (
    MAX_SAFE_LANDING_VY,
    MAX_SAFE_LANDING_VX,
    MAX_SAFE_LANDING_ANGLE_RAD,
    MAX_SAFE_LANDING_OMEGA,
    TARGET_ZONE_X_MIN,
    TARGET_ZONE_X_MAX,
    WORLD_X_MIN,
    WORLD_X_MAX,
    MAX_EPISODE_TIME,
)
from evaluation.test_cases import TestCase, build_curated_test_cases, get_stochastic_unseen_seeds


# Standard target zone boundaries according to course specification
COURSE_TARGET_ZONE_X = 0.8  # meters (|x| <= 0.8m is green center target)


@dataclass
class EpisodeTelemetry:
    """Fine-grained viễn trắc chi tiết cho từng episode đánh giá."""
    episode_id: Union[int, str]
    category: str
    description: str
    initial_state: Dict[str, float]
    final_state: Dict[str, float]
    termination_reason: str

    # The 9 Core Evaluation Criteria
    is_safe_landing: bool
    is_in_target_zone: bool
    landing_time: float
    landing_angle_deg: float
    landing_speed_vy: float
    landing_speed_vx: float
    landing_omega: float
    landing_x: float
    is_collision: bool
    is_timeout: bool
    is_out_of_bounds: bool
    fuel_used_pct: float
    raw_reward: float
    step_count: int

    # Failure Taxonomy Flags (Multi-label analysis)
    fail_upside_down: bool = False
    fail_overshoot: bool = False
    fail_undershoot: bool = False
    fail_too_fast: bool = False
    fail_too_much_rotation: bool = False

    # Primary Mutually-Exclusive Failure Category
    primary_failure_type: str = "NONE"


@dataclass
class EvaluationSummary:
    """Aggregated scorecard covering all 9 criteria and failure taxonomy."""
    total_episodes: int
    eval_mode: str

    # 1. Safe Landing
    safe_landing_count: int
    safe_landing_rate_pct: float

    # 2. Correct Landing Position
    target_zone_count: int
    target_zone_rate_pct: float
    mean_landing_x_error_m: float

    # 3. Landing Time
    mean_landing_time_s: float
    std_landing_time_s: float

    # 4. Landing Orientation
    mean_landing_angle_deg: float
    max_landing_angle_deg: float
    angle_within_5deg_rate_pct: float

    # 5. Landing Velocity
    mean_landing_vy_mps: float
    mean_landing_vx_mps: float
    mean_total_landing_speed_mps: float

    # 6. Collision
    collision_count: int
    collision_rate_pct: float

    # 7. Timeout
    timeout_count: int
    timeout_rate_pct: float

    # 8. Out-of-bounds
    oob_count: int
    oob_rate_pct: float

    # 9. Fuel Consumption
    mean_fuel_used_pct: float
    mean_fuel_remaining_pct: float

    # Raw Reward (Physical performance baseline)
    mean_raw_reward: float

    # Failure Breakdown (Confusion Matrix Counts)
    failure_counts: Dict[str, int] = field(default_factory=dict)
    failure_percentages: Dict[str, float] = field(default_factory=dict)

    # Per-category scorecard (if evaluated across multiple categories)
    category_scorecards: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class StandaloneEvaluator:
    """
    Independent Evaluation Harness.
    Guarantees:
      - zero training steps
      - zero gradient computation
      - zero exploration (epsilon = 0.0)
      - zero replay buffer updates
      - pure deterministic assessment
    """

    def __init__(self, env: Optional[RoadsterEnv] = None):
        self.env = env or RoadsterEnv(use_shaping=False)

    def _classify_failure(
        self,
        is_safe: bool,
        is_collision: bool,
        is_timeout: bool,
        is_oob: bool,
        final_x: float,
        final_y: float,
        final_vx: float,
        final_vy: float,
        final_theta_deg: float,
        final_omega: float,
    ) -> Tuple[bool, bool, bool, bool, bool, str]:
        """
        Perform rigorous failure taxonomy classification:
          - upside down
          - overshoot
          - undershoot
          - too fast
          - too much rotation
          - primary_type
        """
        if is_safe:
            return False, False, False, False, False, "SAFE_LANDING"

        # Check sub-conditions
        abs_angle = abs(final_theta_deg)
        abs_vy = abs(final_vy)
        abs_vx = abs(final_vx)
        abs_omega = abs(final_omega)

        fail_upside_down = (abs_angle > 90.0)
        fail_overshoot = (final_x > COURSE_TARGET_ZONE_X)
        fail_undershoot = (final_x < -COURSE_TARGET_ZONE_X)
        fail_too_fast = (abs_vy >= MAX_SAFE_LANDING_VY or abs_vx >= MAX_SAFE_LANDING_VX)
        fail_too_much_rotation = (abs_angle >= math.degrees(MAX_SAFE_LANDING_ANGLE_RAD) or abs_omega >= MAX_SAFE_LANDING_OMEGA)

        # Primary mutually exclusive assignment
        if is_timeout:
            primary = "TIMEOUT"
        elif is_oob:
            if final_x > 0:
                primary = "OOB_OVERSHOOT"
            else:
                primary = "OOB_UNDERSHOOT"
        elif fail_upside_down:
            primary = "CRASH_UPSIDE_DOWN"
        elif fail_too_fast and fail_too_much_rotation:
            primary = "CRASH_COMPOUND_FAST_AND_ROTATED"
        elif fail_too_fast:
            primary = "CRASH_TOO_FAST"
        elif fail_too_much_rotation:
            primary = "CRASH_TOO_MUCH_ROTATION"
        elif is_collision:
            primary = "CRASH_OTHER"
        else:
            primary = "UNKNOWN_FAILURE"

        return (
            fail_upside_down,
            fail_overshoot,
            fail_undershoot,
            fail_too_fast,
            fail_too_much_rotation,
            primary,
        )

    def evaluate_episode(
        self,
        agent: Any,
        reset_options: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        episode_id: Union[int, str] = 0,
        category: str = "Standard",
        description: str = "",
    ) -> EpisodeTelemetry:
        """
        Execute a single evaluation rollout under strict greedy policy.
        """
        # Ensure agent is in evaluation mode
        if hasattr(agent, "online_net") and hasattr(agent.online_net, "eval"):
            agent.online_net.eval()

        obs, info = self.env.reset(seed=seed, options=reset_options)
        init_raw = self.env.world.get_raw_state()
        init_state = {
            "x": float(init_raw["x"]),
            "y": float(init_raw["y"]),
            "vx": float(init_raw["vx"]),
            "vy": float(init_raw["vy"]),
            "theta_deg": float(init_raw["theta_deg"]),
            "omega": float(init_raw["omega"]),
            "fuel": float(init_raw["fuel"]),
        }

        done = False
        ep_raw_reward = 0.0
        step_count = 0

        with torch.no_grad():
            while not done:
                if hasattr(agent, "select_action"):
                    action = agent.select_action(obs, evaluate=True)
                elif hasattr(agent, "get_q_values"):
                    q_vals = agent.get_q_values(obs)
                    action = int(np.argmax(q_vals))
                else:
                    raise AttributeError("Agent must implement select_action or get_q_values")

                obs, reward, terminated, truncated, step_info = self.env.step(action)
                done = terminated or truncated
                step_count += 1
                ep_raw_reward += float(step_info.get("step_raw_reward", reward))

        # Final physical state
        final_raw = self.env.world.get_raw_state()
        final_x = float(final_raw["x"])
        final_y = float(final_raw["y"])
        final_vx = float(final_raw["vx"])
        final_vy = float(final_raw["vy"])
        final_theta_deg = float(final_raw["theta_deg"])
        final_omega = float(final_raw["omega"])
        final_fuel = float(final_raw["fuel"])
        elapsed_time = float(final_raw["time"])

        final_state = {
            "x": final_x,
            "y": final_y,
            "vx": final_vx,
            "vy": final_vy,
            "theta_deg": final_theta_deg,
            "omega": final_omega,
            "fuel": final_fuel,
            "time": elapsed_time,
        }

        term_reason = str(step_info.get("termination_reason", "UNKNOWN"))
        is_safe = bool(step_info.get("safe_landing", False))
        is_collision = bool(step_info.get("crash", False))
        is_timeout = bool(step_info.get("timeout", False))
        is_oob = bool(step_info.get("out_of_bounds", False))
        is_target = bool(abs(final_x) <= COURSE_TARGET_ZONE_X and is_safe)
        fuel_used = float(step_info.get("fuel_used", 100.0 - final_fuel))

        # Classify failures
        (
            fail_up,
            fail_over,
            fail_under,
            fail_fast,
            fail_rot,
            primary_type,
        ) = self._classify_failure(
            is_safe=is_safe,
            is_collision=is_collision,
            is_timeout=is_timeout,
            is_oob=is_oob,
            final_x=final_x,
            final_y=final_y,
            final_vx=final_vx,
            final_vy=final_vy,
            final_theta_deg=final_theta_deg,
            final_omega=final_omega,
        )

        return EpisodeTelemetry(
            episode_id=episode_id,
            category=category,
            description=description,
            initial_state=init_state,
            final_state=final_state,
            termination_reason=term_reason,
            is_safe_landing=is_safe,
            is_in_target_zone=is_target,
            landing_time=elapsed_time,
            landing_angle_deg=final_theta_deg,
            landing_speed_vy=final_vy,
            landing_speed_vx=final_vx,
            landing_omega=final_omega,
            landing_x=final_x,
            is_collision=is_collision,
            is_timeout=is_timeout,
            is_out_of_bounds=is_oob,
            fuel_used_pct=fuel_used,
            raw_reward=ep_raw_reward,
            step_count=step_count,
            fail_upside_down=fail_up,
            fail_overshoot=fail_over,
            fail_undershoot=fail_under,
            fail_too_fast=fail_fast,
            fail_too_much_rotation=fail_rot,
            primary_failure_type=primary_type,
        )

    def evaluate_curated_suite(
        self,
        agent: Any,
        test_cases: Optional[List[TestCase]] = None,
    ) -> Tuple[EvaluationSummary, List[EpisodeTelemetry]]:
        """
        Evaluate agent against the 60 structured corner test cases.
        """
        cases = test_cases or build_curated_test_cases()
        telemetries: List[EpisodeTelemetry] = []

        for c in cases:
            telem = self.evaluate_episode(
                agent=agent,
                reset_options=c.to_env_options(),
                seed=None,
                episode_id=c.test_id,
                category=c.category,
                description=c.description,
            )
            telemetries.append(telem)

        summary = self._aggregate_telemetries(telemetries, eval_mode="Curated Corner Cases (60 Tests)")
        return summary, telemetries

    def evaluate_stochastic_suite(
        self,
        agent: Any,
        num_episodes: int = 100,
        seed_start: int = 30000,
        level: int = 3,
    ) -> Tuple[EvaluationSummary, List[EpisodeTelemetry]]:
        """
        Evaluate agent across 100 independent unseen random seeds.
        """
        seeds = get_stochastic_unseen_seeds(count=num_episodes, seed_start=seed_start)
        telemetries: List[EpisodeTelemetry] = []

        for idx, s in enumerate(seeds, 1):
            telem = self.evaluate_episode(
                agent=agent,
                reset_options={"level": level},
                seed=s,
                episode_id=s,
                category="Stochastic Level 3",
                description=f"Unseen Seed {s}",
            )
            telemetries.append(telem)

        summary = self._aggregate_telemetries(
            telemetries,
            eval_mode=f"Stochastic Unseen Domain ({num_episodes} Episodes, Seeds {seed_start}..{seed_start+num_episodes-1})",
        )
        return summary, telemetries

    def _aggregate_telemetries(
        self,
        telemetries: List[EpisodeTelemetry],
        eval_mode: str,
    ) -> EvaluationSummary:
        """
        Aggregate list of EpisodeTelemetry into comprehensive scorecard.
        """
        N = len(telemetries)
        if N == 0:
            raise ValueError("Cannot aggregate empty telemetries list")

        safe_count = sum(1 for t in telemetries if t.is_safe_landing)
        target_count = sum(1 for t in telemetries if t.is_in_target_zone)
        collision_count = sum(1 for t in telemetries if t.is_collision)
        timeout_count = sum(1 for t in telemetries if t.is_timeout)
        oob_count = sum(1 for t in telemetries if t.is_out_of_bounds)

        # Metrics calculated over safe landings (or all if applicable)
        safe_landings = [t for t in telemetries if t.is_safe_landing]
        if safe_landings:
            mean_time = float(np.mean([t.landing_time for t in safe_landings]))
            std_time = float(np.std([t.landing_time for t in safe_landings]))
            mean_angle = float(np.mean([abs(t.landing_angle_deg) for t in safe_landings]))
            max_angle = float(np.max([abs(t.landing_angle_deg) for t in safe_landings]))
            angle_ok_count = sum(1 for t in safe_landings if abs(t.landing_angle_deg) <= 5.0)
            angle_ok_pct = (angle_ok_count / len(safe_landings)) * 100.0

            mean_vy = float(np.mean([abs(t.landing_speed_vy) for t in safe_landings]))
            mean_vx = float(np.mean([abs(t.landing_speed_vx) for t in safe_landings]))
            mean_speed = float(np.mean([math.sqrt(t.landing_speed_vx**2 + t.landing_speed_vy**2) for t in safe_landings]))
            mean_x_err = float(np.mean([abs(t.landing_x) for t in safe_landings]))
        else:
            mean_time = 0.0
            std_time = 0.0
            mean_angle = 0.0
            max_angle = 0.0
            angle_ok_pct = 0.0
            mean_vy = 0.0
            mean_vx = 0.0
            mean_speed = 0.0
            mean_x_err = 0.0

        fuel_used_all = [t.fuel_used_pct for t in telemetries]
        raw_rewards_all = [t.raw_reward for t in telemetries]

        # Failure counts by primary failure type
        failure_counts: Dict[str, int] = {
            "SAFE_LANDING": safe_count,
            "CRASH_TOO_FAST": sum(1 for t in telemetries if t.primary_failure_type == "CRASH_TOO_FAST"),
            "CRASH_TOO_MUCH_ROTATION": sum(1 for t in telemetries if t.primary_failure_type == "CRASH_TOO_MUCH_ROTATION"),
            "CRASH_UPSIDE_DOWN": sum(1 for t in telemetries if t.primary_failure_type == "CRASH_UPSIDE_DOWN"),
            "CRASH_COMPOUND_FAST_AND_ROTATED": sum(1 for t in telemetries if t.primary_failure_type == "CRASH_COMPOUND_FAST_AND_ROTATED"),
            "CRASH_OTHER": sum(1 for t in telemetries if t.primary_failure_type == "CRASH_OTHER"),
            "OOB_OVERSHOOT": sum(1 for t in telemetries if t.primary_failure_type == "OOB_OVERSHOOT"),
            "OOB_UNDERSHOOT": sum(1 for t in telemetries if t.primary_failure_type == "OOB_UNDERSHOOT"),
            "TIMEOUT": timeout_count,
        }

        # Multi-label failure dimension totals
        multi_label_counts = {
            "total_collisions": collision_count,
            "total_upside_down": sum(1 for t in telemetries if t.fail_upside_down),
            "total_overshoot": sum(1 for t in telemetries if t.fail_overshoot),
            "total_undershoot": sum(1 for t in telemetries if t.fail_undershoot),
            "total_too_fast": sum(1 for t in telemetries if t.fail_too_fast),
            "total_too_much_rotation": sum(1 for t in telemetries if t.fail_too_much_rotation),
            "total_timeout": timeout_count,
            "total_out_of_bounds": oob_count,
        }
        failure_counts.update(multi_label_counts)

        failure_pcts = {k: round((v / N) * 100.0, 2) for k, v in failure_counts.items()}

        # Per-category scorecard breakdown
        categories = sorted(list(set(t.category for t in telemetries)))
        cat_scorecards: Dict[str, Dict[str, Any]] = {}
        for cat in categories:
            cat_list = [t for t in telemetries if t.category == cat]
            c_safe = sum(1 for t in cat_list if t.is_safe_landing)
            c_crash = sum(1 for t in cat_list if t.is_collision)
            c_target = sum(1 for t in cat_list if t.is_in_target_zone)
            c_safe_items = [t for t in cat_list if t.is_safe_landing]
            cat_scorecards[cat] = {
                "total": len(cat_list),
                "safe_rate_pct": round((c_safe / len(cat_list)) * 100.0, 1),
                "crash_rate_pct": round((c_crash / len(cat_list)) * 100.0, 1),
                "target_rate_pct": round((c_target / len(cat_list)) * 100.0, 1),
                "mean_landing_time_s": round(float(np.mean([t.landing_time for t in c_safe_items])), 2) if c_safe_items else 0.0,
                "mean_speed_mps": round(float(np.mean([math.sqrt(t.landing_speed_vx**2 + t.landing_speed_vy**2) for t in c_safe_items])), 3) if c_safe_items else 0.0,
                "mean_angle_deg": round(float(np.mean([abs(t.landing_angle_deg) for t in c_safe_items])), 2) if c_safe_items else 0.0,
                "mean_fuel_pct": round(float(np.mean([t.fuel_used_pct for t in cat_list])), 1),
            }

        return EvaluationSummary(
            total_episodes=N,
            eval_mode=eval_mode,
            safe_landing_count=safe_count,
            safe_landing_rate_pct=round((safe_count / N) * 100.0, 2),
            target_zone_count=target_count,
            target_zone_rate_pct=round((target_count / N) * 100.0, 2),
            mean_landing_x_error_m=round(mean_x_err, 3),
            mean_landing_time_s=round(mean_time, 2),
            std_landing_time_s=round(std_time, 2),
            mean_landing_angle_deg=round(mean_angle, 2),
            max_landing_angle_deg=round(max_angle, 2),
            angle_within_5deg_rate_pct=round(angle_ok_pct, 2),
            mean_landing_vy_mps=round(mean_vy, 3),
            mean_landing_vx_mps=round(mean_vx, 3),
            mean_total_landing_speed_mps=round(mean_speed, 3),
            collision_count=collision_count,
            collision_rate_pct=round((collision_count / N) * 100.0, 2),
            timeout_count=timeout_count,
            timeout_rate_pct=round((timeout_count / N) * 100.0, 2),
            oob_count=oob_count,
            oob_rate_pct=round((oob_count / N) * 100.0, 2),
            mean_fuel_used_pct=round(float(np.mean(fuel_used_all)), 2),
            mean_fuel_remaining_pct=round(100.0 - float(np.mean(fuel_used_all)), 2),
            mean_raw_reward=round(float(np.mean(raw_rewards_all)), 2),
            failure_counts=failure_counts,
            failure_percentages=failure_pcts,
            category_scorecards=cat_scorecards,
        )


def export_evaluation_report(
    summary: EvaluationSummary,
    telemetries: List[EpisodeTelemetry],
    output_filepath: str,
) -> None:
    """Save full structured evaluation telemetry and scorecard to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    report_dict = {
        "scorecard": asdict(summary),
        "telemetry_records": [asdict(t) for t in telemetries],
    }
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"[Evaluator] Exported detailed evaluation report to: {output_filepath}")
