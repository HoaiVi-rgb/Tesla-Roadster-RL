"""
Independent Policy Evaluator for the Reproducible RL Experiment Suite.
Strictly decoupled from training and demonstration datasets.
All evaluations are conducted under:
  - Disjoint seeds (20000..20049)
  - Pure greedy execution (epsilon = 0.0)
  - Level 3 Full Challenge Domain
  - Pure unshaped physical reward scale (r_raw)
"""

from __future__ import annotations
import os
import sys
from typing import Dict, Any, List
import numpy as np

from environment.roadster_env import RoadsterEnv


def evaluate_policy_on_test_domain(
    agent,
    num_episodes: int = 50,
    seed_base: int = 20000,
    level: int = 3,
) -> Dict[str, Any]:
    """
    Evaluate an agent on the standardized test domain.

    Parameters
    ----------
    agent : trained agent exposing `get_q_values(obs)` or `select_action(obs, evaluate=True)`
    num_episodes : number of test episodes (default: 50)
    seed_base : start of disjoint test seeds (default: 20000)
    level : difficulty level (default: 3 Full Challenge Domain)

    Returns
    -------
    Dictionary of empirical evaluation metrics.
    """
    env = RoadsterEnv(use_shaping=False)

    successes = 0
    crashes = 0
    timeouts = 0
    oobs = 0

    raw_rewards: List[float] = []
    landing_speeds: List[float] = []
    landing_vys: List[float] = []
    landing_vxs: List[float] = []
    landing_angles: List[float] = []
    landing_times: List[float] = []
    fuel_used_list: List[float] = []
    all_q_max: List[float] = []

    for ep in range(num_episodes):
        seed = seed_base + ep
        obs, info = env.reset(seed=seed, options={"level": level})
        ep_raw_reward = 0.0
        done = False

        while not done:
            # Action selection: greedy without exploration
            if hasattr(agent, "get_q_values"):
                q_vals = agent.get_q_values(obs)
                all_q_max.append(float(np.max(q_vals)))
                action = int(np.argmax(q_vals))
            elif hasattr(agent, "select_action"):
                action = int(agent.select_action(obs, evaluate=True))
            else:
                raise AttributeError("Agent must implement get_q_values or select_action")

            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            ep_raw_reward += float(step_info.get("step_raw_reward", reward))

        raw_rewards.append(ep_raw_reward)
        fuel_used = float(step_info.get("fuel_used", 100.0 - step_info.get("fuel", 100.0)))
        fuel_used_list.append(fuel_used)

        is_safe = bool(step_info.get("safe_landing", False))
        is_crash = bool(step_info.get("crash", False))
        is_timeout = bool(step_info.get("timeout", False))
        is_oob = bool(step_info.get("out_of_bounds", False))

        if is_safe:
            successes += 1
            landing_speeds.append(float(step_info.get("landing_speed", 0.0)))
            landing_vys.append(float(step_info.get("landing_speed_vy", 0.0)))
            landing_vxs.append(float(step_info.get("landing_speed_vx", 0.0)))
            landing_angles.append(abs(float(step_info.get("landing_angle_deg", 0.0))))
            landing_times.append(float(step_info.get("time", 0.0)))
        elif is_crash:
            crashes += 1
        elif is_timeout:
            timeouts += 1
        elif is_oob:
            oobs += 1

    # Exact metrics required by user prompt
    success_rate = (successes / num_episodes) * 100.0
    collision_rate = (crashes / num_episodes) * 100.0
    timeout_rate = (timeouts / num_episodes) * 100.0
    oob_rate = (oobs / num_episodes) * 100.0

    mean_reward = float(np.mean(raw_rewards)) if raw_rewards else 0.0
    mean_landing_time = float(np.mean(landing_times)) if landing_times else 0.0
    mean_landing_velocity = float(np.mean(landing_speeds)) if landing_speeds else 0.0
    mean_abs_landing_angle = float(np.mean(landing_angles)) if landing_angles else 0.0
    mean_fuel_used = float(np.mean(fuel_used_list)) if fuel_used_list else 0.0
    mean_q_value = float(np.mean(all_q_max)) if all_q_max else 0.0

    return {
        "num_episodes": num_episodes,
        "seed_range": [seed_base, seed_base + num_episodes - 1],
        "eval_level": level,
        "success_rate": round(success_rate, 2),
        "collision_rate": round(collision_rate, 2),
        "timeout_rate": round(timeout_rate, 2),
        "oob_rate": round(oob_rate, 2),
        "mean_reward": round(mean_reward, 2),
        "mean_landing_time": round(mean_landing_time, 2),
        "mean_landing_velocity": round(mean_landing_velocity, 3),
        "mean_abs_landing_angle": round(mean_abs_landing_angle, 2),
        "mean_fuel_used": round(mean_fuel_used, 2),
        "mean_max_q_value": round(mean_q_value, 3),
        "safe_count": successes,
        "crash_count": crashes,
        "timeout_count": timeouts,
        "oob_count": oobs,
    }
