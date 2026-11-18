"""
Independent Benchmark Runner for 100 Test Episodes.
Evaluates the trained neural network model on fixed seeds (20000..20099)
with exploration disabled (epsilon = 0.0), producing a comprehensive scorecard
aligned with the course evaluation rubric.
"""

from __future__ import annotations
import math
import os
import sys

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath("."))
import json
import time
from typing import Dict, Any, List
import numpy as np
import torch

from environment.roadster_env import RoadsterEnv
from rl.dqfd.agent import DQfDAgent, DQfDConfig


def run_benchmark(
    model_path: str = "checkpoints/best_model.pth",
    num_episodes: int = 100,
    seed_start: int = 20000,
    level: int = 3,
    output_report_path: str = "evaluation/benchmark_results.json",
) -> Dict[str, Any]:
    """
    Execute 100 benchmark episodes on Level 3 (full challenge domain).
    """
    print("=" * 75)
    print(f"OFFICIAL INDEPENDENT BENCHMARK: {num_episodes} EPISODES (Level {level})")
    print(f"Model Checkpoint: {model_path}")
    print(f"Seed Range: {seed_start} to {seed_start + num_episodes - 1}")
    print("=" * 75)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Initialize agent and load checkpoint
    config = DQfDConfig()
    agent = DQfDAgent(config=config)
    agent.load_checkpoint(model_path)
    agent.online_net.eval()

    # Initialize environment with unshaped raw reward
    env = RoadsterEnv(use_shaping=False)

    records: List[Dict[str, Any]] = []
    safe_landings = 0
    in_target_zone = 0
    crashes = 0
    timeouts = 0
    oobs = 0

    landing_speeds: List[float] = []
    landing_vx_list: List[float] = []
    landing_vy_list: List[float] = []
    landing_angles_deg: List[float] = []
    landing_omegas: List[float] = []
    landing_positions_x: List[float] = []
    landing_times: List[float] = []
    fuel_consumptions: List[float] = []
    raw_rewards: List[float] = []

    start_bench_time = time.time()

    for idx in range(num_episodes):
        ep_seed = seed_start + idx
        obs, info = env.reset(seed=ep_seed, options={"level": level})
        ep_raw_reward = 0.0
        done = False
        step_count = 0

        while not done:
            # Deterministic greedy action selection
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
            ep_raw_reward += float(info.get("step_raw_reward", reward))

        term_type = info.get("termination_reason", info.get("termination_type", ""))
        fuel_used = float(info.get("fuel_used", 100.0 - info.get("fuel", 100.0)))
        raw_rewards.append(ep_raw_reward)
        fuel_consumptions.append(fuel_used)

        is_safe = (term_type == "SAFE_LANDING" or info.get("safe_landing", False))
        is_target = bool(info.get("in_target_zone", False))

        if is_safe:
            safe_landings += 1
            spd = float(info.get("landing_speed", 0.0))
            landing_speeds.append(spd)
            landing_vx_list.append(abs(float(info.get("vx", 0.0))))
            landing_vy_list.append(abs(float(info.get("vy", 0.0))))
            landing_angles_deg.append(abs(float(info.get("landing_angle_deg", 0.0))))
            landing_omegas.append(abs(float(info.get("omega", 0.0))))
            landing_positions_x.append(float(info.get("landing_position", 0.0)))
            landing_times.append(float(info.get("time", 0.0)))
            if is_target:
                in_target_zone += 1
        elif term_type == "CRASH" or info.get("crash", False):
            crashes += 1
        elif term_type == "TIMEOUT" or info.get("timeout", False):
            timeouts += 1
        elif term_type == "OUT_OF_BOUNDS" or info.get("out_of_bounds", False):
            oobs += 1

        records.append({
            "episode": idx + 1,
            "seed": ep_seed,
            "termination_type": term_type,
            "safe_landing": is_safe,
            "in_target_zone": is_target,
            "raw_reward": ep_raw_reward,
            "steps": step_count,
            "time_s": float(info.get("time", 0.0)),
            "landing_speed_mps": float(info.get("landing_speed", 0.0)),
            "landing_angle_deg": float(info.get("landing_angle_deg", 0.0)),
            "landing_pos_x": float(info.get("landing_position", 0.0)),
            "fuel_used_pct": fuel_used,
        })

    bench_duration = time.time() - start_bench_time

    # Calculate statistics
    safe_rate = (safe_landings / num_episodes) * 100.0
    target_rate = (in_target_zone / num_episodes) * 100.0
    crash_rate = (crashes / num_episodes) * 100.0
    timeout_rate = (timeouts / num_episodes) * 100.0
    oob_rate = (oobs / num_episodes) * 100.0

    mean_speed = float(np.mean(landing_speeds)) if landing_speeds else 0.0
    mean_vx = float(np.mean(landing_vx_list)) if landing_vx_list else 0.0
    mean_vy = float(np.mean(landing_vy_list)) if landing_vy_list else 0.0
    mean_angle = float(np.mean(landing_angles_deg)) if landing_angles_deg else 0.0
    mean_omega = float(np.mean(landing_omegas)) if landing_omegas else 0.0
    mean_time = float(np.mean(landing_times)) if landing_times else 0.0
    mean_fuel = float(np.mean(fuel_consumptions))
    mean_reward = float(np.mean(raw_rewards))

    # Evaluate against rubrics
    # 1. Safe landing (|vy| < 1.0, |vx| < 0.5, |theta| < 5 deg, |omega| < 0.2): 2.0 pts if rate >= 80%
    pts_safe = 2.0 * min(1.0, safe_rate / 80.0)
    # 2. Target zone accuracy (wheels in [-2.0, 2.0]): 2.0 pts if target rate >= 80%
    pts_target = 2.0 * min(1.0, target_rate / 80.0)
    # 3. Landing time (< 5.0s, penalty -4 pts per 5s over): 2.0 pts if mean_time < 5.0s
    if mean_time <= 5.0:
        pts_time = 2.0
    else:
        pts_time = max(0.0, 2.0 - 4.0 * ((mean_time - 5.0) / 5.0))

    summary = {
        "metadata": {
            "model_path": model_path,
            "num_episodes": num_episodes,
            "seed_start": seed_start,
            "level": level,
            "benchmark_time_seconds": round(bench_duration, 2),
        },
        "rates": {
            "safe_landing_rate_pct": round(safe_rate, 2),
            "target_zone_accuracy_pct": round(target_rate, 2),
            "crash_rate_pct": round(crash_rate, 2),
            "timeout_rate_pct": round(timeout_rate, 2),
            "out_of_bounds_rate_pct": round(oob_rate, 2),
        },
        "averages_safe_landings": {
            "mean_landing_speed_mps": round(mean_speed, 3),
            "mean_landing_vx_mps": round(mean_vx, 3),
            "mean_landing_vy_mps": round(mean_vy, 3),
            "mean_landing_angle_deg": round(mean_angle, 3),
            "mean_angular_velocity_radps": round(mean_omega, 3),
            "mean_landing_time_s": round(mean_time, 2),
        },
        "overall_averages": {
            "mean_fuel_consumed_pct": round(mean_fuel, 2),
            "mean_raw_reward": round(mean_reward, 2),
        },
        "rubric_scorecard": {
            "safe_landing_points": round(pts_safe, 2),
            "target_zone_points": round(pts_target, 2),
            "landing_speed_time_points": round(pts_time, 2),
            "subtotal_landing_points": round(pts_safe + pts_target + pts_time, 2),
        },
        "episodes": records,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_report_path)), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print("BENCHMARK SUMMARY RESULTS:")
    print(f"  Safe Landing Rate:     {safe_rate:.1f}% ({safe_landings}/{num_episodes})")
    print(f"  Target Zone Accuracy:  {target_rate:.1f}% ({in_target_zone}/{num_episodes})")
    print(f"  Crash Rate:            {crash_rate:.1f}%")
    print(f"  Timeout Rate:          {timeout_rate:.1f}%")
    print(f"  Out of Bounds Rate:    {oob_rate:.1f}%")
    print("-" * 50)
    print(f"  Mean Landing Speed:    {mean_speed:.3f} m/s (Limit: < 1.0 m/s)")
    print(f"  Mean Landing Angle:    {mean_angle:.2f} deg (Limit: < 5.0 deg)")
    print(f"  Mean Landing Time:     {mean_time:.2f} s   (Limit: < 5.0 s)")
    print(f"  Mean Raw Reward:       {mean_reward:.2f}")
    print(f"  Mean Fuel Used:        {mean_fuel:.1f}%")
    print("-" * 50)
    print(f"  RUBRIC PREDICTED SCORE:")
    print(f"    - Safe Landing:      {pts_safe:.2f} / 2.0")
    print(f"    - Target Zone:       {pts_target:.2f} / 2.0")
    print(f"    - Time Performance:  {pts_time:.2f} / 2.0")
    print(f"    - Total Flight Pts:  {pts_safe + pts_target + pts_time:.2f} / 6.0")
    print("=" * 75)
    print(f"Report written to: {output_report_path}")

    return summary


if __name__ == "__main__":
    model = "checkpoints/best_model.pth"
    if len(sys.argv) > 1:
        model = sys.argv[1]
    run_benchmark(model_path=model)
