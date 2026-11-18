"""
Independent 3-Way Comparative Evaluation:
1. Standard DQN (Tabula Rasa Baseline)
2. Double DQN (Decoupled Target Baseline)
3. Demonstration-Assisted Double DQN (Guided Double DQN with Supervised Margin Loss)

Evaluates on identical, independent test seeds (20000..20049) with epsilon = 0.0.
Strictly ensures zero demonstration data appears in the evaluation test set.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import time
from typing import Dict, Any, List
import numpy as np

from environment.roadster_env import RoadsterEnv
from rl.dqn.agent import DQNAgent, DQNConfig
from rl.double_dqn.agent import DoubleDQNAgent
from rl.demonstration_agent import DemonstrationDQNAgent, DemonstrationAgentConfig


def evaluate_agent(
    agent,
    env: RoadsterEnv,
    num_episodes: int = 50,
    seed_base: int = 20000,
    level: int = 3,
) -> Dict[str, Any]:
    safe_list = []
    crash_list = []
    timeout_list = []
    oob_list = []
    raw_rewards = []
    landing_speeds = []
    landing_vys = []
    landing_vxs = []
    landing_angles = []
    landing_omegas = []
    landing_times = []
    fuel_used_list = []
    q_max_list = []

    for ep in range(num_episodes):
        seed = seed_base + ep
        obs, info = env.reset(seed=seed, options={"level": level})
        done = False
        ep_raw_rew = 0.0

        while not done:
            q_vals = agent.get_q_values(obs)
            q_max_list.append(float(np.max(q_vals)))
            action = int(np.argmax(q_vals))

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_raw_rew += float(info.get("step_raw_reward", reward))

        is_safe = bool(info.get("safe_landing", False))
        is_crash = bool(info.get("crash", False))
        is_timeout = bool(info.get("timeout", False))
        is_oob = bool(info.get("out_of_bounds", False))

        safe_list.append(1.0 if is_safe else 0.0)
        crash_list.append(1.0 if is_crash else 0.0)
        timeout_list.append(1.0 if is_timeout else 0.0)
        oob_list.append(1.0 if is_oob else 0.0)
        raw_rewards.append(ep_raw_rew)
        fuel_used_list.append(float(info.get("fuel_used", 100.0 - info.get("fuel", 100.0))))

        if is_safe:
            landing_speeds.append(float(info.get("landing_speed", 0.0)))
            landing_vys.append(float(info.get("landing_speed_vy", 0.0)))
            landing_vxs.append(float(info.get("landing_speed_vx", 0.0)))
            landing_angles.append(abs(float(info.get("landing_angle_deg", 0.0))))
            landing_omegas.append(abs(float(info.get("omega", 0.0))))
            landing_times.append(float(info.get("time", 0.0)))

    return {
        "num_episodes": num_episodes,
        "seed_range": [seed_base, seed_base + num_episodes - 1],
        "safe_landing_rate_pct": float(np.mean(safe_list) * 100.0),
        "crash_rate_pct": float(np.mean(crash_list) * 100.0),
        "timeout_rate_pct": float(np.mean(timeout_list) * 100.0),
        "oob_rate_pct": float(np.mean(oob_list) * 100.0),
        "mean_raw_reward": float(np.mean(raw_rewards)),
        "mean_landing_speed_mps": float(np.mean(landing_speeds)) if landing_speeds else 0.0,
        "mean_landing_vy_mps": float(np.mean(landing_vys)) if landing_vys else 0.0,
        "mean_landing_vx_mps": float(np.mean(landing_vxs)) if landing_vxs else 0.0,
        "mean_landing_angle_deg": float(np.mean(landing_angles)) if landing_angles else 0.0,
        "mean_landing_omega_radps": float(np.mean(landing_omegas)) if landing_omegas else 0.0,
        "mean_landing_time_s": float(np.mean(landing_times)) if landing_times else 0.0,
        "mean_fuel_used_pct": float(np.mean(fuel_used_list)),
        "mean_max_q_value": float(np.mean(q_max_list)),
    }


def main():
    print("=" * 80)
    print("INDEPENDENT 3-WAY COMPARATIVE BENCHMARK (50 Episodes, Level 3 Full Domain)")
    print("Models Evaluated:")
    print("  1. Standard DQN (Tabula Rasa)")
    print("  2. Double DQN (Tabula Rasa)")
    print("  3. Demonstration-Assisted Double DQN (Guided DDQN + Margin Loss)")
    print("=" * 80)

    env = RoadsterEnv(use_shaping=False)

    # 1. Load Standard DQN
    dqn_path = "checkpoints/compare_dqn/dqn_best.pth"
    print(f"\n[1/3] Loading Standard DQN from: {dqn_path}")
    dqn_agent = DQNAgent()
    dqn_agent.load_checkpoint(dqn_path)
    dqn_agent.online_net.eval()
    dqn_metrics = evaluate_agent(dqn_agent, env, num_episodes=50, seed_base=20000, level=3)

    # 2. Load Double DQN
    ddqn_path = "checkpoints/compare_ddqn/dqn_best.pth"
    print(f"[2/3] Loading Double DQN from: {ddqn_path}")
    ddqn_agent = DoubleDQNAgent()
    ddqn_agent.load_checkpoint(ddqn_path)
    ddqn_agent.online_net.eval()
    ddqn_metrics = evaluate_agent(ddqn_agent, env, num_episodes=50, seed_base=20000, level=3)

    # 3. Load Demonstration-Assisted DQN
    demo_path = "checkpoints/best_model.pth"
    print(f"[3/3] Loading Demonstration-Assisted DQN from: {demo_path}")
    demo_agent = DemonstrationDQNAgent()
    demo_agent.load_checkpoint(demo_path)
    demo_agent.online_net.eval()
    demo_metrics = evaluate_agent(demo_agent, env, num_episodes=50, seed_base=20000, level=3)

    # Summary table
    print("\n" + "=" * 85)
    print(f"{'Metric':<35} | {'Standard DQN':<14} | {'Double DQN':<14} | {'Demo-Assisted DQN':<16}")
    print("-" * 85)
    print(f"{'Safe Landing Rate (%)':<35} | {dqn_metrics['safe_landing_rate_pct']:<14.1f} | {ddqn_metrics['safe_landing_rate_pct']:<14.1f} | {demo_metrics['safe_landing_rate_pct']:<16.1f}")
    print(f"{'Crash Rate (%)':<35} | {dqn_metrics['crash_rate_pct']:<14.1f} | {ddqn_metrics['crash_rate_pct']:<14.1f} | {demo_metrics['crash_rate_pct']:<16.1f}")
    print(f"{'Timeout Rate (%)':<35} | {dqn_metrics['timeout_rate_pct']:<14.1f} | {ddqn_metrics['timeout_rate_pct']:<14.1f} | {demo_metrics['timeout_rate_pct']:<16.1f}")
    print(f"{'Out of Bounds Rate (%)':<35} | {dqn_metrics['oob_rate_pct']:<14.1f} | {ddqn_metrics['oob_rate_pct']:<14.1f} | {demo_metrics['oob_rate_pct']:<16.1f}")
    print(f"{'Mean Raw Reward':<35} | {dqn_metrics['mean_raw_reward']:<14.2f} | {ddqn_metrics['mean_raw_reward']:<14.2f} | {demo_metrics['mean_raw_reward']:<16.2f}")
    print(f"{'Mean Landing Speed (|vy| m/s)':<35} | {dqn_metrics['mean_landing_vy_mps']:<14.2f} | {ddqn_metrics['mean_landing_vy_mps']:<14.2f} | {demo_metrics['mean_landing_vy_mps']:<16.2f}")
    print(f"{'Mean Landing Angle (deg)':<35} | {dqn_metrics['mean_landing_angle_deg']:<14.2f} | {ddqn_metrics['mean_landing_angle_deg']:<14.2f} | {demo_metrics['mean_landing_angle_deg']:<16.2f}")
    print(f"{'Mean Landing Time (s)':<35} | {dqn_metrics['mean_landing_time_s']:<14.2f} | {ddqn_metrics['mean_landing_time_s']:<14.2f} | {demo_metrics['mean_landing_time_s']:<16.2f}")
    print(f"{'Mean Fuel Used (%)':<35} | {dqn_metrics['mean_fuel_used_pct']:<14.2f} | {ddqn_metrics['mean_fuel_used_pct']:<14.2f} | {demo_metrics['mean_fuel_used_pct']:<16.2f}")
    print(f"{'Mean Predicted Q-Value':<35} | {dqn_metrics['mean_max_q_value']:<14.3f} | {ddqn_metrics['mean_max_q_value']:<14.3f} | {demo_metrics['mean_max_q_value']:<16.3f}")
    print("=" * 85)

    comparison_results = {
        "metadata": {
            "eval_episodes": 50,
            "seed_range": [20000, 20049],
            "curriculum_level": 3,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "standard_dqn": dqn_metrics,
        "double_dqn": ddqn_metrics,
        "demonstration_assisted_dqn": demo_metrics,
    }

    out_file = "checkpoints/3way_comparison_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"\nSaved 3-way comparative evaluation to: {out_file}")


if __name__ == "__main__":
    main()
