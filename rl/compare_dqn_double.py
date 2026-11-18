"""
Comparative Benchmark: Standard DQN vs. Double DQN.
Runs under identical environments, identical training seeds, and identical evaluation seeds.
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
from rl.dqn.trainer import DQNTrainer
from rl.double_dqn.agent import DoubleDQNAgent


def run_benchmark(
    train_episodes: int = 150,
    eval_episodes: int = 30,
    seed_base: int = 42,
    eval_seed_base: int = 20000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run identical benchmark comparing Standard DQN against Double DQN.
    """
    config = DQNConfig(
        warmup_steps=100,
        batch_size=32,
        buffer_size=10000,
        target_update_frequency=50,
        epsilon_decay=train_episodes * 40,
        learning_rate=0.0005,
        gamma=0.995,
    )

    results = {}

    # --------------------------------------------------------------------------
    # 1. RUN STANDARD DQN
    # --------------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 70)
        print(">>> 1/2: TRAINING STANDARD DQN BASELINE")
        print("=" * 70)

    env_dqn = RoadsterEnv()
    agent_dqn = DQNAgent(config)
    trainer_dqn = DQNTrainer(
        env=env_dqn,
        agent=agent_dqn,
        checkpoint_dir="checkpoints/compare_dqn",
    )

    t0_dqn = time.time()
    dqn_logs = trainer_dqn.train(
        num_episodes=train_episodes,
        eval_interval=train_episodes,
        eval_episodes=eval_episodes,
        log_interval=50,
        verbose=verbose,
    )
    dqn_train_time = time.time() - t0_dqn

    # --------------------------------------------------------------------------
    # 2. RUN DOUBLE DQN
    # --------------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 70)
        print(">>> 2/2: TRAINING DOUBLE DQN")
        print("=" * 70)

    env_ddqn = RoadsterEnv()
    agent_ddqn = DoubleDQNAgent(config)
    trainer_ddqn = DQNTrainer(
        env=env_ddqn,
        agent=agent_ddqn,
        checkpoint_dir="checkpoints/compare_ddqn",
    )

    t0_ddqn = time.time()
    ddqn_logs = trainer_ddqn.train(
        num_episodes=train_episodes,
        eval_interval=train_episodes,
        eval_episodes=eval_episodes,
        log_interval=50,
        verbose=verbose,
    )
    ddqn_train_time = time.time() - t0_ddqn

    # --------------------------------------------------------------------------
    # 3. IDENTICAL EVALUATION SET RUN
    # --------------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 70)
        print(f">>> EVALUATING BOTH MODELS ON IDENTICAL {eval_episodes} TEST SEEDS (SEED {eval_seed_base}..)")
        print("=" * 70)

    def evaluate_model(agent: DQNAgent, env: RoadsterEnv) -> Dict[str, Any]:
        success_list = []
        crash_list = []
        timeout_list = []
        oob_list = []
        raw_rewards = []
        learning_rewards = []
        landing_vys = []
        landing_vxs = []
        landing_angles = []
        landing_times = []
        all_q_values = []

        for i in range(eval_episodes):
            seed = eval_seed_base + i
            obs, info = env.reset(seed=seed, options={"level": 3})
            done = False
            ep_raw_rew = 0.0
            ep_learn_rew = 0.0

            while not done:
                q_vals = agent.get_q_values(obs)
                all_q_values.append(np.max(q_vals))
                action = int(np.argmax(q_vals))

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_raw_rew += info.get("raw_reward", 0.0)
                ep_learn_rew += reward

            success_list.append(1.0 if info["safe_landing"] else 0.0)
            crash_list.append(1.0 if info["crash"] else 0.0)
            timeout_list.append(1.0 if info["timeout"] else 0.0)
            oob_list.append(1.0 if info["out_of_bounds"] else 0.0)
            raw_rewards.append(ep_raw_rew)
            learning_rewards.append(ep_learn_rew)
            landing_vys.append(info["landing_speed_vy"])
            landing_vxs.append(info["landing_speed_vx"])
            landing_angles.append(abs(info["landing_angle_deg"]))
            landing_times.append(info["time"])

        return {
            "success_rate_percent": float(np.mean(success_list) * 100.0),
            "collision_rate_percent": float(np.mean(crash_list) * 100.0),
            "timeout_rate_percent": float(np.mean(timeout_list) * 100.0),
            "oob_rate_percent": float(np.mean(oob_list) * 100.0),
            "mean_raw_reward": float(np.mean(raw_rewards)),
            "mean_learning_reward": float(np.mean(learning_rewards)),
            "mean_landing_vy_ms": float(np.mean(landing_vys)),
            "mean_landing_vx_ms": float(np.mean(landing_vxs)),
            "mean_landing_angle_deg": float(np.mean(landing_angles)),
            "mean_landing_time_s": float(np.mean(landing_times)),
            "mean_estimated_q_value": float(np.mean(all_q_values)),
            "max_estimated_q_value": float(np.max(all_q_values)),
        }

    eval_dqn = evaluate_model(agent_dqn, env_dqn)
    eval_ddqn = evaluate_model(agent_ddqn, env_ddqn)

    results = {
        "dqn": {
            "eval_metrics": eval_dqn,
            "train_wall_time_s": dqn_train_time,
            "final_train_reward": dqn_logs[-1]["raw_reward"] if dqn_logs else 0.0,
        },
        "double_dqn": {
            "eval_metrics": eval_ddqn,
            "train_wall_time_s": ddqn_train_time,
            "final_train_reward": ddqn_logs[-1]["raw_reward"] if ddqn_logs else 0.0,
        },
    }

    if verbose:
        print("\n" + "=" * 70)
        print(">>> COMPARATIVE BENCHMARK SUMMARY TABLE")
        print("=" * 70)
        print(f"{'Metric':<32} | {'Standard DQN':<16} | {'Double DQN':<16}")
        print("-" * 70)
        print(f"{'Success Rate (%)':<32} | {eval_dqn['success_rate_percent']:<16.1f} | {eval_ddqn['success_rate_percent']:<16.1f}")
        print(f"{'Collision Rate (%)':<32} | {eval_dqn['collision_rate_percent']:<16.1f} | {eval_ddqn['collision_rate_percent']:<16.1f}")
        print(f"{'Timeout Rate (%)':<32} | {eval_dqn['timeout_rate_percent']:<16.1f} | {eval_ddqn['timeout_rate_percent']:<16.1f}")
        print(f"{'Mean Raw Reward':<32} | {eval_dqn['mean_raw_reward']:<16.2f} | {eval_ddqn['mean_raw_reward']:<16.2f}")
        print(f"{'Mean Landing Speed (|vy| m/s)':<32} | {eval_dqn['mean_landing_vy_ms']:<16.2f} | {eval_ddqn['mean_landing_vy_ms']:<16.2f}")
        print(f"{'Mean Landing Angle (deg)':<32} | {eval_dqn['mean_landing_angle_deg']:<16.2f} | {eval_ddqn['mean_landing_angle_deg']:<16.2f}")
        print(f"{'Mean Landing Time (s)':<32} | {eval_dqn['mean_landing_time_s']:<16.2f} | {eval_ddqn['mean_landing_time_s']:<16.2f}")
        print(f"{'Mean Estimated Q-Value':<32} | {eval_dqn['mean_estimated_q_value']:<16.3f} | {eval_ddqn['mean_estimated_q_value']:<16.3f}")
        print(f"{'Max Estimated Q-Value':<32} | {eval_dqn['max_estimated_q_value']:<16.3f} | {eval_ddqn['max_estimated_q_value']:<16.3f}")
        print("=" * 70)

    # Save to JSON
    with open("checkpoints/compare_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_benchmark(train_episodes=100, eval_episodes=25, verbose=True)
