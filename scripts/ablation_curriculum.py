"""
Curriculum Learning Ablation Experiment:
Compares:
  1. Baseline (No Curriculum: straight training on Level 3 Full Challenge Domain)
  2. +Curriculum (Progressive Curriculum: Level 1 -> Level 2 -> Level 3)

Runs under identical random seeds.
Evaluates both models on identical independent test set (Seeds 20000..20049).
Ensures zero data leakage between training curriculum and evaluation sets.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import time
from typing import Dict, Any, List
import numpy as np
import torch

from environment.roadster_env import RoadsterEnv
from demonstration.dataset import DemonstrationDataset
from rl.dqn.replay_buffer import ReplayBuffer
from rl.demonstration_agent import DemonstrationDQNAgent, DemonstrationAgentConfig
from training.curriculum import RoadsterCurriculum, LevelSpec, STANDARD_LEVELS


def evaluate_model_on_test_set(
    agent: DemonstrationDQNAgent,
    num_episodes: int = 50,
    seed_base: int = 20000,
    level: int = 3,
) -> Dict[str, Any]:
    """
    Evaluate agent on independent evaluation set with epsilon = 0.0.
    """
    env = RoadsterEnv(use_shaping=False)
    successes = 0
    crashes = 0
    timeouts = 0
    oobs = 0
    raw_rewards = []
    landing_speeds = []
    landing_vys = []
    landing_vxs = []
    landing_angles = []
    landing_times = []
    fuel_used_list = []

    for i in range(num_episodes):
        seed = seed_base + i
        obs, info = env.reset(seed=seed, options={"level": level})
        done = False
        ep_raw_reward = 0.0

        while not done:
            q_vals = agent.get_q_values(obs)
            action = int(np.argmax(q_vals))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_raw_reward += float(info.get("step_raw_reward", reward))

        raw_rewards.append(ep_raw_reward)
        fuel_used_list.append(float(info.get("fuel_used", 100.0 - info.get("fuel", 100.0))))

        if info.get("safe_landing", False):
            successes += 1
            landing_speeds.append(float(info.get("landing_speed", 0.0)))
            landing_vys.append(float(info.get("landing_speed_vy", 0.0)))
            landing_vxs.append(float(info.get("landing_speed_vx", 0.0)))
            landing_angles.append(abs(float(info.get("landing_angle_deg", 0.0))))
            landing_times.append(float(info.get("time", 0.0)))
        elif info.get("crash", False):
            crashes += 1
        elif info.get("timeout", False):
            timeouts += 1
        elif info.get("out_of_bounds", False):
            oobs += 1

    return {
        "num_episodes": num_episodes,
        "seed_range": [seed_base, seed_base + num_episodes - 1],
        "safe_landing_rate_pct": float((successes / num_episodes) * 100.0),
        "crash_rate_pct": float((crashes / num_episodes) * 100.0),
        "timeout_rate_pct": float((timeouts / num_episodes) * 100.0),
        "oob_rate_pct": float((oobs / num_episodes) * 100.0),
        "mean_raw_reward": float(np.mean(raw_rewards)),
        "mean_landing_speed_mps": float(np.mean(landing_speeds)) if landing_speeds else 0.0,
        "mean_landing_vy_mps": float(np.mean(landing_vys)) if landing_vys else 0.0,
        "mean_landing_vx_mps": float(np.mean(landing_vxs)) if landing_vxs else 0.0,
        "mean_landing_angle_deg": float(np.mean(landing_angles)) if landing_angles else 0.0,
        "mean_landing_time_s": float(np.mean(landing_times)) if landing_times else 0.0,
        "mean_fuel_used_pct": float(np.mean(fuel_used_list)),
    }


def train_run(
    curriculum_enabled: bool,
    num_episodes: int = 150,
    demo_dataset: DemonstrationDataset = None,
    seed_base: int = 42,
    verbose: bool = True,
) -> Tuple[DemonstrationDQNAgent, RoadsterCurriculum, Dict[str, Any]]:
    """
    Execute a training run with curriculum enabled or disabled.
    """
    mode_name = "+Curriculum (Progressive L1->L2->L3)" if curriculum_enabled else "Baseline (No Curriculum: L3 Only)"
    if verbose:
        print("\n" + "=" * 75)
        print(f"STARTING TRAINING RUN: {mode_name}")
        print("=" * 75)

    # Initialize curriculum controller
    test_levels = [
        LevelSpec(1, "Level 1: Low Altitude", (1.2, 3.0), (-3.0, 3.0), (-0.2, 0.2), min_episodes=20, success_threshold=0.70),
        LevelSpec(2, "Level 2: Mid Altitude", (2.0, 5.0), (-8.0, 8.0), (-0.5, 0.5), min_episodes=25, success_threshold=0.70),
        LevelSpec(3, "Level 3: Full Domain", (4.0, 9.0), (-15.0, 15.0), (-2.0, 2.0), min_episodes=30, success_threshold=0.75),
    ]

    curriculum = RoadsterCurriculum(
        enabled=curriculum_enabled,
        levels=test_levels,
        window_size=15,
        baseline_level=3,
    )

    env = RoadsterEnv(use_shaping=True)
    config = DemonstrationAgentConfig(
        warmup_steps=128,
        batch_size=32,
        buffer_size=10000,
        learning_rate=0.0005,
        target_update_frequency=100,
        epsilon_start=0.15,
        epsilon_end=0.02,
        epsilon_decay=num_episodes * 25,
    )
    agent = DemonstrationDQNAgent(config=config)

    # Load initial pre-trained weights so agent has baseline competency
    if os.path.exists("checkpoints/dqfd/best_pretrained.pth"):
        agent.load_checkpoint("checkpoints/dqfd/best_pretrained.pth")

    online_buffer = ReplayBuffer(capacity=config.buffer_size, state_dim=config.state_dim)
    demo_size = int(config.batch_size * config.demo_ratio)
    online_size = config.batch_size - demo_size
    rng = np.random.default_rng(seed_base)

    t0 = time.time()
    for ep in range(1, num_episodes + 1):
        # Training seed is strictly separated from test seeds (test is 20000+)
        ep_seed = seed_base + ep
        reset_opts = curriculum.get_env_options()
        obs, info = env.reset(seed=ep_seed, options=reset_opts)
        ep_raw_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            online_buffer.push(obs, action, reward, next_obs, done)
            agent.total_env_steps += 1
            ep_raw_reward += float(step_info.get("step_raw_reward", reward))
            obs = next_obs

            # Mixed training update
            if len(online_buffer) >= config.warmup_steps and demo_dataset is not None:
                raw_demo = demo_dataset.sample_batch(demo_size, rng=rng)
                demo_batch = {
                    "states": torch.from_numpy(raw_demo["states"]).float().to(agent.device),
                    "actions": torch.from_numpy(raw_demo["actions"]).long().to(agent.device),
                    "rewards": torch.from_numpy(raw_demo["rewards"]).float().to(agent.device),
                    "next_states": torch.from_numpy(raw_demo["next_states"]).float().to(agent.device),
                    "dones": torch.from_numpy(raw_demo["dones"]).float().to(agent.device),
                }
                raw_online = online_buffer.sample(online_size, device=str(agent.device))
                online_batch = {
                    "states": raw_online[0],
                    "actions": raw_online[1],
                    "rewards": raw_online[2],
                    "next_states": raw_online[3],
                    "dones": raw_online[4],
                }
                agent.mixed_train_step(demo_batch, online_batch)

        # Record episode in curriculum
        promoted, new_level = curriculum.record_episode(step_info, raw_reward=ep_raw_reward)
        if promoted and verbose:
            print(f"  >>> [Curriculum Promotion] Advanced to Level {new_level} at Episode {ep}!")

        if ep % 25 == 0 and verbose:
            print(
                f"  [Ep {ep:03d}/{num_episodes}] L{curriculum.current_level} | "
                f"WinSucc: {curriculum.rolling_success_rate*100:.1f}% | "
                f"Term: {step_info.get('termination_reason')} | "
                f"Speed: {step_info.get('landing_speed', 0.0):.2f}m/s | "
                f"Rew: {ep_raw_reward:.1f}"
            )

    train_time = time.time() - t0
    curriculum_summary = curriculum.get_summary()

    return agent, curriculum, {
        "train_time_seconds": round(train_time, 2),
        "total_episodes": num_episodes,
        "curriculum_summary": curriculum_summary,
    }


def run_curriculum_ablation():
    print("=" * 80)
    print("RUNNING CURRICULUM ABLATION STUDY: BASELINE VS. +CURRICULUM")
    print("=" * 80)

    demo_dataset = DemonstrationDataset.load_jsonl("demonstrations.jsonl")

    # 1. Run Baseline (No Curriculum: straight Level 3)
    agent_base, cur_base, meta_base = train_run(
        curriculum_enabled=False,
        num_episodes=150,
        demo_dataset=demo_dataset,
        seed_base=100,
        verbose=True,
    )
    eval_base = evaluate_model_on_test_set(agent_base, num_episodes=50, seed_base=20000, level=3)

    # 2. Run +Curriculum (Progressive Level 1 -> 2 -> 3)
    agent_cur, cur_cur, meta_cur = train_run(
        curriculum_enabled=True,
        num_episodes=150,
        demo_dataset=demo_dataset,
        seed_base=100,
        verbose=True,
    )
    eval_cur = evaluate_model_on_test_set(agent_cur, num_episodes=50, seed_base=20000, level=3)

    # 3. Print Comparison Table
    print("\n" + "=" * 80)
    print("CURRICULUM ABLATION RESULTS (Evaluated on Independent 50-Episode Level 3 Test Set)")
    print("=" * 80)
    print(f"{'Metric':<36} | {'Baseline (No Curriculum)':<20} | {'+Curriculum (L1->L2->L3)':<20}")
    print("-" * 80)
    print(f"{'Safe Landing Rate (%)':<36} | {eval_base['safe_landing_rate_pct']:<20.1f} | {eval_cur['safe_landing_rate_pct']:<20.1f}")
    print(f"{'Crash Rate (%)':<36} | {eval_base['crash_rate_pct']:<20.1f} | {eval_cur['crash_rate_pct']:<20.1f}")
    print(f"{'Out of Bounds Rate (%)':<36} | {eval_base['oob_rate_pct']:<20.1f} | {eval_cur['oob_rate_pct']:<20.1f}")
    print(f"{'Mean Raw Reward':<36} | {eval_base['mean_raw_reward']:<20.2f} | {eval_cur['mean_raw_reward']:<20.2f}")
    print(f"{'Mean Landing Speed (|vy| m/s)':<36} | {eval_base['mean_landing_vy_mps']:<20.3f} | {eval_cur['mean_landing_vy_mps']:<20.3f}")
    print(f"{'Mean Landing Angle (deg)':<36} | {eval_base['mean_landing_angle_deg']:<20.2f} | {eval_cur['mean_landing_angle_deg']:<20.2f}")
    print(f"{'Mean Landing Time (s)':<36} | {eval_base['mean_landing_time_s']:<20.2f} | {eval_cur['mean_landing_time_s']:<20.2f}")
    print(f"{'Mean Fuel Used (%)':<36} | {eval_base['mean_fuel_used_pct']:<20.2f} | {eval_cur['mean_fuel_used_pct']:<20.2f}")
    print("=" * 80)

    # 4. Save results to JSON
    report = {
        "metadata": {
            "eval_episodes": 50,
            "seed_range": [20000, 20049],
            "evaluation_level": 3,
            "training_episodes": 100,
        },
        "baseline_no_curriculum": {
            "eval_metrics": eval_base,
            "training_meta": meta_base,
        },
        "curriculum_enabled": {
            "eval_metrics": eval_cur,
            "training_meta": meta_cur,
        },
    }

    out_path = "checkpoints/curriculum_ablation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved curriculum ablation report to: {out_path}")


if __name__ == "__main__":
    run_curriculum_ablation()
