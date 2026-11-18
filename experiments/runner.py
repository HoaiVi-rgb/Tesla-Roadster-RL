"""
Reproducible Experiment Execution Framework.
Orchestrates systematic multi-seed experiments across 5 algorithms:
  - Experiment A: Standard DQN (Tabula Rasa)
  - Experiment B: Double DQN (Tabula Rasa)
  - Experiment C: Double DQN + Reward Shaping (PBRS)
  - Experiment D: Double DQN + Progressive Curriculum Learning
  - Experiment E: Demonstration-Assisted Double DQN (Guided DDQN + Margin Loss)

Logs:
  - Full configuration parameters
  - Fixed seeds (PyTorch, NumPy, Python random)
  - Training telemetry curves (episodes, losses, rewards, termination reasons)
  - Model weights checkpoints
  - Independent evaluation benchmarks (Seeds 20000..20049, Level 3 Domain)
  - Code version hash and timestamps
  - Output files: experiments/results.csv and experiments/summary.json
"""

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import json
import random
import datetime
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

from environment.roadster_env import RoadsterEnv
from demonstration.dataset import DemonstrationDataset
from rl.dqn.agent import DQNAgent, DQNConfig
from rl.dqn.replay_buffer import ReplayBuffer
from rl.double_dqn.agent import DoubleDQNAgent
from rl.demonstration_agent import DemonstrationDQNAgent, DemonstrationAgentConfig
from training.curriculum import RoadsterCurriculum, STANDARD_LEVELS

from experiments.config import ExperimentConfig, build_experiment_suite, get_code_version_hash
from experiments.evaluator import evaluate_policy_on_test_domain


def set_seed(seed: int) -> None:
    """Ensure deterministic execution across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment_a_dqn(cfg: ExperimentConfig) -> Tuple[DQNAgent, Dict[str, Any]]:
    """
    Experiment A: Standard DQN (Tabula Rasa).
    Single Q target: y = r + gamma * (1 - done) * max_a' Q_target(s', a')
    """
    set_seed(cfg.seed)
    env = RoadsterEnv(use_shaping=False)

    agent_cfg = DQNConfig(
        warmup_steps=cfg.warmup_steps,
        batch_size=cfg.batch_size,
        buffer_size=cfg.buffer_size,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        target_update_frequency=cfg.target_update_frequency,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay,
        grad_clip=cfg.grad_clip,
    )
    agent = DQNAgent(config=agent_cfg)
    buffer = ReplayBuffer(capacity=cfg.buffer_size, state_dim=agent_cfg.state_dim)

    training_metrics = []
    t0 = time.time()

    for ep in range(1, cfg.train_episodes + 1):
        ep_seed = cfg.seed * 1000 + ep
        obs, info = env.reset(seed=ep_seed, options={"level": 3})
        done = False
        ep_reward = 0.0
        ep_losses = []

        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, term, trunc, step_info = env.step(action)
            done = term or trunc

            buffer.push(obs, action, reward, next_obs, term)
            agent.total_env_steps += 1
            obs = next_obs
            ep_reward += float(step_info.get("step_raw_reward", reward))

            if buffer.can_sample(cfg.batch_size) and agent.total_env_steps >= cfg.warmup_steps:
                batch = buffer.sample(cfg.batch_size, device=str(agent.device))
                loss = agent.train_step(batch)
                ep_losses.append(loss)

        training_metrics.append({
            "episode": ep,
            "raw_reward": ep_reward,
            "mean_loss": float(np.mean(ep_losses)) if ep_losses else 0.0,
            "epsilon": float(agent.get_epsilon()),
            "termination": step_info.get("termination_reason"),
            "safe_landing": bool(step_info.get("safe_landing", False)),
        })

    train_time = time.time() - t0
    return agent, {
        "train_time_seconds": round(train_time, 2),
        "total_env_steps": agent.total_env_steps,
        "total_updates": agent.total_updates,
        "final_epsilon": float(agent.get_epsilon()),
        "episodes_logged": len(training_metrics),
    }


def run_experiment_b_double_dqn(cfg: ExperimentConfig) -> Tuple[DoubleDQNAgent, Dict[str, Any]]:
    """
    Experiment B: Double DQN (Tabula Rasa).
    Decoupled target: a* = argmax_a Q_online(s', a); y = r + gamma * (1 - done) * Q_target(s', a*)
    """
    set_seed(cfg.seed)
    env = RoadsterEnv(use_shaping=False)

    agent_cfg = DQNConfig(
        warmup_steps=cfg.warmup_steps,
        batch_size=cfg.batch_size,
        buffer_size=cfg.buffer_size,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        target_update_frequency=cfg.target_update_frequency,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay,
        grad_clip=cfg.grad_clip,
    )
    agent = DoubleDQNAgent(config=agent_cfg)
    buffer = ReplayBuffer(capacity=cfg.buffer_size, state_dim=agent_cfg.state_dim)

    training_metrics = []
    t0 = time.time()

    for ep in range(1, cfg.train_episodes + 1):
        ep_seed = cfg.seed * 1000 + ep
        obs, info = env.reset(seed=ep_seed, options={"level": 3})
        done = False
        ep_reward = 0.0
        ep_losses = []

        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, term, trunc, step_info = env.step(action)
            done = term or trunc

            buffer.push(obs, action, reward, next_obs, term)
            agent.total_env_steps += 1
            obs = next_obs
            ep_reward += float(step_info.get("step_raw_reward", reward))

            if buffer.can_sample(cfg.batch_size) and agent.total_env_steps >= cfg.warmup_steps:
                batch = buffer.sample(cfg.batch_size, device=str(agent.device))
                loss = agent.train_step(batch)
                ep_losses.append(loss)

        training_metrics.append({
            "episode": ep,
            "raw_reward": ep_reward,
            "mean_loss": float(np.mean(ep_losses)) if ep_losses else 0.0,
            "epsilon": float(agent.get_epsilon()),
            "termination": step_info.get("termination_reason"),
            "safe_landing": bool(step_info.get("safe_landing", False)),
        })

    train_time = time.time() - t0
    return agent, {
        "train_time_seconds": round(train_time, 2),
        "total_env_steps": agent.total_env_steps,
        "total_updates": agent.total_updates,
        "final_epsilon": float(agent.get_epsilon()),
        "episodes_logged": len(training_metrics),
    }


def run_experiment_c_ddqn_shaping(cfg: ExperimentConfig) -> Tuple[DoubleDQNAgent, Dict[str, Any]]:
    """
    Experiment C: Double DQN + Potential-Based Reward Shaping (PBRS).
    Environment provides r_learn = 0.01 * (r_base + gamma * Phi(s') - Phi(s)).
    """
    set_seed(cfg.seed)
    env = RoadsterEnv(use_shaping=True, gamma=cfg.gamma)

    agent_cfg = DQNConfig(
        warmup_steps=cfg.warmup_steps,
        batch_size=cfg.batch_size,
        buffer_size=cfg.buffer_size,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        target_update_frequency=cfg.target_update_frequency,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay,
        grad_clip=cfg.grad_clip,
    )
    agent = DoubleDQNAgent(config=agent_cfg)
    buffer = ReplayBuffer(capacity=cfg.buffer_size, state_dim=agent_cfg.state_dim)

    training_metrics = []
    t0 = time.time()

    for ep in range(1, cfg.train_episodes + 1):
        ep_seed = cfg.seed * 1000 + ep
        obs, info = env.reset(seed=ep_seed, options={"level": 3})
        done = False
        ep_raw_reward = 0.0
        ep_losses = []

        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, shaped_reward, term, trunc, step_info = env.step(action)
            done = term or trunc

            # Agent learns on shaped reward
            buffer.push(obs, action, shaped_reward, next_obs, term)
            agent.total_env_steps += 1
            obs = next_obs
            ep_raw_reward += float(step_info.get("step_raw_reward", shaped_reward))

            if buffer.can_sample(cfg.batch_size) and agent.total_env_steps >= cfg.warmup_steps:
                batch = buffer.sample(cfg.batch_size, device=str(agent.device))
                loss = agent.train_step(batch)
                ep_losses.append(loss)

        training_metrics.append({
            "episode": ep,
            "raw_reward": ep_raw_reward,
            "mean_loss": float(np.mean(ep_losses)) if ep_losses else 0.0,
            "epsilon": float(agent.get_epsilon()),
            "termination": step_info.get("termination_reason"),
            "safe_landing": bool(step_info.get("safe_landing", False)),
        })

    train_time = time.time() - t0
    return agent, {
        "train_time_seconds": round(train_time, 2),
        "total_env_steps": agent.total_env_steps,
        "total_updates": agent.total_updates,
        "final_epsilon": float(agent.get_epsilon()),
        "episodes_logged": len(training_metrics),
    }


def run_experiment_d_ddqn_curriculum(cfg: ExperimentConfig) -> Tuple[DoubleDQNAgent, Dict[str, Any]]:
    """
    Experiment D: Double DQN + Progressive Curriculum Learning.
    Stages: Level 1 (1.2-3m, +-3 deg) -> Level 2 (2-5m, +-8 deg) -> Level 3 (4-9m, +-15 deg).
    """
    set_seed(cfg.seed)
    env = RoadsterEnv(use_shaping=False)
    curriculum = RoadsterCurriculum(
        enabled=True,
        levels=STANDARD_LEVELS,
        window_size=15,
        baseline_level=3,
    )

    agent_cfg = DQNConfig(
        warmup_steps=cfg.warmup_steps,
        batch_size=cfg.batch_size,
        buffer_size=cfg.buffer_size,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        target_update_frequency=cfg.target_update_frequency,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay,
        grad_clip=cfg.grad_clip,
    )
    agent = DoubleDQNAgent(config=agent_cfg)
    buffer = ReplayBuffer(capacity=cfg.buffer_size, state_dim=agent_cfg.state_dim)

    training_metrics = []
    t0 = time.time()

    for ep in range(1, cfg.train_episodes + 1):
        ep_seed = cfg.seed * 1000 + ep
        reset_opts = curriculum.get_env_options()
        obs, info = env.reset(seed=ep_seed, options=reset_opts)
        done = False
        ep_raw_reward = 0.0
        ep_losses = []

        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, term, trunc, step_info = env.step(action)
            done = term or trunc

            buffer.push(obs, action, reward, next_obs, term)
            agent.total_env_steps += 1
            obs = next_obs
            ep_raw_reward += float(step_info.get("step_raw_reward", reward))

            if buffer.can_sample(cfg.batch_size) and agent.total_env_steps >= cfg.warmup_steps:
                batch = buffer.sample(cfg.batch_size, device=str(agent.device))
                loss = agent.train_step(batch)
                ep_losses.append(loss)

        curriculum.record_episode(step_info, raw_reward=ep_raw_reward)

        training_metrics.append({
            "episode": ep,
            "level": curriculum.current_level,
            "raw_reward": ep_raw_reward,
            "mean_loss": float(np.mean(ep_losses)) if ep_losses else 0.0,
            "epsilon": float(agent.get_epsilon()),
            "termination": step_info.get("termination_reason"),
            "safe_landing": bool(step_info.get("safe_landing", False)),
        })

    train_time = time.time() - t0
    return agent, {
        "train_time_seconds": round(train_time, 2),
        "total_env_steps": agent.total_env_steps,
        "total_updates": agent.total_updates,
        "final_epsilon": float(agent.get_epsilon()),
        "final_level": curriculum.current_level,
        "curriculum_summary": curriculum.get_summary(),
        "episodes_logged": len(training_metrics),
    }


def run_experiment_e_demo_assisted(
    cfg: ExperimentConfig,
    demo_dataset: DemonstrationDataset,
) -> Tuple[DemonstrationDQNAgent, Dict[str, Any]]:
    """
    Experiment E: Demonstration-Assisted Double DQN.
    Combines Double DQN TD Loss + Supervised Large Margin Loss (lambda_margin=5.0, margin=0.2)
    + Mixed Replay (25% demo, 75% online) + Initial Demonstration Pre-training.
    """
    set_seed(cfg.seed)
    env = RoadsterEnv(use_shaping=True, gamma=cfg.gamma)

    agent_cfg = DemonstrationAgentConfig(
        warmup_steps=cfg.warmup_steps,
        batch_size=cfg.batch_size,
        buffer_size=cfg.buffer_size,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        target_update_frequency=cfg.target_update_frequency,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay,
        grad_clip=cfg.grad_clip,
        lambda_margin=cfg.lambda_margin,
        margin_value=cfg.margin_value,
        lambda_l2=cfg.lambda_l2,
        demo_ratio=cfg.demo_ratio,
    )
    agent = DemonstrationDQNAgent(config=agent_cfg)

    # Pretraining step: 3,000 updates using seed-specific RNG for reproducible variation
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)
    pretrain_steps = 3000
    for _ in range(pretrain_steps):
        raw_demo = demo_dataset.sample_batch(cfg.batch_size, rng=rng)
        demo_batch = {
            "states": torch.from_numpy(raw_demo["states"]).float().to(agent.device),
            "actions": torch.from_numpy(raw_demo["actions"]).long().to(agent.device),
            "rewards": torch.from_numpy(raw_demo["rewards"]).float().to(agent.device),
            "next_states": torch.from_numpy(raw_demo["next_states"]).float().to(agent.device),
            "dones": torch.from_numpy(raw_demo["dones"]).float().to(agent.device),
        }
        agent.pretrain_step(demo_batch)

    # Online fine-tuning with mixed replay
    online_buffer = ReplayBuffer(capacity=cfg.buffer_size, state_dim=agent_cfg.state_dim)
    demo_size = int(cfg.batch_size * cfg.demo_ratio)
    online_size = cfg.batch_size - demo_size

    training_metrics = []
    for ep in range(1, cfg.train_episodes + 1):
        ep_seed = cfg.seed * 1000 + ep
        obs, info = env.reset(seed=ep_seed, options={"level": 3})
        done = False
        ep_raw_reward = 0.0
        ep_losses = []

        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, shaped_reward, term, trunc, step_info = env.step(action)
            done = term or trunc

            online_buffer.push(obs, action, shaped_reward, next_obs, term)
            agent.total_env_steps += 1
            obs = next_obs
            ep_raw_reward += float(step_info.get("step_raw_reward", shaped_reward))

            if len(online_buffer) >= cfg.warmup_steps:
                # Sample mixed mini-batch
                raw_d = demo_dataset.sample_batch(demo_size, rng=rng)
                d_batch = {
                    "states": torch.from_numpy(raw_d["states"]).float().to(agent.device),
                    "actions": torch.from_numpy(raw_d["actions"]).long().to(agent.device),
                    "rewards": torch.from_numpy(raw_d["rewards"]).float().to(agent.device),
                    "next_states": torch.from_numpy(raw_d["next_states"]).float().to(agent.device),
                    "dones": torch.from_numpy(raw_d["dones"]).float().to(agent.device),
                }
                raw_o = online_buffer.sample(online_size, device=str(agent.device))
                o_batch = {
                    "states": raw_o[0],
                    "actions": raw_o[1],
                    "rewards": raw_o[2],
                    "next_states": raw_o[3],
                    "dones": raw_o[4],
                }
                step_loss = agent.mixed_train_step(d_batch, o_batch)
                ep_losses.append(step_loss.get("loss_total", 0.0))

        training_metrics.append({
            "episode": ep,
            "raw_reward": ep_raw_reward,
            "mean_loss": float(np.mean(ep_losses)) if ep_losses else 0.0,
            "epsilon": float(agent.get_epsilon()),
            "termination": step_info.get("termination_reason"),
            "safe_landing": bool(step_info.get("safe_landing", False)),
        })

    train_time = time.time() - t0
    return agent, {
        "train_time_seconds": round(train_time, 2),
        "pretrain_updates": pretrain_steps,
        "total_env_steps": agent.total_env_steps,
        "total_updates": agent.total_updates,
        "final_epsilon": float(agent.get_epsilon()),
        "episodes_logged": len(training_metrics),
    }


def execute_full_experiment_suite(
    seeds: List[int] = [42, 123, 999],
    train_episodes: int = 150,
    eval_episodes: int = 50,
    output_dir: str = "experiments",
    checkpoint_dir: str = "checkpoints/experiments",
) -> Dict[str, Any]:
    """
    Execute all 5 experiments across all seeds, log metrics, save checkpoints,
    export experiments/results.csv and experiments/summary.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 80)
    print("REPRODUCIBLE RL EXPERIMENT FRAMEWORK: 5 ALGORITHMS x 3 SEEDS = 15 RUNS")
    print(f"Algorithms: A (DQN), B (Double DQN), C (DDQN+Shaping), D (DDQN+Curriculum), E (Demo-Assisted DDQN)")
    print(f"Seeds: {seeds} | Training Episodes: {train_episodes} | Evaluation Episodes: {eval_episodes}")
    print(f"Timestamp: {datetime.datetime.now().isoformat()}")
    print("=" * 80)

    # Load demonstrations for Experiment E
    demo_dataset = DemonstrationDataset.load_jsonl("demonstrations.jsonl")

    all_configs = build_experiment_suite(seeds=seeds)
    # Override train_episodes and eval_episodes if specified
    for c in all_configs:
        c.train_episodes = train_episodes
        c.eval_episodes = eval_episodes
        c.checkpoint_dir = checkpoint_dir
        c.output_dir = output_dir

    run_records: List[Dict[str, Any]] = []
    code_version = get_code_version_hash()
    suite_start_time = time.time()

    total_runs = len(all_configs)
    for idx, cfg in enumerate(all_configs, 1):
        print(f"\n[{idx}/{total_runs}] Running {cfg.algorithm} (Seed {cfg.seed})...")
        t_start = time.time()

        if cfg.exp_id == "exp_a_dqn":
            agent, train_info = run_experiment_a_dqn(cfg)
        elif cfg.exp_id == "exp_b_double_dqn":
            agent, train_info = run_experiment_b_double_dqn(cfg)
        elif cfg.exp_id == "exp_c_ddqn_shaping":
            agent, train_info = run_experiment_c_ddqn_shaping(cfg)
        elif cfg.exp_id == "exp_d_ddqn_curriculum":
            agent, train_info = run_experiment_d_ddqn_curriculum(cfg)
        elif cfg.exp_id == "exp_e_demo_assisted":
            agent, train_info = run_experiment_e_demo_assisted(cfg, demo_dataset)
        else:
            raise ValueError(f"Unknown experiment ID: {cfg.exp_id}")

        # Save checkpoint
        ckpt_filename = f"{cfg.exp_id}_seed_{cfg.seed}.pth"
        ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)
        agent.save_checkpoint(ckpt_path)

        # Standardized independent evaluation on test seeds (20000..20049)
        agent.online_net.eval()
        eval_metrics = evaluate_policy_on_test_domain(
            agent=agent,
            num_episodes=cfg.eval_episodes,
            seed_base=cfg.eval_seed_base,
            level=cfg.eval_level,
        )

        elapsed = time.time() - t_start
        print(
            f"  -> Finished in {elapsed:.1f}s | "
            f"Success: {eval_metrics['success_rate']:.1f}% | "
            f"Crash: {eval_metrics['collision_rate']:.1f}% | "
            f"Reward: {eval_metrics['mean_reward']:.2f} | "
            f"Speed: {eval_metrics['mean_landing_velocity']:.2f}m/s | "
            f"Angle: {eval_metrics['mean_abs_landing_angle']:.1f}deg"
        )

        record = {
            "algorithm": cfg.algorithm,
            "exp_id": cfg.exp_id,
            "seed": cfg.seed,
            "config": cfg.to_dict(),
            "checkpoint_path": ckpt_path,
            "training_info": train_info,
            "eval_metrics": eval_metrics,
            "timestamp": datetime.datetime.now().isoformat(),
            "code_version_hash": code_version,
            "elapsed_seconds": round(elapsed, 2),
        }
        run_records.append(record)

    total_suite_time = time.time() - suite_start_time
    print("\n" + "=" * 80)
    print(f"ALL {total_runs} RUNS COMPLETED in {total_suite_time:.1f}s")
    print("=" * 80)

    # 1. Generate results.csv
    csv_path = os.path.join(output_dir, "results.csv")
    csv_columns = [
        "algorithm",
        "seed",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "mean_reward",
        "mean_landing_time",
        "mean_landing_velocity",
        "mean_abs_landing_angle",
        "mean_fuel_used",
    ]

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(csv_columns) + "\n")
        for rec in run_records:
            em = rec["eval_metrics"]
            row = [
                rec["algorithm"],
                str(rec["seed"]),
                f"{em['success_rate']:.2f}",
                f"{em['collision_rate']:.2f}",
                f"{em['timeout_rate']:.2f}",
                f"{em['mean_reward']:.2f}",
                f"{em['mean_landing_time']:.2f}",
                f"{em['mean_landing_velocity']:.3f}",
                f"{em['mean_abs_landing_angle']:.2f}",
                f"{em['mean_fuel_used']:.2f}",
            ]
            f.write(",".join(row) + "\n")

    print(f"\n[Artifact] Successfully written results CSV to: {csv_path}")

    # 2. Compute aggregated summary per algorithm across seeds
    algo_groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in run_records:
        algo = rec["algorithm"]
        if algo not in algo_groups:
            algo_groups[algo] = []
        algo_groups[algo].append(rec)

    aggregated_summary: Dict[str, Any] = {}
    for algo, runs in algo_groups.items():
        succ = [r["eval_metrics"]["success_rate"] for r in runs]
        crash = [r["eval_metrics"]["collision_rate"] for r in runs]
        timeout = [r["eval_metrics"]["timeout_rate"] for r in runs]
        rew = [r["eval_metrics"]["mean_reward"] for r in runs]
        ltime = [r["eval_metrics"]["mean_landing_time"] for r in runs]
        lspeed = [r["eval_metrics"]["mean_landing_velocity"] for r in runs]
        langle = [r["eval_metrics"]["mean_abs_landing_angle"] for r in runs]
        fuel = [r["eval_metrics"]["mean_fuel_used"] for r in runs]

        aggregated_summary[algo] = {
            "num_seeds": len(runs),
            "seeds": [r["seed"] for r in runs],
            "success_rate": {
                "mean": round(float(np.mean(succ)), 2),
                "std": round(float(np.std(succ, ddof=1)) if len(succ) > 1 else 0.0, 2),
            },
            "collision_rate": {
                "mean": round(float(np.mean(crash)), 2),
                "std": round(float(np.std(crash, ddof=1)) if len(crash) > 1 else 0.0, 2),
            },
            "timeout_rate": {
                "mean": round(float(np.mean(timeout)), 2),
                "std": round(float(np.std(timeout, ddof=1)) if len(timeout) > 1 else 0.0, 2),
            },
            "mean_reward": {
                "mean": round(float(np.mean(rew)), 2),
                "std": round(float(np.std(rew, ddof=1)) if len(rew) > 1 else 0.0, 2),
            },
            "mean_landing_time": {
                "mean": round(float(np.mean(ltime)), 2),
                "std": round(float(np.std(ltime, ddof=1)) if len(ltime) > 1 else 0.0, 2),
            },
            "mean_landing_velocity": {
                "mean": round(float(np.mean(lspeed)), 3),
                "std": round(float(np.std(lspeed, ddof=1)) if len(lspeed) > 1 else 0.0, 3),
            },
            "mean_abs_landing_angle": {
                "mean": round(float(np.mean(langle)), 2),
                "std": round(float(np.std(langle, ddof=1)) if len(langle) > 1 else 0.0, 2),
            },
            "mean_fuel_used": {
                "mean": round(float(np.mean(fuel)), 2),
                "std": round(float(np.std(fuel, ddof=1)) if len(fuel) > 1 else 0.0, 2),
            },
        }

    # 3. Generate summary.json
    summary_data = {
        "metadata": {
            "title": "Roadster 2D Flying Car RL Experiment Suite",
            "timestamp": datetime.datetime.now().isoformat(),
            "code_version_hash": code_version,
            "suite_duration_seconds": round(total_suite_time, 2),
            "total_runs": total_runs,
            "seeds_tested": seeds,
            "eval_episodes_per_run": eval_episodes,
            "eval_seed_base": 20000,
            "eval_domain_level": 3,
            "algorithms_evaluated": list(algo_groups.keys()),
        },
        "aggregate_results": aggregated_summary,
        "runs": run_records,
    }

    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"[Artifact] Successfully written summary JSON to: {json_path}")
    return summary_data


if __name__ == "__main__":
    execute_full_experiment_suite(
        seeds=[42, 123, 999],
        train_episodes=150,
        eval_episodes=50,
        output_dir="experiments",
        checkpoint_dir="checkpoints/experiments",
    )
