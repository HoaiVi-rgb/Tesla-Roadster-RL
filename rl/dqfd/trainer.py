"""
Guided DQfD Trainer.
Coordinates Stage 1 (Guided Pre-training on Demonstration Dataset)
and Stage 2 (Curriculum-guided Interactive RL Fine-Tuning).
"""

from __future__ import annotations
import math
import os
import sys
import json
import time
from typing import Dict, Any, Optional, List
import numpy as np
import torch

from environment.roadster_env import RoadsterEnv
from demonstration.dataset import DemonstrationDataset
from rl.dqn.replay_buffer import ReplayBuffer
from rl.dqfd.agent import DQfDAgent, DQfDConfig
from training.curriculum_manager import CurriculumManager


def export_weights_to_json(agent: DQfDAgent, output_filepath: str) -> None:
    """
    Export neural network weights and biases to a clean JSON file
    for client-side JavaScript execution in HTML5 Canvas.
    """
    state_dict = agent.online_net.state_dict()
    export_dict = {
        "model_architecture": "9 -> 64 -> 64 -> 16",
        "activation": "relu",
        "layers": {
            "fc1_weight": state_dict["fc1.weight"].cpu().numpy().tolist(),  # (64, 9)
            "fc1_bias": state_dict["fc1.bias"].cpu().numpy().tolist(),      # (64,)
            "fc2_weight": state_dict["fc2.weight"].cpu().numpy().tolist(),  # (64, 64)
            "fc2_bias": state_dict["fc2.bias"].cpu().numpy().tolist(),      # (64,)
            "fc3_weight": state_dict["fc3.weight"].cpu().numpy().tolist(),  # (16, 64)
            "fc3_bias": state_dict["fc3.bias"].cpu().numpy().tolist(),      # (16,)
        },
        "normalization": {
            "x": 10.0,
            "h": 15.0,
            "vx": 10.0,
            "vy": 10.0,
            "omega": 5.0,
            "fuel": 100.0,
            "t": 20.0,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(export_dict, f, indent=2)
    print(f"[Exporter] Successfully exported neural network weights to: {output_filepath}")


def evaluate_policy(
    agent: DQfDAgent,
    num_episodes: int = 20,
    seed_base: int = 10000,
    level: int = 3,
) -> Dict[str, float]:
    """
    Evaluate agent greedily (epsilon = 0.0) on fixed deterministic seeds.
    Uses unshaped raw environment rewards for objective measurement.
    """
    env = RoadsterEnv(use_shaping=False)
    successes = 0
    crashes = 0
    timeouts = 0
    oobs = 0
    raw_rewards = []
    landing_speeds = []
    landing_angles = []
    landing_times = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed_base + ep, options={"level": level})
        ep_raw_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_raw_reward += float(info.get("step_raw_reward", reward))

        raw_rewards.append(ep_raw_reward)
        term_type = info.get("termination_reason", info.get("termination_type", ""))
        is_safe = (term_type == "SAFE_LANDING" or info.get("safe_landing", False))
        if is_safe:
            successes += 1
            landing_speeds.append(float(info.get("landing_speed", 0.0)))
            landing_angles.append(float(info.get("landing_angle_deg", 0.0)))
            landing_times.append(float(info.get("time", 0.0)))
        elif term_type == "CRASH" or info.get("crash", False):
            crashes += 1
        elif term_type == "TIMEOUT" or info.get("timeout", False):
            timeouts += 1
        elif term_type == "OUT_OF_BOUNDS" or info.get("out_of_bounds", False):
            oobs += 1

    return {
        "success_rate": successes / float(num_episodes),
        "crash_rate": crashes / float(num_episodes),
        "timeout_rate": timeouts / float(num_episodes),
        "oob_rate": oobs / float(num_episodes),
        "mean_raw_reward": float(np.mean(raw_rewards)) if raw_rewards else 0.0,
        "mean_landing_speed": float(np.mean(landing_speeds)) if landing_speeds else 0.0,
        "mean_landing_angle": float(np.mean(landing_angles)) if landing_angles else 0.0,
        "mean_landing_time": float(np.mean(landing_times)) if landing_times else 0.0,
    }


class DQfDTrainer:
    """
    Full DQfD training orchestrator:
    - Pretrains on demonstration transitions with Large Margin Loss
    - Fine-tunes interactively with curriculum and mixed mini-batches
    """

    def __init__(
        self,
        config: Optional[DQfDConfig] = None,
        demo_filepath: str = "demonstrations.jsonl",
        checkpoint_dir: str = "checkpoints/dqfd",
    ):
        self.config = config or DQfDConfig()
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.agent = DQfDAgent(config=self.config)

        print(f"[DQfD Trainer] Loading demonstration dataset from: {demo_filepath}")
        self.demo_dataset = DemonstrationDataset.load_jsonl(demo_filepath)
        print(f"[DQfD Trainer] Loaded {len(self.demo_dataset)} transitions from {self.demo_dataset.num_episodes} episodes.")

        self.online_buffer = ReplayBuffer(
            capacity=self.config.buffer_size,
            state_dim=self.config.state_dim,
        )
        self.curriculum = CurriculumManager(starting_level=1)
        self.env = RoadsterEnv(use_shaping=True, gamma=self.config.gamma)
        self.rng = np.random.default_rng(42)

    def run_pretraining(
        self,
        num_updates: int = 30000,
        eval_interval: int = 2500,
        eval_episodes: int = 20,
    ) -> DQfDAgent:
        """
        Stage 1: Pre-training on demonstration dataset.
        """
        print("\n" + "=" * 70)
        print(f"STAGE 1: GUIDED PRE-TRAINING ({num_updates} updates, Margin={self.config.margin_value}, Lambda_E={self.config.lambda_margin})")
        print("=" * 70)

        start_time = time.time()
        best_success_rate = -1.0

        for update_step in range(1, num_updates + 1):
            raw_batch = self.demo_dataset.sample_batch(self.config.batch_size, rng=self.rng)
            batch = {
                "states": torch.from_numpy(raw_batch["states"]).float().to(self.agent.device),
                "actions": torch.from_numpy(raw_batch["actions"]).long().to(self.agent.device),
                "rewards": torch.from_numpy(raw_batch["rewards"]).float().to(self.agent.device),
                "next_states": torch.from_numpy(raw_batch["next_states"]).float().to(self.agent.device),
                "dones": torch.from_numpy(raw_batch["dones"]).float().to(self.agent.device),
            }

            metrics = self.agent.pretrain_step(batch)

            if update_step % 1000 == 0:
                elapsed = time.time() - start_time
                fps = update_step / elapsed
                print(
                    f"[Pretrain {update_step:05d}/{num_updates}] "
                    f"Total Loss: {metrics['loss_total']:.4f} | "
                    f"TD: {metrics['loss_td']:.4f} | "
                    f"Margin: {metrics['loss_margin']:.4f} | "
                    f"Q_mean: {metrics['q_mean']:.2f} | "
                    f"{fps:.1f} upd/s"
                )

            if update_step % eval_interval == 0 or update_step == num_updates:
                eval_metrics = evaluate_policy(self.agent, num_episodes=eval_episodes, level=3)
                print(
                    f"  >>> EVAL @ Step {update_step:05d}: "
                    f"Success: {eval_metrics['success_rate']*100:.1f}% | "
                    f"Raw Reward: {eval_metrics['mean_raw_reward']:.1f} | "
                    f"Landing Speed: {eval_metrics['mean_landing_speed']:.2f} m/s | "
                    f"Angle: {eval_metrics['mean_landing_angle']:.1f} deg | "
                    f"Time: {eval_metrics['mean_landing_time']:.2f} s"
                )

                if eval_metrics["success_rate"] >= best_success_rate:
                    best_success_rate = eval_metrics["success_rate"]
                    best_path = os.path.join(self.checkpoint_dir, "best_pretrained.pth")
                    self.agent.save_checkpoint(best_path)
                    print(f"  >>> [Checkpoint] Saved new best model ({best_success_rate*100:.1f}%) to {best_path}")

        print("\n[DQfD Trainer] Pretraining finished.")
        return self.agent

    def run_finetuning(
        self,
        num_episodes: int = 500,
        eval_interval: int = 50,
    ) -> DQfDAgent:
        """
        Stage 2: Fine-tuning interactively with mixed demonstration replay and curriculum.
        """
        print("\n" + "=" * 70)
        print(f"STAGE 2: CURRICULUM RL FINE-TUNING ({num_episodes} episodes)")
        print("=" * 70)

        demo_size = int(self.config.batch_size * self.config.demo_batch_ratio)
        online_size = self.config.batch_size - demo_size

        best_success_rate = 0.0

        for episode in range(1, num_episodes + 1):
            reset_opts = self.curriculum.get_reset_options()
            obs, info = self.env.reset(options=reset_opts)
            ep_reward = 0.0
            done = False

            while not done:
                action = self.agent.select_action(obs, evaluate=False)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.online_buffer.push(obs, action, reward, next_obs, done)
                self.agent.total_env_steps += 1
                ep_reward += reward
                obs = next_obs

                # Training step if enough online samples
                if len(self.online_buffer) >= self.config.warmup_steps:
                    # Sample demo batch
                    raw_demo = self.demo_dataset.sample_batch(demo_size, rng=self.rng)
                    demo_batch = {
                        "states": torch.from_numpy(raw_demo["states"]).float().to(self.agent.device),
                        "actions": torch.from_numpy(raw_demo["actions"]).long().to(self.agent.device),
                        "rewards": torch.from_numpy(raw_demo["rewards"]).float().to(self.agent.device),
                        "next_states": torch.from_numpy(raw_demo["next_states"]).float().to(self.agent.device),
                        "dones": torch.from_numpy(raw_demo["dones"]).float().to(self.agent.device),
                    }
                    # Sample online batch
                    raw_online = self.online_buffer.sample(online_size, device=str(self.agent.device))
                    online_batch = {
                        "states": raw_online[0],
                        "actions": raw_online[1],
                        "rewards": raw_online[2],
                        "next_states": raw_online[3],
                        "dones": raw_online[4],
                    }
                    self.agent.mixed_train_step(demo_batch, online_batch)

            is_safe = (info.get("termination_type") == "SAFE_LANDING")
            raw_rew = float(info.get("step_raw_reward", 0.0))
            promoted = self.curriculum.record_episode(success=is_safe, raw_reward=raw_rew)

            if promoted:
                print(f"[Curriculum] >>> PROMOTED TO LEVEL {self.curriculum.current_level} ({self.curriculum.current_spec.name})!")

            if episode % 10 == 0:
                c_status = self.curriculum.get_status()
                print(
                    f"[Finetune Ep {episode:04d}/{num_episodes}] "
                    f"L{c_status['current_level']} | "
                    f"WinSucc: {c_status['rolling_success_rate']*100:.1f}% | "
                    f"Last: {info.get('termination_type')} | "
                    f"Speed: {info.get('landing_speed', 0.0):.2f}m/s | "
                    f"Time: {info.get('time', 0.0):.2f}s | "
                    f"Eps: {self.agent.get_epsilon():.3f}"
                )

            if episode % eval_interval == 0 or episode == num_episodes:
                eval_metrics = evaluate_policy(self.agent, num_episodes=20, level=3)
                print(
                    f"  >>> EVAL @ Ep {episode}: "
                    f"Success: {eval_metrics['success_rate']*100:.1f}% | "
                    f"Reward: {eval_metrics['mean_raw_reward']:.1f} | "
                    f"Speed: {eval_metrics['mean_landing_speed']:.2f} m/s | "
                    f"Angle: {eval_metrics['mean_landing_angle']:.1f} deg"
                )
                if eval_metrics["success_rate"] >= best_success_rate:
                    best_success_rate = eval_metrics["success_rate"]
                    best_path = os.path.join(self.checkpoint_dir, "best_finetuned.pth")
                    self.agent.save_checkpoint(best_path)

        return self.agent
