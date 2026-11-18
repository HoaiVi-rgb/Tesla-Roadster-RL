"""
DQN Trainer orchestrating the training loop, replay buffer interaction,
logging metrics, evaluation, and checkpointing.
STRICTLY RUNS INDEPENDENTLY OF UI. NO TEACHER USED.
"""

from __future__ import annotations
import os
import time
from collections import deque
from typing import Dict, Any, Optional, List, Callable
import numpy as np

from environment.roadster_env import RoadsterEnv
from rl.dqn.agent import DQNAgent, DQNConfig
from rl.dqn.replay_buffer import ReplayBuffer


class DQNTrainer:
    """
    Headless Trainer for the Baseline Deep Q-Network.
    """

    def __init__(
        self,
        env: Optional[RoadsterEnv] = None,
        agent: Optional[DQNAgent] = None,
        buffer: Optional[ReplayBuffer] = None,
        checkpoint_dir: str = "checkpoints/dqn_baseline",
    ):
        self.env = env or RoadsterEnv(default_curriculum_level=3)
        self.agent = agent or DQNAgent()
        self.buffer = buffer or ReplayBuffer(
            capacity=self.agent.config.buffer_size,
            state_dim=self.agent.config.state_dim,
        )
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Telemetry history for rolling statistics
        self.recent_successes: deque = deque(maxlen=50)
        self.recent_rewards: deque = deque(maxlen=50)
        self.recent_raw_rewards: deque = deque(maxlen=50)

    def train(
        self,
        num_episodes: int = 1000,
        eval_interval: int = 100,
        eval_episodes: int = 20,
        log_interval: int = 10,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run the complete DQN training loop.

        Parameters
        ----------
        num_episodes : total number of episodes to train
        eval_interval : episodes between evaluation checkpoints
        eval_episodes : number of deterministic evaluation runs
        log_interval : episodes between console log printouts
        callback : optional function called with episode metrics dict
        verbose : print progress to stdout

        Returns
        -------
        List of episode log dictionaries.
        """
        training_logs: List[Dict[str, Any]] = []
        best_eval_success_rate = -1.0
        start_wall_time = time.time()

        if verbose:
            print(f"=== STARTING BASELINE DQN TRAINING ({num_episodes} EPISODES) ===")
            print(f"Warmup steps: {self.agent.config.warmup_steps} | Batch size: {self.agent.config.batch_size}")
            print(f"LR: {self.agent.config.learning_rate} | Gamma: {self.agent.config.gamma} | Target Sync: {self.agent.config.target_update_frequency}")
            print("=" * 70)

        for ep in range(1, num_episodes + 1):
            obs, info = self.env.reset(seed=ep, options={"level": 3})

            ep_reward = 0.0
            ep_raw_reward = 0.0
            ep_losses: List[float] = []
            done = False

            while not done:
                # 1. Select action using epsilon-greedy (NO TEACHER)
                action = self.agent.select_action(obs, evaluate=False)

                # 2. Step environment
                next_obs, reward, terminated, truncated, next_info = self.env.step(action)
                done = terminated or truncated

                # 3. Store transition in replay buffer
                self.buffer.push(
                    state=obs,
                    action=action,
                    reward=reward,
                    next_state=next_obs,
                    done=terminated,  # Terminal state (not truncated timeout)
                )

                self.agent.total_env_steps += 1
                obs = next_obs
                info = next_info
                ep_reward += reward
                ep_raw_reward += next_info.get("raw_reward", 0.0)

                # 4. Train online network if warmup threshold reached
                if self.buffer.can_sample(self.agent.config.batch_size) and (
                    self.agent.total_env_steps >= self.agent.config.warmup_steps
                ):
                    batch = self.buffer.sample(
                        batch_size=self.agent.config.batch_size,
                        device=str(self.agent.device),
                    )
                    loss = self.agent.train_step(batch)
                    ep_losses.append(loss)

            # Record episode outcomes
            is_safe = bool(info["safe_landing"])
            self.recent_successes.append(1.0 if is_safe else 0.0)
            self.recent_rewards.append(ep_reward)
            self.recent_raw_rewards.append(ep_raw_reward)

            mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
            success_rate = float(np.mean(self.recent_successes)) * 100.0

            log_entry: Dict[str, Any] = {
                "episode": int(ep),
                "total_steps": int(self.agent.total_env_steps),
                "total_updates": int(self.agent.total_updates),
                "reward": float(ep_reward),
                "raw_reward": float(ep_raw_reward),
                "loss": float(mean_loss),
                "epsilon": float(self.agent.get_epsilon()),
                "success_rate": float(success_rate),
                "landing_velocity": float(info["landing_speed_vy"]),
                "landing_angle": float(abs(info["landing_angle_deg"])),
                "fuel_used": float(100.0 - info["fuel"]),
                "landing_time": float(info["time"]),
                "termination_reason": str(info["termination_reason"]),
                "safe_landing": bool(is_safe),
            }
            training_logs.append(log_entry)

            if callback:
                callback(log_entry)

            # Console logging
            if verbose and (ep % log_interval == 0 or ep == num_episodes):
                elapsed = time.time() - start_wall_time
                print(
                    f"Ep {ep:4d}/{num_episodes} | "
                    f"Steps: {self.agent.total_env_steps:6d} | "
                    f"Rew: {ep_reward:6.2f} (Raw: {ep_raw_reward:7.1f}) | "
                    f"Loss: {mean_loss:7.4f} | "
                    f"Eps: {self.agent.get_epsilon():.3f} | "
                    f"Win50: {success_rate:5.1f}% | "
                    f"T: {info['time']:4.1f}s | "
                    f"Elapsed: {elapsed:5.1f}s"
                )

            # Periodic deterministic evaluation
            if ep % eval_interval == 0 or ep == num_episodes:
                eval_metrics = self.evaluate(num_episodes=eval_episodes, seed_base=10000 + ep)
                if verbose:
                    print(
                        f"--> [EVAL Ep {ep}] Success: {eval_metrics['eval_success_rate']:.1f}% | "
                        f"Avg Rew: {eval_metrics['eval_avg_raw_reward']:.1f} | "
                        f"Avg Vy: {eval_metrics['eval_avg_vy']:.2f} m/s | "
                        f"Avg Time: {eval_metrics['eval_avg_time']:.2f}s"
                    )

                # Save latest and best model checkpoints
                latest_path = os.path.join(self.checkpoint_dir, "dqn_latest.pth")
                self.agent.save_checkpoint(latest_path)

                if eval_metrics["eval_success_rate"] >= best_eval_success_rate:
                    best_eval_success_rate = eval_metrics["eval_success_rate"]
                    best_path = os.path.join(self.checkpoint_dir, "dqn_best.pth")
                    self.agent.save_checkpoint(best_path)
                    # Also export web weights
                    web_weights_path = os.path.join(self.checkpoint_dir, "weights_best.json")
                    self.agent.export_weights_json(web_weights_path)

        return training_logs

    def evaluate(self, num_episodes: int = 20, seed_base: int = 20000) -> Dict[str, float]:
        """
        Evaluate agent deterministically (epsilon = 0.0) across fixed test seeds.
        """
        eval_successes = []
        eval_raw_rewards = []
        eval_vys = []
        eval_vxs = []
        eval_angles = []
        eval_times = []

        for i in range(num_episodes):
            eval_seed = seed_base + i
            obs, info = self.env.reset(seed=eval_seed, options={"level": 3})
            ep_raw_rew = 0.0
            done = False

            while not done:
                action = self.agent.select_action(obs, evaluate=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                ep_raw_rew += info.get("raw_reward", 0.0)

            eval_successes.append(1.0 if info["safe_landing"] else 0.0)
            eval_raw_rewards.append(ep_raw_rew)
            eval_vys.append(info["landing_speed_vy"])
            eval_vxs.append(info["landing_speed_vx"])
            eval_angles.append(abs(info["landing_angle_deg"]))
            eval_times.append(info["time"])

        return {
            "eval_success_rate": float(np.mean(eval_successes)) * 100.0,
            "eval_avg_raw_reward": float(np.mean(eval_raw_rewards)),
            "eval_avg_vy": float(np.mean(eval_vys)),
            "eval_avg_vx": float(np.mean(eval_vxs)),
            "eval_avg_angle_deg": float(np.mean(eval_angles)),
            "eval_avg_time": float(np.mean(eval_times)),
        }
