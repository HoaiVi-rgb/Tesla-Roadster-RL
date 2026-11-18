"""
Demonstration Generator for collecting safe expert landing trajectories.
"""

from __future__ import annotations
import math
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from environment.roadster_env import RoadsterEnv
from demonstration.teacher import HeuristicTeacherController
from demonstration.dataset import DemonstrationDataset, DemonstrationTransition


class DemonstrationGenerator:
    """
    Orchestrates the generation and filtering of expert demonstrations.

    Guarantees:
    - Initial state domain: h in [4, 9] m, x in [-2, 2] m, theta in [-15, +15] deg.
    - ONLY safe landing episodes are retained in the dataset.
    - Every transition stores state, action, reward, next_state, done, episode_id.
    """

    def __init__(
        self,
        env: Optional[RoadsterEnv] = None,
        teacher: Optional[HeuristicTeacherController] = None,
    ):
        self.env = env or RoadsterEnv(default_curriculum_level=3)
        self.teacher = teacher or HeuristicTeacherController()

    def generate(
        self,
        target_safe_episodes: int = 500,
        max_attempts: int = 2000,
        output_filepath: Optional[str] = "demonstrations.jsonl",
        seed_start: int = 42,
        verbose: bool = True,
    ) -> Tuple[DemonstrationDataset, Dict[str, Any]]:
        """
        Generate demonstration dataset containing exactly `target_safe_episodes` safe landing trajectories.

        Parameters
        ----------
        target_safe_episodes : number of safe episodes to collect (e.g. 500)
        max_attempts : maximum total episodes to attempt before aborting
        output_filepath : optional path to save dataset as .jsonl
        seed_start : starting random seed
        verbose : print progress updates

        Returns
        -------
        (dataset, statistics_dict)
        """
        dataset = DemonstrationDataset()

        safe_count = 0
        total_attempts = 0

        landing_times: List[float] = []
        landing_speeds_vy: List[float] = []
        landing_speeds_vx: List[float] = []
        landing_speeds_total: List[float] = []
        landing_angles_deg: List[float] = []
        landing_positions_x: List[float] = []
        target_zone_count: int = 0

        start_wall_time = time.time()

        while safe_count < target_safe_episodes and total_attempts < max_attempts:
            current_seed = seed_start + total_attempts
            obs, info = self.env.reset(seed=current_seed, options={"level": 3})

            episode_transitions: List[DemonstrationTransition] = []
            done = False
            step_idx = 0

            while not done:
                action = self.teacher.select_action(obs=obs, info=info)
                next_obs, reward, terminated, truncated, next_info = self.env.step(action)
                done = terminated or truncated

                # Record transition
                transition = DemonstrationTransition(
                    state=obs.tolist(),
                    action=int(action),
                    reward=float(reward),
                    raw_reward=float(next_info.get("raw_reward", reward)),
                    next_state=next_obs.tolist(),
                    done=bool(done),
                    episode_id=int(safe_count),  # 0-indexed safe episode ID
                    step_idx=int(step_idx),
                    info={
                        "x": next_info["x"],
                        "h": next_info["h"],
                        "vx": next_info["vx"],
                        "vy": next_info["vy"],
                        "theta": next_info["theta"],
                        "omega": next_info["omega"],
                        "time": next_info["time"],
                        "fuel": next_info["fuel"],
                    },
                )
                episode_transitions.append(transition)
                obs = next_obs
                info = next_info
                step_idx += 1

            total_attempts += 1

            # FILTER: ONLY KEEP SAFE LANDING EPISODES
            if info["safe_landing"]:
                dataset.extend(episode_transitions)
                safe_count += 1

                landing_times.append(info["time"])
                landing_speeds_vy.append(info["landing_speed_vy"])
                landing_speeds_vx.append(info["landing_speed_vx"])
                landing_speeds_total.append(info["landing_speed"])
                landing_angles_deg.append(abs(info["landing_angle_deg"]))
                landing_positions_x.append(info["landing_position"])
                if info["in_target_zone"]:
                    target_zone_count += 1

                if verbose and (safe_count % 50 == 0 or safe_count == target_safe_episodes):
                    elapsed = time.time() - start_wall_time
                    success_rate = (safe_count / total_attempts) * 100.0
                    print(
                        f"Collected {safe_count}/{target_safe_episodes} safe episodes "
                        f"({dataset.num_transitions} transitions) | "
                        f"Success rate: {success_rate:.1f}% | Elapsed: {elapsed:.1f}s"
                    )

        elapsed_total = time.time() - start_wall_time
        success_rate = (safe_count / total_attempts) * 100.0 if total_attempts > 0 else 0.0

        stats: Dict[str, Any] = {
            "target_safe_episodes": target_safe_episodes,
            "episodes_collected": safe_count,
            "total_attempts": total_attempts,
            "success_rate_percent": float(success_rate),
            "num_transitions": dataset.num_transitions,
            "avg_transitions_per_episode": float(dataset.num_transitions / max(1, safe_count)),
            "avg_landing_time_s": float(np.mean(landing_times)) if landing_times else 0.0,
            "max_landing_time_s": float(np.max(landing_times)) if landing_times else 0.0,
            "avg_landing_speed_vy_ms": float(np.mean(landing_speeds_vy)) if landing_speeds_vy else 0.0,
            "avg_landing_speed_vx_ms": float(np.mean(landing_speeds_vx)) if landing_speeds_vx else 0.0,
            "avg_landing_speed_total_ms": float(np.mean(landing_speeds_total)) if landing_speeds_total else 0.0,
            "avg_landing_angle_deg": float(np.mean(landing_angles_deg)) if landing_angles_deg else 0.0,
            "avg_landing_position_x_m": float(np.mean(landing_positions_x)) if landing_positions_x else 0.0,
            "target_zone_success_rate_percent": float((target_zone_count / max(1, safe_count)) * 100.0),
            "generation_wall_time_s": float(elapsed_total),
        }

        if output_filepath:
            dataset.save_jsonl(output_filepath)
            if verbose:
                print(f"Successfully saved {dataset.num_transitions} transitions to {output_filepath}")

        return dataset, stats
