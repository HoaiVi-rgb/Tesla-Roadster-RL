"""
Configuration definitions for the Reproducible RL Experiment Suite.
Covers:
  - Experiment A: Standard DQN (Tabula Rasa)
  - Experiment B: Double DQN (Tabula Rasa)
  - Experiment C: Double DQN + Potential-Based Reward Shaping (PBRS)
  - Experiment D: Double DQN + Progressive Curriculum Learning
  - Experiment E: Demonstration-Assisted Double DQN (Guided DDQN + Margin Loss)
"""

from __future__ import annotations
import os
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class ExperimentConfig:
    """
    Complete configuration for an individual experiment seed run.
    Serialized directly to summary.json for end-to-end reproducibility.
    """
    # Identifiers
    exp_id: str                          # e.g., "exp_a_dqn"
    algorithm: str                       # e.g., "Standard DQN"
    seed: int                            # e.g., 42, 123, 999

    # Training Horizon & Sizing
    train_episodes: int = 150            # Total online training episodes
    eval_episodes: int = 50              # Evaluation episodes per benchmark
    eval_seed_base: int = 20000          # Independent evaluation seed start
    eval_level: int = 3                  # Independent test domain (Level 3 Full Challenge)

    # Optimization Hyperparameters
    learning_rate: float = 0.0005        # Adam learning rate
    gamma: float = 0.995                 # Discount factor
    batch_size: int = 32                 # Mini-batch size
    buffer_size: int = 10000             # Experience replay capacity
    warmup_steps: int = 100              # Replay buffer warmup before updates
    target_update_frequency: int = 100   # Steps between target net syncs
    grad_clip: float = 10.0              # L2 gradient norm clipping threshold

    # Exploration Schedule
    epsilon_start: float = 1.0           # Starting epsilon
    epsilon_end: float = 0.02            # Minimum epsilon floor
    epsilon_decay: int = 6000            # Epsilon linear/exponential decay horizon

    # Algorithmic Modifiers
    double_dqn: bool = False             # Whether to use decoupled Double DQN target
    use_shaping: bool = False            # Potential-Based Reward Shaping (PBRS)
    use_curriculum: bool = False         # Progressive Curriculum (L1 -> L2 -> L3)

    # Demonstration Guidance (Experiment E)
    use_demonstrations: bool = False     # Demonstration-Assisted mode
    demo_ratio: float = 0.25             # Demo sample ratio in mixed replay (25% demo, 75% online)
    lambda_margin: float = 5.0           # Weight for Supervised Large Margin Loss
    margin_value: float = 0.2            # Separation margin m
    lambda_l2: float = 1e-5              # Weight for L2 parameter regularization
    pretrained_checkpoint: Optional[str] = "checkpoints/dqfd/best_pretrained.pth"

    # Storage paths
    checkpoint_dir: str = "checkpoints/experiments"
    output_dir: str = "experiments"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_code_version_hash() -> str:
    """
    Generate deterministic hash of the codebase core source files
    to satisfy reproducibility requirements when not inside a git repository.
    """
    core_dirs = ["physics", "environment", "rl", "training", "demonstration"]
    hasher = hashlib.sha256()
    for cdir in core_dirs:
        if os.path.exists(cdir):
            for root, _, files in sorted(os.walk(cdir)):
                for f in sorted(files):
                    if f.endswith(".py"):
                        fpath = os.path.join(root, f)
                        try:
                            with open(fpath, "rb") as fp:
                                hasher.update(fp.read())
                        except Exception:
                            pass
    return hasher.hexdigest()[:16]


def build_experiment_suite(seeds: List[int] = [42, 123, 999]) -> List[ExperimentConfig]:
    """
    Construct the full matrix of 5 algorithms x N seeds = 15 runs.
    """
    configs: List[ExperimentConfig] = []

    for seed in seeds:
        # Experiment A: Standard DQN (Tabula Rasa)
        configs.append(ExperimentConfig(
            exp_id="exp_a_dqn",
            algorithm="Standard DQN",
            seed=seed,
            double_dqn=False,
            use_shaping=False,
            use_curriculum=False,
            use_demonstrations=False,
            epsilon_start=1.0,
            epsilon_end=0.02,
        ))

        # Experiment B: Double DQN (Tabula Rasa)
        configs.append(ExperimentConfig(
            exp_id="exp_b_double_dqn",
            algorithm="Double DQN",
            seed=seed,
            double_dqn=True,
            use_shaping=False,
            use_curriculum=False,
            use_demonstrations=False,
            epsilon_start=1.0,
            epsilon_end=0.02,
        ))

        # Experiment C: Double DQN + Reward Shaping
        configs.append(ExperimentConfig(
            exp_id="exp_c_ddqn_shaping",
            algorithm="Double DQN + Shaping",
            seed=seed,
            double_dqn=True,
            use_shaping=True,
            use_curriculum=False,
            use_demonstrations=False,
            epsilon_start=1.0,
            epsilon_end=0.02,
        ))

        # Experiment D: Double DQN + Curriculum
        configs.append(ExperimentConfig(
            exp_id="exp_d_ddqn_curriculum",
            algorithm="Double DQN + Curriculum",
            seed=seed,
            double_dqn=True,
            use_shaping=False,
            use_curriculum=True,
            use_demonstrations=False,
            epsilon_start=1.0,
            epsilon_end=0.02,
        ))

        # Experiment E: Demonstration-Assisted Double DQN
        configs.append(ExperimentConfig(
            exp_id="exp_e_demo_assisted",
            algorithm="Demonstration-Assisted DDQN",
            seed=seed,
            double_dqn=True,
            use_shaping=True,
            use_curriculum=False,
            use_demonstrations=True,
            pretrained_checkpoint="checkpoints/dqfd/best_pretrained.pth",
            epsilon_start=0.15,
            epsilon_end=0.02,
        ))

    return configs
