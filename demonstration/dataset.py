"""
Demonstration dataset storage, serialization (JSONL), and batch sampling.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Iterator
import numpy as np


@dataclass(frozen=True)
class DemonstrationTransition:
    """A single state-action-reward transition in an expert demonstration."""
    state: List[float]
    action: int
    reward: float
    raw_reward: float
    next_state: List[float]
    done: bool
    episode_id: int
    step_idx: int
    info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "raw_reward": self.raw_reward,
            "next_state": self.next_state,
            "done": self.done,
            "episode_id": self.episode_id,
            "step_idx": self.step_idx,
            "info": self.info or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DemonstrationTransition:
        return cls(
            state=[float(x) for x in data["state"]],
            action=int(data["action"]),
            reward=float(data["reward"]),
            raw_reward=float(data.get("raw_reward", data["reward"])),
            next_state=[float(x) for x in data["next_state"]],
            done=bool(data["done"]),
            episode_id=int(data["episode_id"]),
            step_idx=int(data.get("step_idx", 0)),
            info=data.get("info", {}),
        )


class DemonstrationDataset:
    """
    Dataset container for storing, loading, and sampling expert demonstrations.
    Compatible with JSONL format for streaming and memory efficiency.
    """

    def __init__(self, transitions: Optional[List[DemonstrationTransition]] = None):
        self.transitions: List[DemonstrationTransition] = transitions or []

    def __len__(self) -> int:
        return len(self.transitions)

    def __iter__(self) -> Iterator[DemonstrationTransition]:
        return iter(self.transitions)

    def add(self, transition: DemonstrationTransition) -> None:
        self.transitions.append(transition)

    def extend(self, transitions: List[DemonstrationTransition]) -> None:
        self.transitions.extend(transitions)

    @property
    def num_transitions(self) -> int:
        return len(self.transitions)

    @property
    def num_episodes(self) -> int:
        if not self.transitions:
            return 0
        return len(set(t.episode_id for t in self.transitions))

    def save_jsonl(self, filepath: str) -> None:
        """Save dataset to a JSONL file (one JSON object per line)."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for t in self.transitions:
                f.write(json.dumps(t.to_dict()) + "\n")

    @classmethod
    def load_jsonl(cls, filepath: str) -> DemonstrationDataset:
        """Load dataset from a JSONL file."""
        transitions: List[DemonstrationTransition] = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    transitions.append(DemonstrationTransition.from_dict(data))
                except Exception as e:
                    raise ValueError(f"Error parsing line {line_idx + 1} of {filepath}: {e}")
        return cls(transitions=transitions)

    def sample_batch(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
        """
        Sample a random batch of transitions as numpy arrays for training.
        """
        if len(self.transitions) == 0:
            raise ValueError("Cannot sample from an empty dataset")

        rng = rng or np.random.default_rng()
        indices = rng.choice(len(self.transitions), size=batch_size, replace=(batch_size > len(self.transitions)))

        batch_states = [self.transitions[i].state for i in indices]
        batch_actions = [self.transitions[i].action for i in indices]
        batch_rewards = [self.transitions[i].reward for i in indices]
        batch_raw_rewards = [self.transitions[i].raw_reward for i in indices]
        batch_next_states = [self.transitions[i].next_state for i in indices]
        batch_dones = [self.transitions[i].done for i in indices]

        return {
            "states": np.array(batch_states, dtype=np.float32),
            "actions": np.array(batch_actions, dtype=np.int64),
            "rewards": np.array(batch_rewards, dtype=np.float32),
            "raw_rewards": np.array(batch_raw_rewards, dtype=np.float32),
            "next_states": np.array(batch_next_states, dtype=np.float32),
            "dones": np.array(batch_dones, dtype=np.bool_),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Compute statistical summary of dataset transitions."""
        if not self.transitions:
            return {"num_episodes": 0, "num_transitions": 0}

        rewards = [t.reward for t in self.transitions]
        raw_rewards = [t.raw_reward for t in self.transitions]
        actions = [t.action for t in self.transitions]

        return {
            "num_episodes": self.num_episodes,
            "num_transitions": self.num_transitions,
            "mean_reward": float(np.mean(rewards)),
            "mean_raw_reward": float(np.mean(raw_rewards)),
            "action_distribution": {int(a): int(count) for a, count in zip(*np.unique(actions, return_counts=True))},
        }
