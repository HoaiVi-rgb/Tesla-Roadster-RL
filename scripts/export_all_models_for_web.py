"""
Export PyTorch Checkpoints to JSON for Client-Side JavaScript Neural Network Inference.
Exports:
  1. Standard DQN Baseline (exp_a_dqn_seed_42.pth) -> web/weights/dqn_weights.json
  2. Double DQN Baseline (exp_b_double_dqn_seed_42.pth) -> web/weights/ddqn_weights.json
  3. Demo-Assisted Double DQN (best_model.pth) -> web/weights/trained_weights.json
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import torch

from rl.dqn.agent import DQNAgent
from rl.double_dqn.agent import DoubleDQNAgent
from rl.demonstration_agent import DemonstrationDQNAgent


def export_agent_to_json(agent, output_filepath: str, model_name: str):
    state_dict = agent.online_net.state_dict()
    export_dict = {
        "model_name": model_name,
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
    print(f"Exported {model_name} to: {output_filepath}")


def main():
    # 1. Export Standard DQN Baseline
    dqn_agent = DQNAgent()
    dqn_path = "checkpoints/experiments/exp_a_dqn_seed_42.pth"
    if os.path.exists(dqn_path):
        dqn_agent.load_checkpoint(dqn_path)
        export_agent_to_json(dqn_agent, "web/weights/dqn_weights.json", "Standard DQN (Tabula Rasa)")

    # 2. Export Double DQN Baseline
    ddqn_agent = DoubleDQNAgent()
    ddqn_path = "checkpoints/experiments/exp_b_double_dqn_seed_42.pth"
    if os.path.exists(ddqn_path):
        ddqn_agent.load_checkpoint(ddqn_path)
        export_agent_to_json(ddqn_agent, "web/weights/ddqn_weights.json", "Double DQN (Tabula Rasa)")

    # 3. Export Demo-Assisted Double DQN
    demo_agent = DemonstrationDQNAgent()
    demo_path = "checkpoints/best_model.pth"
    if os.path.exists(demo_path):
        demo_agent.load_checkpoint(demo_path)
        export_agent_to_json(demo_agent, "web/weights/trained_weights.json", "Demonstration-Assisted Double DQN")


if __name__ == "__main__":
    main()
