"""
Execution script for Guided Double DQN (DQfD) Training Pipeline.
Runs Pre-training on demonstrations, evaluates periodically, exports weights to JSON.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

from rl.dqfd.agent import DQfDConfig
from rl.dqfd.trainer import DQfDTrainer, export_weights_to_json, evaluate_policy

def main():
    config = DQfDConfig(
        state_dim=9,
        action_dim=16,
        hidden_dim=64,
        learning_rate=0.001,
        gamma=0.995,
        batch_size=64,
        buffer_size=50000,
        warmup_steps=256,
        target_update_frequency=500,
        epsilon_start=0.20,
        epsilon_end=0.02,
        epsilon_decay=10000,
        lambda_margin=5.0,
        margin_value=0.2,
        lambda_l2=1e-5,
        loss_type="huber",
    )

    trainer = DQfDTrainer(
        config=config,
        demo_filepath="demonstrations.jsonl",
        checkpoint_dir="checkpoints/dqfd",
    )

    # 1. Run Pretraining (30,000 updates)
    agent = trainer.run_pretraining(num_updates=30000, eval_interval=5000, eval_episodes=25)

    # 2. Evaluate final model
    eval_metrics = evaluate_policy(agent, num_episodes=50, seed_base=10000, level=3)
    print("\n" + "=" * 60)
    print("FINAL DQfD PRETRAINED EVALUATION (50 Episodes, Level 3):")
    print(f"  Success Rate:        {eval_metrics['success_rate']*100:.1f}%")
    print(f"  Crash Rate:          {eval_metrics['crash_rate']*100:.1f}%")
    print(f"  Timeout Rate:        {eval_metrics['timeout_rate']*100:.1f}%")
    print(f"  Mean Raw Reward:     {eval_metrics['mean_raw_reward']:.1f}")
    print(f"  Mean Landing Speed:  {eval_metrics['mean_landing_speed']:.2f} m/s")
    print(f"  Mean Landing Angle:  {eval_metrics['mean_landing_angle']:.1f} deg")
    print(f"  Mean Landing Time:   {eval_metrics['mean_landing_time']:.2f} s")
    print("=" * 60)

    # 3. Export weights to JSON for Web Canvas Visualizer
    export_path = "web/weights/trained_weights.json"
    export_weights_to_json(agent, export_path)

    # Also save PyTorch checkpoint as best_model.pth
    best_path = "checkpoints/best_model.pth"
    agent.save_checkpoint(best_path)
    print(f"[Main] Saved final weights checkpoint to: {best_path}")

if __name__ == "__main__":
    main()
