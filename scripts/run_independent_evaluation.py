"""
Comprehensive Independent Evaluation Execution Script.
Runs:
  1. The 60 Curated Corner Test Cases (Deterministic Stress Suite)
  2. The 100 Stochastic Unseen Seeds (30000..30099)
Across models:
  - Demonstration-Assisted Double DQN (Best Production Model)
  - Standard DQN Baseline
  - Double DQN Baseline
Exports detailed JSON reports and prints complete failure matrices.
"""

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import json
from dataclasses import asdict
from typing import Dict, Any

from evaluation.evaluator import StandaloneEvaluator, export_evaluation_report
from evaluation.test_cases import build_curated_test_cases
from rl.demonstration_agent import DemonstrationDQNAgent
from rl.dqn.agent import DQNAgent
from rl.double_dqn.agent import DoubleDQNAgent


def run_full_evaluation():
    print("=" * 80)
    print("STARTING COMPREHENSIVE INDEPENDENT EVALUATION PIPELINE")
    print("Strict Protocol: epsilon = 0.0, eval() mode, torch.no_grad(), zero training")
    print("=" * 80)

    evaluator = StandaloneEvaluator()
    curated_cases = build_curated_test_cases()

    # 1. Evaluate Best Model: Demonstration-Assisted DDQN
    demo_model_path = "checkpoints/best_model.pth"
    if not os.path.exists(demo_model_path):
        demo_model_path = "checkpoints/experiments/exp_e_demo_assisted_seed_123.pth"

    print(f"\n[1/3] Evaluating Demonstration-Assisted Double DQN ({demo_model_path})...")
    demo_agent = DemonstrationDQNAgent()
    demo_agent.load_checkpoint(demo_model_path)
    demo_agent.online_net.eval()

    # (a) Curated Corner Cases (60 Tests)
    summary_demo_curated, telems_demo_curated = evaluator.evaluate_curated_suite(
        agent=demo_agent, test_cases=curated_cases
    )
    export_evaluation_report(
        summary_demo_curated, telems_demo_curated,
        "evaluation/reports/demo_assisted_curated_report.json"
    )

    # (b) Stochastic Unseen Seeds (100 Tests: 30000..30099)
    summary_demo_stochastic, telems_demo_stochastic = evaluator.evaluate_stochastic_suite(
        agent=demo_agent, num_episodes=100, seed_start=30000, level=3
    )
    export_evaluation_report(
        summary_demo_stochastic, telems_demo_stochastic,
        "evaluation/reports/demo_assisted_stochastic_report.json"
    )

    # 2. Evaluate Baseline: Standard DQN
    dqn_model_path = "checkpoints/experiments/exp_a_dqn_seed_42.pth"
    print(f"\n[2/3] Evaluating Standard DQN Baseline ({dqn_model_path})...")
    dqn_agent = DQNAgent()
    dqn_agent.load_checkpoint(dqn_model_path)
    dqn_agent.online_net.eval()

    summary_dqn_stochastic, telems_dqn_stochastic = evaluator.evaluate_stochastic_suite(
        agent=dqn_agent, num_episodes=100, seed_start=30000, level=3
    )
    export_evaluation_report(
        summary_dqn_stochastic, telems_dqn_stochastic,
        "evaluation/reports/dqn_baseline_stochastic_report.json"
    )

    # 3. Evaluate Baseline: Double DQN
    ddqn_model_path = "checkpoints/experiments/exp_b_double_dqn_seed_42.pth"
    print(f"\n[3/3] Evaluating Double DQN Baseline ({ddqn_model_path})...")
    ddqn_agent = DoubleDQNAgent()
    ddqn_agent.load_checkpoint(ddqn_model_path)
    ddqn_agent.online_net.eval()

    summary_ddqn_stochastic, telems_ddqn_stochastic = evaluator.evaluate_stochastic_suite(
        agent=ddqn_agent, num_episodes=100, seed_start=30000, level=3
    )
    export_evaluation_report(
        summary_ddqn_stochastic, telems_ddqn_stochastic,
        "evaluation/reports/double_dqn_stochastic_report.json"
    )

    # 4. Print Executive Summary Table
    print("\n" + "=" * 88)
    print("EXECUTIVE SCORECARD: 100 UNSEEN STOCHASTIC TEST EPISODES (SEEDS 30000..30099)")
    print("=" * 88)
    header = f"{'Evaluation Criterion':<38} | {'Standard DQN':<14} | {'Double DQN':<14} | {'Demo-Assisted DDQN':<16}"
    print(header)
    print("-" * 88)
    print(f"{'1. Safe Landing Rate (%)':<38} | {summary_dqn_stochastic.safe_landing_rate_pct:<14.1f} | {summary_ddqn_stochastic.safe_landing_rate_pct:<14.1f} | {summary_demo_stochastic.safe_landing_rate_pct:<16.1f}")
    print(f"{'2. Target Zone Rate (|x| <= 0.8m) (%)':<38} | {summary_dqn_stochastic.target_zone_rate_pct:<14.1f} | {summary_ddqn_stochastic.target_zone_rate_pct:<14.1f} | {summary_demo_stochastic.target_zone_rate_pct:<16.1f}")
    print(f"{'3. Mean Landing Time (s)':<38} | {summary_dqn_stochastic.mean_landing_time_s:<14.2f} | {summary_ddqn_stochastic.mean_landing_time_s:<14.2f} | {summary_demo_stochastic.mean_landing_time_s:<16.2f}")
    print(f"{'4. Mean Absolute Angle (deg)':<38} | {summary_dqn_stochastic.mean_landing_angle_deg:<14.2f} | {summary_ddqn_stochastic.mean_landing_angle_deg:<14.2f} | {summary_demo_stochastic.mean_landing_angle_deg:<16.2f}")
    print(f"{'5. Mean Landing Speed (|vy| m/s)':<38} | {summary_dqn_stochastic.mean_landing_vy_mps:<14.3f} | {summary_ddqn_stochastic.mean_landing_vy_mps:<14.3f} | {summary_demo_stochastic.mean_landing_vy_mps:<16.3f}")
    print(f"{'6. Collision Rate (%)':<38} | {summary_dqn_stochastic.collision_rate_pct:<14.1f} | {summary_ddqn_stochastic.collision_rate_pct:<14.1f} | {summary_demo_stochastic.collision_rate_pct:<16.1f}")
    print(f"{'7. Timeout Rate (%)':<38} | {summary_dqn_stochastic.timeout_rate_pct:<14.1f} | {summary_ddqn_stochastic.timeout_rate_pct:<14.1f} | {summary_demo_stochastic.timeout_rate_pct:<16.1f}")
    print(f"{'8. Out of Bounds Rate (%)':<38} | {summary_dqn_stochastic.oob_rate_pct:<14.1f} | {summary_ddqn_stochastic.oob_rate_pct:<14.1f} | {summary_demo_stochastic.oob_rate_pct:<16.1f}")
    print(f"{'9. Fuel Used (%)':<38} | {summary_dqn_stochastic.mean_fuel_used_pct:<14.1f} | {summary_ddqn_stochastic.mean_fuel_used_pct:<14.1f} | {summary_demo_stochastic.mean_fuel_used_pct:<16.1f}")
    print(f"{'Mean Raw Reward':<38} | {summary_dqn_stochastic.mean_raw_reward:<14.2f} | {summary_ddqn_stochastic.mean_raw_reward:<14.2f} | {summary_demo_stochastic.mean_raw_reward:<16.2f}")
    print("=" * 88)

    # 5. Print Failure Taxonomy & Confusion Matrix Breakdown
    print("\n" + "=" * 88)
    print("FAILURE MODE TAXONOMY & CONFUSION MATRIX BREAKDOWN (100 Unseen Seeds)")
    print("=" * 88)
    f_header = f"{'Failure Category / Dimension':<38} | {'Standard DQN':<14} | {'Double DQN':<14} | {'Demo-Assisted DDQN':<16}"
    print(f_header)
    print("-" * 88)
    f_keys = [
        ("Safe Touchdowns", "SAFE_LANDING"),
        ("Crash: Too Fast (|vy|>=1 or |vx|>=0.5)", "CRASH_TOO_FAST"),
        ("Crash: Too Much Rotation (angle>=5 deg)", "CRASH_TOO_MUCH_ROTATION"),
        ("Crash: Upside Down (angle > 90 deg)", "CRASH_UPSIDE_DOWN"),
        ("Crash: Compound (Fast + Rotated)", "CRASH_COMPOUND_FAST_AND_ROTATED"),
        ("Out of Bounds: Overshoot (x > +20m)", "OOB_OVERSHOOT"),
        ("Out of Bounds: Undershoot (x < -20m)", "OOB_UNDERSHOOT"),
        ("Timeout (t >= 20.0s)", "TIMEOUT"),
    ]
    for label, key in f_keys:
        cnt_dqn = summary_dqn_stochastic.failure_counts.get(key, 0)
        cnt_ddqn = summary_ddqn_stochastic.failure_counts.get(key, 0)
        cnt_demo = summary_demo_stochastic.failure_counts.get(key, 0)
        print(f"{label:<38} | {cnt_dqn:<14d} | {cnt_ddqn:<14d} | {cnt_demo:<16d}")
    print("=" * 88)

    # 6. Curated Corner Cases Breakdown for Demo-Assisted DDQN
    print("\n" + "=" * 88)
    print("DEMO-ASSISTED DDQN: CURATED CORNER STRESS TESTS BREAKDOWN (60 Tests, 6 Categories)")
    print("=" * 88)
    print(f"{'Stress Category':<30} | {'Tests':<6} | {'Safe (%)':<9} | {'Target (%)':<11} | {'Time (s)':<9} | {'Speed (m/s)':<12} | {'Angle (deg)':<11}")
    print("-" * 88)
    for cat, sc in summary_demo_curated.category_scorecards.items():
        print(
            f"{cat:<30} | {sc['total']:<6d} | {sc['safe_rate_pct']:<9.1f} | {sc['target_rate_pct']:<11.1f} | "
            f"{sc['mean_landing_time_s']:<9.2f} | {sc['mean_speed_mps']:<12.3f} | {sc['mean_angle_deg']:<11.2f}"
        )
    print("=" * 88)

    # Save summary report for Markdown generation
    combined_summary = {
        "curated_summary_demo": asdict(summary_demo_curated),
        "stochastic_summary_demo": asdict(summary_demo_stochastic),
        "stochastic_summary_dqn": asdict(summary_dqn_stochastic),
        "stochastic_summary_ddqn": asdict(summary_ddqn_stochastic),
    }
    with open("evaluation/reports/evaluation_executive_summary.json", "w", encoding="utf-8") as f:
        json.dump(combined_summary, f, indent=2)
    print("\nSaved executive summary to: evaluation/reports/evaluation_executive_summary.json")


if __name__ == "__main__":
    run_full_evaluation()
