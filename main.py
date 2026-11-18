"""
Unified Principal Engineer CLI for the Tesla Roadster 2D Flying Car RL Project.
Final Release Entry Point.

Commands:
  1. simulate: Run physics simulation with trained, DQN, or Double DQN agent
  2. train: Train a fresh agent (DQN, Double DQN, or Demonstration-Assisted)
  3. evaluate: Run independent evaluation on unseen test cases (curated or stochastic)
  4. compare: Display Head-to-Head benchmark comparison matrix
  5. explain: Explain the end-to-end RL pipeline step-by-step
  6. web: Launch the interactive Web Visualizer
"""

from __future__ import annotations
import os
import sys
import argparse
import json
import time
import webbrowser
import http.server
import socketserver
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))

from physics.constants import (
    REST_HEIGHT,
    PHYSICS_DT,
    DECISION_DT,
    MAX_SAFE_LANDING_VY,
    MAX_SAFE_LANDING_VX,
    MAX_SAFE_LANDING_ANGLE_DEG,
    MAX_SAFE_LANDING_OMEGA,
    TARGET_ZONE_X_MIN,
    TARGET_ZONE_X_MAX,
)
from environment.roadster_env import RoadsterEnv
from environment.action import decode_action, get_action_description, action_to_bin_str
from rl.dqn.network import QNetwork
from rl.dqn.agent import DQNAgent, DQNConfig
from rl.double_dqn.agent import DoubleDQNAgent
from rl.demonstration_agent import DemonstrationDQNAgent, DemonstrationAgentConfig
from evaluation.evaluator import StandaloneEvaluator
from evaluation.test_cases import build_curated_test_cases, get_stochastic_unseen_seeds


def load_agent(model_type: str = "trained", custom_path: str = None):
    """Factory helper to load specific trained model checkpoint."""
    if custom_path:
        checkpoint_path = custom_path
    elif model_type == "dqn":
        checkpoint_path = "checkpoints/experiments/exp_a_dqn_seed_42.pth"
    elif model_type == "ddqn":
        checkpoint_path = "checkpoints/experiments/exp_b_double_dqn_seed_42.pth"
    else:  # "trained" / "demo"
        checkpoint_path = "checkpoints/best_model.pth"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Determine agent class based on checkpoint or type
    if model_type == "dqn":
        agent = DQNAgent()
    elif model_type == "ddqn":
        agent = DoubleDQNAgent()
    else:
        agent = DemonstrationDQNAgent()

    agent.load_checkpoint(checkpoint_path)
    agent.online_net.eval()
    return agent, checkpoint_path


def cmd_simulate(args):
    """Run physics simulation with step-by-step telemetry."""
    print("=" * 80)
    print(f"🚀 TESLA ROADSTER 2D SIMULATOR | Model: {args.model.upper()}")
    print("=" * 80)

    agent, ckpt_path = load_agent(args.model, args.model_path)
    print(f"Loaded checkpoint: {ckpt_path}")

    env = RoadsterEnv(default_curriculum_level=args.level, use_shaping=False)
    
    for ep in range(args.episodes):
        ep_seed = args.seed + ep if args.seed is not None else int(time.time() * 1000) % 100000
        obs, info = env.reset(seed=ep_seed)
        
        print(f"\n--- Episode {ep + 1}/{args.episodes} (Seed: {ep_seed}, Level: {args.level}) ---")
        print(f"Initial: x={info['x']:.2f}m, y={info['h']:.2f}m, vx={info['vx']:.2f}m/s, vy={info['vy']:.2f}m/s, theta={info['theta_deg']:.1f}°")
        
        done = False
        step = 0
        total_raw_reward = 0.0

        while not done:
            q_values = agent.get_q_values(obs)
            action = agent.select_action(obs, evaluate=True)
            u1, u2, u3, u4 = decode_action(action)
            bin_str = action_to_bin_str(action)

            obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            total_raw_reward += next_info.get("raw_reward", reward)
            step += 1

            if args.verbose or done:
                print(
                    f"Step {step:03d} | Pos: ({next_info['x']:+5.2f}, {next_info['h']:4.2f})m | "
                    f"Vel: ({next_info['vx']:+5.2f}, {next_info['vy']:+5.2f})m/s | "
                    f"Angle: {next_info['theta_deg']:+5.1f}° | "
                    f"Act: {action:02d} [{bin_str}] | Q_max: {np.max(q_values):+6.2f}"
                )

        print("\n🏁 EPISODE OUTCOME:")
        print(f"  Result:              {next_info['termination_reason']}")
        print(f"  Safe Landing:        {'✅ YES' if next_info['safe_landing'] else '❌ NO'}")
        print(f"  Target Zone:         {'✅ YES' if next_info['in_target_zone'] else '❌ NO'}")
        print(f"  Touchdown |vy|:      {next_info['landing_speed_vy']:.3f} m/s (Limit: <= {MAX_SAFE_LANDING_VY} m/s)")
        print(f"  Touchdown |vx|:      {next_info['landing_speed_vx']:.3f} m/s (Limit: <= {MAX_SAFE_LANDING_VX} m/s)")
        print(f"  Touchdown Angle:     {abs(next_info['landing_angle_deg']):.2f}° (Limit: <= {MAX_SAFE_LANDING_ANGLE_DEG}°)")
        print(f"  Flight Duration:     {next_info['time']:.2f} s")
        print(f"  Fuel Consumed:       {100.0 - next_info['fuel']:.2f}%")
        print(f"  Cumulative Reward:   {total_raw_reward:.2f}")


def cmd_evaluate(args):
    """Run independent evaluation on unseen test suites."""
    print("=" * 80)
    print("📊 INDEPENDENT EVALUATOR (Epsilon=0.0, Zero Gradients, Disjoint Test Sets)")
    print("=" * 80)

    agent, ckpt_path = load_agent(args.model, args.model_path)
    print(f"Evaluating Model: {ckpt_path}\n")

    evaluator = StandaloneEvaluator()

    if args.suite == "curated":
        print(f"Running Curated Edge Cases Suite (60 Structured Tests)...")
        summary, _ = evaluator.evaluate_curated_suite(agent)
    else:
        episodes = args.episodes or 100
        print(f"Running Stochastic Unseen Suite ({episodes} Episodes, Seeds 30000..{30000 + episodes - 1})...")
        summary, _ = evaluator.evaluate_stochastic_suite(agent, num_episodes=episodes, seed_start=30000)

    print("\n" + "=" * 50)
    print("🎯 THE 9 GOLDEN METRICS SCORECARD:")
    print("=" * 50)
    print(f"  1. Safe Landing Rate:       {summary.safe_landing_rate_pct:6.2f}% ({summary.safe_landing_count}/{summary.total_episodes})")
    print(f"  2. Target Zone Accuracy:    {summary.target_zone_rate_pct:6.2f}% (Mean x err: {summary.mean_landing_x_error_m:.3f}m)")
    print(f"  3. Mean Touchdown Speed:    {summary.mean_total_landing_speed_mps:6.3f} m/s (vy: {summary.mean_landing_vy_mps:.3f}, vx: {summary.mean_landing_vx_mps:.3f})")
    print(f"  4. Mean Landing Angle:      {summary.mean_landing_angle_deg:6.2f}° (Max: {summary.max_landing_angle_deg:.2f}°)")
    print(f"  5. Mean Landing Time:       {summary.mean_landing_time_s:6.2f} s")
    print(f"  6. Collision Rate:          {summary.collision_rate_pct:6.2f}% ({summary.collision_count} crashes)")
    print(f"  7. Out of Bounds Rate:      {summary.oob_rate_pct:6.2f}% ({summary.oob_count} OOB)")
    print(f"  8. Timeout Rate:            {summary.timeout_rate_pct:6.2f}% ({summary.timeout_count} timeouts)")
    print(f"  9. Mean Fuel Consumed:      {summary.mean_fuel_used_pct:6.2f}%")
    print("=" * 50)
    print(f"  Mean Unshaped Raw Reward:   {summary.mean_raw_reward:+6.2f}")

    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)
        print(f"\nExported evaluation summary to: {args.export}")


def cmd_compare(args):
    """Print Head-to-Head comparative benchmark table."""
    print("=" * 90)
    print("🏆 BENCHMARK COMPARATIVE MATRIX (Ground Truth from Experiments & Independent Evaluator)")
    print("=" * 90)

    header = f"{'Algorithm / Model':<30} | {'Safe Rate':<10} | {'Collisions':<10} | {'OOB':<8} | {'Vy (m/s)':<8} | {'Angle':<8} | {'Time (s)':<8} | {'Fuel %':<8}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Standard DQN (Baseline)", "0.0%", "0.0%", "100.0%", "N/A", "N/A", "N/A", "7.18%"),
        ("Double DQN (Baseline)", "0.0%", "16.0%", "84.0%", "N/A", "N/A", "N/A", "7.33%"),
        ("Double DQN + Shaping (PBRS)", "0.0%", "8.7%", "91.3%", "N/A", "N/A", "N/A", "7.60%"),
        ("Double DQN + Curriculum", "0.0%", "72.0%", "28.0%", "N/A", "N/A", "N/A", "7.01%"),
        ("Demonstration-Assisted DDQN", "92.0%", "8.0%", "0.0%", "0.561", "2.15°", "4.34s", "7.54%"),
    ]

    for name, safe, coll, oob, vy, ang, t, fuel in rows:
        print(f"{name:<30} | {safe:<10} | {coll:<10} | {oob:<8} | {vy:<8} | {ang:<8} | {t:<8} | {fuel:<8}")
    print("=" * 90)
    print("💡 Key Takeaway: Standard DQN and Double DQN tabula-rasa fail due to sparse reward and")
    print("   16-action discrete thruster dynamics. Guided Large Margin Demonstration Learning")
    print("   provides the critical bootstrap for 92.0% safe landing success without cheating.")


def cmd_explain(args):
    """Explain the complete RL pipeline step-by-step."""
    pipeline = """
========================================================================================
🎓 COMPLETE REINFORCEMENT LEARNING PIPELINE EXPLANATION
========================================================================================

Step 1: STATE (Không gian Trạng thái)
  • Raw Physical State: x, h, vx, vy, theta, omega, fuel, time.
  • Normalized Observation s in R^9:
    s = [x/10, h/15, vx/10, vy/10, sin(theta), cos(theta), omega/5, fuel/100, t/20]
  • (sin(theta), cos(theta)) eliminates angle wrapping discontinuity across [-pi, +pi].

Step 2: ACTION (Không gian Hành động)
  • 16 discrete actions: a in {0..15} mapping to 4 binary cold gas thruster valves:
    a = u1 + 2*u2 + 4*u3 + 8*u4
    - u1 (V1 tail lift): 9000N, arm -1.8m -> pitch down torque -16,200 N*m
    - u2 (V2 head lift): 9000N, arm +1.8m -> pitch up torque +16,200 N*m
    - u3 (V3 push right): 6000N through COM -> zero torque
    - u4 (V4 push left):  6000N through COM -> zero torque
  • Balanced vertical thrust (V1+V2, Action 3): 18,000 N > 11,772 N gravity, torque = 0!

Step 3: ENVIRONMENT DYNAMICS (Mô hình Động lực học)
  • 120 Hz Symplectic Euler integration (12 substeps per 0.1s decision cycle).
  • Rotated force vectoring:
    Fx_world = Fx_local * cos(theta) - Fy_local * sin(theta)
    Fy_world = Fx_local * sin(theta) + Fy_local * cos(theta) - mg

Step 4: REWARD FUNCTION (Hàm Phần thưởng & Định lý PBRS)
  • r_base = r_term (+200 safe, -100 crash, -250 OOB, -500 timeout)
           + r_target (+30 in pad) - 2*(vx^2 + vy^2) - 10*(1 - cos(theta))
           - 0.2*delta_fuel - 1.0*delta_time
  • PBRS Shaping: F(s, s') = gamma * Phi(s') - Phi(s)
  • Potential: Phi(s) = -10*sqrt(x^2 + (h-0.72)^2) - 4*sqrt(vx^2 + vy^2)
  • Invariance Guarantee: Phi(s_terminal) = 0.0 for ALL terminal states!

Step 5: Q-VALUE PREDICTION (Hàm Giá trị Q)
  • Multi-Layer Perceptron: 9 -> 64 (ReLU) -> 64 (ReLU) -> 16 linear outputs.
  • Output vector: Q(s, a; theta) for all 16 actions.

Step 6: LOSS & PARAMETER UPDATE (Cập nhật Học tăng cường)
  • Double DQN TD Target:
    a* = argmax_a Q(s', a; theta_online)
    y  = r + gamma * (1 - done) * Q(s', a*; theta_target)
  • Supervised Large Margin Loss (DQfD, Hester et al. 2017):
    J_E(Q) = max_{a in A} [ Q(s, a; theta) + l(a_E, a) ] - Q(s, a_E; theta)
    where l(a_E, a) = 0 if a == a_E else margin (0.2).
  • Total Loss: L(theta) = L_TD + lambda_margin * J_E(Q) + lambda_l2 * L2(theta).

Step 7: POLICY EXECUTION (Chiến lược Ra quyết định)
  • Training: Epsilon-greedy exploration linearly decaying from 1.0 to 0.05.
  • Inference & Evaluation: Pure Greedy Deterministic (epsilon = 0.0):
    pi*(s) = argmax_a Q(s, a; theta)

Step 8: TERMINAL LANDING VERIFICATION (Tiêu chuẩn Tiếp đất An toàn)
  • Evaluated strictly on physical geometry of 6 vehicle contact vertices:
    |vy| <= 1.0 m/s, |vx| <= 0.5 m/s, |theta| <= 5 deg, |omega| <= 0.2 rad/s.
========================================================================================
"""
    print(pipeline)


def cmd_train(args):
    """Execute training run."""
    print("=" * 80)
    print(f"🏋️ TRAINING AGENT | Algorithm: {args.algo.upper()} | Seed: {args.seed}")
    print("=" * 80)
    from experiments.config import ExperimentConfig
    from experiments.runner import (
        run_experiment_a_dqn,
        run_experiment_b_double_dqn,
        run_experiment_e_demo_assisted,
    )

    cfg = ExperimentConfig(
        algo_name=args.algo,
        seed=args.seed,
        total_episodes=args.episodes or 500,
        checkpoint_dir="checkpoints/custom",
    )

    if args.algo == "dqn":
        agent, history = run_experiment_a_dqn(cfg)
    elif args.algo == "ddqn":
        agent, history = run_experiment_b_double_dqn(cfg)
    else:
        agent, history = run_experiment_e_demo_assisted(cfg)

    out_path = args.output or f"checkpoints/custom/{args.algo}_seed_{args.seed}.pth"
    agent.save_checkpoint(out_path)
    print(f"\nTraining completed! Saved model to: {out_path}")


def cmd_web(args):
    """Start local web server and open Web Visualizer."""
    port = args.port or 8000
    web_dir = os.path.abspath("web")
    os.chdir(web_dir)

    handler = http.server.SimpleHTTPRequestHandler
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            url = f"http://localhost:{port}"
            print("=" * 70)
            print(f"🌐 TESLA ROADSTER 2D WEB VISUALIZER")
            print(f"Serving at: {url}")
            print(f"Opening browser... Press Ctrl+C to terminate.")
            print("=" * 70)
            webbrowser.open(url)
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nWeb server stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Tesla Roadster 2D Flying Car RL Project - Final Release CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 1. Simulate
    sim_p = subparsers.add_parser("simulate", help="Run physics simulation with a model")
    sim_p.add_argument("--model", choices=["trained", "dqn", "ddqn"], default="trained", help="Model to run")
    sim_p.add_argument("--model-path", type=str, default=None, help="Custom checkpoint path")
    sim_p.add_argument("--episodes", type=int, default=3, help="Number of episodes to simulate")
    sim_p.add_argument("--seed", type=int, default=42, help="Starting random seed")
    sim_p.add_argument("--level", type=int, default=3, help="Curriculum domain level (1, 2, or 3)")
    sim_p.add_argument("--verbose", action="store_true", help="Print step-by-step state and valves")

    # 2. Evaluate
    eval_p = subparsers.add_parser("evaluate", help="Run independent evaluation")
    eval_p.add_argument("--model", choices=["trained", "dqn", "ddqn"], default="trained")
    eval_p.add_argument("--model-path", type=str, default=None, help="Custom checkpoint path")
    eval_p.add_argument("--suite", choices=["stochastic", "curated"], default="stochastic", help="Test suite to run")
    eval_p.add_argument("--episodes", type=int, default=100, help="Episodes for stochastic suite")
    eval_p.add_argument("--export", type=str, default=None, help="JSON file path to export summary")

    # 3. Compare
    subparsers.add_parser("compare", help="Display comparative benchmark matrix")

    # 4. Explain
    subparsers.add_parser("explain", help="Explain end-to-end RL pipeline step-by-step")

    # 5. Train
    train_p = subparsers.add_parser("train", help="Train an agent")
    train_p.add_argument("--algo", choices=["dqn", "ddqn", "demo"], default="demo", help="Algorithm")
    train_p.add_argument("--episodes", type=int, default=500, help="Training episodes")
    train_p.add_argument("--seed", type=int, default=42, help="Random seed")
    train_p.add_argument("--output", type=str, default=None, help="Output checkpoint path")

    # 6. Web
    web_p = subparsers.add_parser("web", help="Launch interactive Web Visualizer")
    web_p.add_argument("--port", type=int, default=8000, help="Local HTTP port")

    args = parser.parse_args()

    if args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "explain":
        cmd_explain(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "web":
        cmd_web(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
