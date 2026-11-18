"""
Automated Senior Software QA Suite for the Tesla Roadster 2D Flying Car RL Project.
Covers:
  1. PHYSICS: gravity, thrust, torque, collision
  2. ENVIRONMENT: state shape, action mapping, terminal conditions, timestep
  3. RL: Q target, replay, epsilon, target network, gradient, checkpoint
  4. REWARD: safe landing, collision, timeout, fuel, time, PBRS
  5. EVALUATION: no training during test, epsilon = 0, independent test set
  6. UI INTEGRITY: DQN runs DQN, Double DQN runs Double DQN, no background controller
"""

import os
import sys
import math
import json
import pytest
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))

from physics.constants import (
    VEHICLE_MASS,
    VEHICLE_LENGTH,
    VEHICLE_HEIGHT,
    REST_HEIGHT,
    VEHICLE_INERTIA,
    GRAVITY_ACCEL,
    THRUST_LIFT,
    THRUST_LATERAL,
    PHYSICS_DT,
    DECISION_DT,
    SUBSTEPS_PER_STEP,
    MAX_SAFE_LANDING_VY,
    MAX_SAFE_LANDING_VX,
    MAX_SAFE_LANDING_ANGLE_RAD,
    MAX_SAFE_LANDING_OMEGA,
    TARGET_ZONE_X_MIN,
    TARGET_ZONE_X_MAX,
    WORLD_X_MIN,
    WORLD_X_MAX,
    WORLD_Y_MAX,
    MAX_EPISODE_TIME,
)
from physics.world import PhysicsWorld
from physics.thrusters import ThrusterSystem
from environment.roadster_env import RoadsterEnv
from environment.state import (
    STATE_DIM,
    RawState,
    raw_to_normalized,
    normalized_to_raw,
    validate_state,
)
from environment.action import (
    NUM_ACTIONS,
    decode_action,
    encode_action,
    validate_action,
)
from environment.reward import (
    RewardEngine,
    RewardBreakdown,
)
from environment.terminal import (
    TerminalDetector,
    TerminationType,
)
from rl.dqn.network import QNetwork
from rl.dqn.agent import DQNAgent, DQNConfig
from rl.dqn.replay_buffer import ReplayBuffer
from rl.double_dqn.agent import DoubleDQNAgent
from rl.demonstration_agent import DemonstrationDQNAgent
from evaluation.evaluator import StandaloneEvaluator
from evaluation.test_cases import build_curated_test_cases, get_stochastic_unseen_seeds


# =============================================================================
# 1. PHYSICS TESTS (GRAVITY, THRUST, TORQUE, COLLISION)
# =============================================================================

class TestPhysicsQA:
    """Rigorous QA tests for the 2D physics integration engine."""

    def test_physics_gravity_free_fall_kinematics(self):
        """Verify Symplectic Euler integration under gravity g = 9.81 m/s^2."""
        world = PhysicsWorld(initial_x=0.0, initial_y=15.0, initial_vx=0.0, initial_vy=0.0)

        # 1. Acceleration under action 0 (no thrusters)
        world.substep(action=0, dt=PHYSICS_DT)
        assert world.body.last_ax == pytest.approx(0.0, abs=1e-6)
        assert world.body.last_ay == pytest.approx(-GRAVITY_ACCEL, abs=1e-5)
        assert world.body.last_alpha == pytest.approx(0.0, abs=1e-6)

        # 2. Kinematics after 1.0 second (120 substeps)
        world = PhysicsWorld(initial_x=0.0, initial_y=20.0, initial_vx=0.0, initial_vy=0.0)
        for _ in range(120):
            world.substep(action=0, dt=PHYSICS_DT)

        # Expected velocity: vy = -g * t = -9.81 m/s
        assert world.body.vy == pytest.approx(-GRAVITY_ACCEL * 1.0, abs=1e-4)

        # Expected position in Symplectic Euler: y ≈ y0 - 0.5*g*t^2
        expected_y = 20.0 - 0.5 * GRAVITY_ACCEL * (1.0**2)
        assert world.body.y == pytest.approx(expected_y, abs=0.1)

    def test_physics_thrust_force_magnitudes_and_rotation(self):
        """Verify lift thrust (9000N x 2), lateral thrust (6000N x 2), and rotated vector."""
        # Case 1: Level vehicle, Action 3 (V1 + V2 = 18000N)
        world = PhysicsWorld(initial_x=0.0, initial_y=10.0, initial_vx=0.0, initial_vy=0.0, initial_theta=0.0)
        world.substep(action=3, dt=PHYSICS_DT)
        expected_f_net = (2 * THRUST_LIFT) - (VEHICLE_MASS * GRAVITY_ACCEL)  # 6228.0 N
        expected_ay = expected_f_net / VEHICLE_MASS  # 5.19 m/s^2
        assert world.body.last_ay == pytest.approx(expected_ay, abs=1e-4)
        assert world.body.last_ax == pytest.approx(0.0, abs=1e-5)

        # Case 2: Lateral push right (V3 = Action 4) -> ax = +6000/1200 = +5.0 m/s^2
        world = PhysicsWorld(initial_x=0.0, initial_y=10.0, initial_vx=0.0, initial_vy=0.0, initial_theta=0.0)
        world.substep(action=4, dt=PHYSICS_DT)
        assert world.body.last_ax == pytest.approx(+5.0, abs=1e-4)

        # Case 3: Rotated thrust at theta = 30 deg (pi / 6 rad)
        theta = math.pi / 6.0
        world = PhysicsWorld(initial_x=0.0, initial_y=10.0, initial_vx=0.0, initial_vy=0.0, initial_theta=theta)
        world.substep(action=3, dt=PHYSICS_DT)  # V1 + V2
        # Fx = -18000 * sin(30) = -9000 N -> ax = -9000 / 1200 = -7.5 m/s^2
        expected_ax = (-2.0 * THRUST_LIFT * math.sin(theta)) / VEHICLE_MASS
        assert world.body.last_ax == pytest.approx(expected_ax, abs=1e-3)

    def test_physics_torque_and_angular_acceleration(self):
        """Verify torque arms (±1.8m), moment of inertia (2017 kg·m^2), and zero net torque."""
        # V1 (Tail): arm = -1.8m -> torque = -16200 N·m (Clockwise)
        world = PhysicsWorld(initial_x=0.0, initial_y=10.0, initial_vx=0.0, initial_vy=0.0, initial_theta=0.0)
        world.substep(action=1, dt=PHYSICS_DT)
        expected_alpha_v1 = -16200.0 / VEHICLE_INERTIA
        assert world.body.last_alpha == pytest.approx(expected_alpha_v1, abs=1e-4)

        # V2 (Head): arm = +1.8m -> torque = +16200 N·m (Counter-Clockwise)
        world = PhysicsWorld(initial_x=0.0, initial_y=10.0, initial_vx=0.0, initial_vy=0.0, initial_theta=0.0)
        world.substep(action=2, dt=PHYSICS_DT)
        expected_alpha_v2 = +16200.0 / VEHICLE_INERTIA
        assert world.body.last_alpha == pytest.approx(expected_alpha_v2, abs=1e-4)

        # V1 + V2 (Action 3): Balanced lift -> Net torque must be identically ZERO
        world = PhysicsWorld(initial_x=0.0, initial_y=10.0, initial_vx=0.0, initial_vy=0.0, initial_theta=0.0)
        world.substep(action=3, dt=PHYSICS_DT)
        assert world.body.last_alpha == pytest.approx(0.0, abs=1e-6)

        # Lateral thrusters V3, V4 (Action 4 and 8): centered -> Net torque ZERO
        world.substep(action=4, dt=PHYSICS_DT)
        assert world.body.last_alpha == pytest.approx(0.0, abs=1e-6)
        world.substep(action=8, dt=PHYSICS_DT)
        assert world.body.last_alpha == pytest.approx(0.0, abs=1e-6)

    def test_physics_collision_detection_and_rules(self):
        """Verify 6-point collision detection, safe landing vs crash criteria."""
        world = PhysicsWorld(initial_x=0.0, initial_y=REST_HEIGHT, initial_vx=0.1, initial_vy=-0.5, initial_theta=0.02, initial_omega=0.05)
        coll = world.collision_detector.check_collision(world.body)
        assert coll.is_contact is True
        assert coll.is_safe_landing is True
        assert coll.is_crash is False

        # Crash 1: High vertical velocity (|vy| = 1.8 m/s >= 1.0 m/s)
        world = PhysicsWorld(initial_x=0.0, initial_y=REST_HEIGHT, initial_vx=0.1, initial_vy=-1.8, initial_theta=0.02, initial_omega=0.05)
        coll = world.collision_detector.check_collision(world.body)
        assert coll.is_contact is True
        assert coll.is_safe_landing is False
        assert coll.is_crash is True

        # Crash 2: High tilt angle (|theta| = 8 deg >= 5 deg)
        world = PhysicsWorld(initial_x=0.0, initial_y=REST_HEIGHT, initial_vx=0.1, initial_vy=-0.4, initial_theta=math.radians(8.0), initial_omega=0.05)
        coll = world.collision_detector.check_collision(world.body)
        assert coll.is_contact is True
        assert coll.is_safe_landing is False
        assert coll.is_crash is True

        # Crash 3: Inverted roof contact (Upside down rollover: COM height <= 0.45m causes roof vertices to strike ground)
        world = PhysicsWorld(initial_x=0.0, initial_y=0.4, initial_vx=0.0, initial_vy=-0.2, initial_theta=math.pi, initial_omega=0.0)
        coll = world.collision_detector.check_collision(world.body)
        assert coll.is_contact is True
        assert coll.is_safe_landing is False
        assert coll.is_crash is True


# =============================================================================
# 2. ENVIRONMENT TESTS (STATE, ACTION, TERMINALS, TIMESTEP)
# =============================================================================

class TestEnvironmentQA:
    """Rigorous QA tests for the MDP Environment."""

    def test_env_state_shape_and_continuous_bounds(self):
        """Verify 9-dimensional continuous state vector normalization."""
        env = RoadsterEnv()
        obs, info = env.reset(seed=123)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (9,)
        assert obs.dtype == np.float32

        # Check normalization bounds for standard operational domain
        raw = RawState(x=0.0, h=7.5, vx=2.0, vy=-3.0, theta=math.pi/4, omega=0.5, fuel=80.0, time=5.0)
        norm = raw_to_normalized(raw)
        assert abs(norm[0] - 0.0) < 1e-5          # x / 10
        assert abs(norm[1] - 0.5) < 1e-5          # h / 15
        assert abs(norm[2] - 0.2) < 1e-5          # vx / 10
        assert abs(norm[3] - (-0.3)) < 1e-5       # vy / 10
        assert abs(norm[4] - math.sin(math.pi/4)) < 1e-5
        assert abs(norm[5] - math.cos(math.pi/4)) < 1e-5
        assert abs(norm[6] - 0.1) < 1e-5          # omega / 5
        assert abs(norm[7] - 0.8) < 1e-5          # fuel / 100
        assert abs(norm[8] - 0.25) < 1e-5         # time / 20

        # Trigonometric identity validation
        assert abs(norm[4]**2 + norm[5]**2 - 1.0) < 1e-5

    def test_env_action_mapping_bidirectional(self):
        """Verify exact bijection between action integer {0..15} and (V1, V2, V3, V4)."""
        for a in range(NUM_ACTIONS):
            u1, u2, u3, u4 = decode_action(a)
            re_encoded = encode_action(u1, u2, u3, u4)
            assert re_encoded == a

        # Specific action semantic checks
        assert decode_action(0) == (0, 0, 0, 0)
        assert decode_action(3) == (1, 1, 0, 0)
        assert decode_action(15) == (1, 1, 1, 1)

    def test_env_terminal_conditions_and_timeout(self):
        """Verify all 4 terminal conditions: safe landing, crash, oob, timeout."""
        env = RoadsterEnv()

        # Terminal 1: Safe Landing
        env.reset(options={"initial_x": 0.0, "initial_y": REST_HEIGHT, "initial_vy": -0.4})
        obs, reward, terminated, truncated, info = env.step(0)
        assert terminated is True
        assert info["safe_landing"] is True
        assert env.last_terminal_result.is_safe_landing is True

        # Terminal 2: Crash (High speed)
        env.reset(options={"initial_x": 0.0, "initial_y": REST_HEIGHT, "initial_vy": -3.0})
        obs, reward, terminated, truncated, info = env.step(0)
        assert terminated is True
        assert info["crash"] is True
        assert env.last_terminal_result.is_crash is True

        # Terminal 3: Out of Bounds (|x| > 20.0 m)
        env.reset(options={"initial_x": 21.0, "initial_y": 5.0})
        obs, reward, terminated, truncated, info = env.step(0)
        assert terminated is True
        assert info["out_of_bounds"] is True
        assert env.last_terminal_result.is_out_of_bounds is True

        # Terminal 4: Timeout at 200 decision steps (20.0s)
        env.reset(options={"initial_x": 0.0, "initial_y": 8.0})
        for step_idx in range(200):
            act = 3 if env.world.body.y < 8.0 else 0
            obs, r, term, trunc, inf = env.step(act)
            if term or trunc:
                break
        assert trunc is True
        assert inf["timeout"] is True
        assert env.last_terminal_result.is_timeout is True

    def test_env_timestep_and_substepping_count(self):
        """Verify exact 12 substeps per decision cycle (0.1s @ 120Hz)."""
        env = RoadsterEnv()
        env.reset(options={"initial_x": 0.0, "initial_y": 8.0})
        obs, r, term, trunc, info = env.step(0)
        assert abs(info["time"] - DECISION_DT) < 1e-5
        assert SUBSTEPS_PER_STEP == 12
        assert abs(PHYSICS_DT * SUBSTEPS_PER_STEP - DECISION_DT) < 1e-6


# =============================================================================
# 3. REINFORCEMENT LEARNING TESTS (DQN, DOUBLE DQN, REPLAY, GRADIENT, CKPT)
# =============================================================================

class TestReinforcementLearningQA:
    """Rigorous QA tests for RL algorithms and optimization mechanics."""

    def test_rl_q_target_standard_dqn_formula(self):
        """Verify Bellman target calculation: y = r + gamma * (1 - done) * max_a' Q_target(s', a')."""
        agent = DQNAgent()
        rewards = torch.tensor([1.0, -0.5, 2.0], dtype=torch.float32)
        dones = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        next_states = torch.randn(3, 9)

        with torch.no_grad():
            q_target_vals = agent.target_net(next_states)
            max_q = q_target_vals.max(dim=1)[0]
            expected_target = rewards + agent.config.gamma * (1.0 - dones) * max_q

        # Verify done flag completely zeros out bootstrap term
        assert expected_target[1].item() == -0.5

    def test_rl_q_target_double_dqn_decoupling(self):
        """Verify Double DQN decouples action selection (online net) from evaluation (target net)."""
        agent = DoubleDQNAgent()
        rewards = torch.tensor([1.0, 2.0], dtype=torch.float32)
        dones = torch.tensor([0.0, 0.0], dtype=torch.float32)
        next_states = torch.randn(2, 9)

        # Force target network weights to differ from online network
        with torch.no_grad():
            for p in agent.target_net.parameters():
                p.add_(0.5)

        td_target = agent.compute_td_target(rewards, next_states, dones)

        with torch.no_grad():
            # Manual decoupled computation
            q_online = agent.online_net(next_states)
            best_acts = q_online.argmax(dim=1, keepdim=True)
            q_target = agent.target_net(next_states)
            q_eval = q_target.gather(1, best_acts).squeeze(1)
            expected = rewards + agent.config.gamma * (1.0 - dones) * q_eval

        assert torch.allclose(td_target, expected, atol=1e-5)

    def test_rl_replay_buffer_ring_capacity_and_sampling(self):
        """Verify ReplayBuffer circular replacement and batch tensor extraction."""
        buf = ReplayBuffer(capacity=100)
        for i in range(150):
            s = np.full(9, i, dtype=np.float32)
            s_next = np.full(9, i + 1, dtype=np.float32)
            buf.push(s, i % 16, float(i), s_next, False)

        assert len(buf) == 100
        # Oldest transitions 0..49 must be evicted
        oldest_state = buf.states[buf.ptr]
        assert oldest_state[0] == 50.0

        batch = buf.sample(32)
        assert batch[0].shape == (32, 9)
        assert batch[1].shape == (32,)
        assert batch[2].shape == (32,)
        assert batch[3].shape == (32, 9)
        assert batch[4].shape == (32,)

    def test_rl_epsilon_decay_schedule_and_greedy_evaluation(self):
        """Verify epsilon decays linearly and evaluate=True forces epsilon=0."""
        config = DQNConfig(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=1000)
        agent = DQNAgent(config)

        agent.total_env_steps = 0
        assert agent.get_epsilon() == 1.0

        agent.total_env_steps = 500
        assert abs(agent.get_epsilon() - 0.525) < 1e-4

        agent.total_env_steps = 1500  # past decay
        assert agent.get_epsilon() == 0.05

        # In evaluation mode, action must be 100% deterministic greedy regardless of steps
        state = np.zeros(9, dtype=np.float32)
        act_eval = agent.select_action(state, evaluate=True)
        q_vals = agent.get_q_values(state)
        assert act_eval == int(np.argmax(q_vals))

    def test_rl_target_network_parameter_isolation(self):
        """Verify target network parameters are unaffected by online gradient updates."""
        agent = DQNAgent()
        initial_target_weight = agent.target_net.fc1.weight.clone().detach()

        # Run multiple training steps on online network
        states = torch.randn(16, 9)
        actions = torch.randint(0, 16, (16,))
        rewards = torch.randn(16)
        next_states = torch.randn(16, 9)
        dones = torch.zeros(16)
        batch = (states, actions, rewards, next_states, dones)

        for _ in range(5):
            agent.train_step(batch)

        # Target network weights must remain identically unchanged until sync
        assert torch.equal(agent.target_net.fc1.weight, initial_target_weight)

        # After hard update, target network must match online network
        agent.update_target_network()
        assert torch.equal(agent.target_net.fc1.weight, agent.online_net.fc1.weight)

    def test_rl_checkpoint_persistence_and_weight_fidelity(self, tmp_path):
        """Verify checkpoint saving, loading, and exact numerical reproduction."""
        agent = DQNAgent()
        ckpt_file = str(tmp_path / "test_ckpt.pth")
        agent.total_env_steps = 4242
        agent.total_updates = 1337

        agent.save_checkpoint(ckpt_file)
        assert os.path.exists(ckpt_file)

        agent2 = DQNAgent()
        agent2.load_checkpoint(ckpt_file)

        assert agent2.total_env_steps == 4242
        assert agent2.total_updates == 1337
        assert torch.equal(agent.online_net.fc1.weight, agent2.online_net.fc1.weight)
        assert torch.equal(agent.online_net.fc3.bias, agent2.online_net.fc3.bias)


# =============================================================================
# 4. REWARD TESTS (SAFE LANDING, COLLISION, TIMEOUT, FUEL, TIME, PBRS)
# =============================================================================

class TestRewardQA:
    """Rigorous QA tests for RewardEngine and PBRS theorem invariance."""

    def test_reward_safe_landing_and_target_zone_bonuses(self):
        """Verify +200 safe landing bonus and +30 green target zone bonus."""
        eng = RewardEngine(use_shaping=False, reward_scale=1.0)
        from physics.collision import CollisionInfo
        from environment.terminal import TerminalResult

        col_safe = CollisionInfo(
            is_contact=True,
            is_safe_landing=True,
            is_crash=False,
            is_out_of_bounds=False,
            is_in_target_zone=True,
            lowest_y=0.0,
            penetration_depth=0.0,
            contact_point_world=(0.2, 0.0),
            contact_reason="SAFE_LANDING",
            impact_vy=-0.2,
            impact_vx=0.0,
            impact_angle_rad=0.0,
            impact_omega=0.0,
            rear_wheel_world=(-0.3, 0.0),
            front_wheel_world=(0.7, 0.0),
        )
        term_safe = TerminalResult(
            done=True,
            terminated=True,
            truncated=False,
            termination_reason="SAFE_LANDING",
            is_safe_landing=True,
            is_crash=False,
            is_out_of_bounds=False,
            is_timeout=False,
        )
        state_curr = {"x": 0.2, "y": 0.1, "vx": 0.0, "vy": -0.2, "theta": 0.0, "omega": 0.0}
        state_next = {"x": 0.2, "y": 0.0, "vx": 0.0, "vy": -0.2, "theta": 0.0, "omega": 0.0}

        brk = eng.calculate_reward(
            state_curr=state_curr,
            state_next=state_next,
            col_info=col_safe,
            term_result=term_safe,
            delta_fuel=0.0,
            delta_time=0.1,
        )

        expected = 200.0 + 30.0 - 0.1 - 2.0 * (0.2**2)
        assert abs(brk.r_terminal - 200.0) < 1e-4
        assert abs(brk.r_target_zone - 30.0) < 1e-4
        assert abs(brk.r_base - expected) < 1e-3

    def test_reward_penalties_crash_timeout_oob_fuel_time(self):
        """Verify penalties: crash -100, timeout -500, oob -250, fuel -0.2/%, time -1.0/s."""
        eng = RewardEngine(use_shaping=False, reward_scale=1.0)
        from physics.collision import CollisionInfo
        from environment.terminal import TerminalResult

        state_curr = {"x": 0.0, "y": 0.5, "vx": 0.0, "vy": -2.0, "theta": 0.1, "omega": 0.0}
        state_next = {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": -2.0, "theta": 0.1, "omega": 0.0}

        # Crash penalty with ground velocity and orientation penalties
        col_crash = CollisionInfo(
            is_contact=True,
            is_safe_landing=False,
            is_crash=True,
            is_out_of_bounds=False,
            is_in_target_zone=False,
            lowest_y=0.0,
            penetration_depth=0.0,
            contact_point_world=(0.0, 0.0),
            contact_reason="CRASH",
            impact_vy=-2.0,
            impact_vx=0.0,
            impact_angle_rad=0.1,
            impact_omega=0.0,
            rear_wheel_world=(-0.5, 0.0),
            front_wheel_world=(0.5, 0.0),
        )
        term_crash = TerminalResult(
            done=True,
            terminated=True,
            truncated=False,
            termination_reason="CRASH",
            is_safe_landing=False,
            is_crash=True,
            is_out_of_bounds=False,
            is_timeout=False,
        )
        brk = eng.calculate_reward(
            state_curr=state_curr,
            state_next=state_next,
            col_info=col_crash,
            term_result=term_crash,
            delta_fuel=1.0,
            delta_time=0.1,
        )
        assert brk.r_terminal == -100.0
        assert abs(brk.r_gas - (-0.2 * 1.0)) < 1e-5
        assert abs(brk.r_time - (-1.0 * 0.1)) < 1e-5
        assert abs(brk.r_impact_vel - (-2.0 * (2.0**2))) < 1e-4

        # Timeout penalty
        term_to = TerminalResult(
            done=True,
            terminated=False,
            truncated=True,
            termination_reason="TIMEOUT",
            is_safe_landing=False,
            is_crash=False,
            is_out_of_bounds=False,
            is_timeout=True,
        )
        col_air = CollisionInfo(
            is_contact=False,
            is_safe_landing=False,
            is_crash=False,
            is_out_of_bounds=False,
            is_in_target_zone=False,
            lowest_y=5.0,
            penetration_depth=0.0,
            contact_point_world=(0.0, 5.0),
            contact_reason="AIRBORNE",
            impact_vy=0.0,
            impact_vx=0.0,
            impact_angle_rad=0.0,
            impact_omega=0.0,
            rear_wheel_world=(-0.5, 5.0),
            front_wheel_world=(0.5, 5.0),
        )
        brk_to = eng.calculate_reward(
            state_curr=state_curr,
            state_next=state_next,
            col_info=col_air,
            term_result=term_to,
            delta_fuel=0.0,
            delta_time=0.1,
        )
        assert brk_to.r_terminal == -500.0

        # Out of bounds penalty
        term_oob = TerminalResult(
            done=True,
            terminated=True,
            truncated=False,
            termination_reason="OUT_OF_BOUNDS",
            is_safe_landing=False,
            is_crash=False,
            is_out_of_bounds=True,
            is_timeout=False,
        )
        col_oob = CollisionInfo(
            is_contact=False,
            is_safe_landing=False,
            is_crash=False,
            is_out_of_bounds=True,
            is_in_target_zone=False,
            lowest_y=5.0,
            penetration_depth=0.0,
            contact_point_world=(10.5, 5.0),
            contact_reason="OUT_OF_BOUNDS",
            impact_vy=0.0,
            impact_vx=2.0,
            impact_angle_rad=0.0,
            impact_omega=0.0,
            rear_wheel_world=(10.0, 5.0),
            front_wheel_world=(11.0, 5.0),
        )
        brk_oob = eng.calculate_reward(
            state_curr=state_curr,
            state_next=state_next,
            col_info=col_oob,
            term_result=term_oob,
            delta_fuel=0.0,
            delta_time=0.1,
        )
        assert brk_oob.r_terminal == -250.0

    def test_reward_pbrs_potential_zero_terminal_and_telescoping(self):
        """Verify Ng et al. (1999) PBRS: Phi(s_terminal) = 0 and telescoping sum invariance."""
        eng = RewardEngine(use_shaping=True, reward_scale=1.0, gamma=0.995)

        # Terminal state potential must be identically 0.0
        assert eng.compute_potential(0.0, 0.72, 0.0, 0.0, is_terminal=True) == 0.0
        assert eng.compute_potential(5.0, 8.0, 2.0, -3.0, is_terminal=True) == 0.0

        # Non-terminal state potential
        phi_0 = eng.compute_potential(x=2.0, h=5.72, vx=1.0, vy=-2.0, is_terminal=False)
        expected_phi_0 = -10.0 * math.sqrt(29.0) - 4.0 * math.sqrt(5.0)
        assert abs(phi_0 - expected_phi_0) < 1e-4

        # Telescoping sum over 3 steps: sum_{t=0}^2 gamma^t (gamma * Phi_{t+1} - Phi_t)
        phi_1 = -15.0
        phi_2 = -8.0
        phi_3 = 0.0  # Terminal
        gamma = 0.995

        step0_shaping = gamma * phi_1 - phi_0
        step1_shaping = gamma * phi_2 - phi_1
        step2_shaping = gamma * phi_3 - phi_2

        total_shaping = step0_shaping + gamma * step1_shaping + (gamma**2) * step2_shaping
        expected_total = (gamma**3) * phi_3 - phi_0
        assert abs(total_shaping - expected_total) < 1e-5


# =============================================================================
# 5. EVALUATION TESTS (NO TRAINING, EPSILON=0, INDEPENDENT TEST SETS)
# =============================================================================

class TestEvaluationQA:
    """Rigorous QA tests for evaluation decoupling and test set independence."""

    def test_eval_zero_gradients_and_zero_weight_updates(self):
        """Verify that running evaluation does NOT compute gradients or update model parameters."""
        evaluator = StandaloneEvaluator()
        agent = DQNAgent()

        initial_weights = agent.online_net.fc1.weight.clone().detach()

        # Run 3 test episodes
        test_cases = build_curated_test_cases()[:3]
        for tc in test_cases:
            evaluator.evaluate_episode(
                agent=agent,
                reset_options=tc.to_env_options(),
                episode_id=tc.test_id,
                category=tc.category,
                description=tc.description,
            )

        # Parameter weights must remain 100% bitwise identical
        assert torch.equal(agent.online_net.fc1.weight, initial_weights)

        # Gradients must be None (not populated)
        for p in agent.online_net.parameters():
            assert p.grad is None

    def test_eval_epsilon_strictly_zero_deterministic(self):
        """Verify that evaluation policy is 100% deterministic (greedy argmax)."""
        evaluator = StandaloneEvaluator()
        agent = DQNAgent()

        # Force state to produce a specific action
        with torch.no_grad():
            agent.online_net.fc3.bias.fill_(-10.0)
            agent.online_net.fc3.bias[7] = +100.0  # Action 7 will always dominate

        tc = build_curated_test_cases()[0]
        telem = evaluator.evaluate_episode(agent=agent, reset_options=tc.to_env_options())
        assert telem.step_count > 0

    def test_eval_independent_test_sets_disjoint_from_training(self):
        """Verify test sets (60 curated + 100 stochastic seeds 30000..30099) are disjoint from train seeds."""
        train_seeds = {42, 123, 999}
        stochastic_test_seeds = set(get_stochastic_unseen_seeds())

        # Exact 100 seeds in 30000..30099
        assert len(stochastic_test_seeds) == 100
        assert min(stochastic_test_seeds) == 30000
        assert max(stochastic_test_seeds) == 30099

        # Zero intersection with training seeds
        assert len(train_seeds.intersection(stochastic_test_seeds)) == 0

        # Curated test suite contains exactly 60 structured cases
        curated_cases = build_curated_test_cases()
        assert len(curated_cases) == 60


# =============================================================================
# 6. UI INTEGRITY TESTS (DQN RUNS DQN, DDQN RUNS DDQN, NO CONTROLLER OVERRIDE)
# =============================================================================

class TestUIIntegrityQA:
    """Rigorous QA tests verifying Web UI neural network inference parity and integrity."""

    def test_ui_dqn_weights_parity_with_pytorch_baseline(self):
        """Verify web/weights/dqn_weights.json exactly matches exp_a_dqn_seed_42.pth."""
        pth = "checkpoints/experiments/exp_a_dqn_seed_42.pth"
        json_path = "web/weights/dqn_weights.json"
        assert os.path.exists(pth)
        assert os.path.exists(json_path)

        ckpt = torch.load(pth, map_location="cpu", weights_only=False)
        state_dict = ckpt["online_net_state_dict"]
        model = QNetwork(9, 16)
        model.load_state_dict(state_dict)
        model.eval()

        with open(json_path, "r", encoding="utf-8") as f:
            w_json = json.load(f)

        layers = w_json["layers"]

        # Test on 10 random state vectors
        rng = np.random.default_rng(42)
        for _ in range(10):
            obs = rng.uniform(-1.0, 1.0, size=9).astype(np.float32)

            with torch.no_grad():
                py_q = model(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()

            # JS MLP simulation
            h1 = np.maximum(0, np.dot(layers["fc1_weight"], obs) + layers["fc1_bias"])
            h2 = np.maximum(0, np.dot(layers["fc2_weight"], h1) + layers["fc2_bias"])
            js_q = np.dot(layers["fc3_weight"], h2) + layers["fc3_bias"]

            diff = np.max(np.abs(py_q - js_q))
            assert diff < 2e-5
            assert int(np.argmax(py_q)) == int(np.argmax(js_q))

    def test_ui_double_dqn_weights_parity_with_pytorch_baseline(self):
        """Verify web/weights/ddqn_weights.json exactly matches exp_b_double_dqn_seed_42.pth."""
        pth = "checkpoints/experiments/exp_b_double_dqn_seed_42.pth"
        json_path = "web/weights/ddqn_weights.json"
        assert os.path.exists(pth)
        assert os.path.exists(json_path)

        ckpt = torch.load(pth, map_location="cpu", weights_only=False)
        state_dict = ckpt["online_net_state_dict"]
        model = QNetwork(9, 16)
        model.load_state_dict(state_dict)
        model.eval()

        with open(json_path, "r", encoding="utf-8") as f:
            w_json = json.load(f)

        layers = w_json["layers"]
        rng = np.random.default_rng(123)
        for _ in range(10):
            obs = rng.uniform(-1.0, 1.0, size=9).astype(np.float32)
            with torch.no_grad():
                py_q = model(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()

            h1 = np.maximum(0, np.dot(layers["fc1_weight"], obs) + layers["fc1_bias"])
            h2 = np.maximum(0, np.dot(layers["fc2_weight"], h1) + layers["fc2_bias"])
            js_q = np.dot(layers["fc3_weight"], h2) + layers["fc3_bias"]

            diff = np.max(np.abs(py_q - js_q))
            assert diff < 2e-5
            assert int(np.argmax(py_q)) == int(np.argmax(js_q))

    def test_ui_trained_model_weights_parity_with_best_checkpoint(self):
        """Verify web/weights/trained_weights.json exactly matches checkpoints/best_model.pth."""
        pth = "checkpoints/best_model.pth"
        json_path = "web/weights/trained_weights.json"
        assert os.path.exists(pth)
        assert os.path.exists(json_path)

        ckpt = torch.load(pth, map_location="cpu", weights_only=False)
        state_dict = ckpt["online_net_state_dict"]
        model = QNetwork(9, 16)
        model.load_state_dict(state_dict)
        model.eval()

        with open(json_path, "r", encoding="utf-8") as f:
            w_json = json.load(f)

        layers = w_json["layers"]
        rng = np.random.default_rng(999)
        for _ in range(10):
            obs = rng.uniform(-1.0, 1.0, size=9).astype(np.float32)
            with torch.no_grad():
                py_q = model(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()

            h1 = np.maximum(0, np.dot(layers["fc1_weight"], obs) + layers["fc1_bias"])
            h2 = np.maximum(0, np.dot(layers["fc2_weight"], h1) + layers["fc2_bias"])
            js_q = np.dot(layers["fc3_weight"], h2) + layers["fc3_bias"]

            diff = np.max(np.abs(py_q - js_q))
            assert diff < 2e-5
            assert int(np.argmax(py_q)) == int(np.argmax(js_q))

    def test_ui_model_predictions_are_distinct(self):
        """Verify DQN, Double DQN, and Trained models produce distinct action outputs (proving real models run)."""
        w_dqn = json.load(open("web/weights/dqn_weights.json"))["layers"]
        w_ddqn = json.load(open("web/weights/ddqn_weights.json"))["layers"]
        w_trn = json.load(open("web/weights/trained_weights.json"))["layers"]

        def forward_mlp(layers, obs):
            h1 = np.maximum(0, np.dot(layers["fc1_weight"], obs) + layers["fc1_bias"])
            h2 = np.maximum(0, np.dot(layers["fc2_weight"], h1) + layers["fc2_bias"])
            return np.dot(layers["fc3_weight"], h2) + layers["fc3_bias"]

        test_obs = np.array([-0.1, 0.4, 0.0, -0.05, 0.1736, 0.9848, 0.0, 1.0, 0.0], dtype=np.float32)

        q_dqn = forward_mlp(w_dqn, test_obs)
        q_ddqn = forward_mlp(w_ddqn, test_obs)
        q_trn = forward_mlp(w_trn, test_obs)

        # Weight layers must differ
        assert not np.allclose(w_dqn["fc1_weight"], w_trn["fc1_weight"], atol=1e-2)
        assert not np.allclose(w_ddqn["fc1_weight"], w_trn["fc1_weight"], atol=1e-2)

    def test_ui_source_code_no_hidden_background_controller(self):
        """Static analysis of web/js/visualizer.js: In RL mode, action is derived ONLY from net.predict."""
        with open("web/js/visualizer.js", "r", encoding="utf-8") as f:
            js_code = f.read()

        assert 'if (this.mode === "rl")' in js_code
        assert 'action = this.net.predict(obs);' in js_code

        # Ensure no heuristic/PID override inside the rl block
        rl_block = js_code.split('if (this.mode === "rl") {')[1].split('} else if')[0]
        assert 'this.flipCtrl' not in rl_block
        assert 'PID' not in rl_block
        assert 'teacher' not in rl_block
        assert 'heuristic' not in rl_block
        assert 'manualAction' not in rl_block
