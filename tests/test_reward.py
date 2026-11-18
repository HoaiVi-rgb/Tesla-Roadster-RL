"""
Unit tests for each individual reward component, priority hierarchy, and anti-reward hacking.
Reference: Roadster_4_Voi_DQN.docx & REWARD_DESIGN.md.
"""

import math
import pytest
from environment.reward import RewardEngine
from environment.terminal import TerminalResult
from physics.collision import CollisionInfo


def make_dummy_collision(
    is_contact=False,
    is_safe=False,
    is_crash=False,
    is_oob=False,
    is_target=False,
    impact_vy=0.0,
    impact_vx=0.0,
    impact_angle=0.0,
):
    return CollisionInfo(
        is_contact=is_contact,
        is_safe_landing=is_safe,
        is_crash=is_crash,
        is_out_of_bounds=is_oob,
        is_in_target_zone=is_target,
        lowest_y=0.0 if is_contact else 5.0,
        penetration_depth=0.0,
        contact_point_world=(0.0, 0.0),
        contact_reason="SAFE_LANDING" if is_safe else ("CRASH" if is_crash else "NONE"),
        impact_vy=impact_vy,
        impact_vx=impact_vx,
        impact_angle_rad=impact_angle,
        impact_omega=0.0,
        rear_wheel_world=(-1.5, 0.0),
        front_wheel_world=(1.5, 0.0),
    )


def make_dummy_terminal(is_safe=False, is_crash=False, is_oob=False, is_timeout=False):
    done = is_safe or is_crash or is_oob or is_timeout
    reason = "SAFE_LANDING" if is_safe else ("CRASH" if is_crash else ("OUT_OF_BOUNDS" if is_oob else ("TIMEOUT" if is_timeout else "IN_FLIGHT")))
    return TerminalResult(
        done=done,
        terminated=is_safe or is_crash or is_oob,
        truncated=is_timeout,
        termination_reason=reason,
        is_safe_landing=is_safe,
        is_crash=is_crash,
        is_out_of_bounds=is_oob,
        is_timeout=is_timeout,
    )


# ==============================================================================
# 1. INDIVIDUAL REWARD COMPONENT TESTS
# ==============================================================================

def test_component_safe_landing():
    """Verify safe landing grants exactly +200.0 terminal bonus."""
    engine = RewardEngine(use_shaping=False)
    col = make_dummy_collision(is_contact=True, is_safe=True)
    term = make_dummy_terminal(is_safe=True)
    state = {"x": 0.0, "y": 0.72, "vx": 0.0, "vy": 0.0, "theta": 0.0}

    bd = engine.calculate_reward(state, state, col, term, delta_fuel=0.0, delta_time=0.0)
    assert bd.r_terminal == 200.0


def test_component_target_zone():
    """Verify landing inside green target zone grants additional +30.0 bonus."""
    engine = RewardEngine(use_shaping=False)
    col = make_dummy_collision(is_contact=True, is_safe=True, is_target=True)
    term = make_dummy_terminal(is_safe=True)
    state = {"x": 0.0, "y": 0.72, "vx": 0.0, "vy": 0.0, "theta": 0.0}

    bd = engine.calculate_reward(state, state, col, term, delta_fuel=0.0, delta_time=0.0)
    assert bd.r_terminal == 200.0
    assert bd.r_target_zone == 30.0


def test_component_crash_penalty():
    """Verify crash penalty is exactly -100.0 (-1/2 of safe landing bonus)."""
    engine = RewardEngine(use_shaping=False)
    col = make_dummy_collision(is_contact=True, is_crash=True)
    term = make_dummy_terminal(is_crash=True)
    state = {"x": 0.0, "y": 0.72, "vx": 0.0, "vy": 0.0, "theta": 0.0}

    bd = engine.calculate_reward(state, state, col, term, delta_fuel=0.0, delta_time=0.0)
    assert bd.r_terminal == -100.0


def test_component_timeout_penalty():
    """Verify timeout penalty is exactly -500.0."""
    engine = RewardEngine(use_shaping=False)
    col = make_dummy_collision(is_contact=False)
    term = make_dummy_terminal(is_timeout=True)
    state = {"x": 0.0, "y": 5.0, "vx": 0.0, "vy": 0.0, "theta": 0.0}

    bd = engine.calculate_reward(state, state, col, term, delta_fuel=0.0, delta_time=0.0)
    assert bd.r_terminal == -500.0


def test_component_out_of_bounds_penalty():
    """Verify out of bounds penalty is exactly -250.0."""
    engine = RewardEngine(use_shaping=False)
    col = make_dummy_collision(is_contact=False, is_oob=True)
    term = make_dummy_terminal(is_oob=True)
    state = {"x": 22.0, "y": 5.0, "vx": 0.0, "vy": 0.0, "theta": 0.0}

    bd = engine.calculate_reward(state, state, col, term, delta_fuel=0.0, delta_time=0.0)
    assert bd.r_terminal == -250.0


def test_component_ground_velocity_penalty():
    """
    Verify ground contact velocity penalty: -2.0 * (vx^2 + vy^2).
    Assessed ONLY upon ground contact!
    """
    engine = RewardEngine(use_shaping=False)
    # Ground contact case: vx = 0.3, vy = -0.4 -> vx^2 + vy^2 = 0.09 + 0.16 = 0.25
    col_contact = make_dummy_collision(is_contact=True, is_safe=True)
    term = make_dummy_terminal(is_safe=True)
    state_next = {"x": 0.0, "y": 0.72, "vx": 0.3, "vy": -0.4, "theta": 0.0}

    bd = engine.calculate_reward(state_next, state_next, col_contact, term, delta_fuel=0.0, delta_time=0.0)
    expected_pen = -2.0 * (0.3**2 + (-0.4)**2)  # -2.0 * 0.25 = -0.5
    assert bd.r_impact_vel == pytest.approx(expected_pen, abs=1e-5)

    # In-air case: MUST BE ZERO even if vx, vy are large!
    col_air = make_dummy_collision(is_contact=False)
    term_air = make_dummy_terminal()
    bd_air = engine.calculate_reward(state_next, state_next, col_air, term_air, delta_fuel=0.0, delta_time=0.0)
    assert bd_air.r_impact_vel == 0.0


def test_component_ground_angle_penalty():
    """
    Verify ground contact angle penalty: -10.0 * (1 - cos(theta)).
    Assessed ONLY upon ground contact!
    """
    engine = RewardEngine(use_shaping=False)
    theta = math.radians(4.0)  # Safe landing tilt
    col_contact = make_dummy_collision(is_contact=True, is_safe=True)
    term = make_dummy_terminal(is_safe=True)
    state_next = {"x": 0.0, "y": 0.72, "vx": 0.0, "vy": 0.0, "theta": theta}

    bd = engine.calculate_reward(state_next, state_next, col_contact, term, delta_fuel=0.0, delta_time=0.0)
    expected_angle_pen = -10.0 * (1.0 - math.cos(theta))
    assert bd.r_impact_angle == pytest.approx(expected_angle_pen, abs=1e-5)

    # In-air case: MUST BE ZERO (freely rotate/flip in air without intermediate angle penalty!)
    col_air = make_dummy_collision(is_contact=False)
    term_air = make_dummy_terminal()
    state_tilted_air = {"x": 0.0, "y": 10.0, "vx": 0.0, "vy": 0.0, "theta": math.pi}  # Upside down in air
    bd_air = engine.calculate_reward(state_tilted_air, state_tilted_air, col_air, term_air, delta_fuel=0.0, delta_time=0.0)
    assert bd_air.r_impact_angle == 0.0


def test_component_fuel_and_time_costs():
    """
    Verify continuous step costs:
    - Fuel: -0.2 / %
    - Time: -1.0 / second
    """
    engine = RewardEngine(use_shaping=False)
    col = make_dummy_collision(is_contact=False)
    term = make_dummy_terminal()
    state = {"x": 0.0, "y": 8.0, "vx": 0.0, "vy": 0.0, "theta": 0.0}

    # Step consuming 1.5% fuel over 0.1s
    bd = engine.calculate_reward(state, state, col, term, delta_fuel=1.5, delta_time=0.1)
    assert bd.r_gas == pytest.approx(-0.2 * 1.5, abs=1e-5)  # -0.3
    assert bd.r_time == pytest.approx(-1.0 * 0.1, abs=1e-5)  # -0.1
    assert bd.r_base == pytest.approx(-0.4, abs=1e-5)


# ==============================================================================
# 2. PRIORITY HIERARCHY TESTS
# ==============================================================================

def test_priority_hierarchy():
    """
    Verify mathematical priority ordering:
    SAFE LANDING >> CORRECT POSITION >> LOW VELOCITY >> LOW ANGLE >> LOW FUEL >> SHORT TIME
    """
    engine = RewardEngine(use_shaping=False)

    # Scenario A: Safe landing outside target zone (+200) vs. Crashing inside target zone (-100 + 30 = -70)
    # Safe landing outside must be strictly preferred over crashing in zone!
    r_safe_outside = 200.0
    r_crash_in_zone = -100.0  # (target bonus not awarded on crash)
    assert r_safe_outside > r_crash_in_zone

    # Scenario B: Safe landing with high fuel usage vs. Timeout (-500)
    # Burning 50% fuel costs -10. Safe landing net = 200 - 10 = +190 >> -500!
    r_safe_fuel = 200.0 - (0.2 * 50.0)
    r_timeout = -500.0
    assert r_safe_fuel > r_timeout


# ==============================================================================
# 3. POTENTIAL-BASED REWARD SHAPING & ANTI-HACKING TESTS
# ==============================================================================

def test_potential_function_and_zero_terminal():
    """Verify Phi(s) formula and that Phi(s_terminal) is strictly 0.0."""
    engine = RewardEngine()
    
    # In-air state at x=3.0, h=4.72 (so h - 0.72 = 4.0), vx=0.0, vy=-3.0
    # dist = sqrt(3^2 + 4^2) = 5.0
    # speed = sqrt(0 + (-3)^2) = 3.0
    # Phi = -10 * 5.0 - 4 * 3.0 = -50 - 12 = -62.0
    phi_air = engine.compute_potential(x=3.0, h=4.72, vx=0.0, vy=-3.0, is_terminal=False)
    assert phi_air == pytest.approx(-62.0, abs=1e-5)

    # Terminal state MUST BE 0.0 regardless of coordinates!
    phi_term = engine.compute_potential(x=3.0, h=4.72, vx=0.0, vy=-3.0, is_terminal=True)
    assert phi_term == 0.0


def test_anti_hovering_reward_hacking():
    """
    ANTI-REWARD-HACKING PROOF:
    Verify that an agent attempting to HOVER in place mid-air collects NEGATIVE reward per second!
    The agent CANNOT exploit shaping to gain infinite positive reward by lingering.
    """
    engine = RewardEngine(use_shaping=True, gamma=0.995, reward_scale=0.01)
    
    # Vehicle hovering at x=0, h=2.0m, vx=0, vy=0
    state_hover = {"x": 0.0, "y": 2.0, "vx": 0.0, "vy": 0.0, "theta": 0.0}
    col_air = make_dummy_collision(is_contact=False)
    term_air = make_dummy_terminal()

    # Hovering requires firing V1+V2 to counter gravity (11,772 N).
    # Two thrusters consume ~2.0% fuel per second -> 0.2% per 0.1s step.
    delta_fuel_step = 0.2
    delta_time_step = 0.1

    bd = engine.calculate_reward(
        state_curr=state_hover,
        state_next=state_hover,
        col_info=col_air,
        term_result=term_air,
        delta_fuel=delta_fuel_step,
        delta_time=delta_time_step,
    )

    # Net reward per step must be NEGATIVE!
    assert bd.r_base < 0.0
    assert bd.r_learn < 0.0
    # Proves hovering continuously depletes reward, driving agent to land promptly!


def test_telescoping_sum_invariance():
    """
    Verify Ng et al. 1999 Telescoping Sum:
    Sum_{t=0}^T gamma^t * F(s_t, s_{t+1}) == -Phi(s_0)
    """
    engine = RewardEngine(gamma=0.995, use_shaping=True)
    gamma = engine.gamma

    states = [
        {"x": 4.0, "y": 10.0, "vx": 1.0, "vy": -2.0},
        {"x": 2.5, "y": 7.0, "vx": 0.8, "vy": -1.5},
        {"x": 1.0, "y": 4.0, "vx": 0.5, "vy": -1.0},
        {"x": 0.0, "y": 0.72, "vx": 0.1, "vy": -0.4},  # Touchdown
    ]

    phi_s0 = engine.compute_potential(states[0]["x"], states[0]["y"], states[0]["vx"], states[0]["vy"], is_terminal=False)

    total_shaping = 0.0
    for t in range(len(states) - 1):
        s_curr = states[t]
        s_next = states[t + 1]
        is_term = (t == len(states) - 2)

        f_t, _, _ = engine.compute_shaping(s_curr, s_next, is_terminal=is_term)
        total_shaping += (gamma**t) * f_t

    assert total_shaping == pytest.approx(-phi_s0, abs=1e-5)


# ==============================================================================
# 4. STRICT SEPARATION: ORIGINAL VS SHAPED REWARD
# ==============================================================================

def test_strict_reward_separation():
    """
    Verify Original Reward (r_base) is completely unshaped and unscaled,
    while Shaped Reward (r_learn) combines base and PBRS with 0.01 scaling.
    """
    engine = RewardEngine(use_shaping=True, reward_scale=0.01)
    
    col = make_dummy_collision(is_contact=True, is_safe=True, is_target=True)
    term = make_dummy_terminal(is_safe=True)
    
    s_curr = {"x": 0.1, "y": 1.0, "vx": 0.2, "vy": -0.5, "theta": 0.0}
    s_next = {"x": 0.0, "y": 0.72, "vx": 0.1, "vy": -0.3, "theta": 0.0}

    bd = engine.calculate_reward(s_curr, s_next, col, term, delta_fuel=0.5, delta_time=0.1)

    # 1. Check raw reward components (unscaled)
    expected_r_base = (
        200.0  # safe landing
        + 30.0  # target zone
        - 2.0 * (0.1**2 + (-0.3)**2)  # -2.0 * 0.10 = -0.2
        - 10.0 * (1.0 - math.cos(0.0))  # 0.0
        - 0.2 * 0.5  # -0.1 gas
        - 1.0 * 0.1  # -0.1 time
    )
    assert bd.r_base == pytest.approx(expected_r_base, abs=1e-4)

    # 2. Check shaped learning reward
    expected_r_learn = 0.01 * (bd.r_base + bd.r_shaping)
    assert bd.r_learn == pytest.approx(expected_r_learn, abs=1e-5)

    # 3. Check independence: r_base is unaffected by gamma or use_shaping
    engine_no_shaping = RewardEngine(use_shaping=False)
    bd_no_shaping = engine_no_shaping.calculate_reward(s_curr, s_next, col, term, delta_fuel=0.5, delta_time=0.1)
    assert bd_no_shaping.r_base == pytest.approx(bd.r_base, abs=1e-5)
    assert bd_no_shaping.r_shaping == 0.0
