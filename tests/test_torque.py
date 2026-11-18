"""
Unit test for thruster torques and angular accelerations.
Verifies torque: tau = r x F, and alpha = tau / I_zz.
"""

import pytest
from physics.world import PhysicsWorld
from physics.thrusters import ThrusterSystem
from physics.constants import (
    THRUSTER_V1_TORQUE,
    THRUSTER_V2_TORQUE,
    VEHICLE_INERTIA,
    PHYSICS_DT,
)


def test_v1_torque_and_angular_acceleration():
    """
    V1: Rear lift thruster (Action 1).
    Position: x = -1.8 m, Force = +9,000 N.
    Torque = -1.8 * 9000 = -16,200 N*m.
    Expected alpha = -16,200 / 2,017.0 = -8.03173 rad/s^2 (clockwise).
    """
    world = PhysicsWorld(initial_x=0.0, initial_y=10.0, initial_vx=0.0, initial_vy=0.0, initial_theta=0.0)
    action_v1 = ThrusterSystem.encode_action(1, 0, 0, 0)
    assert action_v1 == 1
    
    world.substep(action=action_v1, dt=PHYSICS_DT)
    
    expected_alpha = THRUSTER_V1_TORQUE / VEHICLE_INERTIA  # -16200.0 / 2017.0
    assert world.body.last_alpha == pytest.approx(expected_alpha, abs=1e-4)
    assert world.body.omega < 0.0  # Angular velocity increases in clockwise direction


def test_v2_torque_and_angular_acceleration():
    """
    V2: Front lift thruster (Action 2).
    Position: x = +1.8 m, Force = +9,000 N.
    Torque = +1.8 * 9000 = +16,200 N*m.
    Expected alpha = +16,200 / 2,017.0 = +8.03173 rad/s^2 (counter-clockwise).
    """
    world = PhysicsWorld(initial_x=0.0, initial_y=10.0, initial_vx=0.0, initial_vy=0.0, initial_theta=0.0)
    action_v2 = ThrusterSystem.encode_action(0, 1, 0, 0)
    assert action_v2 == 2
    
    world.substep(action=action_v2, dt=PHYSICS_DT)
    
    expected_alpha = THRUSTER_V2_TORQUE / VEHICLE_INERTIA  # +16200.0 / 2017.0
    assert world.body.last_alpha == pytest.approx(expected_alpha, abs=1e-4)
    assert world.body.omega > 0.0  # Angular velocity increases in CCW direction


def test_balanced_v1_v2_zero_torque():
    """
    V1 + V2: Both lift thrusters active (Action 3).
    Net torque = -16,200 + 16,200 = 0.0 N*m.
    Angular acceleration must be exactly zero.
    """
    world = PhysicsWorld(initial_x=0.0, initial_y=10.0, initial_vx=0.0, initial_vy=0.0, initial_theta=0.0)
    action_v1_v2 = ThrusterSystem.encode_action(1, 1, 0, 0)
    
    world.substep(action=action_v1_v2, dt=PHYSICS_DT)
    
    assert world.body.last_alpha == pytest.approx(0.0, abs=1e-6)
    assert world.body.omega == pytest.approx(0.0, abs=1e-6)


def test_lateral_thrusters_zero_torque():
    """
    V3 and V4 are positioned on the centerline y = 0.
    Therefore, they must produce zero torque.
    """
    world = PhysicsWorld(initial_x=0.0, initial_y=10.0, initial_vx=0.0, initial_vy=0.0, initial_theta=0.0)
    
    # V3
    world.substep(action=ThrusterSystem.encode_action(0, 0, 1, 0), dt=PHYSICS_DT)
    assert world.body.last_alpha == pytest.approx(0.0, abs=1e-6)
    
    # V4
    world.reset(x=0.0, y=10.0, vx=0.0, vy=0.0, theta=0.0, omega=0.0)
    world.substep(action=ThrusterSystem.encode_action(0, 0, 0, 1), dt=PHYSICS_DT)
    assert world.body.last_alpha == pytest.approx(0.0, abs=1e-6)
