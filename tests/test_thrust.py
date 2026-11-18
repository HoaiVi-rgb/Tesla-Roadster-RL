"""
Unit test for thruster forces and coordinate transformations.
Verifies lift, lateral forces, and rotation-dependent thrust vectors.
"""

import math
import pytest
import numpy as np

from physics.world import PhysicsWorld
from physics.thrusters import ThrusterSystem
from physics.constants import (
    THRUST_LIFT,
    THRUST_LATERAL,
    GRAVITY_ACCEL,
    VEHICLE_MASS,
    PHYSICS_DT,
)


def test_vertical_lift_acceleration():
    """
    Action 3 = V1 (rear lift) + V2 (front lift) = 18,000 N upward.
    Net vertical force = 18,000 N - 1,200 * 9.81 N = 18,000 - 11,772 = +6,228 N.
    Expected a_y = 6,228 / 1,200 = +5.19 m/s^2.
    """
    world = PhysicsWorld(initial_x=0.0, initial_y=10.0, initial_vx=0.0, initial_vy=0.0, initial_theta=0.0)
    
    # Action 3: binary 0011 -> V1=1, V2=1, V3=0, V4=0
    action = ThrusterSystem.encode_action(1, 1, 0, 0)
    assert action == 3
    
    world.substep(action=action, dt=PHYSICS_DT)
    
    expected_f_net = (2 * THRUST_LIFT) - (VEHICLE_MASS * GRAVITY_ACCEL)  # 6228.0 N
    expected_ay = expected_f_net / VEHICLE_MASS  # 5.19 m/s^2
    
    assert world.body.last_ay == pytest.approx(expected_ay, abs=1e-4)
    assert world.body.last_ax == pytest.approx(0.0, abs=1e-5)
    assert world.body.last_alpha == pytest.approx(0.0, abs=1e-5)  # Torques cancel out!


def test_lateral_thrust_acceleration():
    """
    V3 (Action 4: push right, 6000 N) -> a_x = +6000 / 1200 = +5.0 m/s^2.
    V4 (Action 8: push left, 6000 N)  -> a_x = -6000 / 1200 = -5.0 m/s^2.
    V3 + V4 (Action 12)               -> a_x = 0.0 m/s^2.
    """
    world = PhysicsWorld(initial_x=0.0, initial_y=10.0, initial_vx=0.0, initial_vy=0.0, initial_theta=0.0)
    
    # Test V3 (Push right)
    action_v3 = ThrusterSystem.encode_action(0, 0, 1, 0)
    assert action_v3 == 4
    world.substep(action=action_v3, dt=PHYSICS_DT)
    assert world.body.last_ax == pytest.approx(THRUST_LATERAL / VEHICLE_MASS, abs=1e-4)
    
    # Test V4 (Push left)
    world.reset(x=0.0, y=10.0, vx=0.0, vy=0.0, theta=0.0, omega=0.0)
    action_v4 = ThrusterSystem.encode_action(0, 0, 0, 1)
    assert action_v4 == 8
    world.substep(action=action_v4, dt=PHYSICS_DT)
    assert world.body.last_ax == pytest.approx(-THRUST_LATERAL / VEHICLE_MASS, abs=1e-4)
    
    # Test V3 + V4 (Both horizontal thrusters fire -> cancel out)
    world.reset(x=0.0, y=10.0, vx=0.0, vy=0.0, theta=0.0, omega=0.0)
    action_v3_v4 = ThrusterSystem.encode_action(0, 0, 1, 1)
    assert action_v3_v4 == 12
    world.substep(action=action_v3_v4, dt=PHYSICS_DT)
    assert world.body.last_ax == pytest.approx(0.0, abs=1e-4)


def test_rotated_thrust_vector():
    """
    When vehicle is rotated 90 degrees CCW (theta = pi/2):
    Body +y direction points along World -x direction!
    Therefore, lifting thrusters V1 + V2 (which push +y in body) must produce force along -x in world.
    """
    world = PhysicsWorld(initial_x=0.0, initial_y=15.0, initial_vx=0.0, initial_vy=0.0, initial_theta=math.pi / 2.0)
    
    action_lift = ThrusterSystem.encode_action(1, 1, 0, 0)  # V1 + V2 = 18,000 N along body +y
    world.substep(action=action_lift, dt=PHYSICS_DT)
    
    # At theta = 90 deg:
    # F_thrust_world = R(90 deg) * [0, 18000] = [-18000, 0]
    # a_x = -18000 / 1200 = -15.0 m/s^2
    # F_y_net = -mg = -11772 N -> a_y = -9.81 m/s^2
    assert world.body.last_ax == pytest.approx(-18000.0 / VEHICLE_MASS, abs=1e-4)
    assert world.body.last_ay == pytest.approx(-GRAVITY_ACCEL, abs=1e-4)
