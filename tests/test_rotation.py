"""
Unit test for vehicle rotation dynamics and continuous flip tracking.
"""

import math
import pytest
from physics.world import PhysicsWorld
from physics.thrusters import ThrusterSystem
from physics.constants import (
    THRUSTER_V2_TORQUE,
    VEHICLE_INERTIA,
    PHYSICS_DT,
)


def test_continuous_rotation_kinematics():
    """
    Test continuous rotational integration:
    omega(t) = omega_0 + alpha * t
    theta(t) = theta_0 + omega_0 * t + 0.5 * alpha * t^2
    """
    world = PhysicsWorld(initial_x=0.0, initial_y=20.0, initial_vx=0.0, initial_vy=0.0, initial_theta=0.0, initial_omega=0.0)
    
    # Fire V2 for 0.5 seconds (alpha = +8.03173 rad/s^2)
    alpha = THRUSTER_V2_TORQUE / VEHICLE_INERTIA
    duration = 0.5  # 5 steps = 60 substeps
    
    for _ in range(5):
        world.step(action=2, dt_agent=0.1)
    
    expected_omega = alpha * duration
    assert world.body.omega == pytest.approx(expected_omega, abs=1e-3)
    
    # Semi-implicit Euler theta is very close to 0.5 * alpha * t^2
    expected_theta = 0.5 * alpha * (duration**2)
    assert world.body.theta == pytest.approx(expected_theta, abs=0.05)


def test_flip_detection():
    """
    Test counting 360-degree flips:
    A full 360-degree loop requires Delta theta = 2 * pi (~6.28318 rad).
    Verify that get_num_flips() increases accurately.
    """
    world = PhysicsWorld(initial_x=0.0, initial_y=20.0)
    
    # Artificially set initial angular velocity to complete 1 flip in 1 second: omega = 2 * pi rad/s
    two_pi = 2.0 * math.pi
    world.body.omega = two_pi
    
    assert world.get_num_flips() == 0
    
    # Step forward 0.5 seconds -> rotated pi rad (half flip)
    for _ in range(5):
        world.substep(action=0, dt=0.1)
    assert world.get_num_flips() == 0
    
    # Step forward another 0.55 seconds -> total rotation > 2*pi (1 full flip completed)
    for _ in range(6):
        world.substep(action=0, dt=0.1)
    assert world.get_num_flips() == 1
