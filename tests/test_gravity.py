"""
Unit test for free fall and gravitational acceleration.
Verifies Newton's second law: a_y = -g = -9.81 m/s^2.
"""

import math
import pytest
from physics.world import PhysicsWorld
from physics.constants import GRAVITY_ACCEL, VEHICLE_MASS, VEHICLE_INERTIA, PHYSICS_DT


def test_free_fall_acceleration():
    """Verify that without thrusters (action=0), acceleration is exactly -g."""
    world = PhysicsWorld(initial_x=0.0, initial_y=15.0, initial_vx=0.0, initial_vy=0.0)
    
    # Run 1 sub-step with action 0 (no thrusters)
    col, thrust = world.substep(action=0, dt=PHYSICS_DT)
    
    # Check linear acceleration: a_x = 0, a_y = -g
    assert world.body.last_ax == pytest.approx(0.0, abs=1e-6)
    assert world.body.last_ay == pytest.approx(-GRAVITY_ACCEL, abs=1e-5)
    assert world.body.last_alpha == pytest.approx(0.0, abs=1e-6)


def test_free_fall_kinematics():
    """Verify analytical kinematics: v_y(t) = -g*t and y(t) = y_0 - 0.5*g*t^2 (Semi-implicit Euler)."""
    initial_y = 20.0
    world = PhysicsWorld(initial_x=0.0, initial_y=initial_y, initial_vx=0.0, initial_vy=0.0)
    
    # Run for 1.0 second (10 decision steps = 120 substeps)
    total_time = 1.0
    for _ in range(10):
        res = world.step(action=0, dt_agent=0.1)
    
    assert world.body.time == pytest.approx(total_time, abs=1e-5)
    
    # Analytical velocity: v_y = -g * t = -9.81 m/s
    expected_vy = -GRAVITY_ACCEL * total_time
    assert world.body.vy == pytest.approx(expected_vy, abs=1e-4)
    
    # For Semi-implicit Euler, y(t) is extremely close to y0 - 0.5*g*t^2
    expected_y_cont = initial_y - 0.5 * GRAVITY_ACCEL * (total_time**2)
    assert world.body.y == pytest.approx(expected_y_cont, abs=0.1)
    
    # Horizontal position and orientation must remain strictly 0
    assert world.body.x == pytest.approx(0.0, abs=1e-6)
    assert world.body.vx == pytest.approx(0.0, abs=1e-6)
    assert world.body.theta == pytest.approx(0.0, abs=1e-6)
    assert world.body.omega == pytest.approx(0.0, abs=1e-6)
