"""
Unit test for cold-gas thruster fuel consumption and depletion cutoff.
"""

import pytest
from physics.world import PhysicsWorld
from physics.thrusters import ThrusterSystem
from physics.constants import (
    INITIAL_FUEL,
    FUEL_BURN_RATE_LIFT,
    FUEL_BURN_RATE_LATERAL,
    GRAVITY_ACCEL,
    PHYSICS_DT,
)


def test_fuel_consumption_rate():
    """
    Test fuel burn rate:
    - 1 lift thruster (V1) burns 1.0 %/s
    - After 1.0s, fuel should be 100.0 - 1.0 = 99.0%
    - 2 lift thrusters (V1 + V2) burn 2.0 %/s
    """
    world = PhysicsWorld(initial_fuel=100.0)
    
    # Fire V1 for 1.0s (10 steps of 0.1s)
    for _ in range(10):
        world.step(action=1, dt_agent=0.1)
    
    expected_fuel = 100.0 - (FUEL_BURN_RATE_LIFT * 1.0)
    assert world.thrusters.fuel == pytest.approx(expected_fuel, abs=1e-4)


def test_fuel_exhaustion_thruster_cutoff():
    """
    When fuel reaches 0, thrusters must produce exactly 0 force and 0 torque,
    and vehicle must fall under pure gravity.
    """
    # Start with only 0.1% fuel
    world = PhysicsWorld(initial_x=0.0, initial_y=15.0, initial_fuel=0.1)
    
    # Fire V1 + V2 (Action 3, consumes 2.0 %/s).
    # After 0.1s, it needs 0.2% fuel, so fuel depletes to 0.0%.
    world.step(action=3, dt_agent=0.1)
    assert world.thrusters.fuel == 0.0
    
    # Next sub-step: fuel is 0. Even if action 3 is requested, thrust must be 0!
    col, thrust_info = world.substep(action=3, dt=PHYSICS_DT)
    
    assert thrust_info["effective_valves"] == (0, 0, 0, 0)
    assert thrust_info["thrust_world"][0] == pytest.approx(0.0, abs=1e-6)
    assert thrust_info["thrust_world"][1] == pytest.approx(0.0, abs=1e-6)
    assert thrust_info["torque"] == pytest.approx(0.0, abs=1e-6)
    
    # Net acceleration must be purely gravitational: a_y = -g
    assert world.body.last_ay == pytest.approx(-GRAVITY_ACCEL, abs=1e-5)
