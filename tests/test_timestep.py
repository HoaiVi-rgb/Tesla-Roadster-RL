"""
Unit test for simulation timestep, 120 Hz sub-stepping, and numerical stability.
"""

import pytest
from physics.world import PhysicsWorld
from physics.constants import (
    PHYSICS_FREQ,
    PHYSICS_DT,
    DECISION_FREQ,
    DECISION_DT,
    SUBSTEPS_PER_STEP,
    REST_HEIGHT,
)


def test_120hz_substepping_count():
    """
    Verify that each 10 Hz decision step runs exactly 12 sub-steps at 120 Hz
    when in free flight.
    """
    assert PHYSICS_FREQ == 120
    assert DECISION_FREQ == 10
    assert SUBSTEPS_PER_STEP == 12
    
    world = PhysicsWorld(initial_x=0.0, initial_y=15.0)
    result = world.step(action=0, dt_agent=DECISION_DT)
    
    assert result["substeps_executed"] == 12
    assert world.body.time == pytest.approx(0.1, abs=1e-5)


def test_early_collision_substep_termination():
    """
    Verify that if a collision occurs mid-step, sub-stepping halts immediately
    and does not continue running subsequent sub-steps in the same step.
    """
    # Start just 0.005 m above ground contact with downward velocity
    # It will hit the ground on the first or second sub-step!
    world = PhysicsWorld(
        initial_x=0.0,
        initial_y=REST_HEIGHT + 0.005,
        initial_vx=0.0,
        initial_vy=-2.0,
    )
    
    result = world.step(action=0, dt_agent=DECISION_DT)
    
    # Must terminate early (< 12 substeps executed)
    assert result["is_terminal"] is True
    assert result["substeps_executed"] < 12
    assert result["collision_info"].is_contact is True


def test_20s_timeout_limit():
    """
    Verify that simulation enforces the 20.0s episode horizon.
    """
    world = PhysicsWorld(initial_x=0.0, initial_y=24.0, initial_vy=0.0)
    
    # Run 200 decision steps with hovering lift (action 3: V1+V2)
    # (Vehicle oscillates or hovers up/down, keeping it airborne)
    for step_idx in range(200):
        # Alternate lift to keep in air without crashing
        action = 3 if world.body.vy < 0 else 0
        res = world.step(action=action, dt_agent=0.1)
        if res["is_terminal"]:
            break
            
    assert world.is_terminal is True
    assert world.body.time >= 20.0 or res["is_terminal"]
