"""
Unit test for collision detection, safe landing validation, and bounds checking.
"""

import math
import pytest
from physics.world import PhysicsWorld
from physics.constants import (
    REST_HEIGHT,
    MAX_SAFE_LANDING_VY,
    MAX_SAFE_LANDING_VX,
    MAX_SAFE_LANDING_ANGLE_RAD,
    MAX_SAFE_LANDING_OMEGA,
    TARGET_ZONE_X_MIN,
    TARGET_ZONE_X_MAX,
    WORLD_X_MIN,
    WORLD_X_MAX,
    WORLD_Y_MAX,
)


def test_safe_landing_nominal():
    """
    Test nominal gentle landing:
    vy = -0.5 m/s (safe < 1.0)
    vx = 0.2 m/s (safe < 0.5)
    theta = 2 degrees (safe < 5 deg)
    omega = 0.05 rad/s (safe < 0.2)
    Position x = 0.0 (inside green target zone)
    When vehicle touches ground, it must be declared SAFE_LANDING and in target zone.
    """
    # Touchdown height when theta = 0 is h = REST_HEIGHT = 0.72m
    world = PhysicsWorld(
        initial_x=0.0,
        initial_y=REST_HEIGHT,
        initial_vx=0.2,
        initial_vy=-0.5,
        initial_theta=math.radians(2.0),
        initial_omega=0.05,
    )
    
    col_info = world.collision_detector.check_collision(world.body)
    assert col_info.is_contact is True
    assert col_info.is_safe_landing is True
    assert col_info.is_crash is False
    assert col_info.is_in_target_zone is True
    assert col_info.contact_reason == "SAFE_LANDING"


def test_crash_high_vertical_velocity():
    """Crash when vy exceeds safe limit: |vy| = 1.8 m/s > 1.0 m/s."""
    world = PhysicsWorld(
        initial_x=0.0,
        initial_y=REST_HEIGHT,
        initial_vx=0.0,
        initial_vy=-1.8,
        initial_theta=0.0,
        initial_omega=0.0,
    )
    col_info = world.collision_detector.check_collision(world.body)
    assert col_info.is_contact is True
    assert col_info.is_safe_landing is False
    assert col_info.is_crash is True
    assert "vy=1.80>1.0" in col_info.contact_reason


def test_crash_high_horizontal_velocity():
    """Crash when vx exceeds safe limit: |vx| = 0.8 m/s > 0.5 m/s."""
    world = PhysicsWorld(
        initial_x=0.0,
        initial_y=REST_HEIGHT,
        initial_vx=0.8,
        initial_vy=-0.4,
        initial_theta=0.0,
        initial_omega=0.0,
    )
    col_info = world.collision_detector.check_collision(world.body)
    assert col_info.is_contact is True
    assert col_info.is_safe_landing is False
    assert col_info.is_crash is True
    assert "vx=0.80>0.5" in col_info.contact_reason


def test_crash_high_tilt_angle():
    """Crash when angle exceeds 5 degrees: theta = 12 degrees > 5 degrees."""
    world = PhysicsWorld(
        initial_x=0.0,
        initial_y=REST_HEIGHT,
        initial_vx=0.1,
        initial_vy=-0.4,
        initial_theta=math.radians(12.0),
        initial_omega=0.0,
    )
    col_info = world.collision_detector.check_collision(world.body)
    assert col_info.is_contact is True
    assert col_info.is_safe_landing is False
    assert col_info.is_crash is True
    assert "angle=12.0>5.0" in col_info.contact_reason


def test_crash_high_spin_rate():
    """Crash when angular velocity exceeds 0.2 rad/s: omega = 0.6 rad/s."""
    world = PhysicsWorld(
        initial_x=0.0,
        initial_y=REST_HEIGHT,
        initial_vx=0.1,
        initial_vy=-0.4,
        initial_theta=0.0,
        initial_omega=0.6,
    )
    col_info = world.collision_detector.check_collision(world.body)
    assert col_info.is_contact is True
    assert col_info.is_safe_landing is False
    assert col_info.is_crash is True
    assert "omega=0.60>0.2" in col_info.contact_reason


def test_target_zone_boundaries():
    """Verify in_target_zone is False when car lands outside the green zone."""
    # Land at x = 5.0m (outside [-1.5, 1.5]m)
    world = PhysicsWorld(
        initial_x=5.0,
        initial_y=REST_HEIGHT,
        initial_vx=0.0,
        initial_vy=-0.4,
        initial_theta=0.0,
        initial_omega=0.0,
    )
    col_info = world.collision_detector.check_collision(world.body)
    assert col_info.is_contact is True
    assert col_info.is_safe_landing is True  # Physically gentle landing
    assert col_info.is_in_target_zone is False  # But missed green target zone!


def test_out_of_bounds_detection():
    """Verify out of bounds when car flies beyond [-20, 20]m or above 25m."""
    # X exceeds boundary
    world_x_oob = PhysicsWorld(initial_x=22.0, initial_y=10.0)
    col_x = world_x_oob.collision_detector.check_collision(world_x_oob.body)
    assert col_x.is_out_of_bounds is True
    
    # Y exceeds ceiling
    world_y_oob = PhysicsWorld(initial_x=0.0, initial_y=26.0)
    col_y = world_y_oob.collision_detector.check_collision(world_y_oob.body)
    assert col_y.is_out_of_bounds is True
