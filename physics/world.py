"""
Physics World Simulation Engine for the 2D Flying Roadster.
Coordinates the rigid body, thrusters, gravity, and collision detection at 120 Hz.
"""

from __future__ import annotations
import math
from typing import Tuple, Dict, Any, Optional
import numpy as np

from physics.constants import (
    VEHICLE_MASS,
    VEHICLE_INERTIA,
    GRAVITY_ACCEL,
    PHYSICS_FREQ,
    PHYSICS_DT,
    DECISION_FREQ,
    DECISION_DT,
    SUBSTEPS_PER_STEP,
    MAX_EPISODE_TIME,
    INITIAL_FUEL,
)
from physics.rigid_body import RigidBody2D
from physics.thrusters import ThrusterSystem
from physics.collision import CollisionDetector, CollisionInfo


class PhysicsWorld:
    """
    2D Physics Environment Simulation World.

    Features:
    - Pure continuous Newtonian dynamics: F_total = gravity + thrust, tau = r x F.
    - Runs at 120 Hz numerical sub-steps (dt = 1/120 s).
    - RL decision steps run at 10 Hz (0.1 s = 12 sub-steps).
    - Full physics state: (x, y, vx, vy, theta, omega, fuel, time).
    - Observation state: normalized 9D vector matching Roadster_4_Voi_DQN specification.
    - Ground truth collision checking without artificial damping or heuristic snapping.
    """

    def __init__(
        self,
        initial_x: float = 0.0,
        initial_y: float = 6.0,
        initial_vx: float = 0.0,
        initial_vy: float = 0.0,
        initial_theta: float = 0.0,
        initial_omega: float = 0.0,
        initial_fuel: float = INITIAL_FUEL,
    ):
        self.body = RigidBody2D(
            x=initial_x,
            y=initial_y,
            vx=initial_vx,
            vy=initial_vy,
            theta=initial_theta,
            omega=initial_omega,
            mass=VEHICLE_MASS,
            inertia=VEHICLE_INERTIA,
        )
        self.thrusters = ThrusterSystem(initial_fuel=initial_fuel)
        self.collision_detector = CollisionDetector()

        self.last_collision_info: CollisionInfo = self.collision_detector.check_collision(self.body)
        self.is_terminal: bool = False
        self.termination_reason: str = "IN_FLIGHT"

    def reset(
        self,
        x: float = 0.0,
        y: float = 6.0,
        vx: float = 0.0,
        vy: float = 0.0,
        theta: float = 0.0,
        omega: float = 0.0,
        fuel: float = INITIAL_FUEL,
    ) -> np.ndarray:
        """
        Reset the physics world to specified initial conditions and return normalized 9D state.
        """
        self.body.reset(x=x, y=y, vx=vx, vy=vy, theta=theta, omega=omega, time=0.0)
        self.thrusters.reset(initial_fuel=fuel)
        self.last_collision_info = self.collision_detector.check_collision(self.body)
        self.is_terminal = False
        self.termination_reason = "IN_FLIGHT"
        return self.get_normalized_state()

    def substep(self, action: int, dt: float = PHYSICS_DT) -> Tuple[CollisionInfo, Dict[str, Any]]:
        """
        Execute a single 120 Hz physics sub-step (dt ~ 0.00833 s).

        Steps:
        1. Apply gravitational force downward at Center of Mass: F_gravity = [0, -m * g]
        2. Apply thruster forces and torques: F_thrust = R(theta) * F_body, tau_thrust
        3. Burn fuel based on active valves
        4. Integrate equations of motion using Symplectic Euler
        5. Check collisions and boundaries
        """
        if self.is_terminal:
            return self.last_collision_info, {}

        # 1. Apply gravity (downward force in world coordinates at COM -> zero torque)
        gravity_force_y = -self.body.mass * GRAVITY_ACCEL
        self.body.apply_force_world(0.0, gravity_force_y)

        # 2. Apply cold-gas thrusters and update fuel
        thrust_info = self.thrusters.apply_thrust(self.body, action, dt)

        # 3. Integrate rigid body equations of motion
        self.body.integrate(dt)

        # 4. Check collisions
        col_info = self.collision_detector.check_collision(self.body)
        self.last_collision_info = col_info

        # Check terminal condition upon first contact or out of bounds
        if col_info.is_contact or col_info.is_out_of_bounds:
            self.is_terminal = True
            self.termination_reason = col_info.contact_reason

        return col_info, thrust_info

    def step(self, action: int, dt_agent: float = DECISION_DT) -> Dict[str, Any]:
        """
        Execute an agent decision step (default 0.1 s = 12 sub-steps at 120 Hz).

        Terminates early if a collision occurs on any sub-step.
        """
        substeps_to_run = int(round(dt_agent / PHYSICS_DT))
        substeps_executed = 0
        fuel_start = self.thrusters.fuel

        for _ in range(substeps_to_run):
            if self.is_terminal:
                break
            col_info, thrust_info = self.substep(action, dt=PHYSICS_DT)
            substeps_executed += 1

        # Check timeout condition with floating point tolerance
        is_timeout = (self.body.time >= MAX_EPISODE_TIME - 1e-5)
        if is_timeout and not self.is_terminal:
            self.is_terminal = True
            self.termination_reason = "TIMEOUT"

        fuel_consumed = fuel_start - self.thrusters.fuel

        return {
            "obs": self.get_normalized_state(),
            "raw_state": self.get_raw_state(),
            "substeps_executed": substeps_executed,
            "collision_info": self.last_collision_info,
            "is_terminal": self.is_terminal,
            "is_timeout": is_timeout,
            "termination_reason": self.termination_reason,
            "fuel_consumed": fuel_consumed,
            "flips_completed": self.get_num_flips(),
        }

    def get_normalized_state(self) -> np.ndarray:
        """
        Construct the reference 9D state vector matching Roadster_4_Voi_DQN specification:
        s = [x/10, h/15, vx/10, vy/10, sin(theta), cos(theta), omega/5, fuel/100, t/20]
        """
        return np.array(
            [
                self.body.x / 10.0,
                self.body.y / 15.0,
                self.body.vx / 10.0,
                self.body.vy / 10.0,
                math.sin(self.body.theta),
                math.cos(self.body.theta),
                self.body.omega / 5.0,
                self.thrusters.fuel / 100.0,
                self.body.time / MAX_EPISODE_TIME,
            ],
            dtype=np.float32,
        )

    def get_raw_state(self) -> Dict[str, float]:
        """Return the unnormalized physical state variables."""
        return {
            "x": self.body.x,
            "y": self.body.y,
            "vx": self.body.vx,
            "vy": self.body.vy,
            "theta": self.body.theta,
            "theta_deg": math.degrees(self.body.theta),
            "omega": self.body.omega,
            "fuel": self.thrusters.fuel,
            "time": self.body.time,
            "accumulated_rotation": self.body.accumulated_rotation,
        }

    def get_num_flips(self) -> int:
        """
        Calculate the number of complete 360-degree rotations (2 * pi rad) performed.
        Uses accumulated continuous rotation.
        """
        # Full turn is 2 * pi radians (~6.283 rad)
        two_pi = 2.0 * math.pi
        return int(abs(self.body.accumulated_rotation) // two_pi)
