"""
Cold Gas Thrusters System for the 2D Flying Roadster.
Controls 4 fixed thrusters, computes forces, torques, and tracks fuel consumption.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, List
import numpy as np

from physics.constants import (
    THRUST_LIFT,
    THRUST_LATERAL,
    THRUSTER_V1_POS,
    THRUSTER_V2_POS,
    THRUSTER_V3_POS,
    THRUSTER_V4_POS,
    THRUSTER_V1_DIR,
    THRUSTER_V2_DIR,
    THRUSTER_V3_DIR,
    THRUSTER_V4_DIR,
    THRUSTER_V1_TORQUE,
    THRUSTER_V2_TORQUE,
    THRUSTER_V3_TORQUE,
    THRUSTER_V4_TORQUE,
    INITIAL_FUEL,
    FUEL_BURN_RATE_LIFT,
    FUEL_BURN_RATE_LATERAL,
)
from physics.rigid_body import RigidBody2D


class ThrusterSystem:
    """
    4 Fixed Thruster System:
    - V1: Rear lift (tail bottom, 9,000 N upward, tau = -16,200 N*m)
    - V2: Front lift (head bottom, 9,000 N upward, tau = +16,200 N*m)
    - V3: Push right (tail centerline, 6,000 N rightward, tau = 0 N*m)
    - V4: Push left (head centerline, 6,000 N leftward, tau = 0 N*m)

    Action space: discrete integer 0 to 15, decoded bitwise.
    Fuel consumption is physically tracked: thrusters fail if fuel reaches 0.
    """

    def __init__(self, initial_fuel: float = INITIAL_FUEL):
        self.initial_fuel: float = float(initial_fuel)
        self.fuel: float = float(initial_fuel)

        # Thruster physical definitions
        self.thrusters = [
            {
                "name": "V1_REAR_LIFT",
                "pos": THRUSTER_V1_POS,
                "dir": THRUSTER_V1_DIR,
                "magnitude": THRUST_LIFT,
                "torque": THRUSTER_V1_TORQUE,
                "burn_rate": FUEL_BURN_RATE_LIFT,
            },
            {
                "name": "V2_FRONT_LIFT",
                "pos": THRUSTER_V2_POS,
                "dir": THRUSTER_V2_DIR,
                "magnitude": THRUST_LIFT,
                "torque": THRUSTER_V2_TORQUE,
                "burn_rate": FUEL_BURN_RATE_LIFT,
            },
            {
                "name": "V3_PUSH_RIGHT",
                "pos": THRUSTER_V3_POS,
                "dir": THRUSTER_V3_DIR,
                "magnitude": THRUST_LATERAL,
                "torque": THRUSTER_V3_TORQUE,
                "burn_rate": FUEL_BURN_RATE_LATERAL,
            },
            {
                "name": "V4_PUSH_LEFT",
                "pos": THRUSTER_V4_POS,
                "dir": THRUSTER_V4_DIR,
                "magnitude": THRUST_LATERAL,
                "torque": THRUSTER_V4_TORQUE,
                "burn_rate": FUEL_BURN_RATE_LATERAL,
            },
        ]

        # Last applied action and forces
        self.last_action: int = 0
        self.last_valves: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.last_thrust_body: np.ndarray = np.zeros(2, dtype=np.float64)
        self.last_thrust_world: np.ndarray = np.zeros(2, dtype=np.float64)
        self.last_torque: float = 0.0
        self.last_fuel_consumed: float = 0.0

    def reset(self, initial_fuel: float = INITIAL_FUEL) -> None:
        """Reset fuel tank and telemetry."""
        self.initial_fuel = float(initial_fuel)
        self.fuel = float(initial_fuel)
        self.last_action = 0
        self.last_valves = (0, 0, 0, 0)
        self.last_thrust_body = np.zeros(2, dtype=np.float64)
        self.last_thrust_world = np.zeros(2, dtype=np.float64)
        self.last_torque = 0.0
        self.last_fuel_consumed = 0.0

    @staticmethod
    def decode_action(action: int) -> Tuple[int, int, int, int]:
        """
        Decode an action integer (0-15) into binary valve states (u1, u2, u3, u4).

        Bit 0: V1 (Rear lift)
        Bit 1: V2 (Front lift)
        Bit 2: V3 (Push right)
        Bit 3: V4 (Push left)
        """
        a = int(action)
        if not (0 <= a <= 15):
            raise ValueError(f"Action must be an integer in [0, 15], got {action}")

        u1 = a & 1
        u2 = (a >> 1) & 1
        u3 = (a >> 2) & 1
        u4 = (a >> 3) & 1
        return (u1, u2, u3, u4)

    @staticmethod
    def encode_action(u1: int, u2: int, u3: int, u4: int) -> int:
        """Encode binary valve states (u1, u2, u3, u4) into an action integer (0-15)."""
        return (int(u1) & 1) | ((int(u2) & 1) << 1) | ((int(u3) & 1) << 2) | ((int(u4) & 1) << 3)

    def apply_thrust(
        self,
        body: RigidBody2D,
        action: int,
        dt: float,
    ) -> Dict[str, Any]:
        """
        Compute thruster forces and torques, consume fuel, and apply to the rigid body.

        Parameters
        ----------
        body : RigidBody2D to apply forces and torques to
        action : discrete action integer in [0, 15]
        dt : time step for fuel consumption calculation (seconds)

        Returns
        -------
        telemetry dict containing valve states, forces, torque, and fuel delta.
        """
        valves = self.decode_action(action)
        self.last_action = action
        self.last_valves = valves

        # Check if fuel is exhausted
        if self.fuel <= 0.0:
            self.fuel = 0.0
            self.last_thrust_body = np.zeros(2, dtype=np.float64)
            self.last_thrust_world = np.zeros(2, dtype=np.float64)
            self.last_torque = 0.0
            self.last_fuel_consumed = 0.0
            return {
                "valves": valves,
                "effective_valves": (0, 0, 0, 0),
                "thrust_body": self.last_thrust_body.copy(),
                "thrust_world": self.last_thrust_world.copy(),
                "torque": 0.0,
                "fuel_consumed": 0.0,
                "fuel_remaining": 0.0,
            }

        # Calculate fuel consumption rate
        fuel_consumption_rate = 0.0
        for i, active in enumerate(valves):
            if active:
                fuel_consumption_rate += self.thrusters[i]["burn_rate"]

        fuel_delta = fuel_consumption_rate * dt
        actual_fuel_consumed = min(self.fuel, fuel_delta)
        self.fuel = max(0.0, self.fuel - actual_fuel_consumed)
        self.last_fuel_consumed = actual_fuel_consumed

        # Accumulate forces in body coordinates and torque about COM
        body_fx = 0.0
        body_fy = 0.0
        net_torque = 0.0

        for i, active in enumerate(valves):
            if active:
                t = self.thrusters[i]
                mag = t["magnitude"]
                dx, dy = t["dir"]
                body_fx += dx * mag
                body_fy += dy * mag
                net_torque += t["torque"]

        self.last_thrust_body = np.array([body_fx, body_fy], dtype=np.float64)
        self.last_torque = net_torque

        # Transform body thrust to world coordinates: F_world = R(theta) * F_body
        f_world = body.body_to_world_vector(self.last_thrust_body)
        self.last_thrust_world = f_world

        # Apply world force at COM and torque to body
        body.apply_force_world(f_world[0], f_world[1])
        body.apply_torque(net_torque)

        return {
            "valves": valves,
            "effective_valves": valves,
            "thrust_body": self.last_thrust_body.copy(),
            "thrust_world": self.last_thrust_world.copy(),
            "torque": net_torque,
            "fuel_consumed": actual_fuel_consumed,
            "fuel_remaining": self.fuel,
        }
