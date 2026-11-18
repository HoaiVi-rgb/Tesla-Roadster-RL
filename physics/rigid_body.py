"""
2D Rigid Body Mechanics and Numerical Integration.
Implements Newton-Euler equations of motion using Symplectic (Semi-Implicit) Euler.
"""

from __future__ import annotations
import math
from typing import Tuple, Sequence, Optional
import numpy as np

from physics.constants import (
    VEHICLE_MASS,
    VEHICLE_INERTIA,
)


class RigidBody2D:
    """
    Planar (2D) Rigid Body with 3 Degrees of Freedom (DoF):
    - Translation along X (horizontal)
    - Translation along Y (vertical)
    - Rotation about Z-axis perpendicular to the plane (theta)

    Coordinate Conventions:
    - World frame: origin at ground center (x=0, y=0). +x right, +y up.
    - Local body frame: origin at Center of Mass (COM). +x forward/head, +y roof.
    - Angle theta: radians, 0 is horizontal facing right.
      theta > 0: counter-clockwise (nose up / ngẩng đầu).
      theta < 0: clockwise (nose down / chúi đầu).
    - Angular velocity omega: rad/s, positive CCW, negative CW.

    Numerical Integration:
    - Symplectic Euler (Semi-Implicit Euler):
      v(t + dt) = v(t) + a(t) * dt
      x(t + dt) = x(t) + v(t + dt) * dt
      omega(t + dt) = omega(t) + alpha(t) * dt
      theta(t + dt) = theta(t) + omega(t + dt) * dt
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 6.0,
        vx: float = 0.0,
        vy: float = 0.0,
        theta: float = 0.0,
        omega: float = 0.0,
        mass: float = VEHICLE_MASS,
        inertia: float = VEHICLE_INERTIA,
    ):
        self.mass: float = float(mass)
        self.inertia: float = float(inertia)

        # State variables
        self.x: float = float(x)
        self.y: float = float(y)
        self.vx: float = float(vx)
        self.vy: float = float(vy)
        self.theta: float = float(theta)
        self.omega: float = float(omega)

        # Time and telemetry
        self.time: float = 0.0
        # Continuous cumulative rotation for counting full flips (n * 2pi)
        self.accumulated_rotation: float = 0.0

        # Accumulated forces and torques for the current sub-step
        self._fx_net: float = 0.0
        self._fy_net: float = 0.0
        self._torque_net: float = 0.0

        # Telemetry from the most recent integration step
        self.last_ax: float = 0.0
        self.last_ay: float = 0.0
        self.last_alpha: float = 0.0

    def reset(
        self,
        x: float = 0.0,
        y: float = 6.0,
        vx: float = 0.0,
        vy: float = 0.0,
        theta: float = 0.0,
        omega: float = 0.0,
        time: float = 0.0,
    ) -> None:
        """Reset the rigid body state to exact initial conditions."""
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.theta = float(theta)
        self.omega = float(omega)
        self.time = float(time)
        self.accumulated_rotation = 0.0

        self._fx_net = 0.0
        self._fy_net = 0.0
        self._torque_net = 0.0

        self.last_ax = 0.0
        self.last_ay = 0.0
        self.last_alpha = 0.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy], dtype=np.float64)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """2x2 Rotation matrix R(theta) transforming body vectors to world vectors."""
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)
        return np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)

    def body_to_world_vector(self, vec_body: Sequence[float]) -> np.ndarray:
        """Transform a direction vector from body frame to world frame."""
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)
        bx, by = vec_body[0], vec_body[1]
        wx = bx * cos_t - by * sin_t
        wy = bx * sin_t + by * cos_t
        return np.array([wx, wy], dtype=np.float64)

    def body_to_world_point(self, pt_body: Sequence[float]) -> np.ndarray:
        """Transform a position point from body frame to world frame."""
        w_vec = self.body_to_world_vector(pt_body)
        return np.array([self.x + w_vec[0], self.y + w_vec[1]], dtype=np.float64)

    def world_to_body_vector(self, vec_world: Sequence[float]) -> np.ndarray:
        """Transform a direction vector from world frame to body frame."""
        cos_t = math.cos(-self.theta)
        sin_t = math.sin(-self.theta)
        wx, wy = vec_world[0], vec_world[1]
        bx = wx * cos_t - wy * sin_t
        by = wx * sin_t + wy * cos_t
        return np.array([bx, by], dtype=np.float64)

    def world_to_body_point(self, pt_world: Sequence[float]) -> np.ndarray:
        """Transform a position point from world frame to body frame."""
        dx = pt_world[0] - self.x
        dy = pt_world[1] - self.y
        return self.world_to_body_vector([dx, dy])

    def apply_force_world(self, fx: float, fy: float) -> None:
        """Apply a force in world coordinates acting at the Center of Mass (produces zero torque)."""
        self._fx_net += float(fx)
        self._fy_net += float(fy)

    def apply_torque(self, torque: float) -> None:
        """Apply a pure torque tau (N*m) about the Z-axis."""
        self._torque_net += float(torque)

    def apply_force_at_body_point(
        self,
        force_body: Sequence[float],
        point_body: Sequence[float],
    ) -> Tuple[np.ndarray, float]:
        """
        Apply a force defined in local body coordinates at an offset point on the vehicle body.

        Parameters
        ----------
        force_body : (fx, fy) in body coordinates
        point_body : (rx, ry) in body coordinates relative to COM

        Returns
        -------
        force_world : (fx_world, fy_world)
        torque : torque about COM (N*m)
        """
        # Torque: tau = r_x * F_y - r_y * F_x
        rx, ry = float(point_body[0]), float(point_body[1])
        fbx, fby = float(force_body[0]), float(force_body[1])
        torque = rx * fby - ry * fbx

        # Force converted to world coordinates: F_world = R(theta) * F_body
        f_world = self.body_to_world_vector((fbx, fby))

        self._fx_net += float(f_world[0])
        self._fy_net += float(f_world[1])
        self._torque_net += float(torque)

        return f_world, torque

    def integrate(self, dt: float) -> None:
        """
        Step forward by dt seconds using Symplectic Euler numerical integration.
        Accelerations are computed directly from accumulated forces and torques:
            a = F_net / m
            alpha = tau_net / I_zz

        NO heuristic damping, NO clamping, NO artificially forcing angle to 0.
        """
        if dt <= 0.0:
            return

        # 1. Compute linear and angular accelerations
        ax = self._fx_net / self.mass
        ay = self._fy_net / self.mass
        alpha = self._torque_net / self.inertia

        # Record telemetry
        self.last_ax = ax
        self.last_ay = ay
        self.last_alpha = alpha

        # 2. Semi-Implicit Euler integration for velocity & angular velocity
        self.vx += ax * dt
        self.vy += ay * dt
        self.omega += alpha * dt

        # 3. Semi-Implicit Euler integration for position & angle
        self.x += self.vx * dt
        self.y += self.vy * dt

        delta_theta = self.omega * dt
        self.theta += delta_theta
        self.accumulated_rotation += delta_theta

        # Advance elapsed time
        self.time += dt

        # 4. Clear force and torque accumulators for next sub-step
        self._fx_net = 0.0
        self._fy_net = 0.0
        self._torque_net = 0.0

    def get_state_tuple(self) -> Tuple[float, float, float, float, float, float]:
        """Return (x, y, vx, vy, theta, omega)."""
        return (self.x, self.y, self.vx, self.vy, self.theta, self.omega)

    def clone(self) -> RigidBody2D:
        """Create an exact deep copy of the rigid body."""
        new_body = RigidBody2D(
            x=self.x,
            y=self.y,
            vx=self.vx,
            vy=self.vy,
            theta=self.theta,
            omega=self.omega,
            mass=self.mass,
            inertia=self.inertia,
        )
        new_body.time = self.time
        new_body.accumulated_rotation = self.accumulated_rotation
        new_body._fx_net = self._fx_net
        new_body._fy_net = self._fy_net
        new_body._torque_net = self._torque_net
        new_body.last_ax = self.last_ax
        new_body.last_ay = self.last_ay
        new_body.last_alpha = self.last_alpha
        return new_body
