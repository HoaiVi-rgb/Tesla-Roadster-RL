"""
Physics constants and simulation parameters for the 2D Flying Roadster.
Reference: Roadster_4_Voi_DQN.docx & k5-htc-doan.pdf.
"""

import math

# ==============================================================================
# VEHICLE MECHANICAL PROPERTIES (VẬT LÝ XE 2D)
# ==============================================================================
# Mass: 1,200 kg
VEHICLE_MASS: float = 1200.0  # kg

# Dimensions: length 4.4m, height 0.9m
VEHICLE_LENGTH: float = 4.4  # m
VEHICLE_HEIGHT: float = 0.9  # m

# Center of Mass (COM) rest height above ground when vehicle sits on wheels
# Matches the reference formula: Phi = -10 * sqrt(x^2 + max(0, h - 0.72)^2)
REST_HEIGHT: float = 0.72  # m

# Moment of inertia about z-axis through COM: I_zz = (1/12) * m * (L^2 + H^2)
# I_zz = (1/12) * 1200 * (4.4^2 + 0.9^2) = 100 * (19.36 + 0.81) = 2017.0 kg*m^2
VEHICLE_INERTIA: float = (1.0 / 12.0) * VEHICLE_MASS * (VEHICLE_LENGTH**2 + VEHICLE_HEIGHT**2)

# ==============================================================================
# ENVIRONMENT & GRAVITATIONAL FORCES (MÔI TRƯỜNG & TRỌNG LỰC)
# ==============================================================================
# Standard gravitational acceleration: 9.81 m/s^2 downward
GRAVITY_ACCEL: float = 9.81  # m/s^2

# Gravitational force: 1,200 kg * 9.81 m/s^2 = 11,772 N downward
GRAVITY_FORCE: float = VEHICLE_MASS * GRAVITY_ACCEL  # 11,772.0 N

# Spatial boundaries (Không gian bay: rộng 40 m [-20, 20], cao 25 m [0, 25])
WORLD_X_MIN: float = -20.0  # m
WORLD_X_MAX: float = 20.0  # m
WORLD_Y_MIN: float = 0.0  # m (ground level)
WORLD_Y_MAX: float = 25.0  # m

# Target Landing Zone (Vùng đáp xanh tại x = 0, rộng +-2.0 m theo đặc tả docx x +- 2m)
TARGET_ZONE_X_MIN: float = -2.0  # m
TARGET_ZONE_X_MAX: float = 2.0  # m

# ==============================================================================
# COLD GAS THRUSTERS SPECIFICATION (4 VÒI PHUN CỐ ĐỊNH)
# ==============================================================================
# Each lifting thruster: 9,000 N. Each lateral thruster: 6,000 N.
# Total vertical lift: 18,000 N > 11,772 N gravity force.
THRUST_LIFT: float = 9000.0  # N
THRUST_LATERAL: float = 6000.0  # N

# Thruster coordinates relative to COM (local body coordinates, meters):
# Origin at COM: +x towards front/head (right), +y towards roof (up).
THRUSTER_V1_POS = (-1.8, -0.45)  # V1: Rear lift (tail bottom)
THRUSTER_V2_POS = (1.8, -0.45)   # V2: Front lift (head bottom)
THRUSTER_V3_POS = (-2.2, 0.0)    # V3: Push right (tail center line)
THRUSTER_V4_POS = (2.2, 0.0)     # V4: Push left (head center line)

# Thruster directions in local body frame (unit vectors):
THRUSTER_V1_DIR = (0.0, 1.0)     # Upward force on body (+y)
THRUSTER_V2_DIR = (0.0, 1.0)     # Upward force on body (+y)
THRUSTER_V3_DIR = (1.0, 0.0)     # Forward/right force on body (+x)
THRUSTER_V4_DIR = (-1.0, 0.0)    # Backward/left force on body (-x)

# Thruster torques about COM (tau = r_x * F_y - r_y * F_x):
# V1: (-1.8) * 9000 - (-0.45) * 0 = -16,200 N*m (clockwise, pitches nose down)
# V2: (+1.8) * 9000 - (-0.45) * 0 = +16,200 N*m (counter-clockwise, pitches nose up)
# V3: (-2.2) * 0 - (0.0) * 6000 = 0 N*m
# V4: (+2.2) * 0 - (0.0) * (-6000) = 0 N*m
THRUSTER_V1_TORQUE: float = THRUSTER_V1_POS[0] * THRUST_LIFT - THRUSTER_V1_POS[1] * 0.0  # -16200.0 N*m
THRUSTER_V2_TORQUE: float = THRUSTER_V2_POS[0] * THRUST_LIFT - THRUSTER_V2_POS[1] * 0.0  # +16200.0 N*m
THRUSTER_V3_TORQUE: float = 0.0  # N*m
THRUSTER_V4_TORQUE: float = 0.0  # N*m

# Combined V1 + V2:
# Net vertical force = 18,000 N
# Net torque = -16,200 + 16,200 = 0.0 N*m (pure balanced vertical lift)

# ==============================================================================
# FUEL DYNAMICS (TIÊU HAO KHÍ NÉN)
# ==============================================================================
INITIAL_FUEL: float = 100.0  # %
# Fuel burn rate (% per second of continuous firing per nozzle)
# 9000 N thruster consumes 1.0 %/s; 6000 N thruster consumes 0.67 %/s
FUEL_BURN_RATE_LIFT: float = 1.0     # %/s
FUEL_BURN_RATE_LATERAL: float = (THRUST_LATERAL / THRUST_LIFT) * FUEL_BURN_RATE_LIFT  # ~0.667 %/s

# ==============================================================================
# TIME & SIMULATION FREQUENCIES (TẦN SỐ TÍNH TOÁN)
# ==============================================================================
PHYSICS_FREQ: int = 120  # Hz
PHYSICS_DT: float = 1.0 / float(PHYSICS_FREQ)  # ~0.008333 s

DECISION_FREQ: int = 10  # Hz (decision every 0.1 s)
DECISION_DT: float = 1.0 / float(DECISION_FREQ)  # 0.1 s

SUBSTEPS_PER_STEP: int = int(round(DECISION_DT / PHYSICS_DT))  # Exactly 12 substeps

MAX_EPISODE_TIME: float = 20.0  # s (200 decision steps)
MAX_DECISION_STEPS: int = int(round(MAX_EPISODE_TIME / DECISION_DT))  # 200 steps

# ==============================================================================
# SAFE LANDING CRITERIA (TIÊU CHÍ TIẾP ĐẤT AN TOÀN - FILE MẪU)
# ==============================================================================
# Đáp an toàn: |vy| < 1 m/s, |vx| < 0.5 m/s, |góc| < 5°, |omega| < 0.2 rad/s tại tiếp xúc
MAX_SAFE_LANDING_VY: float = 1.0  # m/s
MAX_SAFE_LANDING_VX: float = 0.5  # m/s
MAX_SAFE_LANDING_ANGLE_DEG: float = 5.0  # degrees
MAX_SAFE_LANDING_ANGLE_RAD: float = math.radians(MAX_SAFE_LANDING_ANGLE_DEG)  # ~0.0872665 rad
MAX_SAFE_LANDING_OMEGA: float = 0.2  # rad/s

# Contact geometry: 4 car vertices + 2 wheel contact points in body frame
WHEEL_REAR_LOCAL = (-1.5, -REST_HEIGHT)
WHEEL_FRONT_LOCAL = (1.5, -REST_HEIGHT)
BODY_BOTTOM_LEFT_LOCAL = (-VEHICLE_LENGTH / 2.0, -VEHICLE_HEIGHT / 2.0)
BODY_BOTTOM_RIGHT_LOCAL = (VEHICLE_LENGTH / 2.0, -VEHICLE_HEIGHT / 2.0)
BODY_TOP_RIGHT_LOCAL = (VEHICLE_LENGTH / 2.0, VEHICLE_HEIGHT / 2.0)
BODY_TOP_LEFT_LOCAL = (-VEHICLE_LENGTH / 2.0, VEHICLE_HEIGHT / 2.0)

CONTACT_POINTS_LOCAL = [
    WHEEL_REAR_LOCAL,
    WHEEL_FRONT_LOCAL,
    BODY_BOTTOM_LEFT_LOCAL,
    BODY_BOTTOM_RIGHT_LOCAL,
    BODY_TOP_RIGHT_LOCAL,
    BODY_TOP_LEFT_LOCAL,
]
