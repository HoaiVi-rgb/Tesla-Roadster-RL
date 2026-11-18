/**
 * 2D Physics Engine for Tesla Roadster Landing Simulation.
 * Exact 1:1 mathematical match with Python physics module.
 */

const PhysicsConstants = {
  MASS: 1200.0,
  LENGTH: 4.4,
  HEIGHT: 0.9,
  REST_HEIGHT: 0.72,
  MOMENT_INERTIA: 2017.0,
  GRAVITY: 9.8,
  LIFT_THRUST: 9000.0,
  LATERAL_THRUST: 6000.0,
  PHYSICS_DT: 1.0 / 120.0,
  DECISION_DT: 0.1,
  SUBSTEPS_PER_DECISION: 12,
  INITIAL_FUEL: 100.0,
  FUEL_RATE_LIFT: 1.0,     // % per second per thruster
  FUEL_RATE_LATERAL: 0.67,  // % per second per thruster
  TARGET_PAD_MIN_X: -2.0,
  TARGET_PAD_MAX_X: 2.0,
  SAFE_LANDING_MAX_VY: 1.0,
  SAFE_LANDING_MAX_VX: 0.5,
  SAFE_LANDING_MAX_ANGLE_RAD: 5.0 * Math.PI / 180.0, // 0.087266 rad
  SAFE_LANDING_MAX_OMEGA: 0.2,
  BOUNDS_MIN_X: -10.0,
  BOUNDS_MAX_X: 10.0,
  BOUNDS_MAX_Y: 20.0,
  MAX_EPISODE_TIME: 20.0,
};

class PhysicsBody {
  constructor() {
    this.reset();
  }

  reset(x = 0.0, y = 6.0, vx = 0.0, vy = 0.0, theta = 0.0, omega = 0.0, fuel = 100.0) {
    this.x = x;
    this.y = y;
    this.vx = vx;
    this.vy = vy;
    this.theta = theta;
    this.omega = omega;
    this.fuel = fuel;
    this.time = 0.0;
    this.accumulatedRotation = 0.0;
    this.lastTheta = theta;
    this.isLanded = false;
    this.isCrashed = false;
    this.isOutOfBounds = false;
    this.isTimeout = false;
  }

  get contactPointsLocal() {
    return [
      { name: "rear_wheel", x: -1.5, y: -0.45 },
      { name: "front_wheel", x: 1.5, y: -0.45 },
      { name: "tail_bumper", x: -2.2, y: -0.2 },
      { name: "nose_bumper", x: 2.2, y: -0.2 },
      { name: "roof_rear", x: -1.0, y: 0.45 },
      { name: "roof_front", x: 1.0, y: 0.45 },
    ];
  }

  getWorldContactPoints() {
    const cosT = Math.cos(this.theta);
    const sinT = Math.sin(this.theta);
    return this.contactPointsLocal.map((pt) => ({
      name: pt.name,
      x: this.x + pt.x * cosT - pt.y * sinT,
      y: this.y + pt.x * sinT + pt.y * cosT,
    }));
  }

  checkCollision() {
    const pts = this.getWorldContactPoints();
    let lowestY = Infinity;
    let collidingPoints = [];

    for (const pt of pts) {
      if (pt.y < lowestY) lowestY = pt.y;
      if (pt.y <= 0.0) {
        collidingPoints.push(pt);
      }
    }

    const hasContact = collidingPoints.length > 0;
    const speed = Math.sqrt(this.vx * this.vx + this.vy * this.vy);
    const angleRad = Math.abs(this.theta);
    const omegaAbs = Math.abs(this.omega);

    const rw = pts.find((p) => p.name === "rear_wheel");
    const fw = pts.find((p) => p.name === "front_wheel");
    const wheelsInPad =
      rw.x >= PhysicsConstants.TARGET_PAD_MIN_X &&
      rw.x <= PhysicsConstants.TARGET_PAD_MAX_X &&
      fw.x >= PhysicsConstants.TARGET_PAD_MIN_X &&
      fw.x <= PhysicsConstants.TARGET_PAD_MAX_X;

    const onlyWheelsContact = collidingPoints.every(
      (p) => p.name === "rear_wheel" || p.name === "front_wheel"
    );

    const isSafe =
      hasContact &&
      onlyWheelsContact &&
      Math.abs(this.vy) <= PhysicsConstants.SAFE_LANDING_MAX_VY &&
      Math.abs(this.vx) <= PhysicsConstants.SAFE_LANDING_MAX_VX &&
      angleRad <= PhysicsConstants.SAFE_LANDING_MAX_ANGLE_RAD &&
      omegaAbs <= PhysicsConstants.SAFE_LANDING_MAX_OMEGA &&
      wheelsInPad;

    const isCrash = hasContact && !isSafe;

    return {
      hasContact,
      isSafe,
      isCrash,
      wheelsInPad,
      lowestY,
      collidingPoints,
      speed,
      points: pts,
    };
  }

  /**
   * Integrate 1 physics sub-step (1/120s) with Symplectic Euler.
   */
  substep(valves, dt = PhysicsConstants.PHYSICS_DT) {
    let fxLocal = 0.0;
    let fyLocal = 0.0;
    let torque = 0.0;
    let fuelBurn = 0.0;

    const hasFuel = this.fuel > 0.0;

    if (hasFuel) {
      // V1: Lift tail (x = -1.8)
      if (valves.v1) {
        fyLocal += PhysicsConstants.LIFT_THRUST;
        torque += (-1.8) * PhysicsConstants.LIFT_THRUST; // -16200 Nm (CW pitch down)
        fuelBurn += PhysicsConstants.FUEL_RATE_LIFT * dt;
      }
      // V2: Lift head (x = +1.8)
      if (valves.v2) {
        fyLocal += PhysicsConstants.LIFT_THRUST;
        torque += (+1.8) * PhysicsConstants.LIFT_THRUST; // +16200 Nm (CCW pitch up)
        fuelBurn += PhysicsConstants.FUEL_RATE_LIFT * dt;
      }
      // V3: Push right
      if (valves.v3) {
        fxLocal += PhysicsConstants.LATERAL_THRUST;
        fuelBurn += PhysicsConstants.FUEL_RATE_LATERAL * dt;
      }
      // V4: Push left
      if (valves.v4) {
        fxLocal -= PhysicsConstants.LATERAL_THRUST;
        fuelBurn += PhysicsConstants.FUEL_RATE_LATERAL * dt;
      }
      this.fuel = Math.max(0.0, this.fuel - fuelBurn);
    }

    // Transform local thruster forces to world coordinates
    const cosT = Math.cos(this.theta);
    const sinT = Math.sin(this.theta);
    const fxWorld = fxLocal * cosT - fyLocal * sinT;
    const fyWorld = fxLocal * sinT + fyLocal * cosT - PhysicsConstants.MASS * PhysicsConstants.GRAVITY;

    // Linear and angular accelerations
    const ax = fxWorld / PhysicsConstants.MASS;
    const ay = fyWorld / PhysicsConstants.MASS;
    const alpha = torque / PhysicsConstants.MOMENT_INERTIA;

    // Symplectic Euler integration
    this.vx += ax * dt;
    this.vy += ay * dt;
    this.x += this.vx * dt;
    this.y += this.vy * dt;

    this.omega += alpha * dt;
    this.theta += this.omega * dt;

    // Track continuous accumulated rotation
    const dTheta = this.theta - this.lastTheta;
    this.accumulatedRotation += dTheta;
    this.lastTheta = this.theta;

    this.time += dt;

    return { fxWorld, fyWorld, torque, fuelBurn };
  }

  /**
   * Run 1 decision step (0.1s = 12 substeps).
   */
  step(actionInt) {
    const valves = {
      v1: Boolean(actionInt & 1),
      v2: Boolean(actionInt & 2),
      v3: Boolean(actionInt & 4),
      v4: Boolean(actionInt & 8),
    };

    let col = null;
    for (let sub = 0; sub < PhysicsConstants.SUBSTEPS_PER_DECISION; sub++) {
      this.substep(valves, PhysicsConstants.PHYSICS_DT);
      col = this.checkCollision();
      if (col.hasContact) {
        if (col.isSafe) {
          this.isLanded = true;
          this.vx = 0.0;
          this.vy = 0.0;
          this.omega = 0.0;
        } else {
          this.isCrashed = true;
        }
        break;
      }
    }

    // Check boundary & timeout
    if (this.x < PhysicsConstants.BOUNDS_MIN_X || this.x > PhysicsConstants.BOUNDS_MAX_X || this.y > PhysicsConstants.BOUNDS_MAX_Y) {
      this.isOutOfBounds = true;
    }
    if (this.time >= PhysicsConstants.MAX_EPISODE_TIME - 1e-5) {
      this.isTimeout = true;
    }

    return {
      collision: col,
      valves,
      isDone: this.isLanded || this.isCrashed || this.isOutOfBounds || this.isTimeout,
    };
  }

  getNormalizedState() {
    return [
      this.x / 10.0,
      this.y / 15.0,
      this.vx / 10.0,
      this.vy / 10.0,
      Math.sin(this.theta),
      Math.cos(this.theta),
      this.omega / 5.0,
      this.fuel / 100.0,
      this.time / 20.0,
    ];
  }
}
