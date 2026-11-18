/**
 * Client-side Acrobatic Flip Controller.
 * Ports PhysicsFlipController to JavaScript for the interactive Web visualizer.
 */

class PhysicsFlipController {
  constructor() {
    this.stepIdx = 0;
    this.flipCompleted = false;
    this.alphaBrake = 16200.0 / 2017.0; // ~8.0317 rad/s^2
    this.twoPi = 2.0 * Math.PI;
  }

  wrapAngle(angle) {
    return ((angle + Math.PI) % this.twoPi) - Math.PI;
  }

  compute_action(raw) {
    this.stepIdx++;
    const h = raw.y;
    const vy = raw.vy;
    const vx = raw.vx;
    const theta = raw.theta;
    const omega = raw.omega;
    const accumRot = Math.abs(raw.accumulated_rotation || theta);
    const wrappedTh = this.wrapAngle(theta);

    // Flip completion trigger
    if (accumRot >= 5.8 && Math.abs(wrappedTh) < 0.15 && Math.abs(omega) < 0.6) {
      this.flipCompleted = true;
    }

    if (!this.flipCompleted) {
      // Phase 1: Loft upward
      if (this.stepIdx <= 8) {
        return 3; // 0011: V1 + V2
      }
      // Phase 2: Kick spin
      if (accumRot < 1.6) {
        return 2; // 0010: V2
      }
      // Phase 3: Analytical stopping distance check
      if (omega > 0.0) {
        const stopDist = (omega * omega) / (2.0 * this.alphaBrake);
        const remDist = this.twoPi - accumRot;
        if (remDist <= stopDist + 0.15) {
          return 1; // 0001: V1 counter-torque!
        }
      }
      // Spin under momentum
      if (accumRot < 3.8 && omega < 5.5) {
        return 2; // V2
      }
      return 0; // coast
    }

    // Flare landing descent
    let u3 = 0, u4 = 0;
    if (vx > 0.15) u4 = 8; // V4
    else if (vx < -0.15) u3 = 4; // V3

    if (vy < -0.5) {
      return 3 | u3 | u4; // V1 + V2 + lateral
    } else if (vy > -0.2 && h > 1.2) {
      return 0 | u3 | u4;
    } else {
      return 3 | u3 | u4;
    }
  }
}
