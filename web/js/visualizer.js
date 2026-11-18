/**
 * Canvas 2D Visualizer & Explainability Inspector for Tesla Roadster Flying Car Landing.
 * Parity with Roadster_4_Voi_DQN specification.
 */

class RoadsterVisualizer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext("2d");

    this.body = new PhysicsBody();
    this.net = new NeuralNet();

    // Visual scale: 1 meter = scalePixels
    this.scale = 35.0; // pixels per meter
    this.originX = this.canvas.width / 2;
    this.originY = this.canvas.height - 80; // ground level in canvas Y

    this.trail = [];
    this.maxTrail = 120;
    this.particles = [];

    this.mode = "rl"; // "rl", "flip", "manual"
    this.paused = false;
    this.simSpeed = 1.0;
    this.manualAction = 0;

    // Cumulative session stats
    this.stats = {
      totalRuns: 0,
      safeLandings: 0,
      crashes: 0,
      timeouts: 0,
      oobs: 0,
      speeds: [],
      angles: [],
      times: [],
    };

    this.cumReward = 0.0;
    this.lastRewardBreakdown = {
      stepReward: 0.0,
      cumReward: 0.0,
      rawBase: 0.0,
      shaping: 0.0,
      costs: 0.0,
    };

    this.flipCtrl = null;
    this.initEventListeners();
  }

  async init() {
    // Attempt to load neural network weights from JSON
    const loaded = await this.net.loadWeights("weights/trained_weights.json");
    if (!loaded) {
      console.log("[Visualizer] Using bundled fallback weights.");
    }
    this.reset(3);
    this.animate();
  }

  computePotential(x, y, vx, vy) {
    const h = Math.max(0, y - 0.72);
    const dist = Math.sqrt(x * x + h * h);
    const speed = Math.sqrt(vx * vx + vy * vy);
    return -10.0 * dist - 4.0 * speed;
  }

  reset(level = 3) {
    let x = 0.0, y = 6.0, vx = 0.0, vy = 0.0, theta = 0.0, omega = 0.0;
    const r = Math.random;

    if (level === 1) {
      y = 1.8 + r() * 1.0;
      x = (r() - 0.5) * 0.4;
      vx = (r() - 0.5) * 0.2;
      vy = -r() * 0.2;
      theta = (r() - 0.5) * (4.0 * Math.PI / 180.0);
      omega = (r() - 0.5) * 0.05;
    } else if (level === 2) {
      y = 3.0 + r() * 2.0;
      x = (r() - 0.5) * 1.0;
      vx = (r() - 0.5) * 0.4;
      vy = -r() * 0.5;
      theta = (r() - 0.5) * (12.0 * Math.PI / 180.0);
      omega = (r() - 0.5) * 0.1;
    } else if (level === "flip") {
      y = 8.5;
      x = 0.0;
      vx = 0.0;
      vy = 4.9; // Parabolic launch loft: vehicle reaches apex at t=0.5s during flip inversion
      theta = 0.0;
      omega = 5.5; // Rapid 360 flip in ~1.1s
      if (typeof PhysicsFlipController !== "undefined") {
        this.flipCtrl = new PhysicsFlipController();
      }
    } else {
      // Level 3 (Default full challenge domain)
      y = 4.0 + r() * 5.0;
      x = (r() - 0.5) * 4.0;
      vx = (r() - 0.5) * 0.8;
      vy = -r() * 1.0;
      theta = (r() - 0.5) * (20.0 * Math.PI / 180.0);
      omega = (r() - 0.5) * 0.2;
    }

    this.body.reset(x, y, vx, vy, theta, omega);
    this.trail = [];
    this.particles = [];
    this.cumReward = 0.0;
    this.prevPhi = this.computePotential(x, y, vx, vy);
    this.stats.totalRuns++;
    this.updateHUD();
  }

  toScreen(wx, wy) {
    return {
      x: this.originX + wx * this.scale,
      y: this.originY - wy * this.scale,
    };
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    if (this.paused) return;

    let action = 0;
    if (this.mode === "rl") {
      const obs = this.body.getNormalizedObservation();
      action = this.net.predict(obs);
    } else if (this.mode === "flip" && this.flipCtrl) {
      action = this.flipCtrl.compute_action({
        x: this.body.x,
        y: this.body.y,
        vy: this.body.vy,
        vx: this.body.vx,
        theta: this.body.theta,
        omega: this.body.omega,
        accumulated_rotation: this.body.accumulatedRotation,
      });
    } else if (this.mode === "manual") {
      action = this.manualAction;
    }

    if (!this.body.isLanded && !this.body.isCrashed && !this.body.isOutOfBounds && !this.body.isTimeout) {
      const prevX = this.body.x;
      const prevY = this.body.y;
      const prevVx = this.body.vx;
      const prevVy = this.body.vy;
      const prevFuel = this.body.fuel;

      const stepRes = this.body.step(action);
      this.lastValves = stepRes.valves;
      this.lastAction = action;

      // Compute Reward Breakdown for Explainability
      const dt = 0.1;
      const dFuel = Math.max(0, prevFuel - this.body.fuel);
      const fuelCost = -0.2 * dFuel;
      const timeCost = -1.0 * dt;
      let terminalBonus = 0.0;

      if (this.body.isLanded) {
        terminalBonus += 200.0;
        if (Math.abs(this.body.x) <= 0.8) terminalBonus += 30.0;
      } else if (this.body.isCrashed) {
        terminalBonus -= 100.0;
      } else if (this.body.isTimeout) {
        terminalBonus -= 500.0;
      } else if (this.body.isOutOfBounds) {
        terminalBonus -= 250.0;
      }

      const rawBase = terminalBonus + fuelCost + timeCost;
      const curPhi = (this.body.isLanded || this.body.isCrashed) ? 0.0 : this.computePotential(this.body.x, this.body.y, this.body.vx, this.body.vy);
      const shaping = 0.995 * curPhi - this.prevPhi;
      this.prevPhi = curPhi;

      const stepReward = 0.01 * (rawBase + shaping);
      this.cumReward += stepReward;

      this.lastRewardBreakdown = {
        stepReward: stepReward,
        cumReward: this.cumReward,
        rawBase: rawBase,
        shaping: shaping,
        costs: fuelCost + timeCost,
      };

      this.trail.push({ x: this.body.x, y: this.body.y });
      if (this.trail.length > this.maxTrail) this.trail.shift();
      this.spawnParticles(stepRes.valves);

      if (stepRes.isDone) {
        this.recordTerminalOutcome();
      }
    }

    this.updateParticles();
    this.draw();
    this.updateHUD();
  }

  recordTerminalOutcome() {
    if (this.body.isLanded) {
      this.stats.safeLandings++;
      const spd = Math.sqrt(this.body.vx * this.body.vx + this.body.vy * this.body.vy);
      this.stats.speeds.push(spd);
      const twoPi = 2.0 * Math.PI;
      const wrappedTh = ((this.body.theta + Math.PI) % twoPi + twoPi) % twoPi - Math.PI;
      this.stats.angles.push(Math.abs(wrappedTh * 180.0 / Math.PI));
      this.stats.times.push(this.body.time);
    } else if (this.body.isCrashed) {
      this.stats.crashes++;
    } else if (this.body.isTimeout) {
      this.stats.timeouts++;
    } else if (this.body.isOutOfBounds) {
      this.stats.oobs++;
    }
    this.updateScorecard();
  }

  spawnParticles(valves) {
    const cosT = Math.cos(this.body.theta);
    const sinT = Math.sin(this.body.theta);

    const emit = (lx, ly, dirX, dirY, count, color) => {
      const wx = this.body.x + lx * cosT - ly * sinT;
      const wy = this.body.y + lx * sinT + ly * cosT;
      const wDirX = dirX * cosT - dirY * sinT;
      const wDirY = dirX * sinT + dirY * cosT;

      for (let i = 0; i < count; i++) {
        const spread = (Math.random() - 0.5) * 0.4;
        const spd = 3.0 + Math.random() * 4.0;
        this.particles.push({
          x: wx,
          y: wy,
          vx: this.body.vx + (wDirX + spread * wDirY) * spd,
          vy: this.body.vy + (wDirY - spread * wDirX) * spd,
          life: 0.25 + Math.random() * 0.15,
          maxLife: 0.4,
          color: color,
        });
      }
    };

    // V1 (Tail lift): exhaust goes DOWN (-y_local)
    if (valves.v1) emit(-1.8, -0.45, 0.0, -1.0, 3, "rgba(0, 229, 255, ");
    // V2 (Head lift): exhaust goes DOWN (-y_local)
    if (valves.v2) emit(1.8, -0.45, 0.0, -1.0, 3, "rgba(0, 229, 255, ");
    // V3 (Push right): exhaust goes LEFT (-x_local)
    if (valves.v3) emit(0.0, 0.0, -1.0, 0.0, 2, "rgba(255, 119, 0, ");
    // V4 (Push left): exhaust goes RIGHT (+x_local)
    if (valves.v4) emit(0.0, 0.0, 1.0, 0.0, 2, "rgba(255, 119, 0, ");
  }

  updateParticles() {
    const dt = 0.016;
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.life -= dt;
      if (p.life <= 0) this.particles.splice(i, 1);
    }
  }

  draw() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    ctx.fillStyle = "#0a0d14";
    ctx.fillRect(0, 0, w, h);

    // Grid lines (1 meter grid)
    ctx.strokeStyle = "#141c2b";
    ctx.lineWidth = 1;
    for (let y = 0; y <= 15; y += 1) {
      const pt = this.toScreen(0, y);
      ctx.beginPath();
      ctx.moveTo(0, pt.y);
      ctx.lineTo(w, pt.y);
      ctx.stroke();
    }
    for (let x = -10; x <= 10; x += 1) {
      const pt = this.toScreen(x, 0);
      ctx.beginPath();
      ctx.moveTo(pt.x, 0);
      ctx.lineTo(pt.x, h);
      ctx.stroke();
    }

    // Ground plane
    const ground = this.toScreen(0, 0);
    ctx.strokeStyle = "#385170";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(0, ground.y);
    ctx.lineTo(w, ground.y);
    ctx.stroke();

    // Target Pad [-2.0, 2.0] in glowing green
    const padL = this.toScreen(PhysicsConstants.TARGET_PAD_MIN_X, 0);
    const padR = this.toScreen(PhysicsConstants.TARGET_PAD_MAX_X, 0);

    ctx.fillStyle = "rgba(0, 255, 136, 0.15)";
    ctx.fillRect(padL.x, padL.y, padR.x - padL.x, 6);

    ctx.strokeStyle = "#00ff88";
    ctx.lineWidth = 4;
    ctx.shadowColor = "#00ff88";
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.moveTo(padL.x, padL.y);
    ctx.lineTo(padR.x, padR.y);
    ctx.stroke();
    ctx.shadowBlur = 0; // reset

    // Target pad markers
    ctx.fillStyle = "#00ff88";
    ctx.font = "10px monospace";
    ctx.fillText("TARGET PAD [-0.8m, +0.8m]", padL.x + 8, padL.y + 20);

    // Trajectory trail
    if (this.trail.length > 1) {
      ctx.beginPath();
      for (let i = 0; i < this.trail.length; i++) {
        const pt = this.toScreen(this.trail[i].x, this.trail[i].y);
        if (i === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      }
      ctx.strokeStyle = "rgba(0, 229, 255, 0.35)";
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw Thruster Particles
    for (const p of this.particles) {
      const pt = this.toScreen(p.x, p.y);
      const alpha = Math.max(0, p.life / p.maxLife);
      ctx.fillStyle = p.color + alpha + ")";
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw Tesla Roadster Body
    this.drawRoadster();

    // Draw Terminal Banner if episode finished
    if (this.body.isLanded || this.body.isCrashed || this.body.isTimeout || this.body.isOutOfBounds) {
      this.drawStatusBanner();
    }
  }

  drawRoadster() {
    const ctx = this.ctx;
    const pos = this.toScreen(this.body.x, this.body.y);
    const theta = -this.body.theta; // Canvas Y is inverted

    ctx.save();
    ctx.translate(pos.x, pos.y);
    ctx.rotate(theta);

    const L = PhysicsConstants.BODY_LENGTH * this.scale;
    const H = PhysicsConstants.BODY_HEIGHT * this.scale;

    // Roadster Body Shell
    ctx.fillStyle = "#e82127"; // Tesla Red
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;

    ctx.beginPath();
    ctx.moveTo(-L * 0.50, H * 0.15); // Rear bottom
    ctx.lineTo(L * 0.45, H * 0.15);  // Front bottom
    ctx.lineTo(L * 0.50, -H * 0.05); // Nose tip
    ctx.lineTo(L * 0.30, -H * 0.10); // Hood
    ctx.lineTo(L * 0.10, -H * 0.48); // Windshield top
    ctx.lineTo(-L * 0.25, -H * 0.48);// Roof rear
    ctx.lineTo(-L * 0.45, -H * 0.10);// Rear trunk
    ctx.lineTo(-L * 0.50, H * 0.10); // Rear tail
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Cabin Glass (Cyan tinted)
    ctx.fillStyle = "rgba(0, 229, 255, 0.4)";
    ctx.beginPath();
    ctx.moveTo(L * 0.28, -H * 0.10);
    ctx.lineTo(L * 0.08, -H * 0.44);
    ctx.lineTo(-L * 0.22, -H * 0.44);
    ctx.lineTo(-L * 0.30, -H * 0.10);
    ctx.closePath();
    ctx.fill();

    // Wheels
    const drawWheel = (wx, wy) => {
      ctx.fillStyle = "#111111";
      ctx.strokeStyle = "#888888";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(wx, wy, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#00e5ff";
      ctx.beginPath();
      ctx.arc(wx, wy, 2.5, 0, Math.PI * 2);
      ctx.fill();
    };
    drawWheel(-1.5 * this.scale, 0.45 * this.scale);
    drawWheel(1.5 * this.scale, 0.45 * this.scale);

    // Thruster Nozzle Badges
    ctx.fillStyle = "#ffaa00";
    ctx.fillRect(-1.85 * this.scale, 0.35 * this.scale, 5, 4);
    ctx.fillRect(1.75 * this.scale, 0.35 * this.scale, 5, 4);

    // Center of Mass (COM) marker
    ctx.strokeStyle = "#ffff00";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(0, 0, 4, 0, Math.PI * 2);
    ctx.stroke();

    ctx.restore();
  }

  drawStatusBanner() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    let text = "";
    let color = "";

    if (this.body.isLanded) {
      text = "SAFE LANDING ACQUIRED (+200 PTS)";
      color = "#00ff88";
    } else if (this.body.isCrashed) {
      text = "CRASH DETECTED (-100 PTS)";
      color = "#ff3366";
    } else if (this.body.isTimeout) {
      text = "TIMEOUT EXCEEDED (-500 PTS)";
      color = "#ffaa00";
    } else if (this.body.isOutOfBounds) {
      text = "OUT OF BOUNDS (-250 PTS)";
      color = "#ff3366";
    }

    ctx.save();
    ctx.font = "bold 20px monospace";
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = "rgba(10, 13, 20, 0.85)";
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect((w - tw) / 2 - 20, 50, tw + 40, 46, 8);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.fillText(text, (w - tw) / 2, 80);
    ctx.restore();
  }

  generatePolicyExplanation(raw, v, action, maxQ, marginQ) {
    const elBox = document.getElementById("policy-explanation-box");
    if (!elBox) return;

    if (this.body.isLanded) {
      elBox.innerHTML = `<strong>🏆 HẠ CÁNH THÀNH CÔNG AN TOÀN:</strong> Xe tiếp đất với tốc độ rơi ${Math.abs(raw.vy).toFixed(2)} m/s (&lt; 1.0 m/s), góc lệch ${Math.abs(raw.thetaDeg).toFixed(1)}° (&lt; 5°). Nhận thưởng tối đa +200 điểm!`;
      elBox.style.borderColor = "var(--green)";
      return;
    }
    if (this.body.isCrashed) {
      elBox.innerHTML = `<strong>💥 VA CHẠM (CRASH):</strong> Xe chạm đất vượt ngưỡng an toàn. Replay buffer ghi nhận phạt va chạm -100 điểm.`;
      elBox.style.borderColor = "var(--red)";
      return;
    }

    if (this.mode === "flip") {
      const rotDeg = Math.abs(this.body.accumulatedRotation * 180.0 / Math.PI).toFixed(0);
      if (this.flipCtrl && !this.flipCtrl.flipCompleted) {
        elBox.innerHTML = `<strong>🌀 CHẾ ĐỘ LỘN NHÀO (ACROBATIC FLIP 360°):</strong> Đang thực hiện vòng lộn trên không (Xoay tích lũy: ${rotDeg}° / 360°). Bộ điều khiển pha kích hoạt V2 tạo mô-men xoắn xoay lộn ngược và hãm dừng giải tích bằng V1.`;
      } else {
        elBox.innerHTML = `<strong>🛬 TIẾP ĐẤT SAU CÚ LỘN (FLARE DESCENT):</strong> Đã hoàn tất 1 vòng lộn 360° an toàn! Bộ điều khiển chuyển sang pha hạ cánh tốc độ cao, kích hoạt V1+V2 hãm vận tốc rơi về dưới 1.0 m/s để tiếp đất êm ái.`;
      }
      elBox.style.borderColor = "#c084fc";
      return;
    }

    let reason = "";
    // Scenario 1: Touchdown flare
    if (raw.h <= 2.2 && raw.vy < -0.7) {
      reason = `<strong>GIAI ĐOẠN HÃM TIẾP ĐẤT (FLARE):</strong> Xe đang ở độ cao thấp h = ${raw.h.toFixed(2)}m và rơi nhanh vy = ${raw.vy.toFixed(2)} m/s. Agent chọn Action ${action} để kích hoạt vòi nâng, tạo lực đẩy 18,000 N &gt; 11,772 N triệt tiêu tốc độ rơi về dưới 1.0 m/s trước khi chạm đất.`;
    }
    // Scenario 2: Pitch correction
    else if (Math.abs(raw.thetaDeg) > 3.0) {
      if (raw.thetaDeg > 0) {
        reason = `<strong>HIỆU CHỈNH GÓC NGHIÊNG:</strong> Thân xe đang nghiêng đầu lên +${raw.thetaDeg.toFixed(1)}°. Agent chọn Action ${action} (V1 Tail ON) tạo mô-men xoắn âm -16,200 N·m để hạ đầu xe về vị trí nằm ngang 0°.`;
      } else {
        reason = `<strong>HIỆU CHỈNH GÓC NGHIÊNG:</strong> Thân xe đang chúc đầu xuống ${raw.thetaDeg.toFixed(1)}°. Agent chọn Action ${action} (V2 Head ON) tạo mô-men xoắn dương +16,200 N·m để nâng đầu xe về 0°.`;
      }
    }
    // Scenario 3: Horizontal drift correction
    else if (Math.abs(raw.x) > 0.6 || Math.abs(raw.vx) > 0.4) {
      if (raw.x < -0.4) {
        reason = `<strong>HÚT VỀ TÂM BÃI ĐÁP:</strong> Xe đang lệch trái x = ${raw.x.toFixed(2)}m. Agent chọn Action ${action} (V3 Right ON) tạo lực đẩy ngang +6,000 N để kéo xe về bãi đáp x = 0.`;
      } else {
        reason = `<strong>HÚT VỀ TÂM BÃI ĐÁP:</strong> Xe đang lệch phải x = ${raw.x.toFixed(2)}m. Agent chọn Action ${action} (V4 Left ON) tạo lực đẩy ngang -6,000 N để hãm dạt phải và kéo về tâm x = 0.`;
      }
    }
    // Scenario 4: Coasting
    else if (action === 0) {
      reason = `<strong>BAY QUÁN TÍNH TIẾT KIỆM (COASTING):</strong> Vị trí và góc nghiêng đều nằm trong hành lang an toàn (|θ| &lt; 3°, |x| &lt; 0.5m). Agent tắt toàn bộ 4 vòi để tránh bị phạt tiêu hao nhiên liệu (-0.2/step).`;
    }
    // Scenario 5: General guidance
    else {
      reason = `<strong>DUY TRÌ QUỸ ĐẠO TỐI ƯU:</strong> Agent dự đoán Q(s, A${action}) = ${maxQ.toFixed(2)} vượt trội so với hành động thứ 2 (Margin ΔQ = ${marginQ.toFixed(2)}). Lựa chọn tối ưu cân bằng giữa vận tốc rơi và tư thế tiếp đất.`;
    }

    elBox.innerHTML = reason;
    elBox.style.borderColor = "var(--cyan)";
  }

  updateHUD() {
    const twoPi = 2.0 * Math.PI;
    const wrappedTh = ((this.body.theta + Math.PI) % twoPi + twoPi) % twoPi - Math.PI;
    const raw = {
      h: this.body.y,
      x: this.body.x,
      vx: this.body.vx,
      vy: this.body.vy,
      thetaDeg: (wrappedTh * 180.0 / Math.PI),
      omega: this.body.omega,
      fuel: this.body.fuel,
      time: this.body.time,
    };

    // State Panel Updates
    document.getElementById("val-h").innerText = raw.h.toFixed(2) + " m";
    document.getElementById("val-x").innerText = raw.x.toFixed(2) + " m";
    document.getElementById("val-vy").innerText = raw.vy.toFixed(2) + " m/s";
    document.getElementById("val-vx").innerText = raw.vx.toFixed(2) + " m/s";
    document.getElementById("val-theta").innerText = raw.thetaDeg.toFixed(1) + " deg";
    document.getElementById("val-omega").innerText = raw.omega.toFixed(2) + " rad/s";
    document.getElementById("val-fuel").innerText = raw.fuel.toFixed(1) + " %";
    document.getElementById("val-time").innerText = raw.time.toFixed(2) + " s";

    // Normalized 9D state observation box
    const obs = this.body.getNormalizedObservation();
    const obsStr = `[${obs[0].toFixed(2)}, ${obs[1].toFixed(2)}, ${obs[2].toFixed(2)}, ${obs[3].toFixed(2)}, ${obs[4].toFixed(2)}, ${obs[5].toFixed(2)}, ${obs[6].toFixed(2)}, ${obs[7].toFixed(2)}, ${obs[8].toFixed(2)}]`;
    const elNorm = document.getElementById("norm-obs-box");
    if (elNorm) elNorm.innerText = obsStr;

    // Action Panel Updates
    const v = this.lastValves || { v1: false, v2: false, v3: false, v4: false };
    const act = this.lastAction || 0;

    const elV1 = document.getElementById("th-v1");
    const elV2 = document.getElementById("th-v2");
    const elV3 = document.getElementById("th-v3");
    const elV4 = document.getElementById("th-v4");
    if (elV1) elV1.className = v.v1 ? "thruster-box active" : "thruster-box";
    if (elV2) elV2.className = v.v2 ? "thruster-box active" : "thruster-box";
    if (elV3) elV3.className = v.v3 ? "thruster-box active" : "thruster-box";
    if (elV4) elV4.className = v.v4 ? "thruster-box active" : "thruster-box";

    const tagV1 = document.getElementById("tag-v1");
    const tagV2 = document.getElementById("tag-v2");
    const tagV3 = document.getElementById("tag-v3");
    const tagV4 = document.getElementById("tag-v4");
    if (tagV1) { tagV1.innerText = v.v1 ? "ON" : "OFF"; tagV1.className = v.v1 ? "thruster-status-tag on" : "thruster-status-tag off"; }
    if (tagV2) { tagV2.innerText = v.v2 ? "ON" : "OFF"; tagV2.className = v.v2 ? "thruster-status-tag on" : "thruster-status-tag off"; }
    if (tagV3) { tagV3.innerText = v.v3 ? "ON" : "OFF"; tagV3.className = v.v3 ? "thruster-status-tag on" : "thruster-status-tag off"; }
    if (tagV4) { tagV4.innerText = v.v4 ? "ON" : "OFF"; tagV4.className = v.v4 ? "thruster-status-tag on" : "thruster-status-tag off"; }

    const elActId = document.getElementById("val-act-id");
    if (elActId) elActId.innerText = `A${act} [${act.toString(2).padStart(4, "0")}]`;

    // Net torque calculation: tau = (v2 - v1) * 16200
    const netTorque = ((v.v2 ? 1 : 0) - (v.v1 ? 1 : 0)) * 16200;
    const elTorque = document.getElementById("val-net-torque");
    if (elTorque) elTorque.innerText = `${netTorque > 0 ? "+" : ""}${netTorque.toLocaleString()} N·m`;

    // Reward Panel Updates
    const rb = this.lastRewardBreakdown;
    const elStepRew = document.getElementById("val-step-reward");
    if (elStepRew) {
      elStepRew.innerText = (rb.stepReward >= 0 ? "+" : "") + rb.stepReward.toFixed(3);
      elStepRew.className = rb.stepReward >= 0 ? "reward-val pos" : "reward-val neg";
    }
    const elCumRew = document.getElementById("val-cum-reward");
    if (elCumRew) {
      elCumRew.innerText = (rb.cumReward >= 0 ? "+" : "") + rb.cumReward.toFixed(2);
      elCumRew.className = rb.cumReward >= 0 ? "reward-val pos" : "reward-val neg";
    }
    const elRawBase = document.getElementById("val-raw-base");
    if (elRawBase) elRawBase.innerText = rb.rawBase.toFixed(2);
    const elShaping = document.getElementById("val-shaping");
    if (elShaping) elShaping.innerText = (rb.shaping >= 0 ? "+" : "") + rb.shaping.toFixed(3);
    const elCosts = document.getElementById("val-costs");
    if (elCosts) elCosts.innerText = rb.costs.toFixed(3);

    // Update Q-values in chart & Policy Explainability
    this.updateQChart(raw, v, act);
  }

  updateQChart(raw, v, act) {
    const qVals = this.net.lastQValues;
    const maxQ = Math.max(...qVals);
    const sortedQ = [...qVals].sort((a, b) => b - a);
    const marginQ = sortedQ.length > 1 ? (sortedQ[0] - sortedQ[1]) : 0.0;

    const elMaxQ = document.getElementById("val-max-q");
    if (elMaxQ) elMaxQ.innerText = maxQ.toFixed(2);
    const elMarginQ = document.getElementById("val-margin-q");
    if (elMarginQ) elMarginQ.innerText = marginQ.toFixed(2);

    const container = document.getElementById("q-bars");
    if (!container) return;

    if (container.children.length === 0) {
      for (let i = 0; i < 16; i++) {
        const row = document.createElement("div");
        row.className = "q-bar-row";
        row.innerHTML = `
          <div class="q-bar-label">A${i}</div>
          <div class="q-bar-bin">${i.toString(2).padStart(4, "0")}</div>
          <div class="q-bar-track"><div class="q-bar-fill" id="q-fill-${i}"></div></div>
          <div class="q-bar-val" id="q-val-${i}">0.00</div>
        `;
        container.appendChild(row);
      }
    }

    for (let i = 0; i < 16; i++) {
      const fill = document.getElementById(`q-fill-${i}`);
      const valSpan = document.getElementById(`q-val-${i}`);
      if (fill && valSpan) {
        const q = qVals[i];
        valSpan.innerText = q.toFixed(2);
        // Normalize Q to display bar percentage (assuming range roughly -2.0 to +4.0)
        const pct = Math.max(0, Math.min(100, (q + 2.0) * 16.6));
        fill.style.width = pct + "%";
        if (q === maxQ) {
          fill.className = "q-bar-fill active";
          valSpan.className = "q-bar-val best";
        } else {
          fill.className = "q-bar-fill";
          valSpan.className = "q-bar-val";
        }
      }
    }

    // Update natural language policy explanation
    this.generatePolicyExplanation(raw, v, act, maxQ, marginQ);
  }

  updateScorecard() {
    const s = this.stats;
    const safeRate = s.totalRuns > 0 ? (s.safeLandings / s.totalRuns) * 100 : 0;
    const avgSpd = s.speeds.length > 0 ? (s.speeds.reduce((a, b) => a + b, 0) / s.speeds.length) : 0;
    const avgAng = s.angles.length > 0 ? (s.angles.reduce((a, b) => a + b, 0) / s.angles.length) : 0;
    const avgTime = s.times.length > 0 ? (s.times.reduce((a, b) => a + b, 0) / s.times.length) : 0;

    const elSafe = document.getElementById("stat-safe-rate");
    if (elSafe) elSafe.innerText = `${safeRate.toFixed(1)}% (${s.safeLandings}/${s.totalRuns})`;
    const elSpd = document.getElementById("stat-avg-speed");
    if (elSpd) elSpd.innerText = `${avgSpd.toFixed(2)} m/s`;
    const elAng = document.getElementById("stat-avg-angle");
    if (elAng) elAng.innerText = `${avgAng.toFixed(1)} deg`;
    const elTime = document.getElementById("stat-avg-time");
    if (elTime) elTime.innerText = `${avgTime.toFixed(2)} s`;
  }

  initEventListeners() {
    window.addEventListener("keydown", (e) => {
      if (this.mode !== "manual") return;
      if (e.key === "ArrowUp" || e.key === "w") this.manualAction |= 3;  // V1 + V2
      if (e.key === "ArrowLeft" || e.key === "a") this.manualAction |= 8;// V4
      if (e.key === "ArrowRight" || e.key === "d") this.manualAction |= 4;// V3
      if (e.key === "q") this.manualAction |= 1; // V1 only
      if (e.key === "e") this.manualAction |= 2; // V2 only
    });

    window.addEventListener("keyup", (e) => {
      if (this.mode !== "manual") return;
      if (e.key === "ArrowUp" || e.key === "w") this.manualAction &= ~3;
      if (e.key === "ArrowLeft" || e.key === "a") this.manualAction &= ~8;
      if (e.key === "ArrowRight" || e.key === "d") this.manualAction &= ~4;
      if (e.key === "q") this.manualAction &= ~1;
      if (e.key === "e") this.manualAction &= ~2;
    });
  }
}
