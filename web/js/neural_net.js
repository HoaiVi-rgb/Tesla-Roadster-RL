/**
 * Client-side Neural Network forward pass (Inference Engine).
 * Runs 9 -> 64 -> 64 -> 16 MLP purely in JavaScript.
 * NO Python backend or server required!
 */

class NeuralNet {
  constructor() {
    this.weightsLoaded = false;
    this.layers = null;
    this.lastQValues = new Array(16).fill(0.0);
    this.lastHiddenActivations = new Array(64).fill(0.0);
  }

  async loadWeights(jsonPath = "weights/trained_weights.json") {
    try {
      const response = await fetch(jsonPath);
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }
      const data = await response.json();
      this.layers = data.layers;
      this.weightsLoaded = true;
      console.log("[NeuralNet] Loaded weights successfully from:", jsonPath);
      return true;
    } catch (e) {
      console.warn("[NeuralNet] Failed to load JSON weights (possibly local file:// protocol):", e);
      if (typeof window !== "undefined" && window.DEFAULT_TRAINED_WEIGHTS) {
        console.log("[NeuralNet] Using bundled window.DEFAULT_TRAINED_WEIGHTS fallback!");
        this.layers = window.DEFAULT_TRAINED_WEIGHTS.layers;
        this.weightsLoaded = true;
        return true;
      }
      return false;
    }
  }

  setWeights(weightObj) {
    this.layers = weightObj.layers;
    this.weightsLoaded = true;
    this.modelName = weightObj.model_name || "Unknown Model";
    this.modelArchitecture = weightObj.model_architecture || "9 -> 64 -> 64 -> 16";
  }

  /**
   * Forward pass: 9 -> 64 (ReLU) -> 64 (ReLU) -> 16 (Linear)
   * @param {number[]} state9 - 9-element normalized state vector
   * @returns {{ action: number, qValues: number[], activations: number[] }}
   */
  forward(state9) {
    if (!this.weightsLoaded || !this.layers) {
      // Fallback: zero action if not loaded
      return {
        action: 0,
        qValues: new Array(16).fill(0.0),
        activations: new Array(64).fill(0.0),
      };
    }

    const { fc1_weight, fc1_bias, fc2_weight, fc2_bias, fc3_weight, fc3_bias } = this.layers;

    // Layer 1: 9 -> 64 (ReLU)
    const h1 = new Array(64);
    for (let i = 0; i < 64; i++) {
      let sum = fc1_bias[i];
      const row = fc1_weight[i];
      for (let j = 0; j < 9; j++) {
        sum += row[j] * state9[j];
      }
      h1[i] = sum > 0 ? sum : 0.0; // ReLU
    }

    // Layer 2: 64 -> 64 (ReLU)
    const h2 = new Array(64);
    for (let i = 0; i < 64; i++) {
      let sum = fc2_bias[i];
      const row = fc2_weight[i];
      for (let j = 0; j < 64; j++) {
        sum += row[j] * h1[j];
      }
      h2[i] = sum > 0 ? sum : 0.0; // ReLU
    }
    this.lastHiddenActivations = h2;

    // Layer 3: 64 -> 16 (Linear Q-values)
    const qValues = new Array(16);
    let bestAction = 0;
    let maxQ = -Infinity;

    for (let i = 0; i < 16; i++) {
      let sum = fc3_bias[i];
      const row = fc3_weight[i];
      for (let j = 0; j < 64; j++) {
        sum += row[j] * h2[j];
      }
      qValues[i] = sum;
      if (sum > maxQ) {
        maxQ = sum;
        bestAction = i;
      }
    }

    this.lastQValues = qValues;

    return {
      action: bestAction,
      qValues: qValues,
      activations: h2,
    };
  }

  /**
   * Fast action prediction for 9-element normalized state vector.
   * @param {number[]} state9
   * @returns {number} action (0..15)
   */
  predict(state9) {
    const res = this.forward(state9);
    return res.action;
  }
}
