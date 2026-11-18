"""
Reinforcement Learning Environment for the 2D Flying Roadster Landing Problem.
Implements the Markov Decision Process (MDP): S in R^9, A in {0..15}, P, R, gamma.
"""

from __future__ import annotations
import math
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np

from physics.constants import (
    DECISION_DT,
    INITIAL_FUEL,
    REST_HEIGHT,
)
from physics.world import PhysicsWorld
from environment.state import (
    STATE_DIM,
    RawState,
    raw_to_normalized,
    validate_state,
)
from environment.action import (
    NUM_ACTIONS,
    validate_action,
    decode_action,
    action_to_bin_str,
)
from environment.terminal import (
    TerminalDetector,
    TerminalResult,
    TerminationType,
)
from environment.reward import (
    RewardEngine,
    RewardBreakdown,
)


class RoadsterEnv:
    """
    2D Flying Roadster Landing Environment.

    Observation Space:
        Continuous vector in R^9:
        [x/10, h/15, vx/10, vy/10, sin(theta), cos(theta), omega/5, fuel/100, t/20]

    Action Space:
        Discrete(16): all 16 binary combinations of the 4 thrusters (V1, V2, V3, V4).

    Step Cycle:
        Each action is held for 0.1 s, running 12 sub-steps at 120 Hz.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        use_shaping: bool = True,
        reward_scale: float = 0.01,
        gamma: float = 0.995,
        default_curriculum_level: int = 3,
    ):
        self.world = PhysicsWorld()
        self.terminal_detector = TerminalDetector()
        self.reward_engine = RewardEngine(
            use_shaping=use_shaping,
            reward_scale=reward_scale,
            gamma=gamma,
        )
        self.default_curriculum_level = default_curriculum_level

        self.rng = np.random.default_rng(42)
        self.current_step = 0
        self.total_episodes = 0

        # Most recent transition cache
        self.last_state_raw: Dict[str, float] = {}
        self.last_terminal_result: Optional[TerminalResult] = None
        self.last_reward_breakdown: Optional[RewardBreakdown] = None

    @property
    def action_space_size(self) -> int:
        return NUM_ACTIONS

    @property
    def observation_space_dim(self) -> int:
        return STATE_DIM

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to start a new episode.

        Parameters
        ----------
        seed : optional integer random seed
        options : optional dict with initial state overrides or curriculum level:
                  {'level': 1 | 2 | 3, 'initial_x': float, 'initial_y': float, ...}

        Returns
        -------
        (obs, info)
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        options = options or {}
        level = options.get("level", self.default_curriculum_level)

        # 1. Determine initial physical conditions based on curriculum level or overrides
        if "initial_x" in options and "initial_y" in options:
            init_x = float(options["initial_x"])
            init_y = float(options["initial_y"])
            init_vx = float(options.get("initial_vx", 0.0))
            init_vy = float(options.get("initial_vy", 0.0))
            init_theta = float(options.get("initial_theta", 0.0))
            init_omega = float(options.get("initial_omega", 0.0))
        elif level == 1:
            # Level 1: Low altitude, small angle, near center
            init_y = float(self.rng.uniform(1.2, 3.0))
            init_x = float(self.rng.uniform(-0.2, 0.2))
            init_vx = float(self.rng.uniform(-0.1, 0.1))
            init_vy = float(self.rng.uniform(-0.2, 0.0))
            init_theta = float(math.radians(self.rng.uniform(-3.0, 3.0)))
            init_omega = float(self.rng.uniform(-0.05, 0.05))
        elif level == 2:
            # Level 2: Medium altitude, medium angle, moderate offset
            init_y = float(self.rng.uniform(2.0, 5.0))
            init_x = float(self.rng.uniform(-0.5, 0.5))
            init_vx = float(self.rng.uniform(-0.3, 0.3))
            init_vy = float(self.rng.uniform(-0.5, 0.0))
            init_theta = float(math.radians(self.rng.uniform(-8.0, 8.0)))
            init_omega = float(self.rng.uniform(-0.1, 0.1))
        elif level in (4, "flip", "acrobatic"):
            # Level 4 / Flip: High altitude, inverted or large angle for flip maneuvers
            init_y = float(self.rng.uniform(7.0, 9.5))
            init_x = float(self.rng.uniform(-1.0, 1.0))
            init_vx = float(self.rng.uniform(-0.3, 0.3))
            init_vy = float(self.rng.uniform(-0.5, 0.0))
            # 50% chance clockwise flip, 50% counter-clockwise
            sign = 1.0 if self.rng.random() < 0.5 else -1.0
            init_theta = float(sign * math.radians(self.rng.uniform(90.0, 180.0)))
            init_omega = float(self.rng.uniform(-0.2, 0.2))
        else:
            # Level 3 (Default full challenge domain):
            # h in [4.0, 9.0] m, x in [-2.0, 2.0] m, theta in [-15, +15] deg
            init_y = float(self.rng.uniform(4.0, 9.0))
            init_x = float(self.rng.uniform(-2.0, 2.0))
            init_vx = float(self.rng.uniform(-0.5, 0.5))
            init_vy = float(self.rng.uniform(-1.0, 0.0))
            init_theta = float(math.radians(self.rng.uniform(-15.0, 15.0)))
            init_omega = float(self.rng.uniform(-0.15, 0.15))

        init_fuel = float(options.get("initial_fuel", INITIAL_FUEL))

        # 2. Reset physics world
        self.world.reset(
            x=init_x,
            y=init_y,
            vx=init_vx,
            vy=init_vy,
            theta=init_theta,
            omega=init_omega,
            fuel=init_fuel,
        )

        self.current_step = 0
        self.total_episodes += 1
        self.last_state_raw = self.world.get_raw_state()
        self.last_terminal_result = None
        self.last_reward_breakdown = None

        obs = self.world.get_normalized_state()
        validate_state(obs)

        info = self._construct_info(
            col_info=self.world.last_collision_info,
            term_result=TerminalResult(
                done=False,
                terminated=False,
                truncated=False,
                termination_reason=TerminationType.IN_FLIGHT.value,
                is_safe_landing=False,
                is_crash=False,
                is_out_of_bounds=False,
                is_timeout=False,
            ),
            reward_bd=None,
            action=0,
            delta_fuel=0.0,
        )

        return obs, info

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one decision step (0.1s held action, 12 physics substeps at 120Hz).

        Parameters
        ----------
        action : integer in [0, 15]

        Returns
        -------
        obs : np.ndarray (shape (9,), float32)
        reward : float (scaled learning reward r_learn)
        terminated : bool (safe landing, crash, or out of bounds)
        truncated : bool (timeout 20s)
        info : dict containing detailed state, telemetry, and unscaled raw reward
        """
        # Validate action
        act = validate_action(action)
        state_curr_raw = self.world.get_raw_state()

        # Step physics world for 0.1s (12 substeps at 120Hz)
        step_result = self.world.step(action=act, dt_agent=DECISION_DT)

        self.current_step += 1
        state_next_raw = self.world.get_raw_state()
        self.last_state_raw = state_next_raw
        col_info = step_result["collision_info"]

        # Evaluate terminal condition
        term_result = self.terminal_detector.evaluate(
            col_info=col_info,
            current_time=state_next_raw["time"],
        )
        self.last_terminal_result = term_result

        # Calculate reward
        reward_bd = self.reward_engine.calculate_reward(
            state_curr=state_curr_raw,
            state_next=state_next_raw,
            col_info=col_info,
            term_result=term_result,
            delta_fuel=step_result["fuel_consumed"],
            delta_time=DECISION_DT,
        )
        self.last_reward_breakdown = reward_bd

        # Get and validate next observation
        obs = step_result["obs"]
        validate_state(obs)

        # Construct comprehensive info dictionary
        info = self._construct_info(
            col_info=col_info,
            term_result=term_result,
            reward_bd=reward_bd,
            action=act,
            delta_fuel=step_result["fuel_consumed"],
        )

        return obs, reward_bd.r_learn, term_result.terminated, term_result.truncated, info

    def step_done(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Classic Gym API: returns (obs, reward, done, info) where done = terminated or truncated.
        """
        obs, reward, terminated, truncated, info = self.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def _construct_info(
        self,
        col_info: Any,
        term_result: TerminalResult,
        reward_bd: Optional[RewardBreakdown],
        action: int,
        delta_fuel: float,
    ) -> Dict[str, Any]:
        """
        Construct structured info dictionary with all required physical quantities.
        """
        raw = self.world.get_raw_state()
        valves = decode_action(action)

        speed_mag = math.sqrt(raw["vx"]**2 + raw["vy"]**2)

        info: Dict[str, Any] = {
            # Required core state telemetry
            "x": float(raw["x"]),
            "h": float(raw["y"]),
            "vx": float(raw["vx"]),
            "vy": float(raw["vy"]),
            "theta": float(raw["theta"]),
            "theta_deg": float(raw["theta_deg"]),
            "omega": float(raw["omega"]),
            "fuel": float(raw["fuel"]),
            "time": float(raw["time"]),
            # Landing specific telemetry
            "landing_speed": float(speed_mag),
            "landing_speed_vy": float(abs(raw["vy"])),
            "landing_speed_vx": float(abs(raw["vx"])),
            "landing_angle": float(raw["theta"]),
            "landing_angle_deg": float(raw["theta_deg"]),
            "landing_position": float(raw["x"]),
            # Termination flags
            "termination_reason": term_result.termination_reason,
            "termination_type": term_result.termination_reason,
            "safe_landing": term_result.is_safe_landing,
            "crash": term_result.is_crash,
            "out_of_bounds": term_result.is_out_of_bounds,
            "timeout": term_result.is_timeout,
            "done": term_result.done,
            # Target zone
            "in_target_zone": col_info.is_in_target_zone,
            # Action and controls
            "action": action,
            "valves": valves,
            "bin_action": action_to_bin_str(action),
            # Acrobatic bonus
            "flips_completed": self.world.get_num_flips(),
            # Fuel and step accounting
            "step_count": self.current_step,
            "fuel_consumed_step": delta_fuel,
        }

        if reward_bd is not None:
            info.update(
                {
                    "raw_reward": reward_bd.r_base,
                    "learning_reward": reward_bd.r_learn,
                    "shaping_reward": reward_bd.r_shaping,
                    "reward_terminal": reward_bd.r_terminal,
                    "reward_target_zone": reward_bd.r_target_zone,
                    "reward_impact_vel": reward_bd.r_impact_vel,
                    "reward_impact_angle": reward_bd.r_impact_angle,
                    "reward_gas": reward_bd.r_gas,
                    "reward_time": reward_bd.r_time,
                }
            )

        return info
