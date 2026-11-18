import sys
import os
sys.path.insert(0, os.path.abspath("."))

from environment.roadster_env import RoadsterEnv
from demonstration.flip_controller import PhysicsFlipController

env = RoadsterEnv(use_shaping=False)
ctrl = PhysicsFlipController()
obs, info = env.reset(seed=42, options={"initial_x": 0.0, "initial_y": 9.0, "initial_vx": 0.0, "initial_vy": 0.0, "initial_theta": 0.0, "initial_omega": 0.0})
done = False
for step in range(70):
    raw = env.world.get_raw_state()
    action = ctrl.compute_action(raw)
    obs, r, term, trunc, info = env.step(action)
    print(f"step {step:02d}: h={info['h']:.2f} | vy={info['vy']:.2f} | th_deg={info['theta_deg']:.1f} | rot={raw.get('accumulated_rotation', 0.0):.2f} | a={action:04b}")
    if term or trunc:
        break

print("\nResult:")
print("Term reason:", info.get("termination_reason"))
print("Safe landing:", info.get("safe_landing"))
print("Flips completed:", info.get("flips_completed"))
print("Landing speed:", info.get("landing_speed"))
print("Landing angle deg:", info.get("landing_angle_deg"))
print("Landing time:", info.get("time"))
