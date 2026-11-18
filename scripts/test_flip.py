import sys
import os
sys.path.insert(0, os.path.abspath("."))

from environment.roadster_env import RoadsterEnv
from demonstration.flip_controller import PhysicsFlipController

env = RoadsterEnv(use_shaping=False)
ctrl = PhysicsFlipController()
obs, info = env.reset(seed=42, options={
    "initial_x": 0.0,
    "initial_y": 8.5,
    "initial_vx": 0.0,
    "initial_vy": 4.9,
    "initial_theta": 0.0,
    "initial_omega": 5.5,
})
done = False
for step in range(80):
    raw = env.world.get_raw_state()
    action = ctrl.compute_action(raw)
    obs, r, term, trunc, info = env.step(action)
    print(f"step {step:02d} (t={info['time']:.2f}s): h={info['h']:.2f}m | vy={info['vy']:+.2f}m/s | vx={info['vx']:+.2f}m/s | th_deg={info['theta_deg']:+.1f}° | a={action:04b}")
    if term or trunc:
        break

print("\n" + "=" * 50)
print("Result:")
print("Term reason:     ", info.get("termination_reason"))
print("Safe landing:    ", info.get("safe_landing"))
print("Flips completed: ", info.get("flips_completed"))
print(f"Landing speed:    {info.get('landing_speed', 0):.3f} m/s")
print(f"Landing angle:    {info.get('landing_angle_deg', 0):.2f}°")
print(f"Landing time:     {info.get('time', 0):.2f} s")
print("=" * 50)
