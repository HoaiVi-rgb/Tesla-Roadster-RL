import sys
import os
sys.path.insert(0, os.path.abspath("."))

from environment.roadster_env import RoadsterEnv
from rl.dqfd.agent import DQfDAgent

agent = DQfDAgent()
agent.load_checkpoint("checkpoints/dqfd/best_pretrained.pth")
env = RoadsterEnv(use_shaping=False)
obs, info = env.reset(seed=10000, options={"level": 3})
print("Initial raw state:", env.world.get_raw_state())
for step in range(50):
    q_vals = agent.get_q_values(obs)
    action = int(agent.select_action(obs, evaluate=True))
    obs, r, term, trunc, info = env.step(action)
    print(f"Step {step:02d}: a={action} ({action:04b}) | h={info['h']:.2f} | vy={info['vy']:.2f} | vx={info['vx']:.2f} | th={info['theta_deg']:.1f} | term={info.get('termination_type')}")
    if term or trunc:
        print("Terminated at step", step, "reason:", info.get("termination_type"), "col_type:", info.get("collision_type"))
        break
