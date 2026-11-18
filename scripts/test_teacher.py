import sys; sys.path.insert(0, '.')
import math
from environment.roadster_env import RoadsterEnv
from environment.action import encode_action

def teacher_policy(raw_state):
    x = raw_state['x']
    y = raw_state['h']
    vx = raw_state['vx']
    vy = raw_state['vy']
    theta = raw_state['theta']
    omega = raw_state['omega']
    
    # 1. Lateral control (pure lateral force, zero torque)
    target_vx = -1.0 * x
    target_vx = max(-1.0, min(1.0, target_vx))
    vx_err = vx - target_vx
    u3 = 1 if vx_err < -0.1 else 0
    u4 = 1 if vx_err > 0.1 else 0
    
    # 2. Vertical descent profile
    h_above = max(0.0, y - 0.72)
    if h_above > 4.5:
        target_vy = -3.0
    elif h_above > 2.5:
        target_vy = -2.0
    elif h_above > 1.2:
        target_vy = -1.0
    elif h_above > 0.4:
        target_vy = -0.5
    else:
        target_vy = -0.2
        
    need_lift = (vy < target_vy)
    
    # Terminal cushion: prevent free fall into ground
    if h_above < 0.35 and vy < -0.1:
        need_lift = True
    elif h_above < 0.8 and vy < -0.5:
        need_lift = True
        
    # 3. Attitude & Angular Rate Damping Control
    u1 = 0
    u2 = 0
    
    # Check if attitude is already stable
    if h_above < 1.5 and abs(theta) < math.radians(3.5) and abs(omega) < 0.25:
        if need_lift:
            u1 = 1
            u2 = 1
        else:
            u1 = 0
            u2 = 0
    else:
        theta_next = theta + omega * 0.1
        if omega > 0.25:
            u1 = 1
            u2 = 0
        elif omega < -0.25:
            u2 = 1
            u1 = 0
        elif theta_next > math.radians(2.5):
            u1 = 1
            u2 = 0
        elif theta_next < -math.radians(2.5):
            u2 = 1
            u1 = 0
        else:
            if need_lift:
                u1 = 1
                u2 = 1
                
    return encode_action(u1, u2, u3, u4)

env = RoadsterEnv()
success = 0
total = 100
landing_times = []
landing_speeds = []
landing_angles = []

for ep in range(total):
    obs, info = env.reset(seed=ep, options={'level': 3})
    done = False
    while not done:
        action = teacher_policy(info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    if info['safe_landing']:
        success += 1
        landing_times.append(info['time'])
        landing_speeds.append(info['landing_speed'])
        landing_angles.append(abs(info['landing_angle_deg']))

print(f"Success: {success}/{total} ({success/total*100:.1f}%)")
if landing_times:
    print(f"Avg Landing Time: {sum(landing_times)/len(landing_times):.2f}s (Max: {max(landing_times):.2f}s)")
    print(f"Avg Landing Speed: {sum(landing_speeds)/len(landing_speeds):.2f} m/s")
    print(f"Avg Landing Angle: {sum(landing_angles)/len(landing_angles):.2f} deg")
