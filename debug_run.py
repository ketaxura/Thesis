"""
debug_run_mpc.py
Visualize a single MPC (hard) run to inspect collision behaviour.
Drop into MPC project folder and run.

Change SEED and PATH_ID to investigate different runs.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse

from dynamics import unicycle_dynamics
from obstacles import ObstacleManager

from obs import Obstacle
from run_experiment_mpc_v2 import (
    build_mpc_solver, _ellipse_surface_dist,
    dt, N, nx, nu, r_robot, safety_buffer,
    NEAR_MISS_MARGIN
)
# ============================================================
# CHANGE THESE
SEED     = 0
PATH_ID  = 3    # 3=zigzag, 1=sine
MAX_STEPS = 800
# ============================================================

print(f"Debug MPC (hard): path_id={PATH_ID}, seed={SEED}")

# if PATH_ID == 3:
#     ref_traj = make_zigzag()
# else:
#     ref_traj = make_sine()

# STATIC_RECTS = make_static_obs(PATH_ID)
# obstacles, num_dyn_obs = make_dynamic_obs(PATH_ID, SEED)


env = Obstacle(PATH_ID)

ref_traj     = env.path_selector(PATH_ID)
STATIC_RECTS = env.static_obs()
dyn_obs      = env.dynamic_obs_seeded(SEED)
num_dyn_obs  = len(dyn_obs)

# solver, lbx, ubx, lbg, ubg, nX, nU, nS = build_mpc_solver(STATIC_RECTS, num_dyn_obs)
solver, lbx, ubx, lbg, ubg, nX, nU = build_mpc_solver(STATIC_RECTS, dyn_obs, use_hard=True)

p0  = ref_traj[:, 0]; p1 = ref_traj[:, 1]
th0 = float(np.arctan2(p1[1]-p0[1], p1[0]-p0[0]))
x_current     = np.array([p0[0], p0[1], th0])
X_goal_global = np.array([ref_traj[0,-1], ref_traj[1,-1], 0.0])

u_prev = np.zeros(nu)
prev_z = None
ref_k  = 0
_at_end_count = 0

x_hist   = [x_current[0]]
y_hist   = [x_current[1]]
th_hist  = [x_current[2]]
obs_hist = []   # list of (num_dyn_obs, 2) arrays

collision_steps = []   # (step, obs_idx, rx, ry, ox, oy)
static_collision_steps = []   # (step, rx, ry, cx, cy)
goal_reached = False
steps_taken  = 0

for k in range(MAX_STEPS):

    R_horizon = ref_traj[:, ref_k : ref_k + N + 1]
    if R_horizon.shape[1] < N + 1:
        last = R_horizon[:, -1].reshape(2,1)
        pad  = np.repeat(last, N+1-R_horizon.shape[1], axis=1)
        R_horizon = np.hstack([R_horizon, pad])

    X_goal_val = np.array([R_horizon[0,-1], R_horizon[1,-1], 0.0])
    # obsx_h, obsy_h = obstacles.predict_horizon()
    
    if dyn_obs:
        preds = [obs.predict_horizon(N, dt) for obs in dyn_obs]
        obsx_h = np.array([p[0] for p in preds])
        obsy_h = np.array([p[1] for p in preds])
    else:
        obsx_h = np.zeros((0, N))
        obsy_h = np.zeros((0, N))

    p = np.concatenate([
        x_current, u_prev,
        R_horizon.flatten(order='F'),
        obsx_h.flatten(order='F'),
        obsy_h.flatten(order='F'),
        X_goal_val,
    ])

    kwargs = dict(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)
    if prev_z is not None:
        kwargs["x0"] = prev_z

    sol    = solver(**kwargs)
    prev_z = sol["x"]

    z     = sol["x"].full().flatten()
    U_opt = z[nX : nX+nU].reshape((nu, N), order="F")
    v, omega = U_opt[:, 0]
    u_prev   = np.array([v, omega])

    ref_k = min(ref_k + 1, ref_traj.shape[1] - 1)

    x_current = np.array([
        x_current[0] + dt * v * np.cos(x_current[2]),
        x_current[1] + dt * v * np.sin(x_current[2]),
        x_current[2] + dt * omega,
    ])
    steps_taken += 1

    # Step obstacles, then record (matches MPCC timing)
    for obs in dyn_obs:
        obs.step(dt, STATIC_RECTS)

    obs_pos_now = np.array([[obs.x, obs.y] for obs in dyn_obs])
        
    
    obs_hist.append(obs_pos_now)

    x_hist.append(x_current[0])
    y_hist.append(x_current[1])
    th_hist.append(x_current[2])

    # Collision detection
    # for i in range(num_dyn_obs):
    #     ox, oy = obs_pos_now[i]
    #     dx = x_current[0] - ox
    #     dy = x_current[1] - oy
    #     if (dx/A_OBS)**2 + (dy/B_OBS)**2 <= 1.0:
    #         print(f"  [COLLISION] step {k}: obs {i}  "
    #               f"robot=({x_current[0]:.2f},{x_current[1]:.2f})  "
    #               f"obs=({ox:.2f},{oy:.2f})  "
    #               f"dist={np.sqrt((dx/A_OBS)**2+(dy/B_OBS)**2):.3f}")
    #         collision_steps.append((k, i, x_current[0], x_current[1], ox, oy))
    
    for obs in dyn_obs:
        dx = x_current[0] - obs.x
        dy = x_current[1] - obs.y

        if (dx/obs.a)**2 + (dy/obs.b)**2 <= 1.0:
            print(f"[DYN COLLISION] step {k}  robot=({x_current[0]:.2f},{x_current[1]:.2f})  obs=({obs.x:.2f},{obs.y:.2f})")
                
                
        # Static obstacle collisions
    for (cx, cy, hw, hh) in STATIC_RECTS:
        if abs(x_current[0]-cx) < hw + r_robot and \
        abs(x_current[1]-cy) < hh + r_robot:

            print(f"  [STATIC COLLISION] step {k}  "
                f"robot=({x_current[0]:.2f},{x_current[1]:.2f})  "
                f"rect_center=({cx:.2f},{cy:.2f})")

            static_collision_steps.append(
                (k, x_current[0], x_current[1], cx, cy)
            )

    dist_to_goal  = np.linalg.norm(x_current[:2] - X_goal_global[:2])
    path_complete = ref_k >= ref_traj.shape[1] - 2

    if dist_to_goal < 0.5 and path_complete:
        goal_reached = True; break
    if path_complete:
        _at_end_count += 1
        if _at_end_count >= 30:
            goal_reached = True; break
    else:
        _at_end_count = 0

print(f"\nGoal reached: {goal_reached}  Steps: {steps_taken}")
print(f"Dynamic collisions: {len(collision_steps)}")
print(f"Static collisions: {len(static_collision_steps)}")
print(f"Total collisions: {len(collision_steps) + len(static_collision_steps)}")

# ============================================================
# Animate
# ============================================================
plt.ion()
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(ref_traj[0], ref_traj[1], 'k--', lw=2, label='Reference', alpha=0.7)
path_line, = ax.plot([], [], 'b-', lw=2, label='Trajectory')

robot_body = plt.Circle((x_hist[0], y_hist[0]), r_robot,
    edgecolor='blue', facecolor='lightblue', alpha=0.7, lw=2)
robot_buf = plt.Circle((x_hist[0], y_hist[0]), r_robot + safety_buffer,
    edgecolor='blue', facecolor='none', lw=1.5, linestyle='--', alpha=0.4)
ax.add_patch(robot_body)
ax.add_patch(robot_buf)

arrow = ax.quiver(x_hist[0], y_hist[0],
    0.4*np.cos(th_hist[0]), 0.4*np.sin(th_hist[0]),
    angles='xy', scale_units='xy', scale=1, color='darkblue', width=0.012)

for (cx, cy, hw, hh) in STATIC_RECTS:
    ax.add_patch(patches.Rectangle(
        (cx-hw, cy-hh), 2*hw, 2*hh,
        edgecolor='black', facecolor='gray', alpha=0.5))

dyn_bodies  = []
dyn_margins = []
init_obs = obs_hist[0] if obs_hist else np.array([[obs.x, obs.y] for obs in dyn_obs])

for i, obs in enumerate(dyn_obs):
    ox, oy = init_obs[i]
    body = Ellipse((ox, oy), 2*obs.a, 2*obs.b,
                   edgecolor='red', facecolor='salmon', alpha=0.5, lw=2)
    margin = Ellipse((ox, oy),
                     2*(obs.a + r_robot + safety_buffer),
                     2*(obs.b + r_robot + safety_buffer),
                     edgecolor='red', facecolor='none', lw=1.5,
                     linestyle='--', alpha=0.5)
    ax.add_patch(body);   dyn_bodies.append(body)
    ax.add_patch(margin); dyn_margins.append(margin)

# Mark collision points
# Dynamic collisions → RED
for (step, i, rx, ry, ox, oy) in collision_steps:
    ax.plot(rx, ry, 'rx', markersize=12, markeredgewidth=3)
    ax.annotate(f'D{k}', (rx, ry), textcoords='offset points',
                xytext=(5,5), color='red', fontsize=8)

# Static collisions → ORANGE
for (step, rx, ry, cx, cy) in static_collision_steps:
    ax.plot(rx, ry, 'x', color='orange', markersize=12, markeredgewidth=3)
    ax.annotate(f'S{k}', (rx, ry), textcoords='offset points',
                xytext=(5,5), color='orange', fontsize=8)

step_txt = ax.text(0.02, 0.97, 'step: 0', transform=ax.transAxes,
    fontsize=10, va='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

ax.set_xlim(-2, 23); ax.set_ylim(-4, 7)
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=8)
ax.set_title(f'MPC Hard — path_id={PATH_ID} seed={SEED}  '
             f'dyn={len(collision_steps)}, static={len(static_collision_steps)}')

T = min(len(x_hist), len(obs_hist))
for k in range(T):
    robot_body.center = (x_hist[k], y_hist[k])
    robot_buf.center  = (x_hist[k], y_hist[k])
    arrow.set_offsets([x_hist[k], y_hist[k]])
    arrow.set_UVC(0.4*np.cos(th_hist[k]), 0.4*np.sin(th_hist[k]))
    path_line.set_data(x_hist[:k+1], y_hist[:k+1])

    for i in range(num_dyn_obs):
        ox, oy = obs_hist[k][i]
        dyn_bodies[i].set_center((ox, oy))
        dyn_margins[i].set_center((ox, oy))

    step_txt.set_text(f'step: {k}')
    plt.pause(0.04)

plt.ioff()
plt.show(block=True)