"""
debug_run.py
Run a single MPCC seed with full visualization to diagnose poor performance.
Change SEED and PATH_ID to investigate different runs.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import time

from mpcc import build_mpcc_solver
from obs import Obstacle
from visualization import visualize
from subplots import subplots

from params import (
    dt, N, nx, nu, nr,
    r_robot, safety_buffer,
    v_max, omega_max, dv_max, dw_max,
    kappa_w, eps,
    cont_err_hist, lag_err_hist, x_state_hist, u_hist, mu_hist
)
from dynamics import unicycle_dynamics, main_init

# ============================================================
# CHANGE THESE TO INVESTIGATE DIFFERENT RUNS
# ============================================================
SEED       = 2       # which seed failed
PATH_ID    = 3       # 3=zigzag, 1=sine
MAX_STEPS  = 800     # increased from 600
# ============================================================

print(f"Debug run: path_id={PATH_ID}, seed={SEED}, max_steps={MAX_STEPS}")

env          = Obstacle(PATH_ID)
ref_traj     = env.path_selector(PATH_ID)
STATIC_RECTS = env.static_obs()
dyn_obs      = env.dynamic_obs_seeded(seed_offset=SEED)

mpcc_data = build_mpcc_solver(ref_traj, STATIC_RECTS, dyn_obs)
solver = mpcc_data.solver
lbx, ubx = mpcc_data.lbx, mpcc_data.ubx
lbg, ubg = mpcc_data.lbg, mpcc_data.ubg

x_current, x_history, y_history, theta_history = main_init(ref_traj)

mu            = 0.0
X_goal_global = np.array([ref_traj[0, -1], ref_traj[1, -1], 0.0], dtype=float)
u_prev        = np.zeros(nu)
prev_z        = None
dyn_obs_hist  = []
_at_end_count = 0

solve_times   = []
goal_reached  = False

dyn_body_log      = []
dyn_exclusion_log = []
static_body_log   = []

for k in range(MAX_STEPS):

    # Reference window
    k_ref     = max(int(np.floor(mu)), 0)
    s_local   = mu - k_ref
    R_horizon = ref_traj[:, k_ref : k_ref + N + 1]
    if R_horizon.shape[1] < N + 1:
        last = R_horizon[:, -1].reshape(-1, 1)
        pad  = np.repeat(last, N + 1 - R_horizon.shape[1], axis=1)
        R_horizon = np.hstack((R_horizon, pad))

    # Horizon predictions
    if dyn_obs:
        preds    = [obs.predict_horizon(N, dt) for obs in dyn_obs]
        obs_x_h  = np.array([pr[0] for pr in preds])
        obs_y_h  = np.array([pr[1] for pr in preds])
        obs_th_h = np.array([pr[2] for pr in preds])
    else:
        obs_x_h  = np.zeros((0, N))
        obs_y_h  = np.zeros((0, N))
        obs_th_h = np.zeros((0, N))

    p = np.concatenate([
        x_current,
        u_prev,
        np.array([s_local]),
        R_horizon.flatten(order='F'),
        X_goal_global,
        obs_x_h.flatten(order='F'),
        obs_y_h.flatten(order='F'),
        obs_th_h.flatten(order='F'),
    ])

    kwargs = dict(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)
    if prev_z is not None:
        kwargs["x0"] = prev_z

    t0  = time.perf_counter()
    sol = solver(**kwargs)
    solve_times.append((time.perf_counter() - t0) * 1000)
    prev_z = sol["x"]

    z      = sol["x"].full().flatten()
    offset = 0
    X_opt  = z[offset:offset + mpcc_data.nX].reshape((nx, N+1), order="F"); offset += mpcc_data.nX
    U_opt  = z[offset:offset + mpcc_data.nU].reshape((nu, N),   order="F"); offset += mpcc_data.nU
    s_opt  = z[offset:offset + mpcc_data.nProg].reshape((N+1,), order="F"); offset += mpcc_data.nProg
    offset += mpcc_data.nSlack_static

    v, omega = U_opt[:, 0]
    u_prev   = np.array([v, omega])

    delta_s = float(s_opt[1]) - float(s_opt[0])
    mu      = float(np.clip(mu + delta_s, 0.0, ref_traj.shape[1] - 1))
    mu_hist.append(mu)

    # Record BEFORE stepping
    x_history.append(x_current[0])
    y_history.append(x_current[1])
    theta_history.append(x_current[2])
    dyn_obs_hist.append([(obs.x, obs.y, obs.theta) for obs in dyn_obs])

    x_current = np.array([
        x_current[0] + dt * v * np.cos(x_current[2]),
        x_current[1] + dt * v * np.sin(x_current[2]),
        x_current[2] + dt * omega,
    ])
    x_state_hist.append(x_current.copy())
    u_hist.append(U_opt[:, 0].copy())

    for obs in dyn_obs:
        obs.step(dt, STATIC_RECTS)

    # Collision detection
    for i, obs in enumerate(dyn_obs):
        dx = x_current[0] - obs.x
        dy = x_current[1] - obs.y
        if (dx / obs.a)**2 + (dy / obs.b)**2 <= 1.0:
            dyn_body_log.append((k, i))
        a_e = obs.a + r_robot + safety_buffer
        b_e = obs.b + r_robot + safety_buffer
        if (dx / a_e)**2 + (dy / b_e)**2 <= 1.0:
            dyn_exclusion_log.append((k, i))

    for i, (cx, cy, hw, hh) in enumerate(STATIC_RECTS):
        if abs(x_current[0]-cx) < hw+r_robot and abs(x_current[1]-cy) < hh+r_robot:
            static_body_log.append((k, i))

    # Errors
    k2    = int(np.clip(np.floor(mu), 0, ref_traj.shape[1]-2))
    R_now = ref_traj[:, k2:k2+2]
    dR    = R_now[:, 1] - R_now[:, 0]
    t_hat = dR / (np.linalg.norm(dR) + 1e-12)
    n_hat = np.array([-t_hat[1], t_hat[0]])
    ev    = x_current[:2] - R_now[:, 0]
    cont_err_hist.append(float(np.dot(n_hat, ev)))
    lag_err_hist.append(float(np.dot(t_hat, ev)))

    dist_to_goal  = np.linalg.norm(x_current[:2] - X_goal_global[:2])
    path_complete = mu >= ref_traj.shape[1] - 2
    prog_pct      = mu / (ref_traj.shape[1] - 1) * 100

    print(f"k={k:4d}  mu={mu:6.1f}/{ref_traj.shape[1]-1}  ({prog_pct:4.1f}%)  "
          f"dist_goal={dist_to_goal:.2f}  solve={solve_times[-1]:.1f}ms  "
          f"v={v:.2f}  delta_s={delta_s:.4f}")

    if dist_to_goal < 0.5 and path_complete:
        print(f"\n*** GOAL REACHED at step {k} ***")
        goal_reached = True
        break

    if path_complete:
        _at_end_count += 1
        if _at_end_count >= 30:
            print(f"\n*** PATH COMPLETE at step {k} (obstacle near goal, dist={dist_to_goal:.2f}) ***")
            goal_reached = True
            break
    else:
        _at_end_count = 0

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*50}")
print(f"Path id={PATH_ID}  seed={SEED}")
print(f"Goal reached: {goal_reached}")
print(f"Steps taken:  {k+1} / {MAX_STEPS}")
print(f"Final mu:     {mu:.1f} / {ref_traj.shape[1]-1}  ({mu/(ref_traj.shape[1]-1)*100:.1f}%)")
print(f"Mean solve:   {np.mean(solve_times):.1f} ms")
print(f"Max solve:    {np.max(solve_times):.1f} ms")
print(f"Body collisions:      {len(dyn_body_log) + len(static_body_log)}")
print(f"Exclusion violations: {len(dyn_exclusion_log)}")
print(f"{'='*50}\n")

x_state_hist_arr = np.array(x_state_hist)
u_hist_arr       = np.array(u_hist)

subplots(x_state_hist_arr, u_hist_arr, cont_err_hist, lag_err_hist, mu_hist)

# Cap T so visualization never overruns dyn_obs_hist
T = min(len(x_history), len(dyn_obs_hist))
visualize(ref_traj, x_history[:T], y_history[:T], theta_history[:T],
          STATIC_RECTS, dyn_obs, dyn_obs_hist)