"""
UNICYCLE MODEL PREDICTIVE CONTOURING CONTROL 
Usukhbayar Amgalanbat
"""




import params
from mpcc import build_mpcc_solver
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import numpy as np

from obs import Obstacle
from params import *
from dynamics import *
from visualization import visualize
from subplots import subplots



"""
Reference Path Generation:

Path 0 = Straight line path
Path 1 = Sine wave
Path 2 = Figure 8
Path 3 = Sharp zig zag
Path 4 = Big spiral
"""

path_id = 3
env = Obstacle(3)



ref_traj = env.path_selector(path_id)
STATIC_RECTS  = env.static_obs()
dyn_obs = env.dynamic_obs()


mpcc_data = build_mpcc_solver(ref_traj, STATIC_RECTS, dyn_obs)

solver = mpcc_data.solver
lbx = mpcc_data.lbx
ubx = mpcc_data.ubx
lbg = mpcc_data.lbg
ubg = mpcc_data.ubg




# =============================================================
# Initial state — always derived from the path defined above
# =============================================================

log_file = open("mpcc_main_run_log.txt", "w")
sys.stdout = log_file
sys.stderr = log_file

x_current, x_history, y_history, theta_history = main_init(ref_traj)

mu = 0.0   # global progress index along ref_traj




X_goal_global = np.array(
    [ref_traj[0, -1], ref_traj[1, -1], 0.0],
    dtype=float
)

u_prev = np.zeros((nu,))
prev_z = None
dyn_obs_hist = []
_at_end_count = 0   # consecutive steps at max path progress

# Collision tracking
# "body"      — robot centre inside raw obstacle (true physical contact)
# "exclusion" — robot centre inside solver exclusion zone (a + r_robot + safety_buffer)
dyn_body_log      = []   # (step, obs_index)
dyn_exclusion_log = []   # (step, obs_index)
static_body_log   = []   # (step, obs_index)


# =============================================================
# Main loop
# =============================================================
for k in range(500):

    # -------------------------------------------------
    # 1. Build local reference window
    # -------------------------------------------------
    k_ref = int(np.floor(mu))
    k_ref = max(k_ref, 0)
    s_local = mu - k_ref

    R_horizon = ref_traj[:, k_ref : k_ref + N + 1]

    if R_horizon.shape[1] < N + 1:
        last_col = R_horizon[:, -1].reshape(-1, 1)
        pad = np.repeat(last_col, N + 1 - R_horizon.shape[1], axis=1)
        R_horizon = np.hstack((R_horizon, pad))

    X_goal = X_goal_global

    # -------------------------------------------------
    # 2. Build parameter vector
    # -------------------------------------------------
    # Pre-compute full horizon predictions for each dynamic obstacle
    if dyn_obs:
        horizon_preds = [obs.predict_horizon(N, dt) for obs in dyn_obs]
        obs_x_h     = np.array([pr[0] for pr in horizon_preds])  # (num_dyn_obs, N)
        obs_y_h     = np.array([pr[1] for pr in horizon_preds])  # (num_dyn_obs, N)
        obs_theta_h = np.array([pr[2] for pr in horizon_preds])  # (num_dyn_obs, N)
    else:
        obs_x_h     = np.zeros((0, N))
        obs_y_h     = np.zeros((0, N))
        obs_theta_h = np.zeros((0, N))

    p = np.concatenate([
        x_current,
        u_prev,
        np.array([s_local]),
        R_horizon.flatten(order='F'),
        X_goal,
        obs_x_h.flatten(order='F'),
        obs_y_h.flatten(order='F'),
        obs_theta_h.flatten(order='F'),
    ])

    # -------------------------------------------------
    # 3. Solve NLP
    # -------------------------------------------------
    kwargs = {
        "lbx": lbx,
        "ubx": ubx,
        "lbg": lbg,
        "ubg": ubg,
        "p":   p,
    }

    if prev_z is not None:
        kwargs["x0"] = prev_z

    sol    = solver(**kwargs)
    prev_z = sol["x"]

    # -------------------------------------------------
    # 4. Extract optimal control
    # -------------------------------------------------
    z = sol["x"].full().flatten()
    offset = 0

    X_opt = z[offset:offset + mpcc_data.nX].reshape((nx, N + 1), order="F")
    offset += mpcc_data.nX

    U_opt = z[offset:offset + mpcc_data.nU].reshape((nu, N), order="F")
    offset += mpcc_data.nU

    s_opt = z[offset:offset + mpcc_data.nProg].reshape((N + 1,), order="F")
    offset += mpcc_data.nProg


    S_obs_opt = z[offset:offset + mpcc_data.nSlack_static].reshape((len(STATIC_RECTS), N), order="F")
    offset += mpcc_data.nSlack_static

    # No S_dyn_opt — dynamic obstacles use hard constraints


    assert offset == z.size, (offset, z.size)

    print("s min:", np.min(s_opt), " s max:", np.max(s_opt))

    # -------------------------------------------------
    # 5. Apply first control (receding horizon)
    # -------------------------------------------------
    v, omega = U_opt[:, 0]
    u_prev   = np.array([v, omega])

    # tangent from the local horizon
    dR0     = R_horizon[:, 1] - R_horizon[:, 0]
    t0      = dR0 / (np.linalg.norm(dR0) + 1e-12)

    # -------------------------------------------------
    # 6. Advance global progress using local s_opt
    # -------------------------------------------------
    delta_s = float(s_opt[1]) - float(s_opt[0])   # incremental progress this step
    mu      = mu + delta_s
    mu      = np.clip(mu, 0.0, ref_traj.shape[1] - 1)

    mu_hist.append(mu)

    print(f"mu: {mu:.4f}")

    # -------------------------------------------------
    # 7. Simulate robot forward one step
    # -------------------------------------------------

    # Record robot and obstacles at time k (same snapshot the solver used)
    x_history.append(x_current[0])
    y_history.append(x_current[1])
    theta_history.append(x_current[2])
    dyn_obs_hist.append([(obs.x, obs.y, obs.theta) for obs in dyn_obs])

    # Now advance both to k+1
    x_current = np.array([
        x_current[0] + dt * v * np.cos(x_current[2]),
        x_current[1] + dt * v * np.sin(x_current[2]),
        x_current[2] + dt * omega
    ])

    x_state_hist.append(x_current.copy())
    u_hist.append(U_opt[:, 0].copy())

    for obs in dyn_obs:
        obs.step(dt, STATIC_RECTS)

    # -------------------------------------------------
    # 7b. Collision detection
    # -------------------------------------------------
    for i, obs in enumerate(dyn_obs):
        dx = x_current[0] - obs.x
        dy = x_current[1] - obs.y

        # Physical body: robot centre inside raw obstacle ellipse
        if (dx / obs.a) ** 2 + (dy / obs.b) ** 2 <= 1.0:
            dyn_body_log.append((k, i))
            print(f"  [BODY COLLISION]      step {k}: dyn obs {i} "
                  f"robot=({x_current[0]:.2f},{x_current[1]:.2f}) obs=({obs.x:.2f},{obs.y:.2f})")

        # Exclusion zone: same margin the solver enforces
        a_excl = obs.a + r_robot + safety_buffer
        b_excl = obs.b + r_robot + safety_buffer
        if (dx / a_excl) ** 2 + (dy / b_excl) ** 2 <= 1.0:
            dyn_exclusion_log.append((k, i))
            print(f"  [EXCLUSION VIOLATION] step {k}: dyn obs {i} "
                  f"robot=({x_current[0]:.2f},{x_current[1]:.2f}) obs=({obs.x:.2f},{obs.y:.2f})")

    for i, (cx, cy, hw, hh) in enumerate(STATIC_RECTS):
        if (abs(x_current[0] - cx) < hw + r_robot and
                abs(x_current[1] - cy) < hh + r_robot):
            static_body_log.append((k, i))
            print(f"  [BODY COLLISION]      step {k}: static obs {i} "
                  f"robot=({x_current[0]:.2f},{x_current[1]:.2f}) rect=({cx},{cy})")


    

    # -------------------------------------------------
    # 8. Compute geometric errors for logging
    # -------------------------------------------------
    k_ref2    = int(np.clip(np.floor(mu), 0, ref_traj.shape[1] - 2))
    R_now     = ref_traj[:, k_ref2 : k_ref2 + 2]
    r0        = R_now[:, 0]
    dR        = R_now[:, 1] - R_now[:, 0]
    t         = dR / (np.linalg.norm(dR) + 1e-12)
    n         = np.array([-t[1], t[0]])
    error_vec = x_current[0:2] - r0
    e_cont_val = np.dot(n, error_vec)
    e_lag_val  = np.dot(t, error_vec)

    cont_err_hist.append(e_cont_val)
    lag_err_hist.append(e_lag_val)

    # -------------------------------------------------
    # 9. Debug print
    # -------------------------------------------------
    print(f"\nstep: {k}")
    print(f"x vector: {x_current}")
    print(f"control input: {U_opt[:, 0]}")
    print(f"contouring error: {e_cont_val:.4f}")
    print(f"lag error:        {e_lag_val:.4f}")
    print()
    


    # -------------------------------------------------
    # 10. Termination check
    # -------------------------------------------------
    dist_to_goal  = np.linalg.norm(x_current[0:2] - X_goal_global[0:2])
    path_complete = mu >= ref_traj.shape[1] - 2

    # Primary: close enough to goal AND path done
    if dist_to_goal < 0.5 and path_complete:
        print(f"Goal reached at step {k}, distance={dist_to_goal:.4f}")
        break

    # Fallback: path is 100% done but an obstacle is blocking the goal point.
    # Declare success after 30 consecutive steps at max progress.
    if path_complete:
        _at_end_count += 1
        if _at_end_count >= 30:
            print(f"Path complete at step {k} (obstacle near goal, dist={dist_to_goal:.2f})")
            break
    else:
        _at_end_count = 0


# =============================================================
# Done
# =============================================================
x_state_hist = np.array(x_state_hist)
u_hist       = np.array(u_hist)

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_file.close()

print("Simulation complete, producing visualization")

print("\n========== COLLISION SUMMARY ==========")
print(f"Dyn obs — body collisions      : {len(dyn_body_log)}")
for step, idx in dyn_body_log:
    print(f"    step {step:4d}  dyn obs {idx}")
print(f"Dyn obs — exclusion violations : {len(dyn_exclusion_log)}")
for step, idx in dyn_exclusion_log:
    print(f"    step {step:4d}  dyn obs {idx}")
print(f"Static obs — body collisions   : {len(static_body_log)}")
for step, idx in static_body_log:   
    print(f"    step {step:4d}  static obs {idx}")
print(f"---")
print(f"Total body collisions          : {len(dyn_body_log) + len(static_body_log)}")
print(f"Total exclusion violations     : {len(dyn_exclusion_log)}")
print("========================================\n")

subplots(x_state_hist, u_hist, cont_err_hist, lag_err_hist, mu_hist)
visualize(ref_traj, x_history, y_history, theta_history, STATIC_RECTS, dyn_obs, dyn_obs_hist)