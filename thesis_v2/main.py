"""
UNICYCLE MODEL PREDICTIVE CONTOURING CONTROL 
Usukhbayar Amgalanbat
"""
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import params
import mpcc

from params import *
from dynamics import *
from visualization import visualize
from subplots import subplots
from dynamics import path_selector

"""
Reference Path Generation:

Path 0 = Straight line path
Path 1 = Sine wave
Path 2 = Figure 8
Path 3 = Sharp zig zag
Path 4 = Big spiral
"""

ref_traj = path_selector(3)


# =============================================================
# Initial state — always derived from the path defined above
# =============================================================
solver = mpcc.solver
lbx    = mpcc.lbx
ubx    = mpcc.ubx
lbg    = mpcc.lbg
ubg    = mpcc.ubg

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


# =============================================================
# Main loop
# =============================================================
for k in range(1000):

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
    p = np.concatenate([
        x_current,
        u_prev,
        np.array([s_local]),
        R_horizon.flatten(order='F'),
        X_goal
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
    z      = sol["x"].full().flatten()
    offset = 0

    X_opt  = z[offset : offset + mpcc.nX].reshape((nx, N + 1), order="F");  offset += mpcc.nX
    U_opt  = z[offset : offset + mpcc.nU].reshape((nu, N),     order="F");  offset += mpcc.nU
    s_opt  = z[offset : offset + mpcc.nProg].reshape((N + 1,), order="F");  offset += mpcc.nProg

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
    x_current = np.array([
        x_current[0] + dt * v * np.cos(x_current[2]),
        x_current[1] + dt * v * np.sin(x_current[2]),
        x_current[2] + dt * omega
    ])

    x_state_hist.append(x_current.copy())
    u_hist.append(U_opt[:, 0].copy())

    x_history.append(x_current[0])
    y_history.append(x_current[1])
    theta_history.append(x_current[2])


    

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
    dist_to_goal = np.linalg.norm(x_current[0:2] - X_goal_global[0:2])
    if dist_to_goal < 0.1 and mu >= ref_traj.shape[1] - 2:
        print(f"Goal reached at step {k}, distance={dist_to_goal:.4f}")
        break


# =============================================================
# Done
# =============================================================
x_state_hist = np.array(x_state_hist)
u_hist       = np.array(u_hist)

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_file.close()

print("Simulation complete, producing visualization")

subplots(x_state_hist, u_hist, cont_err_hist, lag_err_hist, mu_hist)
visualize(ref_traj, x_history, y_history, theta_history)