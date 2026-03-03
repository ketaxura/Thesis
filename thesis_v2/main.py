# A simple MPCC simulation of a unicycle path tracking a reference path and reaching the goal pose

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import mpcc

from params import *
from dynamics import *
from matplotlib.patches import Ellipse
from visualization import visualize
from subplots import subplots

log_file = open("mpcc_main_run_log.txt", "w")
sys.stdout = log_file
sys.stderr = log_file   # optional: capture errors too

solver = mpcc.solver
lbx = mpcc.lbx
ubx = mpcc.ubx
lbg = mpcc.lbg
ubg = mpcc.ubg
print("Running MPCC")


# Reference Trajectory Definition
# A straight line for this simple case


def arc_length_parameterize(xy: np.ndarray):
    """
    xy: (M,2) array of waypoints
    returns:
      s_grid: (M,) cumulative arc-length
    """
    diffs = np.diff(xy, axis=0)
    ds = np.linalg.norm(diffs, axis=1)
    s_grid = np.concatenate(([0.0], np.cumsum(ds)))
    return s_grid


ref_x = np.linspace(0, -10, 100)
ref_y = np.linspace(0, 10, 100)
ref_traj=[]

ref_traj = np.vstack((ref_x, ref_y))
s_grid = arc_length_parameterize(ref_traj)      # (M,)

mu = 0.0      # GLOBAL progress along full reference

X_goal_global = np.array(
    [ref_traj[0, -1], ref_traj[1, -1], 0.0],
    dtype=float
)

u_prev = np.zeros((nu,))   # [v_prev, omega_prev]

prev_z = None


# max_steps = min(5000, ref_traj.shape[1] - 1)
for k in range(750):

    # -------------------------------------------------
    # 1. Build local reference window
    # -------------------------------------------------
    k_ref = int(np.floor(mu))
    k_ref = max(k_ref, 0)
    s_local = mu - k_ref
    R_horizon = ref_traj[:, k_ref : k_ref + N + 1]

    if R_horizon.shape[1] < N + 1:
        last_col = R_horizon[:, -1].reshape(-1,1)
        pad = np.repeat(last_col, N+1 - R_horizon.shape[1], axis=1)
        R_horizon = np.hstack((R_horizon, pad))
        
    # X_goal = np.array([R_horizon[0, -1], R_horizon[1, -1], 0.0])
    X_goal = X_goal_global
    

    # -------------------------------------------------
    # 2. Build parameter vector
    # -------------------------------------------------
    p = np.concatenate([
        x_current,
        u_prev,
        np.array([s_local]),
        R_horizon.flatten(),
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
        "p": p
    }

    if prev_z is not None:
        kwargs["x0"] = prev_z

    sol = solver(**kwargs)

    prev_z = sol["x"]

    # -------------------------------------------------
    # 4. Extract optimal control
    # -------------------------------------------------
    z = sol["x"].full().flatten()
    offset = 0

    X_opt = z[offset : offset + mpcc.nX].reshape((nx, N+1), order="F")
    offset += mpcc.nX

    U_opt = z[offset : offset + mpcc.nU].reshape((nu, N), order="F")
    offset += mpcc.nU

    s_opt = z[offset : offset + mpcc.nProg].reshape((N+1,), order="F")
    offset += mpcc.nProg
    
    print("s min:", np.min(s_opt), " s max:", np.max(s_opt))

    
    assert offset == z.size, (offset, z.size)


    # -------------------------------------------------
    # 5. Apply first control (receding horizon)
    # -------------------------------------------------
    v, omega = U_opt[:, 0]
    u_prev = np.array([v, omega])

    # tangent from the local horizon (always correct)
    dR0 = R_horizon[:, 1] - R_horizon[:, 0]          # shape (2,)
    t0 = dR0 / (np.linalg.norm(dR0) + 1e-12)         # unit tangent

    heading = np.array([np.cos(x_current[2]), np.sin(x_current[2])], dtype=float)
    v_par_mps = float(v) * float(np.dot(t0, heading))
    
    if k == 0:
        print("tangent t0 =", t0)

    e_cont_vals = mpcc.e_cont_horizon_fun(X_opt, s_opt, R_horizon)
    e_cont_vals = np.array(e_cont_vals.full()).flatten()

    print("True contouring horizon:", e_cont_vals)



    # # waypoint spacing (meters per index)
    # dR = ref_traj[:, 1] - ref_traj[:, 0]
    # ds_ref = float(np.linalg.norm(dR))


    # # mu += dt * vs0
    # mu += dt * v
    
    dR = ref_traj[:, 1] - ref_traj[:, 0]
    ds_ref = float(np.linalg.norm(dR))  # meters per waypoint-index
    heading = np.array([np.cos(x_current[2]), np.sin(x_current[2])])
    v_par = float(v) * float(np.dot(t0, heading))

    # s_opt[0] should equal s_local (because of s[0] = s_prev constraint)
    # Use s_opt[1] as the applied progress after one step
    mu = k_ref + float(s_opt[1])
    
    print(f"mu:{mu}")

    # clamp to valid waypoint index range
    mu = np.clip(mu, 0.0, ref_traj.shape[1] - 1)

    mu_hist.append(mu)

    # Unicycle forward simulation
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
    
    
    
    
    cont_vals = mpcc.cont_cost_fun(X_opt, U_opt, s_opt, R_horizon)
    lag_vals  = mpcc.lag_cost_fun(X_opt, U_opt, s_opt, R_horizon)
    prog_vals = mpcc.prog_cost_fun(X_opt, U_opt, s_opt, R_horizon)
    ctrl_vals = mpcc.ctrl_cost_fun(X_opt, U_opt, s_opt, R_horizon)
    align_vals = mpcc.align_cost_fun(X_opt, U_opt, s_opt, R_horizon)

    cont_vals = np.array(cont_vals.full()).flatten()
    lag_vals  = np.array(lag_vals.full()).flatten()
    prog_vals = np.array(prog_vals.full()).flatten()
    ctrl_vals = np.array(ctrl_vals.full()).flatten()
    align_vals = np.array(align_vals.full()).flatten()

    print("SUM contour:", np.sum(cont_vals))
    print("SUM lag:", np.sum(lag_vals))
    print("SUM progress:", np.sum(prog_vals))
    print("SUM control:", np.sum(ctrl_vals))
    print("SUM align:", np.sum(align_vals))
    print("TOTAL:", np.sum(cont_vals)+np.sum(lag_vals)+np.sum(prog_vals)+np.sum(ctrl_vals)+np.sum(align_vals))

    # --- Compute numerical contouring + lag error for debug ---

    # Get local reference window again
    k_ref = int(np.floor(mu))
    R_horizon = ref_traj[:, k_ref : k_ref + N + 1]

    # Pick first reference point for simplicity
    r0 = R_horizon[:, 0]

    # Compute simple geometric error (straight line case)
    error_vec = x_current[0:2] - r0
    
    # Tangent from the current local reference window
    if R_horizon.shape[1] >= 2:
        dR = R_horizon[:, 1] - R_horizon[:, 0]
    else:
        dR = ref_traj[:, 1] - ref_traj[:, 0]   # fallback
    t = dR / (np.linalg.norm(dR) + 1e-12)

    n = np.array([-t[1], t[0]])

    n = np.array([-t[1], t[0]])

    e_cont_val = np.dot(n, error_vec)
    e_lag_val  = np.dot(t, error_vec)


    cont_err_hist.append(e_cont_val)
    lag_err_hist.append(e_lag_val)
    
    """
    Debug Metrics for a given sim step k:
    Position of robot
    Control input U that was applied
    Control U limit
    Contouring error
    Lag error
    Geometric error
    """
    
    #################
    # Debug Metrics #
    #################
    print(" ")
    print(f"step: {k}")
    print(f"x vector: {x_current}")
    print(f"control input: {U_opt[:, 0]}")
    print(f"control input linear velocity limit: {dv_max}")
    print(f"control input angular velocity limit: {dw_max}")

    print("contouring error numeric:", e_cont_val)
    print("lag error numeric:", e_lag_val)

    print(f"error: {error_vec}")
    print(" ")

x_state_hist = np.array(x_state_hist)   # shape (K,3)
u_hist = np.array(u_hist)               # shape (K,2)

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_file.close()

print("Simulation complete, moving to produce to visualization")

subplots(x_state_hist, u_hist, cont_err_hist, lag_err_hist, mu_hist)

visualize(ref_traj, x_history, y_history, theta_history)

