# A simple MPCC simulation of a unicycle path tracking a reference path and reaching the goal pose

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

from params import *
from dynamics import *
import mpcc
# from mpcc import e_cont, e_lag_np, e

from matplotlib.patches import Ellipse
from visualization import visualize

import sys




# print(mpcc.nS)
# time.sleep(9999999)

log_file = open("mpcc_main_run_log.txt", "w")
sys.stdout = log_file
sys.stderr = log_file   # optional: capture errors too





mpcc_run = True

if mpcc_run:
    solver = mpcc.solver
    lbx = mpcc.lbx
    ubx = mpcc.ubx
    lbg = mpcc.lbg
    ubg = mpcc.ubg
    print("Running MPCC")
else:
    solver = mpc.solver
    lbx = mpc.lbx
    ubx = mpc.ubx
    lbg = mpc.lbg
    ubg = mpc.ubg
    print("Running standard MPC")


    


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

ref_x = np.linspace(0, 10, 100)
ref_y = np.linspace(0, 10, 100)
ref_traj=[]

# ref_traj = np.column_stack((ref_x, ref_y))

ref_traj = np.vstack((ref_x, ref_y))
s_grid = arc_length_parameterize(ref_traj)      # (M,)

mu = 0.0      # GLOBAL progress along full reference





X_goal_global = np.array(
    [ref_traj[0, -1], ref_traj[1, -1], 0.0],
    dtype=float
)


u_prev = np.zeros((nu,))   # [v_prev, omega_prev]

prev_z = None

cont_err_hist = []
lag_err_hist = []

x_state_hist = []
u_hist = []

mu_hist = []

# max_steps = min(5000, ref_traj.shape[1] - 1)
for k in range(500):

    # -------------------------------------------------
    # 1. Build local reference window
    # -------------------------------------------------
    k_ref = int(np.floor(mu))
    k_ref = min(k_ref, ref_traj.shape[1] - (N + 1))
    k_ref = max(k_ref, 0)
    s_local = mu - k_ref
    R_horizon = ref_traj[:, k_ref : k_ref + N + 1]

    # if R_horizon.shape[1] < N + 1:
    #     last = ref_traj[:, -1].reshape(2, 1)
    #     pad = np.repeat(last, N + 1 - R_horizon.shape[1], axis=1)
    #     R_horizon = np.hstack([R_horizon, pad])

    X_goal = np.array([R_horizon[0, -1], R_horizon[1, -1], 0.0])

    

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

    assert offset == z.size, (offset, z.size)
    # z = sol["x"].full().flatten()

    # offset = 0
    # offset += mpcc.nX
    
    # X_opt = z[0:mpcc.nX].reshape(nx, N+1)

    # U_opt = z[offset : offset + mpcc.nU].reshape(nu, N)
    # offset += mpcc.nU



    # """
    # Major bug here, nS was used here before to account for the dynamic obstacles for our receding horizon problem
    # But we also are using ns as the number of progress state values we need for the OCP
    # And since we turned off dynamic obstacles to do MPCC, we set nS to zero
    # Which means that this offset below was not actually moving forward in index, since mpcc.nS is zero
    # Which means that our s_opt was actually not getting s, but u_opt values
    # """
    # # offset += mpcc.nS   #??? Are we okay in the head

    # offset += mpcc.nProg

    # s_opt = z[offset : offset + mpcc.nProg]

    # # offset += mpcc.nProg
    # # vs_opt = z[offset : offset + mpcc.nVs]
    # # print("k", k, "mu", mu, "s_local", s_local)
    
    s0_opt = float(s_opt[0])
    sN_opt = float(s_opt[-1])

    print("Predicted s0:", s0_opt)
    print("Predicted sN:", sN_opt)
    print("Predicted progress over horizon (sN - s0):", sN_opt - s0_opt)




    k_debug = 0   # or any k you want to inspect

    # Extract predicted values
    xk = X_opt[:, k_debug]
    uk = U_opt[:, k_debug]
    sk = s_opt[k_debug]

    R_ca = ca.DM(R_horizon)
    sk_ca = ca.DM(s_opt[k_debug])

    r_k, t_k, n_k = mpcc.ref_eval_fun(R_ca, sk_ca)

    r_k = np.array(r_k.full()).flatten()
    t_k = np.array(t_k.full()).flatten()
    n_k = np.array(n_k.full()).flatten()


    p_xy = xk[0:2]
    e_vec = p_xy - r_k

    e_cont = np.dot(n_k, e_vec)
    e_lag  = np.dot(t_k, e_vec)

    v = uk[0]
    omega = uk[1]

    running_cost = (
        mpcc.q_cont * e_cont**2
        + mpcc.q_lag * e_lag**2
        + mpcc.r_v * v**2
        + mpcc.r_w * omega**2
    )

    print(f"Running cost at k={k_debug}: {running_cost:.6f}")




    # -------------------------------------------------
    # 5. Update progress
    # -------------------------------------------------



    # -------------------------------------------------
    # 6. Apply first control (receding horizon)
    # -------------------------------------------------
    v, omega = U_opt[:, 0]
    u_prev = np.array([v, omega])


    # tangent for your straight line reference (diagonal)
    t0 = np.array([1.0, 1.0], dtype=float)
    t0 /= np.linalg.norm(t0)

    heading = np.array([np.cos(x_current[2]), np.sin(x_current[2])], dtype=float)
    v_par_mps = float(v) * float(np.dot(t0, heading))

    # waypoint spacing (meters per index)
    dR = ref_traj[:, 1] - ref_traj[:, 0]
    ds_ref = float(np.linalg.norm(dR))

    # convert to index increment
    mu += dt * (v_par_mps / ds_ref)

    # keep mu from going backwards (optional but recommended)
    mu = max(mu, 0.0)
    mu = min(mu, ref_traj.shape[1] - 1)


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





    # --- Compute numerical contouring + lag error for debug ---

    # Get local reference window again
    k_ref = int(np.floor(mu))
    R_horizon = ref_traj[:, k_ref : k_ref + N + 1]

    # Pick first reference point for simplicity
    r0 = R_horizon[:, 0]

    # Compute simple geometric error (straight line case)
    error_vec = x_current[0:2] - r0

    # Tangent for straight line (since your ref is diagonal line)
    t = np.array([1.0, 1.0])
    t = t / np.linalg.norm(t)

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




        
    # -------------------------
    # Critical Stall Diagnostics
    # -------------------------

    # 1) Linear velocity
    print("v:", float(v))

    # 2) Heading projection onto tangent
    heading_vec = np.array([np.cos(x_current[2]), np.sin(x_current[2])])
    t_vec = np.array([1.0, 1.0])
    t_vec = t_vec / np.linalg.norm(t_vec)

    dot_th = float(np.dot(t_vec, heading_vec))
    print("dot(t, heading):", dot_th)

    # 3) Parallel velocity component
    v_par_debug = float(v) * dot_th
    print("v_par:", v_par_debug)

    # 4) Lag error (already computed)
    print("e_lag:", float(e_lag_val))

    print("-----")


    # dot product (cosine of angle difference)
    align_val = float(np.dot(t_k, heading_vec))

    # raw misalignment error
    misalign_error = (1.0 - align_val)

    # squared alignment cost contribution
    misalign_cost = mpcc.q_theta * (misalign_error ** 2)

    print("alignment dot(t,heading):", align_val)
    print("misalignment error (1 - dot):", misalign_error)
    print("misalignment cost contribution:", misalign_cost)

    print("e_lag:", e_lag_val)
    print("e_lag_behind:", max(0, -e_lag_val))


x_state_hist = np.array(x_state_hist)   # shape (K,3)
u_hist = np.array(u_hist)               # shape (K,2)

steps = np.arange(len(x_state_hist))

    



sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_file.close()


print("Simulation complete, moving to produce to visualization")

steps = np.arange(len(cont_err_hist))

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# -----------------------------
# Top Left — States
# -----------------------------
axs[0, 0].plot(steps, x_state_hist[:, 0], label="x")
axs[0, 0].plot(steps, x_state_hist[:, 1], label="y")
axs[0, 0].plot(steps, x_state_hist[:, 2], label="theta")
axs[0, 0].set_title("State Evolution")
axs[0, 0].set_xlabel("Step k")
axs[0, 0].set_ylabel("State Value")
axs[0, 0].legend()
axs[0, 0].grid(True)

# -----------------------------
# Top Right — Controls
# -----------------------------
axs[0, 1].plot(steps, u_hist[:, 0], label="v")
axs[0, 1].plot(steps, u_hist[:, 1], label="omega")
axs[0, 1].set_title("Control Inputs")
axs[0, 1].set_xlabel("Step k")
axs[0, 1].set_ylabel("Control Value")
axs[0, 1].legend()
axs[0, 1].grid(True)

# -----------------------------
# Bottom Left — Errors
# -----------------------------
axs[1, 0].plot(steps, cont_err_hist, label="Contouring Error")
axs[1, 0].plot(steps, lag_err_hist, label="Lag Error")
axs[1, 0].set_title("MPCC Errors")
axs[1, 0].set_xlabel("Step k")
axs[1, 0].set_ylabel("Error")
axs[1, 0].legend()
axs[1, 0].grid(True)

# -----------------------------
# Bottom Right — (Optional empty or extra metric)
# -----------------------------
axs[1, 1].plot(steps, mu_hist, label="s progress")
axs[1, 1].set_title("Progress Variable (s)")
axs[1, 1].set_xlabel("Step k")
axs[1, 1].set_ylabel("Progress")
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

visualize(ref_traj, x_history, y_history, theta_history)



