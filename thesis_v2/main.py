# A simple MPCC simulation of a unicycle path tracking a reference path and reaching the goal pose

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

from params import *
from dynamics import *
import mpcc


from matplotlib.patches import Ellipse
from visualization import visualize

import sys

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

ref_x = np.linspace(0, 10, 100)
ref_y = np.linspace(0, 10, 100)
ref_traj=[]

# ref_traj = np.column_stack((ref_x, ref_y))

ref_traj = np.vstack((ref_x, ref_y))


mu = 0.0      # GLOBAL progress along full reference





X_goal_global = np.array(
    [ref_traj[0, -1], ref_traj[1, -1], 0.0],
    dtype=float
)


u_prev = np.zeros((nu,))   # [v_prev, omega_prev]

prev_z = None

# max_steps = min(5000, ref_traj.shape[1] - 1)

for k in range(1000):

    # Path-tracking reference window (2 x (N+1))
    # R_horizon = ref_traj[:, k : k + N + 1]

    k_ref = int(np.floor(mu))
    s_local = mu - k_ref   # local progress inside horizon

    R_horizon = ref_traj[:, k_ref : k_ref + N + 1]

    if R_horizon.shape[1] < N + 1:
        last = ref_traj[:, -1].reshape(2, 1)
        pad = np.repeat(last, N + 1 - R_horizon.shape[1], axis=1)
        R_horizon = np.hstack([R_horizon, pad])

    X_goal_val = np.array([R_horizon[0, -1], R_horizon[1, -1], 0.0])

    p = np.concatenate([
        x_current,
        u_prev,
        np.array([s_local]),     # <-- PASS LOCAL s
        R_horizon.flatten(),
        X_goal_val
    ])





    #what is being done here?
    kwargs = dict(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p) 

    if prev_z is not None:
        kwargs["x0"] = prev_z   # warm start

    # sol = solver(**kwargs)
    t_solve_start = time.perf_counter()
    sol = solver(**kwargs)
    t_solve_end = time.perf_counter()

    solve_time = t_solve_end - t_solve_start
    print(f"k={k}, solve_time = {solve_time*1000:.2f} ms")

    st = solver.stats()

    if sol is None:
        print("Solver returned None:", st["return_status"])
        break

    if st["return_status"] not in {
        "Solve_Succeeded",
        "Solved_To_Acceptable_Level",
        "Maximum_Iterations_Exceeded"
    }:
        print("Bad solve:", st["return_status"])
        break

    # Save solution for warm start next iteration
    prev_z = sol["x"]



    # S_opt = z_opt[s_start:].reshape(num_dyn_obs, N)
    z_opt = sol["x"].full().flatten()
    nx = 3          # state dimension
    nu = 2          # control dimension

    nX = nx * (N + 1)
    nU = nu * N
    nS = num_dyn_obs * N
    z_opt = sol["x"].full().flatten()

    nX = mpcc.nX
    nU = mpcc.nU
    nS = mpcc.nS
    nProg = mpcc.nProg
    nVs = mpcc.nVs

    offset = 0
    X_opt = z_opt[offset:offset+nX].reshape(nx, N+1); offset += nX
    U_opt = z_opt[offset:offset+nU].reshape(nu, N);   offset += nU
    S_opt = z_opt[offset:offset+nS].reshape(num_dyn_obs, N); offset += nS

    s_opt  = z_opt[offset:offset+nProg]               ; offset += nProg
    vs_opt = z_opt[offset:offset+nVs]                 ; offset += nVs

    # Update progress for next iteration (use the next step)
    mu = k_ref + float(s_opt[1])




    # First control to apply
    v, omega = U_opt[:, 0]

    
    eps = 1e-6  # numerical threshold


    
    u_prev = np.array([v, omega], dtype=float)
    
    if k > 0:
        du = u_prev - u_prev_last
        print("du =", du, " | limits =", [dv_max, dw_max])
    u_prev_last = u_prev.copy()



    x_current = np.array([
        x_current[0] + dt * v * np.cos(x_current[2]),
        x_current[1] + dt * v * np.sin(x_current[2]),
        x_current[2] + dt * omega
    ])

    x_history.append(x_current[0])
    y_history.append(x_current[1])
    theta_history.append(x_current[2])
    



sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_file.close()


print("DASODAIJSOIDAJSODIASJODIAJSDOISJDOIASJDOAIJDOAISJDOASIDJAIOS")
visualize(ref_traj, x_history, y_history, theta_history)
