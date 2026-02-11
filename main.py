import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

from params import *
from dynamics import *
from mpc import *
from obstacles import *

from matplotlib.patches import Ellipse
from s_curve_env import generate_s_curve_ref
from visualization import visualize




# To keep our simulation environment variable 
# We have a curved S shaped path and also a straight variable
USE_S_CURVE = True


if USE_S_CURVE:
    ref_traj = generate_s_curve_ref(    #Using this function we are generating a sine wave
        x_start=0.0,
        x_end=20.0,
        num_points=401,   # more points = smoother
        amplitude=2.0,
        frequency=0.6
    )
else:
    # This is where the old ref traj assignment is supposed to take place
    # The old ref traj being a straight line from the origin to some point
    # ref_traj = ...
    pass






X_goal_global = np.array(
    [ref_traj[0, -1], ref_traj[1, -1], 0.0],
    dtype=float
)

# # --- Reset histories AFTER overriding x_current ---
# x_history = [x_current[0]]
# y_history = [x_current[1]]
# theta_history = [x_current[2]]

# -----------------------------
# Dynamic obstacle parameters
# -----------------------------


# goal_xy = np.array([10.0, 10.0])
# goal_theta = 0.0
# # X_goal_val = np.array([goal_xy[0], goal_xy[1], goal_theta])
# # # Default goal (used only for termination + terminal cost)
# # Start with the last point of the whole reference path
# X_goal_val = np.array([ref_traj[0, -1], ref_traj[1, -1], 0.0])

X_goal_val = np.array([ref_traj[0, -1], ref_traj[1, -1], 0.0], dtype=float)

# X_goal_xy = goal_xy

solve_times = []

# BEFORE loop (once)
vx_obs, vy_obs = 0.5, 0.0



# Obstacle motion limits
v_obs_max = 0.8          # 
a_obs_max = 0.5          # m/s^2  (MATCH robot order)
omega_obs_max = 0.6      # rad/s  

dtheta_max = omega_obs_max * dt * change_horizon


obs_pos = np.zeros((num_dyn_obs, 2))
for i in range(num_dyn_obs):
    obs_pos[i] = [2.0 + i, 0.5 * i]



collision_count = 0
collision_log = []

slack_activation_count = 0
slack_sum = 0.0
slack_max = 0.0


# setup once
obstacles = ObstacleManager(
    num_dyn_obs=num_dyn_obs,
    dt=dt,
    N=N,
    rng=rng,
    v_max=v_obs_max,
    a_max=a_obs_max,
    omega_max=omega_obs_max,
    init_positions=obs_pos,
    wall_y_min=WALL_Y_MIN,
    wall_y_max=WALL_Y_MAX,
    y_margin=b_obs,   # important: keep ellipse fully inside
)



obs_pos_hist = [obstacles.pos.copy()]

u_prev = np.zeros((nu,))   # [v_prev, omega_prev]

prev_z = None

max_steps = min(5000, ref_traj.shape[1] - 1)

# print(ref_traj)
# time.sleep(312312312)

# This function outputs the receding horizon
def receding_horizon(k, ref_traj):
    # Path-tracking reference window, (2 x (N+1)) matrix
    # Remember that I am defining my ref traj as just a series of x and y points 
    # That is why it is a 2 row and N column matrix. If we had x and y and heading values to track, then that would mean our row would now have to be 3
    # This window moves along the ref path as the simulation steps forward, as k goes towards the max_steps
    R_horizon = ref_traj[:, k : k + N + 1]      #we are doing some slicing here
    # so consider all rows, but for the columns we are going to only take the columns starting from k all the way to k+N+1, so basically if k is at 100 right now and we set N(our horizon length) to 20. Then we consider the x/y ref points starting at 100 all the way to 111.

    # If we're near the end of ref_traj, pad with the last point
    if R_horizon.shape[1] < N + 1:
        last = ref_traj[:, -1].reshape(2, 1)
        pad = np.repeat(last, N + 1 - R_horizon.shape[1], axis=1)
        R_horizon = np.hstack([R_horizon, pad])

    return R_horizon

# This function constructs the NLP and actually solves it
# Then it outputs the solution and updates the prev_z for warmstart
def nlp_solve(lbx, ubx, lbg, ubg, p):
    global prev_z

    #what is being done here?
    kwargs = dict(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p) #Defining a dictionary of arguments that will be passed to the NLP

    if prev_z is not None:
        kwargs["x0"] = prev_z   # warm start

    # Performance metrics, we are timing how long it takes to solve this nlp once
    t_solve_start = time.perf_counter()
    
    sol = solver(**kwargs)
    
    t_solve_end = time.perf_counter()

    solve_time = t_solve_end - t_solve_start
    print(f"k={k}, solve_time = {solve_time*1000:.2f} ms")

    st = solver.stats()

    if sol is None:
        print("Solver returned None:", st["return_status"])


    if st["return_status"] not in {
        "Solve_Succeeded",
        "Solved_To_Acceptable_Level",
        "Maximum_Iterations_Exceeded"
    }:
        print("Bad solve:", st["return_status"])


    # Save solution for warm start next iteration
    prev_z = sol["x"]
    print(prev_z)


    return sol, solve_time



#ACTUAL SIMULATION STEPPING
for k in range(max_steps):

    R_horizon = receding_horizon(k, ref_traj)
    
    # Optional: make terminal goal consistent with the horizon end
    X_goal_val = np.array([R_horizon[0, -1], R_horizon[1, -1], 0.0])

    obstacles.step()

    obsx_h, obsy_h = obstacles.predict_horizon()
    
    obs_pos_hist.append(obstacles.pos.copy())

    p = np.concatenate([
        x_current,                       # X0 (nx,)
        u_prev,                          # u_prev (nu,)  <-- REQUIRED
        R_horizon.flatten(order="F"),    # (2*(N+1),) so this is what causes the time and geometry coupling
        obsx_h.flatten(order="F"),       # (num_dyn_obs*N,)
        obsy_h.flatten(order="F"),       # (num_dyn_obs*N,)
        X_goal_val                       # (nx,)
    ])

    # print("x_current:", x_current.shape)
    # print("R_horizon:", R_horizon.shape)
    # print("obsx_h:", obsx_h.shape)
    # print("obsy_h:", obsy_h.shape)
    # print("X_goal_val:", X_goal_val.shape)
    # print("p length:", p.shape[0])


    # #what is being done here?
    # kwargs = dict(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p) #Defining a dictionary of arguments that will be passed to the NLP

    # if prev_z is not None:
    #     kwargs["x0"] = prev_z   # warm start

    # # sol = solver(**kwargs)
    # t_solve_start = time.perf_counter()
    # sol = solver(**kwargs)
    # t_solve_end = time.perf_counter()

    # solve_time = t_solve_end - t_solve_start
    # print(f"k={k}, solve_time = {solve_time*1000:.2f} ms")

    # st = solver.stats()

    # if sol is None:
    #     print("Solver returned None:", st["return_status"])
    #     break

    # if st["return_status"] not in {
    #     "Solve_Succeeded",
    #     "Solved_To_Acceptable_Level",
    #     "Maximum_Iterations_Exceeded"
    # }:
    #     print("Bad solve:", st["return_status"])
    #     break

    # # Save solution for warm start next iteration
    # prev_z = sol["x"]
    
    sol, solve_time = nlp_solve(lbx, ubx, lbg, ubg, p)


    z_opt = sol["x"].full().flatten()
    offset = nx * (N + 1)
    v, omega = z_opt[offset: offset + nu]
    
    
    # decision vector layout
    x_block = nx * (N + 1)
    u_block = nu * N
    s_start = x_block + u_block

    S_opt = z_opt[s_start:].reshape(num_dyn_obs, N)
    
    
    eps = 1e-6  # numerical threshold

    active_slack = S_opt > eps

    if np.any(active_slack):
        slack_activation_count += 1

    slack_sum += np.sum(S_opt)
    slack_max = max(slack_max, np.max(S_opt))


    slack_per_obstacle = np.sum(S_opt, axis=1)


    
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
    
    
    for i in range(num_dyn_obs):
        dx = x_current[0] - obs_pos[i, 0]
        dy = x_current[1] - obs_pos[i, 1]

        a_eff = a_obs + r_robot
        b_eff = b_obs + r_robot

        if (dx**2 / a_eff**2 + dy**2 / b_eff**2) <= 1.0:
            collision_count += 1
            collision_log.append((k, i))
            print(f" PHYSICAL COLLISION with obstacle {i} at step {k}")





    if np.linalg.norm(x_current[:2] - X_goal_global[:2]) < 0.2:
        print("Global goal reached.")
        break
    
    
    # progress index = closest reference point
    dist_to_path = np.linalg.norm(
        ref_traj.T - x_current[:2], axis=1
    )
    closest_idx = np.argmin(dist_to_path)

    print(f"k={k}, progress index={closest_idx}")
    
    solve_times.append(solve_time)




solve_times = np.array(solve_times)


#DEBUG AND PERFORMANCE PRINTS
print("\n=== MPC TIMING SUMMARY ===")
print(f"Mean solve time: {solve_times.mean()*1000:.2f} ms")
print(f"Median solve time: {np.median(solve_times)*1000:.2f} ms")
print(f"Max solve time: {solve_times.max()*1000:.2f} ms")
print(f"Min solve time: {solve_times.min()*1000:.2f} ms")

if solve_time > dt:
    print("⚠️ solve_time exceeds dt")



print("\n=== COLLISION SUMMARY ===")
print(f"Total physical collisions: {collision_count}")

print("\n=== SLACK USAGE SUMMARY ===")
print(f"Steps with slack active: {slack_activation_count}")
print(f"Total slack used: {slack_sum:.4f}")
print(f"Max slack value: {slack_max:.4f}")



#MATPLOTLIB OF THE SIMULATION
visualize(ref_traj, x_history, y_history, theta_history, obs_pos_hist)
