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


USE_S_CURVE = True

    
if USE_S_CURVE:
    ref_traj = generate_s_curve_ref(
        x_start=0.0,
        x_end=10.0,
        num_points=401,   # more points = smoother
        amplitude=2.0,
        frequency=0.6
    )
else:
    # keep your old straight path definition here
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

#23123123
#31231230
#2312312
#3012831
#45165165
#389999 here the robot tries to speed up to catch up to the path, by accelerating aggresively
#3123125 cool circle behavior, but again speeds up to get back on the path
#821113 weird and unnecsarry circle?
#11101012 actual obs collision

rng = np.random.default_rng(389999)




# Obstacle motion limits
v_obs_max = 0.8          # 
a_obs_max = 0.5          # m/s^2  (MATCH robot order)
omega_obs_max = 0.6      # rad/s  

dtheta_max = omega_obs_max * dt * change_horizon


# Continuous obstacle state
obs_speed = np.zeros(num_obs)
obs_theta = rng.uniform(0, 2*np.pi, size=num_obs)



static_obs_num=2

# for i in range(num_obs-static_obs_num):
#     obs_pos[i] = [2.0 + i, 0.5*i]
#     speed = rng.uniform(0.2, obs_v_max)
#     obs_vel[i] = speed * np.array([
#         np.cos(obs_theta[i]),
#         np.sin(obs_theta[i])
#     ])
    
for i in range(num_obs - static_obs_num):
    obs_pos[i] = [2.0 + i, 0.5*i]
    speed = rng.uniform(0.2, v_obs_max)
    obs_speed[i] = speed
    obs_vel[i] = speed * np.array([np.cos(obs_theta[i]), np.sin(obs_theta[i])])
    



# for i in range(static_obs_num):
#     obs_pos[i+(num_obs-static_obs_num)] = [1.0, 1.0]
#     speed = 1
#     obs_vel[i+(num_obs-static_obs_num)] = speed * np.array([
#         np.cos(obs_theta[i+(num_obs-static_obs_num)]),
#         np.sin(obs_theta[i+(num_obs-static_obs_num)])
#     ])



start = num_obs - static_obs_num
for j in range(static_obs_num):
    idx = start + j
    obs_pos[idx] = [1.0, 1.0]   # or unique positions per static obstacle
    obs_speed[idx] = 0.0
    obs_vel[idx] = np.array([0.0, 0.0])


obs_pos_hist = [obs_pos.copy()]


collision_count = 0
collision_log = []

slack_activation_count = 0
slack_sum = 0.0
slack_max = 0.0



u_prev = np.zeros((nu,))   # [v_prev, omega_prev]

prev_z = None

max_steps = min(5000, ref_traj.shape[1] - 1)

for k in range(max_steps):

    # Path-tracking reference window (2 x (N+1))
    R_horizon = ref_traj[:, k : k + N + 1]

    # If we're near the end of ref_traj, pad with the last point
    if R_horizon.shape[1] < N + 1:
        last = ref_traj[:, -1].reshape(2, 1)
        pad = np.repeat(last, N + 1 - R_horizon.shape[1], axis=1)
        R_horizon = np.hstack([R_horizon, pad])
        
    # Optional: make terminal goal consistent with the horizon end
    X_goal_val = np.array([R_horizon[0, -1], R_horizon[1, -1], 0.0])

    # propagate true obstacle state
    # obs_x_current += obs_vx * dt
    # obs_y_current += obs_vy * dt
    
    # obsx_h = np.zeros((num_obs, N))
    # obsy_h = np.zeros((num_obs, N))
    
    # #the first x obs of the num_obs should be static obs. Their obs_pos shouldnt move and their obs history x and y and theta shouldnt update
    # static_obs_num=static_obs_num

    # for i in range(num_obs-static_obs_num):

    #     # if change_timer[i] <= 0:
    #     #     obs_theta[i] += rng.uniform(-dtheta_max, dtheta_max)

    #     #     speed = rng.uniform(0.2, obs_v_max)
    #     #     # obs_vel[i] = speed * np.array([np.cos(obs_theta[i]), np.sin(obs_theta[i])])               #instant velocity change

    #     #     change_timer[i] = change_horizon
    #     # else:
    #     #     change_timer[i] -= 1
            
    #     # Desired targets (slowly varying intent)
    #     v_des = rng.uniform(0.2, v_obs_max)
    #     theta_des = obs_theta[i] + rng.uniform(-dtheta_max, dtheta_max)

    #     # ----- speed slew (bounded acceleration) -----
    #     dv = np.clip(
    #         v_des - obs_speed[i],
    #         -a_obs_max * dt,
    #         +a_obs_max * dt
    #     )
    #     obs_speed[i] += dv

    #     # ----- heading slew (bounded turn rate) -----
    #     dtheta = np.clip(
    #         theta_des - obs_theta[i],
    #         -omega_obs_max * dt,
    #         +omega_obs_max * dt
    #     )
    #     obs_theta[i] += dtheta

    #     # ----- velocity vector -----
    #     obs_vel[i] = obs_speed[i] * np.array([
    #         np.cos(obs_theta[i]),
    #         np.sin(obs_theta[i])
    #     ])



    #     # advance true obstacle one step
    #     obs_pos[i] += obs_vel[i] * dt

    #     # predict obstacle over horizon
    #     x_tmp, y_tmp = obs_pos[i]
    #     for j in range(N):
    #         x_tmp += obs_vel[i, 0] * dt
    #         y_tmp += obs_vel[i, 1] * dt
    #         obsx_h[i, j] = x_tmp
    #         obsy_h[i, j] = y_tmp
            
    #     start = num_obs - static_obs_num
    #     for i in range(static_obs_num):
    #         idx = start + i
    #         x_tmp, y_tmp = obs_pos[idx]
    #         for j in range(N):
    #             obsx_h[idx, j] = x_tmp
    #             obsy_h[idx, j] = y_tmp
    
    obsx_h = np.zeros((num_obs, N))
    obsy_h = np.zeros((num_obs, N))

    # ---- dynamic obstacles (0 ... num_obs-static_obs_num-1)
    for i in range(num_obs - static_obs_num):

        v_des = rng.uniform(0.2, v_obs_max)
        theta_des = obs_theta[i] + rng.uniform(-dtheta_max, dtheta_max)

        dv = np.clip(v_des - obs_speed[i], -a_obs_max * dt, +a_obs_max * dt)
        obs_speed[i] += dv

        dtheta = np.clip(theta_des - obs_theta[i], -omega_obs_max * dt, +omega_obs_max * dt)
        obs_theta[i] += dtheta

        obs_vel[i] = obs_speed[i] * np.array([np.cos(obs_theta[i]), np.sin(obs_theta[i])])

        # advance true obstacle one step
        obs_pos[i] += obs_vel[i] * dt

        # predict obstacle over horizon (constant velocity)
        x_tmp, y_tmp = obs_pos[i]
        for j in range(N):
            x_tmp += obs_vel[i, 0] * dt
            y_tmp += obs_vel[i, 1] * dt
            obsx_h[i, j] = x_tmp
            obsy_h[i, j] = y_tmp

    # ---- static obstacles (num_obs-static_obs_num ... num_obs-1)
    start = num_obs - static_obs_num
    for s in range(static_obs_num):
        idx = start + s
        x_tmp, y_tmp = obs_pos[idx]
        for j in range(N):
            obsx_h[idx, j] = x_tmp
            obsy_h[idx, j] = y_tmp


    # The solver only sees obs that are inside this
    # obsx_h
    # obsy_h


    # store for plotting
    obs_pos_hist.append(obs_pos.copy())


    print("x_current:", x_current.shape)
    print("R_horizon:", R_horizon.shape)
    print("obsx_h:", obsx_h.shape)
    print("obsy_h:", obsy_h.shape)
    print("X_goal_val:", X_goal_val.shape)

    p = np.concatenate([
        x_current,                       # X0 (nx,)
        u_prev,                          # u_prev (nu,)  <-- REQUIRED
        R_horizon.flatten(order="F"),    # (2*(N+1),) so this is what causes the time and geometry coupling
        obsx_h.flatten(order="F"),       # (num_obs*N,)
        obsy_h.flatten(order="F"),       # (num_obs*N,)
        X_goal_val                       # (nx,)
    ])



    print("p length:", p.shape[0])


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



    z_opt = sol["x"].full().flatten()
    offset = nx * (N + 1)
    v, omega = z_opt[offset: offset + nu]
    
    
    # decision vector layout
    x_block = nx * (N + 1)
    u_block = nu * N
    s_start = x_block + u_block

    S_opt = z_opt[s_start:].reshape(num_obs, N)
    
    
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
    
    
    for i in range(num_obs):
        dx = x_current[0] - obs_pos[i, 0]
        dy = x_current[1] - obs_pos[i, 1]

        a_eff = a_obs + r_robot
        b_eff = b_obs + r_robot

        if (dx**2 / a_eff**2 + dy**2 / b_eff**2) <= 1.0:
            collision_count += 1
            collision_log.append((k, i))
            print(f"⚠️ PHYSICAL COLLISION with obstacle {i} at step {k}")





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


visualize(ref_traj, x_history, y_history, theta_history, obs_pos_hist)