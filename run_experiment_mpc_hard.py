"""
run_experiment_mpc.py
Self-contained MPC experiment runner for the MPC project folder.
Drop this file into the MPC project directory and run batch_runner_mpc.py.

Uses the same paths, obstacle sizes, safety params, and seeds as the MPCC
experiment so results are directly comparable.
"""

import time
import numpy as np
import casadi as ca

from dynamics import unicycle_dynamics
from obstacles import ObstacleManager

# ----------------------------------------------------------------
# Shared params — must match MPCC project values for fair comparison
# ----------------------------------------------------------------
dt           = 0.1
N            = 30       # same horizon as MPCC
nx, nu       = 3, 2
r_robot      = 0.25
safety_buffer = 0.2
v_max        = 1.0
omega_max    = 2.0
dv_max       = 0.25     # same slew limits as MPCC
dw_max       = 0.6

# Obstacle sizes — same as MPCC obs.py
A_OBS = 0.6
B_OBS = 0.4

# Near-miss threshold — same as MPCC
NEAR_MISS_MARGIN = 0.3

# MPC cost weights
Q_xy   = np.diag([10.0, 10.0])
R_u    = np.diag([0.1,  0.5])
Q_term = 50.0 * np.eye(2)
rho_slack = 1e5


# ----------------------------------------------------------------
# Path generators  (mirror of MPCC obs.py path_selector)
# ----------------------------------------------------------------
def make_zigzag():
    ref_x = np.array([0, 5, 10, 15, 20], dtype=float)
    ref_y = np.array([0, 4,  0,  4,  0], dtype=float)
    ref_x = np.interp(np.linspace(0, 4, 300), np.arange(5), ref_x)
    ref_y = np.interp(np.linspace(0, 4, 300), np.arange(5), ref_y)
    return np.vstack((ref_x, ref_y))

def make_sine():
    t     = np.linspace(0, 20, 300)
    ref_x = t
    ref_y = 2.0 * np.sin(0.5 * t)
    return np.vstack((ref_x, ref_y))


# ----------------------------------------------------------------
# Static obstacles  (mirror of MPCC obs.py static_obs)
# ----------------------------------------------------------------
def make_static_obs(path_id):
    if path_id == 3:   # zigzag
        return [
            (7.0,  1.0, 0.6, 1.2),
            (10.0, 0.0, 0.5, 0.5),
            (10.0,  5.0, 20.0, 0.2),   # top wall
            (10.0, -2.5, 20.0, 0.2),   # bottom wall
        ]
    elif path_id == 1:  # sine
        return [
            (7.0,  0.0, 0.5, 0.5),
            (13.0, 0.0, 0.5, 0.5),
            (10.0,  3.5, 20.0, 0.2),   # top wall
            (10.0, -3.5, 20.0, 0.2),   # bottom wall
        ]
    return []


# ----------------------------------------------------------------
# Dynamic obstacles  (mirror of MPCC obs.py dynamic_obs_seeded)
# Uses ObstacleManager from the MPC project with matched seeds/speeds
# ----------------------------------------------------------------
def make_dynamic_obs(path_id, seed_offset, num_obs=5):
    if path_id == 3:
        init_pos = np.array([
            [2.0,  3.0],
            [18.0, 2.0],
            [14.0, 2.0],
            [5.5,  3.5],
            [10.0,-1.0],
        ], dtype=float)
    elif path_id == 1:
        init_pos = np.array([
            [5.0,  1.0],
            [10.0,-1.0],
            [15.0, 1.5],
        ], dtype=float)
        num_obs = 3
    else:
        init_pos = np.zeros((0, 2))
        num_obs = 0

    rng = np.random.default_rng(seed_offset * 100 + 42)

    obs = ObstacleManager(
        num_dyn_obs   = num_obs,
        dt            = dt,
        N             = N,
        rng           = rng,
        v_max         = 0.5,
        a_max         = 0.3,
        omega_max     = 0.6,
        init_positions= init_pos[:num_obs],
        wall_y_min    = -2.5 if path_id == 3 else -3.5,
        wall_y_max    =  5.0 if path_id == 3 else  3.5,
        y_margin      = B_OBS,
    )
    return obs, num_obs


# ----------------------------------------------------------------
# Build MPC solver (called once per run)
# ----------------------------------------------------------------
def build_mpc_solver(static_rects, num_dyn_obs):

    X      = ca.SX.sym("X",     nx, N + 1)
    U      = ca.SX.sym("U",     nu, N)
    # No S — hard constraints, no slack
    X0     = ca.SX.sym("X0",    nx)
    u_prev = ca.SX.sym("u_prev", nu)
    R      = ca.SX.sym("R",     2,  N + 1)
    obs_x  = ca.SX.sym("obs_x", num_dyn_obs, N)
    obs_y  = ca.SX.sym("obs_y", num_dyn_obs, N)
    X_goal = ca.SX.sym("X_goal", nx)

    cost = 0
    for k in range(N):
        x_err = X[0:2, k] - R[0:2, k]
        cost += ca.mtimes([x_err.T, Q_xy, x_err])
        cost += ca.mtimes([U[:, k].T, R_u, U[:, k]])
    e_term = X[0:2, N] - X_goal[0:2]
    cost  += ca.mtimes([e_term.T, Q_term, e_term])
    # No slack cost

    g_list, lbg, ubg = [], [], []

    # Initial condition
    g_list.append(X[:, 0] - X0)
    lbg += [0.0] * nx;  ubg += [0.0] * nx

    a_e = A_OBS + r_robot + safety_buffer
    b_e = B_OBS + r_robot + safety_buffer

    for k in range(N):
        g_list.append(X[:, k+1] - unicycle_dynamics(X[:, k], U[:, k]))
        lbg += [0.0]*nx;  ubg += [0.0]*nx

        p_next = X[0:2, k+1]

        # Wall constraints
        g_list.append(X[1, k+1] - (-2.3 + r_robot + safety_buffer))
        lbg.append(0.0); ubg.append(ca.inf)
        g_list.append((4.8 - r_robot - safety_buffer) - X[1, k+1])
        lbg.append(0.0); ubg.append(ca.inf)

        # Dynamic obstacles — HARD constraint, no slack
        for i in range(num_dyn_obs):
            dx   = p_next[0] - obs_x[i, k]
            dy   = p_next[1] - obs_y[i, k]
            dist = dx**2 / a_e**2 + dy**2 / b_e**2
            g_list.append(dist - 1.0)   # dist >= 1, no slack term
            lbg.append(0.0); ubg.append(ca.inf)

        # Static obstacles — hard ellipse approx
        for (cx, cy, hw, hh) in static_rects:
            hx_e = hw + r_robot + safety_buffer
            hy_e = hh + r_robot + safety_buffer
            dx   = p_next[0] - cx
            dy   = p_next[1] - cy
            g_list.append(dx**2/hx_e**2 + dy**2/hy_e**2 - 1.0)
            lbg.append(0.0); ubg.append(ca.inf)

    # Slew between horizon steps
    for k in range(N - 1):
        g_list.append(U[0, k+1] - U[0, k])
        lbg.append(-dv_max); ubg.append(dv_max)
        g_list.append(U[1, k+1] - U[1, k])
        lbg.append(-dw_max); ubg.append(dw_max)

    # Slew from previous control
    g_list.append(U[:, 0] - ca.reshape(u_prev, 2, 1))
    lbg += [-dv_max, -dw_max];  ubg += [dv_max, dw_max]

    g = ca.vertcat(*g_list)

    lbx, ubx = [], []
    for _ in range(N+1):
        lbx += [-ca.inf]*nx;  ubx += [ca.inf]*nx
    for _ in range(N):
        lbx += [0.0, -omega_max];  ubx += [v_max, omega_max]
    # No slack bounds

    OPT_vars = ca.vertcat(
        ca.reshape(X, -1, 1),
        ca.reshape(U, -1, 1),
        # No S in decision variables
    )
    p_vec = ca.vertcat(
        X0, u_prev,
        ca.reshape(R, -1, 1),
        ca.reshape(obs_x, -1, 1),
        ca.reshape(obs_y, -1, 1),
        X_goal,
    )

    solver = ca.nlpsol("solver_mpc_hard", "ipopt",
        {"x": OPT_vars, "f": cost, "g": g, "p": p_vec},
        {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 120,
            "ipopt.tol": 1e-3,
            "ipopt.acceptable_tol": 1e-2,
            "ipopt.acceptable_iter": 5,
            "ipopt.warm_start_init_point": "yes",
            "print_time": 0,
        }
    )

    nX = nx * (N+1)
    nU = nu * N
    nS = 0   # no slack
    return solver, lbx, ubx, lbg, ubg, nX, nU, nS


# ----------------------------------------------------------------
# Helper
# ----------------------------------------------------------------
def _ellipse_surface_dist(rx, ry, ox, oy, a, b):
    dx = rx - ox;  dy = ry - oy
    d  = np.sqrt((dx/a)**2 + (dy/b)**2)
    return (d - 1.0) * np.sqrt(a*b)


# ----------------------------------------------------------------
# Main run function
# ----------------------------------------------------------------
def run_mpc_hard(path_id: int, seed_offset: int, max_steps: int = 800) -> dict:

    # Environment
    if path_id == 3:
        ref_traj = make_zigzag()
    elif path_id == 1:
        ref_traj = make_sine()
    else:
        raise ValueError(f"Unsupported path_id {path_id}")

    STATIC_RECTS = make_static_obs(path_id)
    obstacles, num_dyn_obs = make_dynamic_obs(path_id, seed_offset)

    straight_line = float(np.linalg.norm(ref_traj[:, -1] - ref_traj[:, 0]))

    solver, lbx, ubx, lbg, ubg, nX, nU, nS = build_mpc_solver(
        STATIC_RECTS, num_dyn_obs
    )

    # Initial state
    p0  = ref_traj[:, 0];  p1 = ref_traj[:, 1]
    th0 = float(np.arctan2(p1[1]-p0[1], p1[0]-p0[0]))
    x_current     = np.array([p0[0], p0[1], th0])
    X_goal_global = np.array([ref_traj[0,-1], ref_traj[1,-1], 0.0])

    u_prev        = np.zeros(nu)
    prev_z        = None
    ref_k         = 0
    _at_end_count = 0

    cont_errors = [];  lag_errors  = []
    solve_times = [];  v_hist      = [];  omega_hist = []
    path_length = 0.0; prev_pos    = x_current[:2].copy()

    dyn_body_count = 0;  dyn_excl_count  = 0
    static_body    = 0;  near_miss_count = 0
    min_clearance  = np.inf;  danger_zone = 0
    solver_fails   = 0
    goal_reached   = False;  steps_taken = 0

    for k in range(max_steps):

        # Reference window
        R_horizon = ref_traj[:, ref_k : ref_k + N + 1]
        if R_horizon.shape[1] < N + 1:
            last = R_horizon[:, -1].reshape(2, 1)
            pad  = np.repeat(last, N+1 - R_horizon.shape[1], axis=1)
            R_horizon = np.hstack([R_horizon, pad])

        X_goal_val = np.array([R_horizon[0,-1], R_horizon[1,-1], 0.0])

        # Obstacle predictions
        obsx_h, obsy_h = obstacles.predict_horizon()

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

        t0  = time.perf_counter()
        sol = solver(**kwargs)
        solve_times.append(time.perf_counter() - t0)

        st = solver.stats()
        if st["return_status"] not in {
            "Solve_Succeeded", "Solved_To_Acceptable_Level",
            "Maximum_Iterations_Exceeded"
        }:
            solver_fails += 1

        prev_z = sol["x"]

        z     = sol["x"].full().flatten()
        U_opt = z[nX : nX+nU].reshape((nu, N), order="F")
        v, omega = U_opt[:, 0]
        u_prev   = np.array([v, omega])
        v_hist.append(float(v))
        omega_hist.append(float(omega))

        ref_k = min(ref_k + 1, ref_traj.shape[1] - 1)

        x_current = np.array([
            x_current[0] + dt * v * np.cos(x_current[2]),
            x_current[1] + dt * v * np.sin(x_current[2]),
            x_current[2] + dt * omega,
        ])
        steps_taken += 1
        path_length += float(np.linalg.norm(x_current[:2] - prev_pos))
        prev_pos = x_current[:2].copy()

        # Step obstacles to k+1 first, then check collision
        # This matches MPCC which checks robot at k+1 vs obstacle at k+1
        obstacles.step()
        obs_pos_now = obstacles.pos

        # Safety metrics
        step_near = False;  step_danger = False

        for i in range(num_dyn_obs):
            ox, oy = obs_pos_now[i]
            dx = x_current[0] - ox;  dy = x_current[1] - oy

            if (dx/A_OBS)**2 + (dy/B_OBS)**2 <= 1.0:
                dyn_body_count += 1

            a_e = A_OBS + r_robot + safety_buffer
            b_e = B_OBS + r_robot + safety_buffer
            if (dx/a_e)**2 + (dy/b_e)**2 <= 1.0:
                dyn_excl_count += 1

            cl = _ellipse_surface_dist(x_current[0], x_current[1], ox, oy, A_OBS, B_OBS)
            min_clearance = min(min_clearance, cl)
            if cl < NEAR_MISS_MARGIN:
                step_near = True
            if cl < NEAR_MISS_MARGIN + 2*safety_buffer:
                step_danger = True

        for (cx, cy, hw, hh) in STATIC_RECTS:
            if abs(x_current[0]-cx) < hw+r_robot and abs(x_current[1]-cy) < hh+r_robot:
                static_body += 1
            dx_s = max(abs(x_current[0]-cx)-hw, 0.0)
            dy_s = max(abs(x_current[1]-cy)-hh, 0.0)
            cl_s = np.sqrt(dx_s**2+dy_s**2) - r_robot
            min_clearance = min(min_clearance, cl_s)
            if cl_s < NEAR_MISS_MARGIN:
                step_near = True
            if cl_s < NEAR_MISS_MARGIN + 2*safety_buffer:
                step_danger = True

        if step_near:   near_miss_count += 1
        if step_danger: danger_zone     += 1

        # Geometric errors
        dists   = np.linalg.norm(ref_traj.T - x_current[:2], axis=1)
        k2      = min(int(np.argmin(dists)), ref_traj.shape[1]-2)
        R_now   = ref_traj[:, k2:k2+2]
        dR      = R_now[:, 1] - R_now[:, 0]
        t_hat   = dR / (np.linalg.norm(dR) + 1e-12)
        n_hat   = np.array([-t_hat[1], t_hat[0]])
        ev      = x_current[:2] - R_now[:, 0]
        cont_errors.append(float(np.dot(n_hat, ev)))
        lag_errors.append(float(np.dot(t_hat, ev)))

        # Termination
        dist_to_goal  = np.linalg.norm(x_current[:2] - X_goal_global[:2])
        path_complete = ref_k >= ref_traj.shape[1] - 2

        if dist_to_goal < 0.5 and path_complete:
            goal_reached = True;  break

        if path_complete:
            _at_end_count += 1
            if _at_end_count >= 30:
                goal_reached = True;  break
        else:
            _at_end_count = 0

    # Pack metrics
    cont_arr = np.abs(cont_errors)
    lag_arr  = np.array(lag_errors)
    st_arr   = np.array(solve_times) * 1000
    v_arr    = np.array(v_hist)
    om_arr   = np.abs(omega_hist)
    eff      = round(straight_line / path_length, 4) if path_length > 0 else 0.0

    return {
        "path_id":             path_id,
        "seed_offset":         seed_offset,
        "goal_reached":        int(goal_reached),
        "steps_taken":         steps_taken,
        "completion_time_s":   round(steps_taken * dt, 2),
        "path_length":         round(path_length, 4),
        "path_efficiency":     eff,
        "mean_speed":          round(float(v_arr.mean()), 4),
        "std_speed":           round(float(v_arr.std()),  4),
        "smoothness":          round(float(om_arr.sum() * dt), 4),
        "mean_cont_err":       round(float(cont_arr.mean()), 4),
        "rms_cont_err":        round(float(np.sqrt((cont_arr**2).mean())), 4),
        "max_cont_err":        round(float(cont_arr.max()),  4),
        "std_cont_err":        round(float(cont_arr.std()),  4),
        "mean_lag_err":        round(float(lag_arr.mean()),  4),
        "rms_lag_err":         round(float(np.sqrt((lag_arr**2).mean())), 4),
        "max_lag_err":         round(float(np.abs(lag_arr).max()), 4),
        "std_lag_err":         round(float(lag_arr.std()),   4),
        "mean_solve_ms":       round(float(st_arr.mean()),   2),
        "max_solve_ms":        round(float(st_arr.max()),    2),
        "std_solve_ms":        round(float(st_arr.std()),    2),
        "solver_fail_count":   solver_fails,
        "dyn_body_collisions":      dyn_body_count,
        "static_body_collisions":   static_body,
        "total_body_collisions":    dyn_body_count + static_body,
        "dyn_exclusion_violations": dyn_excl_count,
        "near_miss_count":          near_miss_count,
        "danger_zone_steps":        danger_zone,
        "min_clearance_m":          round(float(min_clearance), 4),
    }