import time
import numpy as np
import casadi as ca

from dynamics import unicycle_dynamics
from obs import Obstacle, DynamicObstacle

from params import (
    dt, nx, nu,
    r_robot, safety_buffer,
    v_max, omega_max,
)

N = 30

dv_max = 0.25
dw_max = 0.6

Q_xy      = np.diag([10.0, 10.0])
R_u       = np.diag([0.1,  0.5])
Q_term    = 50.0 * np.eye(2)
rho_slack = 1e5

REF_SLOWDOWN = 1
NEAR_MISS_MARGIN = 0.3


def _ellipse_surface_dist(rx, ry, ox, oy, a, b):
    dx = rx - ox
    dy = ry - oy
    d  = np.sqrt((dx / a) ** 2 + (dy / b) ** 2)
    return (d - 1.0) * np.sqrt(a * b)


def build_mpc_solver(static_rects, dyn_obs, use_hard=False):
    num_dyn_obs = len(dyn_obs)

    X      = ca.SX.sym("X", nx, N + 1)
    U      = ca.SX.sym("U", nu, N)
    X0     = ca.SX.sym("X0", nx)
    u_prev = ca.SX.sym("u_prev", nu)
    R      = ca.SX.sym("R", 2, N + 1)

    obs_x  = ca.SX.sym("obs_x",  num_dyn_obs, N)
    obs_y  = ca.SX.sym("obs_y",  num_dyn_obs, N)
    obs_th = ca.SX.sym("obs_th", num_dyn_obs, N)   # NEW

    X_goal = ca.SX.sym("X_goal", nx)

    if not use_hard:
        S = ca.SX.sym("S", num_dyn_obs, N)

    cost = 0
    for k in range(N):
        x_err = X[0:2, k] - R[0:2, k]
        cost += ca.mtimes([x_err.T, Q_xy, x_err])
        cost += ca.mtimes([U[:, k].T, R_u, U[:, k]])

    e_term = X[0:2, N] - X_goal[0:2]
    cost += ca.mtimes([e_term.T, Q_term, e_term])

    if not use_hard:
        for i in range(num_dyn_obs):
            for k in range(N):
                cost += rho_slack * S[i, k] ** 2

    g_list, lbg, ubg = [], [], []

    # Initial condition
    g_list.append(X[:, 0] - X0)
    lbg += [0.0] * nx
    ubg += [0.0] * nx

    for k in range(N):
        g_list.append(X[:, k + 1] - unicycle_dynamics(X[:, k], U[:, k]))
        lbg += [0.0] * nx
        ubg += [0.0] * nx

        p_next = X[0:2, k + 1]

        # Dynamic obstacles: rotated ellipses
        for i, obs in enumerate(dyn_obs):
            a_e = obs.a + r_robot + safety_buffer
            b_e = obs.b + r_robot + safety_buffer

            dx = p_next[0] - obs_x[i, k]
            dy = p_next[1] - obs_y[i, k]
            th = obs_th[i, k]

            c = ca.cos(th)
            s = ca.sin(th)

            # rotate robot-obstacle relative position into obstacle body frame
            x_body =  c * dx + s * dy
            y_body = -s * dx + c * dy

            dist = (x_body ** 2) / (a_e ** 2) + (y_body ** 2) / (b_e ** 2)

            if use_hard:
                g_list.append(dist - 1.0)
                lbg.append(0.0)
                ubg.append(ca.inf)
            else:
                g_list.append(dist - 1.0 + S[i, k])
                lbg.append(0.0)
                ubg.append(ca.inf)

        # Static obstacles
        for (cx, cy, hw, hh) in static_rects:
            dx = p_next[0] - cx
            dy = p_next[1] - cy

            p_norm = 6
            obs_margin = r_robot + safety_buffer

            dist_p = (
                (ca.fabs(dx) / (hw + obs_margin)) ** p_norm
                + (ca.fabs(dy) / (hh + obs_margin)) ** p_norm
            )

            g_list.append(dist_p)
            lbg.append(1.0)
            ubg.append(ca.inf)

    # Slew between horizon steps
    for k in range(N - 1):
        g_list.append(U[0, k + 1] - U[0, k])
        lbg.append(-dv_max)
        ubg.append(dv_max)

        g_list.append(U[1, k + 1] - U[1, k])
        lbg.append(-dw_max)
        ubg.append(dw_max)

    # Slew from previous control
    g_list.append(U[:, 0] - ca.reshape(u_prev, 2, 1))
    lbg += [-dv_max, -dw_max]
    ubg += [dv_max, dw_max]

    g = ca.vertcat(*g_list)

    lbx, ubx = [], []
    for _ in range(N + 1):
        lbx += [-ca.inf] * nx
        ubx += [ca.inf] * nx

    for _ in range(N):
        lbx += [0.0, -omega_max]
        ubx += [v_max, omega_max]

    if not use_hard:
        for _ in range(num_dyn_obs * N):
            lbx.append(0.0)
            ubx.append(ca.inf)

    if use_hard:
        OPT_vars = ca.vertcat(
            ca.reshape(X, -1, 1),
            ca.reshape(U, -1, 1),
        )
    else:
        OPT_vars = ca.vertcat(
            ca.reshape(X, -1, 1),
            ca.reshape(U, -1, 1),
            ca.reshape(S, -1, 1),
        )

    p_vec = ca.vertcat(
        X0,
        u_prev,
        ca.reshape(R, -1, 1),
        ca.reshape(obs_x, -1, 1),
        ca.reshape(obs_y, -1, 1),
        ca.reshape(obs_th, -1, 1),   # NEW
        X_goal,
    )

    solver_name = "solver_mpc_hard" if use_hard else "solver_mpc_soft"
    solver = ca.nlpsol(
        solver_name,
        "ipopt",
        {"x": OPT_vars, "f": cost, "g": g, "p": p_vec},
        {
            "ipopt.print_level": 1,
            "ipopt.max_iter": 200,
            "ipopt.tol": 1e-6,
            "ipopt.constr_viol_tol": 1e-6,
            "ipopt.acceptable_tol": 1e-6,
            "ipopt.acceptable_iter": 0,
            "ipopt.warm_start_init_point": "yes",
            "print_time": 0,
        },
    )

    nX = nx * (N + 1)
    nU = nu * N
    return solver, lbx, ubx, lbg, ubg, nX, nU


def run_mpc(path_id: int, seed_offset: int, max_steps: int = 800, use_hard: bool = False) -> dict:
    env          = Obstacle(path_id)
    ref_traj     = env.path_selector(path_id)
    STATIC_RECTS = env.static_obs()
    dyn_obs      = env.dynamic_obs_seeded(seed_offset)

    straight_line = float(np.linalg.norm(ref_traj[:, -1] - ref_traj[:, 0]))

    solver, lbx, ubx, lbg, ubg, nX, nU = build_mpc_solver(
        STATIC_RECTS, dyn_obs, use_hard=use_hard
    )

    p0  = ref_traj[:, 0]
    p1  = ref_traj[:, 1]
    th0 = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    x_current     = np.array([p0[0], p0[1], th0])
    X_goal_global = np.array([ref_traj[0, -1], ref_traj[1, -1], 0.0])

    u_prev        = np.zeros(nu)
    prev_z        = None
    ref_k         = 0
    _at_end_count = 0

    cont_errors = []
    lag_errors = []
    solve_times = []
    v_hist = []
    omega_hist = []
    path_length = 0.0
    prev_pos = x_current[:2].copy()

    dyn_body_count = 0
    dyn_excl_count = 0
    static_body = 0
    near_miss_count = 0
    min_clearance = np.inf
    danger_zone = 0
    solver_fails = 0
    goal_reached = False
    steps_taken = 0

    for k in range(max_steps):
        R_horizon = ref_traj[:, ref_k : ref_k + N + 1]
        if R_horizon.shape[1] < N + 1:
            last = R_horizon[:, -1].reshape(2, 1)
            pad = np.repeat(last, N + 1 - R_horizon.shape[1], axis=1)
            R_horizon = np.hstack([R_horizon, pad])

        X_goal_val = np.array([R_horizon[0, -1], R_horizon[1, -1], 0.0])

        if dyn_obs:
            preds    = [obs.predict_horizon(N, dt) for obs in dyn_obs]
            obs_x_h  = np.array([pr[0] for pr in preds])
            obs_y_h  = np.array([pr[1] for pr in preds])
            obs_th_h = np.array([pr[2] for pr in preds])   # NEW
        else:
            obs_x_h  = np.zeros((0, N))
            obs_y_h  = np.zeros((0, N))
            obs_th_h = np.zeros((0, N))                    # NEW

        p = np.concatenate([
            x_current,
            u_prev,
            R_horizon.flatten(order="F"),
            obs_x_h.flatten(order="F"),
            obs_y_h.flatten(order="F"),
            obs_th_h.flatten(order="F"),                   # NEW
            X_goal_val,
        ])

        kwargs = dict(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)
        if prev_z is not None:
            kwargs["x0"] = prev_z

        t0 = time.perf_counter()
        sol = solver(**kwargs)
        solve_times.append(time.perf_counter() - t0)

        stats = solver.stats()
        if stats["return_status"] not in {
            "Solve_Succeeded",
            "Solved_To_Acceptable_Level",
            "Maximum_Iterations_Exceeded",
        }:
            solver_fails += 1

        acceptable_statuses = {
            "Solve_Succeeded",
            "Solved_To_Acceptable_Level",
            "Maximum_Iterations_Exceeded",
        }

        if stats["return_status"] not in acceptable_statuses:
            print(f"❌ Solver failed at step {k}: {stats['return_status']}")
            break

        if stats["return_status"] != "Solve_Succeeded":
            solver_fails += 1

        prev_z = sol["x"]

        z = sol["x"].full().flatten()
        U_opt = z[nX : nX + nU].reshape((nu, N), order="F")
        v, omega = U_opt[:, 0]
        u_prev = np.array([v, omega])

        v_hist.append(float(v))
        omega_hist.append(float(omega))

        if k % REF_SLOWDOWN == 0:
            ref_k = min(ref_k + 1, ref_traj.shape[1] - 1)

        x_current = np.array([
            x_current[0] + dt * v * np.cos(x_current[2]),
            x_current[1] + dt * v * np.sin(x_current[2]),
            x_current[2] + dt * omega,
        ])
        steps_taken += 1
        path_length += float(np.linalg.norm(x_current[:2] - prev_pos))
        prev_pos = x_current[:2].copy()

        for obs in dyn_obs:
            obs.step(dt, STATIC_RECTS)

        step_near = False
        step_danger = False

        for obs in dyn_obs:
            dx = x_current[0] - obs.x
            dy = x_current[1] - obs.y

            if (dx / obs.a) ** 2 + (dy / obs.b) ** 2 <= 1.0:
                dyn_body_count += 1

            a_e = obs.a + r_robot + safety_buffer
            b_e = obs.b + r_robot + safety_buffer

            # Optional: use rotated body-frame check here too
            c = np.cos(obs.theta)
            s = np.sin(obs.theta)
            x_body =  c * dx + s * dy
            y_body = -s * dx + c * dy

            if (x_body / a_e) ** 2 + (y_body / b_e) ** 2 <= 1.0:
                dyn_excl_count += 1

            cl = _ellipse_surface_dist(
                x_current[0], x_current[1], obs.x, obs.y, obs.a, obs.b
            )
            min_clearance = min(min_clearance, cl)

            if cl < NEAR_MISS_MARGIN:
                step_near = True
            if cl < NEAR_MISS_MARGIN + 2 * safety_buffer:
                step_danger = True

        for (cx, cy, hw, hh) in STATIC_RECTS:
            if abs(x_current[0] - cx) < hw + r_robot and abs(x_current[1] - cy) < hh + r_robot:
                static_body += 1

            dx_s = max(abs(x_current[0] - cx) - hw, 0.0)
            dy_s = max(abs(x_current[1] - cy) - hh, 0.0)
            cl_s = np.sqrt(dx_s ** 2 + dy_s ** 2) - r_robot
            min_clearance = min(min_clearance, cl_s)

            if cl_s < NEAR_MISS_MARGIN:
                step_near = True
            if cl_s < NEAR_MISS_MARGIN + 2 * safety_buffer:
                step_danger = True

        if step_near:
            near_miss_count += 1
        if step_danger:
            danger_zone += 1

        dists = np.linalg.norm(ref_traj.T - x_current[:2], axis=1)
        k2 = min(int(np.argmin(dists)), ref_traj.shape[1] - 2)
        R_now = ref_traj[:, k2:k2 + 2]
        dR = R_now[:, 1] - R_now[:, 0]
        t_hat = dR / (np.linalg.norm(dR) + 1e-12)
        n_hat = np.array([-t_hat[1], t_hat[0]])
        ev = x_current[:2] - R_now[:, 0]
        cont_errors.append(float(np.dot(n_hat, ev)))
        lag_errors.append(float(np.dot(t_hat, ev)))

        dist_to_goal = np.linalg.norm(x_current[:2] - X_goal_global[:2])
        path_complete = ref_k >= ref_traj.shape[1] - 2

        if dist_to_goal < 0.5 and path_complete:
            goal_reached = True
            break

        if path_complete:
            _at_end_count += 1
            if _at_end_count >= 30:
                goal_reached = True
                break
        else:
            _at_end_count = 0

    cont_arr = np.abs(cont_errors)
    lag_arr = np.array(lag_errors)
    st_arr = np.array(solve_times) * 1000
    v_arr = np.array(v_hist)
    om_arr = np.abs(omega_hist)
    eff = round(straight_line / path_length, 4) if path_length > 0 else 0.0

    return {
        "path_id": path_id,
        "seed_offset": seed_offset,
        "goal_reached": int(goal_reached),
        "steps_taken": steps_taken,
        "completion_time_s": round(steps_taken * dt, 2),
        "path_length": round(path_length, 4),
        "path_efficiency": eff,
        "mean_speed": round(float(v_arr.mean()), 4),
        "std_speed": round(float(v_arr.std()), 4),
        "smoothness": round(float(om_arr.sum() * dt), 4),
        "mean_cont_err": round(float(cont_arr.mean()), 4),
        "rms_cont_err": round(float(np.sqrt((cont_arr ** 2).mean())), 4),
        "max_cont_err": round(float(cont_arr.max()), 4),
        "std_cont_err": round(float(cont_arr.std()), 4),
        "mean_lag_err": round(float(lag_arr.mean()), 4),
        "rms_lag_err": round(float(np.sqrt((lag_arr ** 2).mean())), 4),
        "max_lag_err": round(float(np.abs(lag_arr).max()), 4),
        "std_lag_err": round(float(lag_arr.std()), 4),
        "mean_solve_ms": round(float(st_arr.mean()), 2),
        "max_solve_ms": round(float(st_arr.max()), 2),
        "std_solve_ms": round(float(st_arr.std()), 2),
        "solver_fail_count": solver_fails,
        "dyn_body_collisions": dyn_body_count,
        "static_body_collisions": static_body,
        "total_body_collisions": dyn_body_count + static_body,
        "dyn_exclusion_violations": dyn_excl_count,
        "near_miss_count": near_miss_count,
        "danger_zone_steps": danger_zone,
        "min_clearance_m": round(float(min_clearance), 4),
    }