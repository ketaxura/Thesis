"""
run_experiment.py
Single MPCC experiment runner — returns a comprehensive metrics dict.
Called by batch_runner.py.
"""

import time
import numpy as np
from mpcc import build_mpcc_solver
from obs import Obstacle

from params import (
    dt, N, nx, nu, nr,
    r_robot, safety_buffer,
    v_max, omega_max, dv_max, dw_max,
    kappa_w, eps,
)
from dynamics import unicycle_dynamics

# Near-miss threshold: distance from robot centre to obstacle surface
NEAR_MISS_MARGIN = 0.3   # metres beyond the raw obstacle body


def _ellipse_surface_dist(rx, ry, ox, oy, a, b):
    """Approximate signed distance from robot centre to ellipse surface."""
    dx = rx - ox
    dy = ry - oy
    d  = np.sqrt((dx / a)**2 + (dy / b)**2)
    r_eff = np.sqrt(a * b)
    return (d - 1.0) * r_eff


def run_mpcc(path_id: int, seed_offset: int, max_steps: int = 800) -> dict:

    env          = Obstacle(path_id)
    ref_traj     = env.path_selector(path_id)
    STATIC_RECTS = env.static_obs()
    dyn_obs      = env.dynamic_obs_seeded(seed_offset)

    straight_line_dist = float(np.linalg.norm(ref_traj[:, -1] - ref_traj[:, 0]))

    mpcc_data = build_mpcc_solver(ref_traj, STATIC_RECTS, dyn_obs)
    solver    = mpcc_data.solver
    lbx, ubx  = mpcc_data.lbx, mpcc_data.ubx
    lbg, ubg  = mpcc_data.lbg, mpcc_data.ubg

    p0  = ref_traj[:, 0]
    p1  = ref_traj[:, 1]
    th0 = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    x_current     = np.array([p0[0], p0[1], th0])
    X_goal_global = np.array([ref_traj[0, -1], ref_traj[1, -1], 0.0])

    mu            = 0.0
    u_prev        = np.zeros(nu)
    prev_z        = None
    _at_end_count = 0

    cont_errors       = []
    lag_errors        = []
    solve_times       = []
    omega_hist        = []
    v_hist            = []
    path_length       = 0.0
    prev_pos          = x_current[:2].copy()

    dyn_body_count      = 0
    dyn_exclusion_count = 0
    static_body_count   = 0
    near_miss_count     = 0
    min_clearance       = np.inf
    danger_zone_count   = 0
    solver_fail_count   = 0

    goal_reached = False
    steps_taken  = 0

    for k in range(max_steps):

        k_ref     = max(int(np.floor(mu)), 0)
        s_local   = mu - k_ref
        R_horizon = ref_traj[:, k_ref : k_ref + N + 1]
        if R_horizon.shape[1] < N + 1:
            last = R_horizon[:, -1].reshape(-1, 1)
            pad  = np.repeat(last, N + 1 - R_horizon.shape[1], axis=1)
            R_horizon = np.hstack((R_horizon, pad))

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
            x_current, u_prev, np.array([s_local]),
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
        solve_times.append(time.perf_counter() - t0)

        stats = solver.stats()
        if stats["return_status"] not in {
            "Solve_Succeeded", "Solved_To_Acceptable_Level",
            "Maximum_Iterations_Exceeded"
        }:
            solver_fail_count += 1

        prev_z = sol["x"]

        z      = sol["x"].full().flatten()
        offset = 0
        X_opt  = z[offset:offset + mpcc_data.nX].reshape((nx, N+1), order="F"); offset += mpcc_data.nX
        U_opt  = z[offset:offset + mpcc_data.nU].reshape((nu, N),   order="F"); offset += mpcc_data.nU
        s_opt  = z[offset:offset + mpcc_data.nProg].reshape((N+1,), order="F"); offset += mpcc_data.nProg
        offset += mpcc_data.nSlack_static

        v, omega = U_opt[:, 0]
        u_prev   = np.array([v, omega])
        v_hist.append(float(v))
        omega_hist.append(float(omega))

        delta_s = float(s_opt[1]) - float(s_opt[0])
        mu      = float(np.clip(mu + delta_s, 0.0, ref_traj.shape[1] - 1))

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

        # Safety metrics
        step_near_miss   = False
        step_danger_zone = False

        for obs in dyn_obs:
            dx = x_current[0] - obs.x
            dy = x_current[1] - obs.y

            if (dx / obs.a)**2 + (dy / obs.b)**2 <= 1.0:
                dyn_body_count += 1

            a_e = obs.a + r_robot + safety_buffer
            b_e = obs.b + r_robot + safety_buffer
            if (dx / a_e)**2 + (dy / b_e)**2 <= 1.0:
                dyn_exclusion_count += 1

            clearance = _ellipse_surface_dist(
                x_current[0], x_current[1], obs.x, obs.y, obs.a, obs.b
            )
            min_clearance = min(min_clearance, clearance)

            if clearance < NEAR_MISS_MARGIN:
                step_near_miss = True
            if clearance < NEAR_MISS_MARGIN + 2 * safety_buffer:
                step_danger_zone = True

        for (cx, cy, hw, hh) in STATIC_RECTS:
            if abs(x_current[0]-cx) < hw+r_robot and abs(x_current[1]-cy) < hh+r_robot:
                static_body_count += 1

            dx_s = max(abs(x_current[0] - cx) - hw, 0.0)
            dy_s = max(abs(x_current[1] - cy) - hh, 0.0)
            clearance_s = np.sqrt(dx_s**2 + dy_s**2) - r_robot
            min_clearance = min(min_clearance, clearance_s)
            if clearance_s < NEAR_MISS_MARGIN:
                step_near_miss = True
            if clearance_s < NEAR_MISS_MARGIN + 2 * safety_buffer:
                step_danger_zone = True

        if step_near_miss:
            near_miss_count += 1
        if step_danger_zone:
            danger_zone_count += 1

        # Geometric errors
        k2    = int(np.clip(np.floor(mu), 0, ref_traj.shape[1] - 2))
        R_now = ref_traj[:, k2:k2+2]
        dR    = R_now[:, 1] - R_now[:, 0]
        t_hat = dR / (np.linalg.norm(dR) + 1e-12)
        n_hat = np.array([-t_hat[1], t_hat[0]])
        ev    = x_current[:2] - R_now[:, 0]
        cont_errors.append(float(np.dot(n_hat, ev)))
        lag_errors.append(float(np.dot(t_hat, ev)))

        # Termination
        dist_to_goal  = np.linalg.norm(x_current[:2] - X_goal_global[:2])
        path_complete = mu >= ref_traj.shape[1] - 2

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

    # Pack metrics
    cont_arr = np.abs(cont_errors)
    lag_arr  = np.array(lag_errors)
    st_arr   = np.array(solve_times) * 1000
    v_arr    = np.array(v_hist)
    om_arr   = np.abs(omega_hist)

    path_efficiency = round(straight_line_dist / path_length, 4) if path_length > 0 else 0.0

    return {
        "path_id":              path_id,
        "seed_offset":          seed_offset,
        "goal_reached":         int(goal_reached),
        "steps_taken":          steps_taken,
        "completion_time_s":    round(steps_taken * dt, 2),

        "path_length":          round(path_length, 4),
        "path_efficiency":      path_efficiency,
        "mean_speed":           round(float(v_arr.mean()), 4),
        "std_speed":            round(float(v_arr.std()),  4),
        "smoothness":           round(float(om_arr.sum() * dt), 4),

        "mean_cont_err":        round(float(cont_arr.mean()), 4),
        "rms_cont_err":         round(float(np.sqrt((cont_arr**2).mean())), 4),
        "max_cont_err":         round(float(cont_arr.max()),  4),
        "std_cont_err":         round(float(cont_arr.std()),  4),

        "mean_lag_err":         round(float(lag_arr.mean()),  4),
        "rms_lag_err":          round(float(np.sqrt((lag_arr**2).mean())), 4),
        "max_lag_err":          round(float(np.abs(lag_arr).max()), 4),
        "std_lag_err":          round(float(lag_arr.std()),   4),

        "mean_solve_ms":        round(float(st_arr.mean()),   2),
        "max_solve_ms":         round(float(st_arr.max()),    2),
        "std_solve_ms":         round(float(st_arr.std()),    2),
        "solver_fail_count":    solver_fail_count,

        "dyn_body_collisions":       dyn_body_count,
        "static_body_collisions":    static_body_count,
        "total_body_collisions":     dyn_body_count + static_body_count,
        "dyn_exclusion_violations":  dyn_exclusion_count,
        "near_miss_count":           near_miss_count,
        "danger_zone_steps":         danger_zone_count,
        "min_clearance_m":           round(float(min_clearance), 4),
    }