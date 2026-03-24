import time
import inspect
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import casadi as ca

from dynamics import unicycle_dynamics
from obs import Obstacle
from mpcc import build_mpcc_solver

from params import (
    dt, N, nx, nu, nr,
    r_robot, safety_buffer,
    v_max, omega_max, dv_max, dw_max,
    kappa_w, eps, v_obs_max,
)

# ============================================================
# Shared experiment constants
# ============================================================

REF_SLOWDOWN = 1
NEAR_MISS_MARGIN = 0.3

# Dynamic-obstacle relevance filtering for faster solves
MAX_ACTIVE_DYN_OBS = 4
OBS_HORIZON_EXTRA_MARGIN = 1.0
DUMMY_OBS_FAR_AWAY = 1e6


@dataclass
class DynObsTemplate:
    a: float
    b: float


def _default_observation_radius(dyn_obs: list) -> float:
    if not dyn_obs:
        return 0.0
    max_obs_radius = max(max(float(obs.a), float(obs.b)) for obs in dyn_obs)
    horizon_time = N * dt
    return (
        v_max * horizon_time
        + v_obs_max * horizon_time
        + (r_robot + safety_buffer)
        + max_obs_radius
        + OBS_HORIZON_EXTRA_MARGIN
    )


def _make_solver_dyn_templates(dyn_obs: list, max_active_dyn: int) -> list:
    if max_active_dyn <= 0:
        return []
    if dyn_obs:
        max_a = max(float(obs.a) for obs in dyn_obs)
        max_b = max(float(obs.b) for obs in dyn_obs)
    else:
        max_a = 0.6
        max_b = 0.4
    return [DynObsTemplate(a=max_a, b=max_b) for _ in range(max_active_dyn)]


def _select_active_dyn_obs(
    x_current: np.ndarray,
    dyn_obs: list,
    max_active_dyn: int,
    observation_radius: float,
) -> list:
    if max_active_dyn <= 0 or not dyn_obs:
        return []

    rx, ry = float(x_current[0]), float(x_current[1])
    scored = []
    for obs in dyn_obs:
        dist = float(np.hypot(obs.x - rx, obs.y - ry))
        if dist <= observation_radius:
            scored.append((dist, obs))

    if not scored:
        return []

    scored.sort(key=lambda item: item[0])
    return [obs for _, obs in scored[:max_active_dyn]]


def _predict_and_pad_dyn_obs(
    active_dyn_obs: list,
    solver_dyn_slots: int,
    static_rects: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if solver_dyn_slots <= 0:
        return np.zeros((0, N)), np.zeros((0, N)), np.zeros((0, N))

    obs_x_h = np.full((solver_dyn_slots, N), DUMMY_OBS_FAR_AWAY, dtype=float)
    obs_y_h = np.full((solver_dyn_slots, N), DUMMY_OBS_FAR_AWAY, dtype=float)
    obs_th_h = np.zeros((solver_dyn_slots, N), dtype=float)

    for i, obs in enumerate(active_dyn_obs[:solver_dyn_slots]):
        pred = (
            obs.predict_horizon_with_rects(N, dt, static_rects)
            if hasattr(obs, "predict_horizon_with_rects")
            else obs.predict_horizon(N, dt)
        )
        obs_x_h[i, :] = np.asarray(pred[0], dtype=float)
        obs_y_h[i, :] = np.asarray(pred[1], dtype=float)
        obs_th_h[i, :] = np.asarray(pred[2], dtype=float)

    return obs_x_h, obs_y_h, obs_th_h

SCENARIOS = [
    {"path_id": 3, "label": "Zigzag"},
    {"path_id": 1, "label": "Sine"},
]

METRICS = [
    "goal_reached", "steps_taken", "completion_time_s",
    "path_length", "path_efficiency",
    "mean_speed", "std_speed", "smoothness",
    "mean_cont_err", "rms_cont_err", "max_cont_err", "std_cont_err",
    "mean_lag_err",  "rms_lag_err",  "max_lag_err",  "std_lag_err",
    "mean_solve_ms", "max_solve_ms", "std_solve_ms", "solver_fail_count",
    "dyn_body_collisions", "static_body_collisions", "total_body_collisions",
    "dyn_exclusion_violations", "near_miss_count",
    "danger_zone_steps", "min_clearance_m",
]

SUMMARY_METRICS = [
    "goal_reached", "completion_time_s", "path_efficiency",
    "mean_speed", "smoothness",
    "mean_cont_err", "rms_cont_err", "max_cont_err",
    "mean_lag_err", "rms_lag_err",
    "mean_solve_ms", "max_solve_ms", "solver_fail_count",
    "total_body_collisions", "dyn_exclusion_violations",
    "near_miss_count", "danger_zone_steps", "min_clearance_m",
]

# ============================================================
# Shared geometry helpers
# ============================================================

def _rotate_into_body_frame(
    rx: float, ry: float,
    ox: float, oy: float,
    th: float,
) -> Tuple[float, float]:
    dx = rx - ox
    dy = ry - oy
    c = np.cos(th)
    s = np.sin(th)
    x_body = c * dx + s * dy
    y_body = -s * dx + c * dy
    return x_body, y_body


def _rotated_ellipse_level(
    rx: float, ry: float,
    ox: float, oy: float,
    a: float, b: float,
    th: float,
) -> float:
    x_body, y_body = _rotate_into_body_frame(rx, ry, ox, oy, th)
    return (x_body / a) ** 2 + (y_body / b) ** 2


def _rotated_ellipse_surface_dist(
    rx: float, ry: float,
    ox: float, oy: float,
    a: float, b: float,
    th: float,
) -> float:
    """
    Approximate signed distance from point to rotated ellipse surface.
    Negative means inside.
    """
    x_body, y_body = _rotate_into_body_frame(rx, ry, ox, oy, th)
    d = np.sqrt((x_body / a) ** 2 + (y_body / b) ** 2)
    r_eff = np.sqrt(a * b)
    return (d - 1.0) * r_eff


def _compute_common_tracking_errors(
    x_current: np.ndarray,
    ref_traj: np.ndarray,
) -> Tuple[float, float]:
    """
    Shared geometric error metric for BOTH MPC and MPCC.
    Uses nearest reference segment so the comparison is apples-to-apples.
    """
    dists = np.linalg.norm(ref_traj.T - x_current[:2], axis=1)
    k2 = min(int(np.argmin(dists)), ref_traj.shape[1] - 2)

    R_now = ref_traj[:, k2:k2 + 2]
    dR = R_now[:, 1] - R_now[:, 0]
    t_hat = dR / (np.linalg.norm(dR) + 1e-12)
    n_hat = np.array([-t_hat[1], t_hat[0]])
    ev = x_current[:2] - R_now[:, 0]

    cont_err = float(np.dot(n_hat, ev))
    lag_err = float(np.dot(t_hat, ev))
    return cont_err, lag_err


# ============================================================
# Shared MPC solver builder
# ============================================================

Q_xy = np.diag([10.0, 10.0])
R_u = np.diag([0.1, 0.5])
Q_term = 50.0 * np.eye(2)
rho_slack = 1e5


def build_mpc_solver(static_rects, dyn_obs, use_hard: bool = False):
    num_dyn_obs = len(dyn_obs)

    X = ca.SX.sym("X", nx, N + 1)
    U = ca.SX.sym("U", nu, N)
    X0 = ca.SX.sym("X0", nx)
    u_prev = ca.SX.sym("u_prev", nu)
    R = ca.SX.sym("R", 2, N + 1)

    obs_x = ca.SX.sym("obs_x", num_dyn_obs, N)
    obs_y = ca.SX.sym("obs_y", num_dyn_obs, N)
    obs_th = ca.SX.sym("obs_th", num_dyn_obs, N)

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

    g_list.append(X[:, 0] - X0)
    lbg += [0.0] * nx
    ubg += [0.0] * nx

    for k in range(N):
        g_list.append(X[:, k + 1] - unicycle_dynamics(X[:, k], U[:, k]))
        lbg += [0.0] * nx
        ubg += [0.0] * nx

        p_next = X[0:2, k + 1]

        for i, obs in enumerate(dyn_obs):
            a_e = obs.a + r_robot + safety_buffer
            b_e = obs.b + r_robot + safety_buffer

            dx = p_next[0] - obs_x[i, k]
            dy = p_next[1] - obs_y[i, k]
            th = obs_th[i, k]

            c = ca.cos(th)
            s = ca.sin(th)

            x_body = c * dx + s * dy
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

    for k in range(N - 1):
        g_list.append(U[0, k + 1] - U[0, k])
        lbg.append(-dv_max)
        ubg.append(dv_max)

        g_list.append(U[1, k + 1] - U[1, k])
        lbg.append(-dw_max)
        ubg.append(dw_max)

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
        opt_vars = ca.vertcat(
            ca.reshape(X, -1, 1),
            ca.reshape(U, -1, 1),
        )
    else:
        opt_vars = ca.vertcat(
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
        ca.reshape(obs_th, -1, 1),
        X_goal,
    )

    solver_name = "solver_mpc_hard" if use_hard else "solver_mpc_soft"
    solver = ca.nlpsol(
        solver_name,
        "ipopt",
        {"x": opt_vars, "f": cost, "g": g, "p": p_vec},
        {
            "ipopt.print_level": 0,
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


# ============================================================
# Controller result struct
# ============================================================

@dataclass
class SolveStep:
    ok: bool
    status: str
    solve_time_s: float
    u: np.ndarray
    progress_index: int
    solver_next_state: Optional[np.ndarray] = None
    rollout_next_state: Optional[np.ndarray] = None
    state_mismatch_norm: float = 0.0


    predict_time_s: float = 0.0
    select_time_s: float = 0.0
    pack_time_s: float = 0.0
    post_time_s: float = 0.0
    active_dyn_count: int = 0


# ============================================================
# Controller wrappers
# ============================================================

class MPCController:
    def __init__(self, static_rects, dyn_obs, ref_traj, use_hard: bool):
        self.ref_traj = ref_traj
        self.use_hard = use_hard
        self.static_rects = static_rects
        self.max_active_dyn = min(MAX_ACTIVE_DYN_OBS, len(dyn_obs))
        self.observation_radius = _default_observation_radius(dyn_obs)
        self.solver_dyn_templates = _make_solver_dyn_templates(dyn_obs, self.max_active_dyn)

        self.solver, self.lbx, self.ubx, self.lbg, self.ubg, self.nX, self.nU = (
            build_mpc_solver(static_rects, self.solver_dyn_templates, use_hard=use_hard)
        )

        self.u_prev = np.zeros(nu)
        self.prev_z = None
        self.ref_k = 0

    def solve(self, x_current: np.ndarray, dyn_obs, goal_global: np.ndarray) -> SolveStep:
        R_horizon = self.ref_traj[:, self.ref_k:self.ref_k + N + 1]
        if R_horizon.shape[1] < N + 1:
            last = R_horizon[:, -1].reshape(2, 1)
            pad = np.repeat(last, N + 1 - R_horizon.shape[1], axis=1)
            R_horizon = np.hstack([R_horizon, pad])

        X_goal_val = np.array([R_horizon[0, -1], R_horizon[1, -1], 0.0])

        t_sel0 = time.perf_counter()
        active_dyn_obs = _select_active_dyn_obs(
            x_current, dyn_obs, self.max_active_dyn, self.observation_radius
        )
        select_time_s = time.perf_counter() - t_sel0

        t_pred0 = time.perf_counter()
        obs_x_h, obs_y_h, obs_th_h = _predict_and_pad_dyn_obs(
            active_dyn_obs, self.max_active_dyn, self.static_rects
        )
        predict_time_s = time.perf_counter() - t_pred0

        t_pack0 = time.perf_counter()
        p = np.concatenate([
            x_current,
            self.u_prev,
            R_horizon.flatten(order="F"),
            obs_x_h.flatten(order="F"),
            obs_y_h.flatten(order="F"),
            obs_th_h.flatten(order="F"),
            X_goal_val,
        ])
        pack_time_s = time.perf_counter() - t_pack0

        kwargs = dict(
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
            p=p,
        )

        X_guess = np.tile(x_current.reshape(-1, 1), (1, N + 1))
        U_guess = np.zeros((nu, N))

        parts = [
            X_guess.flatten(order="F"),
            U_guess.flatten(order="F"),
        ]

        if not self.use_hard:
            n_dyn = self.max_active_dyn
            S_guess = np.zeros(n_dyn * N)
            parts.append(S_guess)

        z0 = np.concatenate(parts)

        expected_n = len(self.lbx)

        if self.prev_z is None or self.prev_z.shape[0] != expected_n:
            kwargs["x0"] = z0
        else:
            kwargs["x0"] = self.prev_z

        t0 = time.perf_counter()
        sol = self.solver(**kwargs)
        solve_time_s = time.perf_counter() - t0

        stats = self.solver.stats()
        status = stats["return_status"]

        acceptable_statuses = {
            "Solve_Succeeded",
            "Solved_To_Acceptable_Level",
            "Maximum_Iterations_Exceeded",
        }

        if status not in acceptable_statuses:
            return SolveStep(
                ok=False,
                status=status,
                solve_time_s=solve_time_s,
                u=np.zeros(nu),
                progress_index=self.ref_k,
            )

        self.prev_z = sol["x"]

        t_post0 = time.perf_counter()

        z = sol["x"].full().flatten()
        X_opt = z[:self.nX].reshape((nx, N + 1), order="F")
        U_opt = z[self.nX:self.nX + self.nU].reshape((nu, N), order="F")
        v, omega = U_opt[:, 0]
        self.u_prev = np.array([v, omega])

        solver_next_state = np.array(X_opt[:, 1], dtype=float)
        rollout_next_state = np.array([
            x_current[0] + dt * v * np.cos(x_current[2]),
            x_current[1] + dt * v * np.sin(x_current[2]),
            x_current[2] + dt * omega,
        ], dtype=float)
        state_mismatch_norm = float(np.linalg.norm(solver_next_state - rollout_next_state))

        self.ref_k = min(self.ref_k + REF_SLOWDOWN, self.ref_traj.shape[1] - 1)

        post_time_s = time.perf_counter() - t_post0

        return SolveStep(
            ok=True,
            status=status,
            solve_time_s=solve_time_s,
            u=np.array([v, omega]),
            progress_index=self.ref_k,
            solver_next_state=solver_next_state,
            rollout_next_state=rollout_next_state,
            state_mismatch_norm=state_mismatch_norm,
            select_time_s=select_time_s,
            predict_time_s=predict_time_s,
            pack_time_s=pack_time_s,
            post_time_s=post_time_s,
            active_dyn_count=len(active_dyn_obs),
        )





class MPCCController:
    _warned_no_true_hard_soft = False

    def __init__(self, static_rects, dyn_obs, ref_traj, use_hard: bool):
        self.ref_traj = ref_traj
        self.use_hard = use_hard
        self.static_rects = static_rects
        self.max_active_dyn = min(MAX_ACTIVE_DYN_OBS, len(dyn_obs))
        self.observation_radius = _default_observation_radius(dyn_obs)
        self.solver_dyn_templates = _make_solver_dyn_templates(dyn_obs, self.max_active_dyn)

        sig = inspect.signature(build_mpcc_solver)
        if "use_hard" in sig.parameters:
            self.mpcc_data = build_mpcc_solver(
                ref_traj,
                static_rects,
                self.solver_dyn_templates,
                use_hard=use_hard,
            )
        else:
            if not MPCCController._warned_no_true_hard_soft:
                mode_msg = (
                    "WARNING: build_mpcc_solver() does not accept use_hard. "
                    "MPCC SOFT/HARD batch runs will both use the same underlying MPCC solver "
                    "until mpcc.py is updated."
                )
                print(mode_msg)
                MPCCController._warned_no_true_hard_soft = True

            self.mpcc_data = build_mpcc_solver(
                ref_traj,
                static_rects,
                self.solver_dyn_templates,
            )

        self.solver = self.mpcc_data.solver
        self.lbx = self.mpcc_data.lbx
        self.ubx = self.mpcc_data.ubx
        self.lbg = self.mpcc_data.lbg
        self.ubg = self.mpcc_data.ubg

        self.mu = 0.0
        self.u_prev = np.zeros(nu)
        self.prev_z = None

    def solve(self, x_current: np.ndarray, dyn_obs, goal_global: np.ndarray) -> SolveStep:
        k_ref = max(int(np.floor(self.mu)), 0)
        s_local = self.mu - k_ref

        R_horizon = self.ref_traj[:, k_ref:k_ref + N + 1]
        if R_horizon.shape[1] < N + 1:
            last = R_horizon[:, -1].reshape(-1, 1)
            pad = np.repeat(last, N + 1 - R_horizon.shape[1], axis=1)
            R_horizon = np.hstack((R_horizon, pad))

        active_dyn_obs = _select_active_dyn_obs(
            x_current, dyn_obs, self.max_active_dyn, self.observation_radius
        )
        obs_x_h, obs_y_h, obs_th_h = _predict_and_pad_dyn_obs(
            active_dyn_obs, self.max_active_dyn, self.static_rects
        )

        p = np.concatenate([
            x_current,
            self.u_prev,
            np.array([s_local]),
            R_horizon.flatten(order="F"),
            goal_global,
            obs_x_h.flatten(order="F"),
            obs_y_h.flatten(order="F"),
            obs_th_h.flatten(order="F"),
        ])

        kwargs = dict(
            lbx=self.lbx, ubx=self.ubx,
            lbg=self.lbg, ubg=self.ubg,
            p=p,
        )
        if self.prev_z is not None:
            kwargs["x0"] = self.prev_z

        t0 = time.perf_counter()



        # print("\nPre-solve parameter sanity check")
        # print("x_current =", x_current)
        # print("self.u_prev =", self.u_prev)
        # print("s_local =", s_local)
        # print("k_ref =", k_ref)
        # print("mu =", self.mu)
        # print("R_horizon shape =", R_horizon.shape)
        # print("R_horizon first col =", R_horizon[:, 0])
        # print("R_horizon last  col =", R_horizon[:, -1])

        # # print("self.u_prev[0] type/value =", type(self.u_prev[0]), self.u_prev[0])
        # # print("self.u_prev[1] type/value =", type(self.u_prev[1]), self.u_prev[1])
        # print("s_local type/value        =", type(s_local), s_local)
        # print("goal_global =", goal_global)



        X_guess = np.tile(x_current.reshape(-1, 1), (1, N + 1))
        U_guess = np.zeros((nu, N))
        s_guess = np.zeros(N + 1)

        parts = [
            X_guess.flatten(order="F"),
            U_guess.flatten(order="F"),
            s_guess.flatten(order="F"),
        ]

        if not self.use_hard:
            n_dyn = self.max_active_dyn
            Sdyn_guess = np.zeros(n_dyn * N)
            parts.append(Sdyn_guess)

        z0 = np.concatenate(parts)

        expected_n = len(self.lbx)

        if self.prev_z is None or self.prev_z.shape[0] != expected_n:
            kwargs["x0"] = z0
        else:
            kwargs["x0"] = self.prev_z

        sol = self.solver(**kwargs)
        solve_time_s = time.perf_counter() - t0

        stats = self.solver.stats()
        status = stats["return_status"]

        acceptable_statuses = {
            "Solve_Succeeded",
            "Solved_To_Acceptable_Level",
            "Maximum_Iterations_Exceeded",
        }

        if status not in acceptable_statuses:
            return SolveStep(
                ok=False,
                status=status,
                solve_time_s=solve_time_s,
                u=np.zeros(nu),
                progress_index=int(np.floor(self.mu)),
            )

        self.prev_z = sol["x"]

        z = sol["x"].full().flatten()
        offset = 0
        _X_opt = z[offset:offset + self.mpcc_data.nX].reshape((nx, N + 1), order="F")
        offset += self.mpcc_data.nX
        U_opt = z[offset:offset + self.mpcc_data.nU].reshape((nu, N), order="F")
        offset += self.mpcc_data.nU
        s_opt = z[offset:offset + self.mpcc_data.nProg].reshape((N + 1,), order="F")

        v, omega = U_opt[:, 0]
        self.u_prev = np.array([v, omega])

        solver_next_state = np.array(_X_opt[:, 1], dtype=float)
        rollout_next_state = np.array([
            x_current[0] + dt * v * np.cos(x_current[2]),
            x_current[1] + dt * v * np.sin(x_current[2]),
            x_current[2] + dt * omega,
        ], dtype=float)
        state_mismatch_norm = float(np.linalg.norm(solver_next_state - rollout_next_state))

        delta_s = float(s_opt[1]) - float(s_opt[0])
        self.mu = float(np.clip(self.mu + delta_s, 0.0, self.ref_traj.shape[1] - 1))

        
        return SolveStep(
            ok=True,
            status=status,
            solve_time_s=solve_time_s,
            u=np.array([v, omega]),
            progress_index=int(np.floor(self.mu)),
            solver_next_state=solver_next_state,
            rollout_next_state=rollout_next_state,
            state_mismatch_norm=state_mismatch_norm,
            active_dyn_count=len(active_dyn_obs),
        )

# ============================================================
# Shared experiment runner
# ============================================================

def run_experiment(
    controller_type: str,
    path_id: int,
    seed_offset: int,
    max_steps: int = 800,
    use_hard: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Unified experiment runner.

    controller_type:
        "mpc"  -> requires use_hard=True/False
        "mpcc" -> requires use_hard=True/False at runner level; actual
                  solver internals only differ if mpcc.py supports it
    """
    controller_type = controller_type.lower()
    if controller_type not in {"mpc", "mpcc"}:
        raise ValueError("controller_type must be 'mpc' or 'mpcc'")

    env = Obstacle(path_id)
    ref_traj = env.path_selector(path_id)
    static_rects = env.static_obs()
    dyn_obs = env.dynamic_obs_seeded(seed_offset)

    

    straight_line_dist = float(np.linalg.norm(ref_traj[:, -1] - ref_traj[:, 0]))

    p0 = ref_traj[:, 0]
    p1 = ref_traj[:, 1]
    th0 = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    x_current = np.array([p0[0], p0[1], th0])
    X_goal_global = np.array([ref_traj[0, -1], ref_traj[1, -1], 0.0])

    if controller_type == "mpc":
        if use_hard is None:
            raise ValueError("For MPC runs, pass use_hard=True or False")

        controller = MPCController(
            static_rects,
            dyn_obs,
            ref_traj,
            use_hard=use_hard,
        )

    elif controller_type == "mpcc":
        if use_hard is None:
            raise ValueError("For MPCC runs, pass use_hard=True or False")

        controller = MPCCController(
            static_rects,
            dyn_obs,
            ref_traj,
            use_hard=use_hard,
        )

    cont_errors = []
    lag_errors = []
    solve_times = []
    omega_hist = []
    v_hist = []
    path_length = 0.0
    prev_pos = x_current[:2].copy()

    dyn_body_count = 0
    dyn_exclusion_count = 0
    static_body_count = 0
    near_miss_count = 0
    min_clearance = np.inf
    danger_zone_count = 0
    solver_fail_count = 0

    goal_reached = False
    steps_taken = 0
    _at_end_count = 0

    for k in range(max_steps):
        step_result = controller.solve(x_current, dyn_obs, X_goal_global)
        solve_times.append(step_result.solve_time_s)

        if step_result.status != "Solve_Succeeded":
            solver_fail_count += 1

        if not step_result.ok:
            print(f"❌ Solver failed at step {k}: {step_result.status}")
            break

        v, omega = step_result.u
        v_hist.append(float(v))
        omega_hist.append(float(omega))

        x_current = np.array([
            x_current[0] + dt * v * np.cos(x_current[2]),
            x_current[1] + dt * v * np.sin(x_current[2]),
            x_current[2] + dt * omega,
        ])
        steps_taken += 1
        path_length += float(np.linalg.norm(x_current[:2] - prev_pos))
        prev_pos = x_current[:2].copy()

        for obs in dyn_obs:
            obs.step(dt, static_rects)

        step_near_miss = False
        step_danger_zone = False

        for obs in dyn_obs:
            body_level = _rotated_ellipse_level(
                x_current[0], x_current[1],
                obs.x, obs.y,
                obs.a, obs.b,
                obs.theta,
            )
            if body_level <= 1.0:
                dyn_body_count += 1

            a_e = obs.a + r_robot + safety_buffer
            b_e = obs.b + r_robot + safety_buffer
            excl_level = _rotated_ellipse_level(
                x_current[0], x_current[1],
                obs.x, obs.y,
                a_e, b_e,
                obs.theta,
            )
            if excl_level <= 1.0:
                dyn_exclusion_count += 1

            clearance = _rotated_ellipse_surface_dist(
                x_current[0], x_current[1],
                obs.x, obs.y,
                obs.a, obs.b,
                obs.theta,
            )
            min_clearance = min(min_clearance, clearance)

            if clearance < NEAR_MISS_MARGIN:
                step_near_miss = True
            if clearance < NEAR_MISS_MARGIN + 2 * safety_buffer:
                step_danger_zone = True

        for (cx, cy, hw, hh) in static_rects:
            if abs(x_current[0] - cx) < hw + r_robot and abs(x_current[1] - cy) < hh + r_robot:
                static_body_count += 1

            dx_s = max(abs(x_current[0] - cx) - hw, 0.0)
            dy_s = max(abs(x_current[1] - cy) - hh, 0.0)
            clearance_s = np.sqrt(dx_s ** 2 + dy_s ** 2) - r_robot
            min_clearance = min(min_clearance, clearance_s)

            if clearance_s < NEAR_MISS_MARGIN:
                step_near_miss = True
            if clearance_s < NEAR_MISS_MARGIN + 2 * safety_buffer:
                step_danger_zone = True

        if step_near_miss:
            near_miss_count += 1
        if step_danger_zone:
            danger_zone_count += 1

        cont_err, lag_err = _compute_common_tracking_errors(x_current, ref_traj)
        cont_errors.append(cont_err)
        lag_errors.append(lag_err)

        progress_index = step_result.progress_index
        path_complete = progress_index >= ref_traj.shape[1] - 2
        dist_to_goal = np.linalg.norm(x_current[:2] - X_goal_global[:2])

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

    cont_arr = np.abs(np.array(cont_errors)) if cont_errors else np.array([0.0])
    lag_arr = np.array(lag_errors) if lag_errors else np.array([0.0])
    st_arr = np.array(solve_times) * 1000 if solve_times else np.array([0.0])
    v_arr = np.array(v_hist) if v_hist else np.array([0.0])
    om_arr = np.abs(np.array(omega_hist)) if omega_hist else np.array([0.0])

    path_efficiency = round(straight_line_dist / path_length, 4) if path_length > 0 else 0.0

    result = {
        "path_id": path_id,
        "seed_offset": seed_offset,
        "goal_reached": int(goal_reached),
        "steps_taken": steps_taken,
        "completion_time_s": round(steps_taken * dt, 2),

        "path_length": round(path_length, 4),
        "path_efficiency": path_efficiency,
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
        "solver_fail_count": solver_fail_count,

        "dyn_body_collisions": dyn_body_count,
        "static_body_collisions": static_body_count,
        "total_body_collisions": dyn_body_count + static_body_count,
        "dyn_exclusion_violations": dyn_exclusion_count,
        "near_miss_count": near_miss_count,
        "danger_zone_steps": danger_zone_count,
        "min_clearance_m": round(float(min_clearance), 4),
    }

    if controller_type == "mpc":
        result["controller"] = f"mpc_{'hard' if use_hard else 'soft'}"
    else:
        result["controller"] = f"mpcc_{'hard' if use_hard else 'soft'}"

    return result