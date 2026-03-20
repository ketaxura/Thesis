from dataclasses import dataclass
import casadi as ca
import numpy as np

from params import *
from dynamics import unicycle_dynamics


@dataclass
class MPCCConfig:
    q_cont: float = 6.0
    q_lag: float = 0.5
    q_theta: float = 5.0
    q_goal: float = 100.0
    rho_obs: float = 1e4
    rho_dyn: float = 1e5
    q_vs: float = 20.0
    q_s_terminal: float = 3.0
    r_v: float = 0.1
    r_w: float = 0.5
    r_dv: float = 2.0
    r_dw: float = 4.0


@dataclass
class MPCCArtifacts:
    solver: any
    lbx: list
    ubx: list
    lbg: list
    ubg: list
    nX: int
    nU: int
    nProg: int
    nSlack_static: int
    nSlack_dyn: int
    ref_eval_fun: any
    e_cont_horizon_fun: any
    cont_cost_fun: any
    lag_cost_fun: any
    prog_cost_fun: any
    ctrl_cost_fun: any
    align_cost_fun: any


def soft_ref_and_frames(R, s_scalar, N_local, kappa_w_local, eps_local):
    idx = ca.DM(np.arange(N_local + 1)).reshape((N_local + 1, 1))

    diff = idx - s_scalar
    w_unn = ca.exp(-kappa_w_local * ca.power(diff, 2))
    w = w_unn / (ca.sum1(w_unn) + eps_local)

    r = ca.mtimes(R, w)

    t_sum = ca.SX.zeros(2, 1)
    for j in range(N_local):
        dR = R[:, j + 1] - R[:, j]
        nrm = ca.sqrt(ca.dot(dR, dR) + eps_local)
        t_j = dR / nrm
        wseg = 0.5 * (w[j] + w[j + 1])
        t_sum += wseg * t_j

    t_nrm = ca.sqrt(ca.dot(t_sum, t_sum) + eps_local)
    t = t_sum / t_nrm
    n = ca.vertcat(-t[1], t[0])

    return r, t, n


def build_mpcc_solver(
    ref_traj: np.ndarray,
    static_rects: list,
    dyn_obs: list,
    use_hard: bool = False,
    config: MPCCConfig | None = None,
) -> MPCCArtifacts:
    if config is None:
        config = MPCCConfig()

    if ref_traj.shape[0] != 2:
        raise ValueError(f"ref_traj must have shape (2, M), got {ref_traj.shape}")

    num_static_obs = len(static_rects)
    num_dyn_obs = len(dyn_obs)

    X = ca.SX.sym("X", nx, N + 1)
    U = ca.SX.sym("U", nu, N)
    s = ca.SX.sym("s", N + 1, 1)
    if not use_hard:
        S_dyn = ca.SX.sym("S_dyn", num_dyn_obs, N)

    X0 = ca.SX.sym("X0", nx)
    u_prev = ca.SX.sym("u_prev", nu)
    s_prev = ca.SX.sym("s_prev", 1)
    R = ca.SX.sym("R", nr, N + 1)
    X_goal = ca.SX.sym("X_goal", nx)
    obs_x = ca.SX.sym("obs_x", num_dyn_obs, N)     # predicted x at horizon steps 1..N
    obs_y = ca.SX.sym("obs_y", num_dyn_obs, N)     # predicted y at horizon steps 1..N
    obs_theta = ca.SX.sym("obs_theta", num_dyn_obs, N)  # predicted heading at horizon steps 1..N

    g_list = []
    lbg = []
    ubg = []

    g_list.append(X[:, 0] - X0)
    lbg += [0.0, 0.0, 0.0]
    ubg += [0.0, 0.0, 0.0]

    g_list.append(s[0] - s_prev)
    lbg.append(0.0)
    ubg.append(0.0)

    dR0 = R[:, 1] - R[:, 0]
    ds_ref = ca.sqrt(ca.dot(dR0, dR0) + eps)

    cost = 0

    for k in range(N):
        g_list.append(X[:, k + 1] - unicycle_dynamics(X[:, k], U[:, k]))
        lbg += [0.0, 0.0, 0.0]
        ubg += [0.0, 0.0, 0.0]

        r_k, t_k, n_k = soft_ref_and_frames(R, s[k], N, kappa_w, eps)

        theta_k = X[2, k]
        heading_vec = ca.vertcat(ca.cos(theta_k), ca.sin(theta_k))
        v_par = U[0, k] * ca.dot(t_k, heading_vec)

        g_list.append(s[k + 1] - (s[k] + dt * v_par / (ds_ref + 1e-9)))
        lbg.append(0.0)
        ubg.append(0.0)

        g_list.append(s[k + 1] - s[k])
        lbg.append(0.0)
        ubg.append(ca.inf)

        p_xy = X[0:2, k]
        e = p_xy - r_k
        e_cont = ca.dot(n_k, e)
        e_lag = ca.dot(t_k, e)

        cost += config.q_cont * (e_cont ** 2)
        cost += config.q_lag * ca.fmax(0, -e_lag) ** 2
        cost += config.r_v * U[0, k] ** 2 + config.r_w * U[1, k] ** 2
        cost += -config.q_vs * v_par

        for i, (cx, cy, hw, hh) in enumerate(static_rects):
            dx = p_xy[0] - cx
            dy = p_xy[1] - cy
            p_norm = 6
            obs_margin = r_robot + safety_buffer

            dist_p = (
                (ca.fabs(dx) / (hw + obs_margin)) ** p_norm
                + (ca.fabs(dy) / (hh + obs_margin)) ** p_norm
            )

            g_list.append(dist_p)
            lbg.append(1.0)
            ubg.append(ca.inf)



        # Dynamic obstacle avoidance — applied on X[:,k+1] (next state).
        # Ellipse is rotated by obs_theta so it aligns with the obstacle's
        # heading direction (major axis = direction of travel).
        p_xy_next = X[0:2, k + 1]
        for i in range(num_dyn_obs):
            ox = obs_x[i, k]
            oy = obs_y[i, k]
            oth = obs_theta[i, k]

            # Rotate offset into obstacle's local frame
            dx_w = p_xy_next[0] - ox
            dy_w = p_xy_next[1] - oy
            dx_local =  ca.cos(oth) * dx_w + ca.sin(oth) * dy_w
            dy_local = -ca.sin(oth) * dx_w + ca.cos(oth) * dy_w

            a = dyn_obs[i].a + r_robot + safety_buffer   # major (along heading)
            b = dyn_obs[i].b + r_robot + safety_buffer   # minor (perpendicular)

            dist = (dx_local / a) ** 2 + (dy_local / b) ** 2

            if use_hard:
                g_list.append(dist)
                lbg.append(1.0)
                ubg.append(ca.inf)
            else:
                slack = S_dyn[i, k]
                g_list.append(dist + slack)
                lbg.append(1.0)
                ubg.append(ca.inf)

                cost += config.rho_dyn * slack ** 2

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

    cost += -config.q_s_terminal * s[N]

    g = ca.vertcat(*g_list)

    lbx = []
    ubx = []

    for _ in range(N + 1):
        lbx += [-ca.inf, -ca.inf, -ca.inf]
        ubx += [ca.inf, ca.inf, ca.inf]

    for _ in range(N):
        lbx += [0.0, -omega_max]
        ubx += [v_max, omega_max]

    for _ in range(N + 1):
        lbx.append(0.0)
        ubx.append(float(N))

    if not use_hard:
        for _ in range(num_dyn_obs * N):
            lbx.append(0.0)
            ubx.append(1.0)

    # Static obstacles are always hard.
    # Dynamic obstacles are softened only in soft mode via S_dyn.

    if use_hard:
        OPT_vars = ca.vertcat(
            ca.reshape(X, -1, 1),
            ca.reshape(U, -1, 1),
            ca.reshape(s, -1, 1),
        )
    else:
        OPT_vars = ca.vertcat(
            ca.reshape(X, -1, 1),
            ca.reshape(U, -1, 1),
            ca.reshape(s, -1, 1),
            ca.reshape(S_dyn, -1, 1),
        )

    p_vec = ca.vertcat(
        X0,
        u_prev,
        s_prev,
        ca.reshape(R, -1, 1),
        X_goal,
        ca.reshape(obs_x, -1, 1),
        ca.reshape(obs_y, -1, 1),
        ca.reshape(obs_theta, -1, 1),
    )

    nlp = {"x": OPT_vars, "f": cost, "g": g, "p": p_vec}

    solver_name = "solver_mpcc_hard" if use_hard else "solver_mpcc_soft"

    solver = ca.nlpsol(
        solver_name,
        "ipopt",
        nlp,
        {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 120,
            "ipopt.tol": 1e-3,
            "ipopt.acceptable_tol": 1e-2,
            "ipopt.acceptable_iter": 5,
            "ipopt.warm_start_init_point": "yes",
            "print_time": 0,
        },
    )

    nX = nx * (N + 1)
    nU = nu * N
    nProg = N + 1
    nSlack_static = 0
    nSlack_dyn = 0 if use_hard else num_dyn_obs * N

    r_sym, t_sym, n_sym = soft_ref_and_frames(R, s[0], N, kappa_w, eps)
    ref_eval_fun = ca.Function("ref_eval_fun", [R, s[0]], [r_sym, t_sym, n_sym])

    e_cont_list = []
    cont_list = []
    lag_list = []
    prog_list = []
    ctrl_list = []
    align_list = []

    for k in range(N):
        r_k, t_k, n_k = soft_ref_and_frames(R, s[k], N, kappa_w, eps)
        theta_k = X[2, k]
        heading_vec = ca.vertcat(ca.cos(theta_k), ca.sin(theta_k))
        p_xy = X[0:2, k]
        e = p_xy - r_k

        e_cont = ca.dot(n_k, e)
        e_lag = ca.dot(t_k, e)
        v_par = U[0, k] * ca.dot(t_k, heading_vec)
        align = ca.dot(t_k, heading_vec)

        e_cont_list.append(e_cont)
        cont_list.append(config.q_cont * (e_cont ** 2))
        lag_list.append(config.q_lag * (e_lag ** 2))
        prog_list.append(-config.q_vs * v_par)
        ctrl_list.append(config.r_v * U[0, k] ** 2 + config.r_w * U[1, k] ** 2)
        align_list.append(config.q_theta * (1 - align) ** 2)

    e_cont_horizon_fun = ca.Function("e_cont_horizon_fun", [X, s, R], [ca.vertcat(*e_cont_list)])
    cont_cost_fun = ca.Function("cont_cost_fun", [X, U, s, R], [ca.vertcat(*cont_list)])
    lag_cost_fun = ca.Function("lag_cost_fun", [X, U, s, R], [ca.vertcat(*lag_list)])
    prog_cost_fun = ca.Function("prog_cost_fun", [X, U, s, R], [ca.vertcat(*prog_list)])
    ctrl_cost_fun = ca.Function("ctrl_cost_fun", [X, U, s, R], [ca.vertcat(*ctrl_list)])
    align_cost_fun = ca.Function("align_cost_fun", [X, U, s, R], [ca.vertcat(*align_list)])

    return MPCCArtifacts(
        solver=solver,
        lbx=lbx,
        ubx=ubx,
        lbg=lbg,
        ubg=ubg,
        nX=nX,
        nU=nU,
        nProg=nProg,
        nSlack_static=nSlack_static,
        nSlack_dyn=nSlack_dyn,
        ref_eval_fun=ref_eval_fun,
        e_cont_horizon_fun=e_cont_horizon_fun,
        cont_cost_fun=cont_cost_fun,
        lag_cost_fun=lag_cost_fun,
        prog_cost_fun=prog_cost_fun,
        ctrl_cost_fun=ctrl_cost_fun,
        align_cost_fun=align_cost_fun,
    )