# mpcc.py
# MPCC with scalar progress state s_k (NO W matrix).
#
# Parameter vector layout (matches your current p length = 211 for N=20, num_dyn_obs=4):
# p = [ X0(3),
#       u_prev(2),
#       s_prev(1),
#       vec(R_horizon)      (2*(N+1)),
#       vec(obs_x_horizon)  (num_dyn_obs*N),
#       vec(obs_y_horizon)  (num_dyn_obs*N),
#       X_goal(3) ]
#
# Decision variables:
#   X (3 x (N+1))
#   U (2 x N)
#   S (num_dyn_obs x N)      slack for dynamic obstacles
#   s (N+1)                  progress along horizon [0..N]
#   vs (N)                   progress rate >= 0

import casadi as ca
import numpy as np

from params import (
    dt, N, nx, nu, nr,
    R_u,
    v_max, omega_max,
    num_dyn_obs,
    a_obs, b_obs,
    r_robot, safety_buffer,
    WALL_Y_MIN, WALL_Y_MAX,
    STATIC_RECTS,
)

from dynamics import unicycle_dynamics


# -----------------------------
# MPCC tuning knobs (start here)
# -----------------------------
q_cont = 100.0          # contouring error weight (main path-following term)
q_lag  = 200.0           # lag error weight (prevents weird orthogonal sticking)
q_vs   = 15.0           # reward on progress rate vs (push forward)
q_s_terminal = 5.0      # reward on terminal progress s_N

q_du_v = 2.0            # smooth v changes
q_du_w = 20.0            # smooth omega changes

q_dvs = 50.0

q_goal = 20.0           # terminal goal xy weight
q_terminal_cont = 30.0  # terminal contouring weight

rho_slack = 2e4
rho_slack_lin = 200


# If your solver reports "feasible but infeasible", usually slack penalty too high
# or vs bounds too tight, or walls too tight. Adjust here if needed.

# Progress bounds
s_min = 0.0
s_max = float(N)        # since s is “index along the horizon”
vs_max = 2.0            # [index/sec]  -> ds = vs*dt. With dt=0.1, ds<=0.2 if vs_max=2

# Slew bounds (MATCH main.py dv_max/dw_max)
dv_max = 0.25
dw_max = 0.6


# Smooth soft “localization” of reference point around s_k
# Larger -> more “one segment only”, smaller -> more averaged reference (less jitter).
kappa_w = 1.5

eps = 1e-8


# -----------------------------
# Helper: soft reference r(s), tangent t(s), normal n(s)
# Uses a smooth weight vector over indices 0..N centered at s.
# -----------------------------
def soft_ref_and_frames(R, s_scalar):
    """
    R: (2 x (N+1)) CasADi
    s_scalar: scalar CasADi
    Returns:
      r (2x1), t (2x1 unit), n (2x1 unit)
    """
    idx = ca.DM(np.arange(N + 1)).reshape((N + 1, 1))


    # w_j = exp(-kappa*(j - s)^2), normalized
    diff = idx - s_scalar
    w_unn = ca.exp(-kappa_w * ca.power(diff, 2))
    w = w_unn / (ca.sum1(w_unn) + eps)         # (N+1,1)

    r = ca.mtimes(R, w)                         # (2,1)

    # Build a smooth tangent as weighted average of segment tangents
    t_sum = ca.SX.zeros(2, 1)
    for j in range(N):
        dR = R[:, j + 1] - R[:, j]              # (2,1)
        nrm = ca.sqrt(ca.dot(dR, dR) + eps)
        t_j = dR / nrm

        wseg = 0.5 * (w[j] + w[j + 1])          # scalar
        t_sum += wseg * t_j

    t_nrm = ca.sqrt(ca.dot(t_sum, t_sum) + eps)
    t = t_sum / t_nrm

    n = ca.vertcat(-t[1], t[0])
    return r, t, n


# -----------------------------
# Symbols
# -----------------------------
X = ca.SX.sym("X", nx, N + 1)
U = ca.SX.sym("U", nu, N)
S = ca.SX.sym("S", num_dyn_obs, N)          # slack for dynamic obstacles
s = ca.SX.sym("s", N + 1, 1)                # progress
vs = ca.SX.sym("vs", N, 1)                  # progress rate

X0 = ca.SX.sym("X0", nx)
u_prev = ca.SX.sym("u_prev", nu)
s_prev = ca.SX.sym("s_prev", 1)

R = ca.SX.sym("R", nr, N + 1)               # horizon ref points
obs_x = ca.SX.sym("obs_x", num_dyn_obs, N)
obs_y = ca.SX.sym("obs_y", num_dyn_obs, N)
X_goal = ca.SX.sym("X_goal", nx)


# -----------------------------
# Constraints g(x,p)
# -----------------------------
g_list = []
lbg = []
ubg = []

# Initial state
g_list.append(X[:, 0] - X0)
lbg += [0.0, 0.0, 0.0]
ubg += [0.0, 0.0, 0.0]

# Initial progress anchored to previous progress
g_list.append(s[0] - s_prev)
lbg.append(0.0)
ubg.append(0.0)

# Corridor bounds (inflate by robot + safety)
y_min = WALL_Y_MIN + r_robot + safety_buffer
y_max = WALL_Y_MAX - r_robot - safety_buffer

# Effective ellipse for dynamic obstacles (robot ⊕ obstacle ⊕ safety)
a_eff = a_obs + r_robot + safety_buffer
b_eff = b_obs + r_robot + safety_buffer

# Dynamics + path constraints per step
for k in range(N):
    # Robot dynamics
    g_list.append(X[:, k + 1] - unicycle_dynamics(X[:, k], U[:, k]))
    lbg += [0.0, 0.0, 0.0]
    ubg += [0.0, 0.0, 0.0]

    # Progress dynamics
    g_list.append(s[k + 1] - (s[k] + vs[k] * dt))
    lbg.append(0.0)
    ubg.append(0.0)


    # # --- compute average segment length (once per k) ---
    # ell_k = 0
    # for j in range(N):
    #     dR = R[:, j + 1] - R[:, j]
    #     ell_k += ca.sqrt(ca.dot(dR, dR))
    # ell_k = ell_k / N

    # # --- vs coupling constraint ---
    # g_list.append(U[0, k] - ell_k * vs[k])
    # lbg.append(0.0)
    # ubg.append(ca.inf)


    # Corridor walls
    g_list.append(X[1, k + 1] - y_min)   # y >= y_min
    lbg.append(0.0)
    ubg.append(ca.inf)

    # Enforce forward progress
    g_list.append(vs[k])
    lbg.append(0.05)    # minimum progress rate
    ubg.append(ca.inf)


    g_list.append(y_max - X[1, k + 1])   # y <= y_max
    lbg.append(0.0)
    ubg.append(ca.inf)

    # Dynamic ellipse obstacles with slack
    for i in range(num_dyn_obs):
        dx = X[0, k + 1] - obs_x[i, k]
        dy = X[1, k + 1] - obs_y[i, k]
        g_list.append(dx**2 / (a_eff**2) + dy**2 / (b_eff**2) - 1.0 + S[i, k])
        lbg.append(0.0)
        ubg.append(ca.inf)

    # Static rectangles (ellipse-approx)
    for (cx, cy, hx, hy) in STATIC_RECTS:
        hx_eff = hx + r_robot + safety_buffer
        hy_eff = hy + r_robot + safety_buffer
        dxr = X[0, k + 1] - cx
        dyr = X[1, k + 1] - cy
        g_list.append(dxr**2 / (hx_eff**2) + dyr**2 / (hy_eff**2) - 1.0)
        lbg.append(0.0)
        ubg.append(ca.inf)

# Slew constraints on U
for k in range(N - 1):
    g_list.append(U[0, k + 1] - U[0, k])
    lbg.append(-dv_max)
    ubg.append(dv_max)

    g_list.append(U[1, k + 1] - U[1, k])
    lbg.append(-dw_max)
    ubg.append(dw_max)

# First-step slew vs previous applied input
g_list.append(U[:, 0] - ca.reshape(u_prev, 2, 1))
lbg += [-dv_max, -dw_max]
ubg += [ dv_max,  dw_max]

g = ca.vertcat(*g_list)


# -----------------------------
# Cost
# -----------------------------
cost = 0

# Stage costs
for k in range(N):
    p_xy = X[0:2, k]                        # (2,)

    r_k, t_k, n_k = soft_ref_and_frames(R, s[k])

    e = p_xy - r_k                          # (2,1)
    e_lag = ca.dot(t_k, e)                  # scalar
    e_cont = ca.dot(n_k, e)                 # scalar

    cost += q_cont * (e_cont**2) + q_lag * (e_lag**2)

    # Control effort
    cost += ca.mtimes([U[:, k].T, R_u, U[:, k]])

    # Encourage forward progress
    cost += -q_vs * vs[k]

    # # Penalize oscillations in progress
    # if k == 0:
    #     ds = vs[0]
    # else:
    #     ds = vs[k] - vs[k-1]

    # cost += q_dvs * ds**2


    # # Slack penalties
    # for i in range(num_dyn_obs):
    #     cost += rho_slack * (S[i, k] ** 2) + rho_slack_lin * S[i, k]

    # Smooth inputs (soft, in addition to hard slew constraints)
    if k == 0:
        dv = U[0, 0] - u_prev[0]
        dw = U[1, 0] - u_prev[1]
    else:
        dv = U[0, k] - U[0, k - 1]
        dw = U[1, k] - U[1, k - 1]
    cost += q_du_v * (dv**2) + q_du_w * (dw**2)

# Terminal tracking / goal
pN = X[0:2, N]
rN, tN, nN = soft_ref_and_frames(R, s[N])
eN = pN - rN
eN_cont = ca.dot(nN, eN)

cost += q_terminal_cont * (eN_cont**2)

goal_err = pN - X_goal[0:2]
cost += q_goal * ca.dot(goal_err, goal_err)

# Reward being “far along” the horizon
cost += -q_s_terminal * s[N]


# -----------------------------
# Bounds (lbx/ubx)
# Decision vector packing order:
#   vec(X), vec(U), vec(S), vec(s), vec(vs)
# -----------------------------
lbx = []
ubx = []

# X bounds
for _ in range(N + 1):
    lbx += [-ca.inf, -ca.inf, -ca.inf]
    ubx += [ ca.inf,  ca.inf,  ca.inf]

# U bounds
for _ in range(N):
    lbx += [-v_max, -omega_max]
    ubx += [ v_max,  omega_max]

# S bounds (>=0)
for _ in range(num_dyn_obs * N):
    lbx.append(0.0)
    ubx.append(ca.inf)

# s bounds
for _ in range(N + 1):
    lbx.append(s_min)
    ubx.append(s_max)

# vs bounds
for _ in range(N):
    lbx.append(0.0)
    ubx.append(vs_max)


# -----------------------------
# Pack decision vars
# -----------------------------
OPT_vars = ca.vertcat(
    ca.reshape(X, -1, 1),
    ca.reshape(U, -1, 1),
    ca.reshape(S, -1, 1),
    ca.reshape(s, -1, 1),
    ca.reshape(vs, -1, 1),
)

# -----------------------------
# NLP
# -----------------------------
p = ca.vertcat(
    X0,
    u_prev,
    s_prev,
    ca.reshape(R, -1, 1),
    ca.reshape(obs_x, -1, 1),
    ca.reshape(obs_y, -1, 1),
    X_goal
)

nlp = {"x": OPT_vars, "f": cost, "g": g, "p": p}

solver = ca.nlpsol(
    "solver_mpcc",
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

# Export for main.py
# (same names your code expects)
# lbg/ubg already built above
# lbx/ubx already built above
# solver already built above

# Also export sizes so main.py can unpack cleanly
nX = nx * (N + 1)
nU = nu * N
nS = num_dyn_obs * N
nProg = (N + 1)          # s
nVs = N                  # vs
