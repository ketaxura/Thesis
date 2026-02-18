import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

from params import *
import casadi as ca

from dynamics import unicycle_dynamics



# -----------------------------
# MPCC tuning knobs (start here)
# -----------------------------
q_cont = 50.0          # contouring error weight (main path-following term)
q_lag  = 1.0  # lag error weight (prevents falling behind)
# q_vs   = 5.0  # Stronger reward on progress rate vs (push forward)
q_s_terminal = 75.0      # reward on terminal progress s_N

q_goal = 20.0           # terminal goal xy weight
q_terminal_cont = 30.0  # terminal contouring weight

r_v = 0.01
r_w = 0.01

w_v = 0.1
w_omega = 0.1

r_dv = 2.0
r_dw = 2.0

rho_vs = 0.5  # Much smaller - don't over-penalize progress rate

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
s = ca.SX.sym("s", N + 1, 1)                # progress

X0 = ca.SX.sym("X0", nx)
u_prev = ca.SX.sym("u_prev", nu)
s_prev = ca.SX.sym("s_prev", 1)


R = ca.SX.sym("R", nr, N + 1)               # horizon ref points


"""
g_list is a long python that contains all the constraint expressions
lbg 
ubg 
"""

# -----------------------------
# Constraints g(x,p)
# -----------------------------
g_list = []                     
lbg = []
ubg = []



"""
This expression below, assigns the first predicted state to equal the real measured state
The actual expression is X-X0
But because the lower and upper bounds are exactly zero
This means that each component of X[:, 0] is being forced to be exactly the same as each component of X0

X is the state vector 
X = [x]
    [y]
    [theta]
    
X0 is the initial state vector 
X0  =   [x0]
        [y0]
        [theta0]
"""
# Initial state
g_list.append(X[:,0]-X0)
lbg += [0.0, 0.0, 0.0]
ubg += [0.0, 0.0, 0.0]



"""
s is defined as a casadi variable that tracks the progress along the reference path
it is a N+1 dimensional column vector

s_prev is parameter that is passed from the previous MPC iteration, the previous progress

g_list.append(s[0] - s_prev) does this 
s0 - s_prev
and since the upper and lower bounds are 0

This means that the solver is forced to always pick s0 such that it equals s_prev
"""
# Initial progress anchored to previous progress
g_list.append(s[0] - s_prev)
lbg.append(0.0)
ubg.append(0.0)




"""
There is a for loop here because the conditions within loop have to be met and satisfied throughout the entire horizon.
And the reason why the previous two conditions were outside of the loop is because they are initial boundary conditions.
So these conditions:
    g_list.append(X[:,0]-X0)
    g_list.append(s[0] - s_prev)
    
Always have to come before the for loop conditions, they structure constraints such that one horizon leads into the next 
horizon smoothly and without error and while mainting progress.

"""

# Dynamics + path constraints per step
for k in range(N):
    
    
    """
    This constraint is the dynamic constraint
    At step k, the state vector values for k+1 must be from the dynamics equation. 
    Essentially the robot is not allowed to instantly teleport, but follow the dynamics equation and make small movements.
    
    And because both bounds are zero, the state k+1 has to be equal to the output of dynamics equation
    So it has to obey physics
    """
    # Robot dynamics
    g_list.append(X[:, k + 1] - unicycle_dynamics(X[:, k], U[:, k]))
    lbg += [0.0, 0.0, 0.0]
    ubg += [0.0, 0.0, 0.0]
    
    
    """
    State transition progress hard constraint
    
    s is a symbolic variable that the solver chooses, because of this to keep choosing in a realistic manner
    We are imbuing a state transtion progress, where future progress s[k+1] is a function of:
    current progress s[k] and current rate of progress vs[k] 
    """
    # # Progress dynamics
    # g_list.append(s[k + 1] - (s[k] + vs[k] * dt))
    # lbg.append(0.0)
    # ubg.append(0.0)


    # Compute reference frames
    r_k, t_k, n_k = soft_ref_and_frames(R, s[k])

    theta_k = X[2, k]

    # heading unit vector
    heading_vec = ca.vertcat(ca.cos(theta_k), ca.sin(theta_k))

    # tangential speed in m/s (can be negative if facing backward)
    v_par_mps = U[0, k] * ca.dot(t_k, heading_vec)

    # meters per waypoint-index (for your straight line it’s constant)
    dR0 = R[:, 1] - R[:, 0]
    ds_ref = ca.sqrt(ca.dot(dR0, dR0) + 1e-9)     # [m/index]

    # convert to index/s
    v_par_idx = v_par_mps / ds_ref               # [index/s]


    
    """
    Forward/positive progress soft constraint
    vs[k] is a rate of progress that we choose
    This term vs[k] can be at its lowest 0.05 and the highest it can be is infinity
    
    So this constraint keeps the solver from making negative or completely zero progress
    """
    # # Enforce forward progress
    # g_list.append(vs[k])
    # lbg.append(0.0)    # minimum progress rate
    # ubg.append(ca.inf)

    # # enforce: s[k+1] >= s[k] + dt * v_par_idx   (lower bound)
    # g_list.append(s[k+1] - s[k] - dt * v_par_idx)
    # lbg.append(0.0)
    # ubg.append(ca.inf)

    # # optional: also cap how fast progress can increase
    # g_list.append(s[k+1] - s[k])
    # lbg.append(0.0)
    # ubg.append(dt * (v_max / ds_ref))


    # Convert meters/sec to index/sec
    v_par_idx = v_par_mps / ds_ref

    g_list.append(s[k + 1] - (s[k] + dt * v_par_idx))
    lbg.append(0.0)
    ubg.append(0.0)






"""
Slew constraints on U

This is a soft constraint on how much one control input at k+1 be different from k

Remember U is a control input with two components in our system, v linear velocity and w(omega) angular velocity
g_list.append(U[0, k + 1] - U[0, k]), constraint on how much linear velocity can increase/decrease
g_list.append(U[1, k + 1] - U[1, k]), constraint on how much angular velocity can increase/decrease

dv_max = linear velocity slew limit
dw_max = angular velocity slew limit
"""
# Slew constraints on U
for k in range(N - 1):
    g_list.append(U[0, k + 1] - U[0, k])
    lbg.append(-dv_max)
    ubg.append(dv_max)

    g_list.append(U[1, k + 1] - U[1, k])
    lbg.append(-dw_max)
    ubg.append(dw_max)
    
"""
Slew constraints on initial boundary condition U

In a receding horizon method like MPC/MPCC we take the first input of the U vector and then apply it,
and then we solve the problem again. This constraint below is a soft constraint that makes sure that the 
first control input u that we are applying to the system doesnt jump too far away from the first control 
input that was applied last problem solve.

Remember U is a control input with two components in our system, v linear velocity and w(omega) angular velocity
g_list.append(U[0, k + 1] - U[0, k]), constraint on how much linear velocity can increase/decrease
g_list.append(U[1, k + 1] - U[1, k]), constraint on how much angular velocity can increase/decrease

dv_max = linear velocity slew limit
dw_max = angular velocity slew limit
"""

# First-step slew vs previous applied input
g_list.append(U[:, 0] - ca.reshape(u_prev, 2, 1))
lbg += [-dv_max, -dw_max]
ubg += [ dv_max,  dw_max]



"""
So the NLP problem that we are trying to solve here doesnt accept a python list of constraint conditions, it requires 
a single casadi object.  
"""


cost = 0

for k in range(N):

    p_xy = X[0:2, k]

    # Compute tangent and heading
    r_k, t_k, n_k = soft_ref_and_frames(R, s[k])

    theta_k = X[2, k]
    heading_vec = ca.vertcat(ca.cos(theta_k), ca.sin(theta_k))

    # Tangential velocity
    v_par = U[0, k] * ca.dot(t_k, heading_vec)


    e = p_xy - r_k

    e_cont = ca.dot(n_k, e)


    e_lag  = ca.dot(t_k, e)

    cost += q_cont * (e_cont**2)
    cost += q_lag * (e_lag**2)  # Prevent falling behind
    # cost += -q_vs * vs[k]

    # cost += w_v * U[0, k]**2
    # cost += w_omega * U[1, k]**2


    cost += r_v * U[0,k]**2 + r_w * U[1,k]**2

    if k > 0:
        cost += r_dv * (U[0,k] - U[0,k-1])**2
        cost += r_dw * (U[1,k] - U[1,k-1])**2




e_cont_fun = ca.Function(
    'e_cont_fun',
    [X, s, R],   # inputs
    [e_cont]     # output
)

cost += -q_s_terminal * s[N]


g = ca.vertcat(*g_list)


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
    # lbx += [ 0.0, -omega_max]
    lbx += [ -v_max, -omega_max]
    ubx += [ v_max,  omega_max]

# # S bounds (>=0)
# for _ in range(num_dyn_obs * N):
#     lbx.append(0.0)
#     ubx.append(ca.inf)

# s bounds (LOCAL horizon index)
for _ in range(N + 1):
    lbx.append(0.0)
    ubx.append(float(ref_traj.shape[1] - 1)) 



# for _ in range(N + 1):
#     lbx.append(0.0)
#     ubx.append(N)


# # vs bounds
# for _ in range(N):
#     lbx.append(0.0)
#     ubx.append(vs_max)



# -----------------------------
# Pack decision vars
# -----------------------------
OPT_vars = ca.vertcat(
    ca.reshape(X, -1, 1),
    ca.reshape(U, -1, 1),
    # ca.reshape(S, -1, 1),
    ca.reshape(s, -1, 1)
)

# -----------------------------
# NLP
# -----------------------------
p = ca.vertcat(
    X0,
    u_prev,
    s_prev,
    ca.reshape(R, -1, 1),
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



r_sym, t_sym, n_sym = soft_ref_and_frames(R, s[0])

ref_eval_fun = ca.Function(
    "ref_eval_fun",
    [R, s[0]],
    [r_sym, t_sym, n_sym]
)
