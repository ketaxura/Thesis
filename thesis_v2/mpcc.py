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
q_cont = 300.0          # contouring error weight (main path-following term)
q_lag  = 200.0           # lag error weight (prevents weird orthogonal sticking)
q_vs   = 1.0           # reward on progress rate vs (push forward)
q_s_terminal = 1.0      # reward on terminal progress s_N

q_du_v = 2.0            # smooth v changes
q_du_w = 20.0            # smooth omega changes

q_dvs = 50.0

q_goal = 20.0           # terminal goal xy weight
q_terminal_cont = 30.0  # terminal contouring weight

rho_slack = 2e4
rho_slack_lin = 200



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
vs = ca.SX.sym("vs", N, 1)                  # progress rate


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
    # Progress dynamics
    g_list.append(s[k + 1] - (s[k] + vs[k] * dt))
    lbg.append(0.0)
    ubg.append(0.0)
    
    
    """
    Forward/positive progress soft constraint
    vs[k] is a rate of progress that we choose
    This term vs[k] can be at its lowest 0.05 and the highest it can be is infinity
    
    So this constraint keeps the solver from making negative or completely zero progress
    """
    # Enforce forward progress
    g_list.append(vs[k])
    lbg.append(0.0)    # minimum progress rate
    ubg.append(ca.inf)




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
g = ca.vertcat(*g_list)




cost = 0

for k in range(N):

    p_xy = X[0:2, k]

    r_k, t_k, n_k = soft_ref_and_frames(R, s[k])

    e = p_xy - r_k
    e_cont = ca.dot(n_k, e)
    e_lag  = ca.dot(t_k, e)

    cost += q_cont * (e_cont**2)
    cost += 5.0 * (e_lag**2)      # small lag penalty
    cost += -q_vs * vs[k]




# # -----------------------------
# # Cost
# # -----------------------------
# cost = 0

# # Stage costs
# for k in range(N):
    
#     """
#     p_xy is the x and y coordinates of the robot at step k
#     """
#     p_xy = X[0:2, k]                        # (2,)
    
    
    
#     """
    
#     """
#     r_k, t_k, n_k = soft_ref_and_frames(R, s[k])

#     e = p_xy - r_k                          # (2,1)
#     e_lag = ca.dot(t_k, e)                  # scalar
#     e_cont = ca.dot(n_k, e)                 # scalar

#     cost += q_cont * (e_cont**2) + q_lag * (e_lag**2)

#     # Control effort
#     cost += ca.mtimes([U[:, k].T, R_u, U[:, k]])

#     # # Encourage forward progress
#     # cost += -q_vs * vs[k]

#     # Penalize oscillations in progress
#     if k == 0:
#         ds = vs[0]
#     else:
#         ds = vs[k] - vs[k-1]

#     cost += q_dvs * ds**2


#     # # Slack penalties
#     # for i in range(num_dyn_obs):
#     #     cost += rho_slack * (S[i, k] ** 2) + rho_slack_lin * S[i, k]

#     # Smooth inputs (soft, in addition to hard slew constraints)
#     if k == 0:
#         dv = U[0, 0] - u_prev[0]
#         dw = U[1, 0] - u_prev[1]
#     else:
#         dv = U[0, k] - U[0, k - 1]
#         dw = U[1, k] - U[1, k - 1]
#     cost += q_du_v * (dv**2) + q_du_w * (dw**2)

# # Terminal tracking / goal
# pN = X[0:2, N]
# rN, tN, nN = soft_ref_and_frames(R, s[N])
# eN = pN - rN
# eN_cont = ca.dot(nN, eN)

# cost += q_terminal_cont * (eN_cont**2)

# goal_err = pN - X_goal[0:2]
# cost += q_goal * ca.dot(goal_err, goal_err)

# # Reward being “far along” the horizon
# cost += -q_s_terminal * s[N]


cost += -10.0 * s[N]




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
    
    
# for _ in range(N + 1):
#     lbx.append(0.0)
#     ubx.append(N)


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
