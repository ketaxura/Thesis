from params import *
import casadi as ca
from dynamics import *

# -----------------------------
# Cost
# -----------------------------


cost = 0
for k in range(N):
    x_err = X[0:2, k] - R[0:2, k]
    cost += ca.mtimes([x_err.T, Q, x_err])
    cost += ca.mtimes([U[:, k].T, R_u, U[:, k]])

# Terminal cost
x_err_terminal = X[0:2, N] - R[0:2, N]
cost += ca.mtimes([x_err_terminal.T, Q, x_err_terminal])

# -----------------------------
# Constraints
# -----------------------------
g = []
g.append(X[:, 0] - X0)

for k in range(N):
    x_next = unicycle_dynamics(X[:, k], U[:, k])
    g.append(X[:, k + 1] - x_next)

# -----------------------------
# Bounds
# -----------------------------
lbx = []
ubx = []

# State bounds (N+1 states)
for _ in range(N + 1):
    lbx += [-ca.inf, -ca.inf, -ca.inf]
    ubx += [ ca.inf,  ca.inf,  ca.inf]

# Control bounds (N inputs)
v_max = 1.0
omega_max = 2.0
for _ in range(N):
    lbx += [-v_max, -omega_max]
    ubx += [ v_max,  omega_max]

# -----------------------------
# Decision variables + NLP
# -----------------------------
OPT_vars = ca.vertcat(
    ca.reshape(X, -1, 1),
    ca.reshape(U, -1, 1)
)

g = ca.vertcat(*g)

nlp = {
    "x": OPT_vars,
    "f": cost,
    "g": g,
    "p": ca.vertcat(X0, ca.reshape(R, -1, 1))
}

solver = ca.nlpsol(
    "solver",
    "ipopt",
    nlp,
    {
        "ipopt.print_level": 0,
        "print_time": 0
    }
)

