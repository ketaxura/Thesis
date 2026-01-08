import casadi as ca
import numpy as np


# -----------------------------
# History buffers
# -----------------------------
x_history = []
y_history = []
x_ref_history = []
y_ref_history = []

# -----------------------------
# MPC Params
# -----------------------------
dt = 0.01
N = 10

nx = 3
nu = 2
nr = 2

# -----------------------------
# Symbolic variables
# -----------------------------
X = ca.SX.sym("X", nx, N + 1)
U = ca.SX.sym("U", nu, N)
X0 = ca.SX.sym("X0", nx)
R = ca.SX.sym("R", nr, N + 1)


# -----------------------------
# Reference trajectory
# -----------------------------
t_ref = np.arange(0, 2 * np.pi, dt)
x_ref = t_ref
y_ref = np.sin(t_ref)
ref_traj = np.vstack((x_ref, y_ref))  # 2 x T


# Cost function weights
Q = np.diag([30.0, 30.0])          # tracking error weight (x,y only)
R_u = np.diag([0.01, 0.01])        # control effort weight (v, omega)


# -----------------------------
# MPC loop + live plot
# -----------------------------
x_current = np.array([0.0, 0.0, 0.0])