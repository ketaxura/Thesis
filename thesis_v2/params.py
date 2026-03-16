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
dt = 0.1
N = 30

nx = 3
nu = 2
nr = 2

# Cost function weights
Q = np.diag([10.0, 10.0])
R_u = np.diag([0.01, 0.01])

# -----------------------------
# Robot
# -----------------------------
r_robot = 0.25

# -----------------------------
# Obstacle
# -----------------------------
num_dyn_obs = 0
obs_pos = np.zeros((num_dyn_obs, 2))
obs_vel = np.zeros((num_dyn_obs, 2))
change_timer = np.zeros(num_dyn_obs, dtype=int)
change_horizon = 40

a_obs = 0.6
b_obs = 0.3

# -----------------------------
# History Variables
# -----------------------------
obs_pos_hist = []
cont_err_hist = []
lag_err_hist = []
x_state_hist = []
u_hist = []
mu_hist = []

# -----------------------------
# Configurable Params
# -----------------------------
safety_buffer = 0.2

v_obs_max = 0.8
a_obs_max = 0.5
omega_obs_max = 0.6

v_max = 1.0
omega_max = 2.0

dv_max = 0.25
dw_max = 0.6

kappa_w = 1.5
eps = 1e-8

s_min = 0.0
vs_max = 2.0

# -----------------------------
# Symbolic variables
# -----------------------------
X = ca.SX.sym("X", nx, N + 1)
U = ca.SX.sym("U", nu, N)
X0 = ca.SX.sym("X0", nx)
R = ca.SX.sym("R", nr, N + 1)

obs_x = ca.SX.sym("obs_x", num_dyn_obs, N)
obs_y = ca.SX.sym("obs_y", num_dyn_obs, N)

S = ca.SX.sym("S", num_dyn_obs, N)
u_prev = ca.SX.sym("u_prev", nu)
X_goal = ca.SX.sym("X_goal", nx)

mu_prev = ca.SX.sym("mu_prev")

# -----------------------------
# Static walls
# -----------------------------
WALL_Y_MIN = -2.75
WALL_Y_MAX =  2.75

rng = np.random.default_rng(389999)


STATIC_RECTS=[]


# STATIC_RECTS = [
#     (7.0,  1.0, 0.6, 1.2),
#     (10.0, 0.0, 0.5, 0.5),
# ]

## NOTE: ref_traj, x_current, x_history, y_history, theta_history
# are all defined in main.py AFTER the path is set.
# Do NOT define them here.