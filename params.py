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
N = 20

nx = 3
nu = 2
nr = 2


# -----------------------------
# Reference trajectory
# -----------------------------
t_ref = np.arange(0, 2 * np.pi, dt)
x_ref = t_ref
y_ref = np.sin(t_ref)

t_ref_xy = np.arange(0, 10, dt)
x_straight_ref = t_ref_xy
y_straight_ref = t_ref_xy

x_ref = x_straight_ref
y_ref = y_straight_ref
ref_traj = np.vstack((x_straight_ref, y_straight_ref))

# ref_traj = np.vstack((x_ref, y_ref))  # 2 x T


# Cost function weights
Q = np.diag([10.0, 10.0])          # tracking error weight (x,y only)
R_u = np.diag([0.01, 0.01])       # control effort weight (v, omega)


# -----------------------------
# MPC loop + live plot
# -----------------------------
# x_current = np.array([0.0, 0.0, 0.0])

x_current = np.array([
    ref_traj[0, 0],
    ref_traj[1, 0],
    0.0
], dtype=float)




# -----------------------------
# Robot circle
# -----------------------------
r_robot = 0.25   # meters (example)


# -----------------------------
# Obstacle
# -----------------------------
num_obs = 8
obs_pos = np.zeros((num_obs, 2))
obs_vel = np.zeros((num_obs, 2))
change_timer = np.zeros(num_obs, dtype=int)
change_horizon = 40

#Ellipse obs semi-principal axis def
a_obs = 0.6
b_obs = 0.3



obs_pos_hist = []



##CONFIGURABLE PARAMS
#Safety buffer for the robot, this inflates the minkowsky sum further
safety_buffer = 0.2

# Obstacle motion limits
v_obs_max = 0.8          # 
a_obs_max = 0.5          # m/s^2  (MATCH robot order)
omega_obs_max = 0.6      # rad/s  


# Control bounds for robot
v_max = 1.0
omega_max = 2.0
##CONFIGURABLE PARAMS



# --- Override initial condition to start exactly on the reference ---
x_current = np.array([ref_traj[0, 0], ref_traj[1, 0], 0.0], dtype=float)

# ---- BEFORE LOOP: init histories (important)
x_history = [x_current[0]]
y_history = [x_current[1]]
theta_history = [x_current[2]]




# -----------------------------
# Symbolic variables
# -----------------------------
X = ca.SX.sym("X", nx, N + 1)
U = ca.SX.sym("U", nu, N)
X0 = ca.SX.sym("X0", nx)
R = ca.SX.sym("R", nr, N + 1)

# Obstacle parameters (as CasADi symbols, because they are in p)
obs_x = ca.SX.sym("obs_x", num_obs, N)
obs_y = ca.SX.sym("obs_y", num_obs, N)

S = ca.SX.sym("S", num_obs, N)
u_prev = ca.SX.sym("u_prev", nu)
X_goal = ca.SX.sym("X_goal", nx)




# -----------------------------
# Static walls (corridor)
# -----------------------------
WALL_Y_MIN = -2.75   # bottom wall
WALL_Y_MAX =  2.75   # top wall
