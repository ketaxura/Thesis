from params import *
import casadi as ca
from dynamics import *
import time

# -----------------------------
# Cost
# ----------------------------
cost = 0
for k in range(N):
    x_err = X[0:2, k] - R[0:2, k]
    cost += ca.mtimes([x_err.T, Q, x_err])
    cost += ca.mtimes([U[:, k].T, R_u, U[:, k]])


# -----------------------------
# Constraints SECTION START
# -----------------------------
g_list = []
lbg = []
ubg = []

Q_terminal = 20 * np.eye(2)

#what is the eN?
eN = X[0:2, N] - X_goal[0:2]

cost += ca.mtimes([eN.T, Q_terminal, eN])

rho_slack = 1e5
rho_slack_lin = 1e3

for i in range(num_obs):
    for k in range(N):
        cost += rho_slack * S[i, k]**2
        cost += rho_slack_lin * S[i, k]
        



a_eff = a_obs + r_robot + safety_buffer
b_eff = b_obs + r_robot + safety_buffer

# Initial condition
g_list.append(X[:, 0] - X0)         #X[] is a casadi decision variable, X0 is the initial observed state
                                    #X[:, 0] - X0  is saying that the difference between the first predicted X state value by the solver and the first initial observed values have to be exactly the same. This is hard inequality constraint
                                    
lbg += [0, 0, 0]                    #ubg and lbg being zero means that we want X[:,0]=X0 to be exact
ubg += [0, 0, 0]
#this hard constraint is critical because we dont want the solver to naively think the robot starts someplace it doesnt
#ofc, lbg and ubg here are 3 terms because the state vector X in our problem is a n=3 vector

# print(lbg)
# time.sleep(132312312)



#we are projecting the robot dynamics over the horizon
#and checking the dynamic elliptical obstacle avoidance constraint for each dyn obs at each horizon step, so here we can see that our solver can "see" with perfect clarity all dyn obs relative position throughout the entire horizon
for k in range(N):
    # dynamics
    g_list.append(X[:, k+1] - unicycle_dynamics(X[:, k], U[:, k]))      #this is the x+1=x+u constraint, a hard constraint that tells the solver over the receding horizon the solver must respect the dynamics constraint at all times
    lbg += [0, 0, 0]
    ubg += [0, 0, 0]
    
    # -----------------------------
    # Static corridor walls
    # -----------------------------
    y_min = WALL_Y_MIN + r_robot + safety_buffer
    y_max = WALL_Y_MAX - r_robot - safety_buffer

    # Bottom wall: y >= y_min
    g_list.append(X[1, k+1] - y_min)
    lbg.append(0.0)
    ubg.append(ca.inf)

    # Top wall: y <= y_max
    g_list.append(y_max - X[1, k+1])
    lbg.append(0.0)
    ubg.append(ca.inf)


    for i in range(num_obs):
        dx = X[0, k+1] - obs_x[i, k]        #checking each 
        dy = X[1, k+1] - obs_y[i, k]

        # OUTSIDE ellipse condition
        g_list.append(
            dx**2 / a_eff**2 + dy**2 / b_eff**2 - 1 + S[i, k]     #slack enabled
            # dx**2 / a_eff**2 + dy**2 / b_eff**2 - 1                 #slack not allowed
        )
        lbg.append(0.0)
        ubg.append(ca.inf)
        
    



# -----------------------------
# Actuation rate limits (slew)
# -----------------------------
dv_max = 0.05   # m/s per step
dw_max = 0.20   # rad/s per step

for k in range(N - 1):
    # v_{k+1} - v_k
    g_list.append(U[0, k+1] - U[0, k])
    lbg.append(-dv_max)
    ubg.append(dv_max)

    # w_{k+1} - w_k
    g_list.append(U[1, k+1] - U[1, k])
    lbg.append(-dw_max)
    ubg.append(dw_max)
    
    
    
# Slew constraint from previous applied control -> first input in horizon
g_list.append(U[:, 0] - u_prev)
lbg += [-dv_max, -dw_max]
ubg += [ dv_max,  dw_max]
#The ubg, lbg here only have two terms because the control input is n=2 dim vector






#All constraints must be pack into g_list before the line below
g = ca.vertcat(*g_list)


# -----------------------------
# Constraints SECTION END
# -----------------------------




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
    # lbx += [0, -omega_max]        #no reversing
    lbx += [-v_max, -omega_max]   #reversing allowed
    ubx += [ v_max,  omega_max]



# Slack bounds (S >= 0) 
for _ in range(num_obs * N):
    lbx.append(0.0)
    ubx.append(ca.inf)


# -----------------------------
# Decision variables + NLP
# -----------------------------
OPT_vars = ca.vertcat(
    ca.reshape(X, -1, 1),
    ca.reshape(U, -1, 1),
    ca.reshape(S, -1, 1)
)

nlp = {
    "x": OPT_vars,                  #decision variables, all of the variables the solver is able to manipulate and change
    "f": cost,                      #cost function, a static function that has our optimization objectives
    "g": g,                         #constraints, all of the constraints/inequalities that must be met
    "p": ca.vertcat(                #the parameter vector, all of the variables that are dynamic and not changeable, but is pivotal to optimization 
        X0,
        u_prev,
        ca.reshape(R, -1, 1),
        ca.reshape(obs_x, -1, 1),
        ca.reshape(obs_y, -1, 1),
        X_goal
    )
}


solver = ca.nlpsol(
    "solver",
    "ipopt",
    nlp,
    {
        "ipopt.print_level": 0,
        "ipopt.warm_start_init_point": "yes",
        "ipopt.mu_init": 1e-2,
        "ipopt.max_iter": 60,
        "ipopt.tol": 1e-2,
        "ipopt.acceptable_tol": 1e-1,
        "ipopt.acceptable_iter": 3,
        "print_time": 0
    }
)


