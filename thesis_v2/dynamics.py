# Unicycle dynamics
import casadi as ca
from params import *

def unicycle_dynamics(x, u):
    v = u[0]
    omega = u[1]
    return ca.vertcat(
        x[0] + dt * v * ca.cos(x[2]),
        x[1] + dt * v * ca.sin(x[2]),
        x[2] + dt * omega
    )