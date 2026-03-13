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
    

def path_selector(path):

    if(path==0):
        # Straight Diagonal line
        ref_x = np.linspace(0, -10, 100)
        ref_y = np.linspace(0, 0, 100)
        ref_traj = np.vstack((ref_x, ref_y))   # shape (2, 100)
        return ref_traj

    elif(path==1):
        # Sine wave from x=0 to x=20
        t = np.linspace(0, 20, 300)
        ref_x = t
        ref_y = 2.0 * np.sin(0.5 * t)   # amplitude=2, frequency=0.5
        ref_traj = np.vstack((ref_x, ref_y))
        return ref_traj

    elif(path==2):
        # Figure-8 (requires negative x, direction reversal at crossing)
        t = np.linspace(0, 2*np.pi, 400)
        ref_x = 10 * np.sin(t)
        ref_y = 5 * np.sin(2*t)
        ref_traj = np.vstack((ref_x, ref_y))
        return ref_traj

    elif(path==3):
        # Sharp zigzag (tests corner handling)
        ref_x = np.array([0,5,10,15,20], dtype=float)
        ref_y = np.array([0,4,0,4,0], dtype=float)
        ref_x = np.interp(np.linspace(0,4,300), np.arange(5), ref_x)
        ref_y = np.interp(np.linspace(0,4,300), np.arange(5), ref_y)
        ref_traj = np.vstack((ref_x, ref_y))
        return ref_traj

    elif(path==3):
        # Tight spiral (continuously increasing curvature)
        t = np.linspace(0, 4*np.pi, 400)
        ref_x = t * np.cos(t)
        ref_y = t * np.sin(t)
        ref_traj = np.vstack((ref_x, ref_y))   
        return ref_traj


def main_init(ref_traj):
    # Compute initial heading from the first path segment
    d0 = ref_traj[:, 1] - ref_traj[:, 0]
    theta0 = float(np.arctan2(d0[1], d0[0]))   # point robot along path at start

    x_current = np.array([ref_traj[0, 0], ref_traj[1, 0], theta0], dtype=float)

    # History buffers — init here so they match x_current
    x_history     = [x_current[0]]
    y_history     = [x_current[1]]
    theta_history = [x_current[2]]


    print("Running MPCC")
    print(f"Path: start={ref_traj[:,0]}, end={ref_traj[:,-1]}, points={ref_traj.shape[1]}")
    print(f"Initial state: {x_current}")


    return x_current, x_history, y_history, theta_history





