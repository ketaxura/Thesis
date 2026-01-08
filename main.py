import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
# import roboticstoolbox as rtb
# import dynamics, mpc
from params import *
from dynamics import *
from mpc import *


plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(x_ref, y_ref, color="red", label="Reference")
robot_plot, = ax.plot([], [], "bo", label="Robot")
ax.set_xlim(-1, 2 * np.pi + 1)
ax.set_ylim(-2, 2)
ax.legend()
ax.grid(True)

for k in range(len(t_ref) - N):
    # Horizon slice: (2 x (N+1))
    R_horizon = ref_traj[:, k : k + N + 1]

    # Parameter vector p = [X0 ; vec(R)]
    p = np.concatenate([
        x_current,
        R_horizon.flatten(order="F")
    ])

    # (Optional) sanity check that packing/unpacking is consistent
    R_from_p = p[nx:].reshape((nr, N + 1), order="F")
    # print("R match:", np.max(np.abs(R_from_p - R_horizon)))

    sol = solver(
        lbx=lbx, ubx=ubx,
        lbg=0, ubg=0,
        p=p
    )

    z_opt = sol["x"].full().flatten()

    offset = nx * (N + 1)
    u0 = z_opt[offset : offset + nu]  # first control [v, omega]

    v = u0[0]
    omega = u0[1]
    x_current = np.array([
        x_current[0] + dt * v * np.cos(x_current[2]),
        x_current[1] + dt * v * np.sin(x_current[2]),
        x_current[2] + dt * omega
    ])

    robot_plot.set_data([x_current[0]], [x_current[1]])
    plt.pause(0.001)

    x_history.append(x_current[0])
    y_history.append(x_current[1])
    x_ref_history.append(R_horizon[0, 0])
    y_ref_history.append(R_horizon[1, 0])

# -----------------------------
# Final plot
# -----------------------------
plt.ioff()
plt.figure(figsize=(6, 6))
plt.plot(x_ref, y_ref, color="red", label="Reference Trajectory")
plt.plot(x_history, y_history, color="blue", label="Robot Trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.show()

# sys.stdout.close()
