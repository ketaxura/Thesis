#All plotting
#Function:
# -Draws the robot and dynamic obstacles
# -Plots the computed the robot and obstacle state tracjectories in matplotlib
# -Draws the reference path
# -

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

import sys
import time

from matplotlib.patches import Ellipse
from params import *



def visualize(ref_traj, x_history, y_history, theta_history, obs_pos_hist):
        
    # ---- PLOTTING (single animation loop)
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 12))

    # --- Reference path (static) ---
    ref_line, = ax.plot(
        ref_traj[0, :],
        ref_traj[1, :],
        "k--",
        linewidth=2,
        alpha=0.8,
        label="Reference path"
    )


    ref_horizon_line, = ax.plot([], [], "g-", linewidth=2, alpha=0.9, label="Ref horizon")


    robot_body, = ax.plot([], [], "b-", lw=2)
    L = 0.35   # body length
    W = 0.20   # body half-width


    robot_circle = plt.Circle(
        (x_history[0], y_history[0]),
        r_robot,
        edgecolor="b",
        facecolor="none",
        linewidth=2
    )
    ax.add_patch(robot_circle)

    path_line,  = ax.plot([], [], "b-", lw=2, label="Trajectory")
    obs_plot,   = ax.plot([], [], "ro", markersize=8    )


    ellipses = []
    for i in range(num_obs):
        e = Ellipse(
            xy=obs_pos[i],
            width=2*a_obs,
            height=2*b_obs,
            edgecolor="r",
            facecolor="none",
            linewidth=2,
            label="Obstacle" if i == 0 else None
        )
        ax.add_patch(e)
        ellipses.append(e)
        
        
        
    robot_buffer = plt.Circle(
        (x_history[0], y_history[0]),
        r_robot + safety_buffer,
        edgecolor="b",
        facecolor="none",
        linestyle="--",
        alpha=0.4,
        linewidth=1.5,
        label="Safety buffer"
    )
    ax.add_patch(robot_buffer)

    # safe bounds (no min() crash)
    ax.set_xlim(-5, 12)
    ax.set_ylim(-5, 12)
    
    
    # -----------------------------
    # Draw static corridor walls
    # -----------------------------
    ax.axhline(WALL_Y_MIN, linewidth=2, linestyle="-", alpha=0.8, label="Bottom wall")
    ax.axhline(WALL_Y_MAX, linewidth=2, linestyle="-", alpha=0.8, label="Top wall")


    ax.legend()
    ax.grid(True)

    T = min(len(x_history), len(obs_pos_hist))



    for k in range(T):
        x = x_history[k]
        y = y_history[k]
        th = theta_history[k]

        R = np.array([[np.cos(th), -np.sin(th)],
                    [np.sin(th),  np.cos(th)]])

        robot_circle.center = (x, y)
        robot_buffer.center = (x, y)

        # plot active reference horizon
        k_ref_end = min(k + N + 1, ref_traj.shape[1])
        ref_horizon_line.set_data(
            ref_traj[0, k:k_ref_end],
            ref_traj[1, k:k_ref_end]
        )

        path_line.set_data(x_history[:k+1], y_history[:k+1])
        # obs_plot.set_data([obs_x_hist[k]], [obs_y_hist[k]])
        for i in range(num_obs):
            ellipses[i].center = (
                obs_pos_hist[k][i, 0],
                obs_pos_hist[k][i, 1]
            )

        plt.pause(0.04)

    plt.ioff()
    plt.show()
