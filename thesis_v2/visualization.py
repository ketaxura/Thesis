#All plotting
#Function:
# -Draws the robot and dynamic obstacles
# -Plots the computed the robot and obstacle state tracjectories in matplotlib
# -Draws the reference path
# -

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from params import STATIC_RECTS
import matplotlib.patches as patches



import sys
import time

from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from params import *


"""
Simulation plot of the robot moving
"""
def visualize(ref_traj, x_history, y_history, theta_history):
    # ---- PLOTTING (single animation loop)
    plt.ion()                                   #interactive plot on
    fig, ax = plt.subplots(figsize=(10, 8))     #defining the size of the plot window

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



    L_arrow = 0.6

    # create empty quiver (will update inside loop)
    heading_arrow = ax.quiver(
        x_history[0],
        y_history[0],
        L_arrow * np.cos(theta_history[0]),
        L_arrow * np.sin(theta_history[0]),
        angles='xy',
        scale_units='xy',
        scale=1,
        width=0.01,
        color='red'
    )
        
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


    for (cx, cy, hw, hh) in STATIC_RECTS:
        rect = patches.Rectangle(
            (cx - hw, cy - hh), 2*hw, 2*hh,
            linewidth=1, edgecolor='black', facecolor='gray', alpha=0.5
        )
        ax.add_patch(rect)

    # safe bounds (no min() crash)
    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)
    

    ax.legend()
    ax.grid(True)

    # T = min(len(x_history), len(obs_pos_hist))
    T = len(x_history)



    for k in range(T):
        x = x_history[k]
        y = y_history[k]
        th = theta_history[k]

        

        R = np.array([[np.cos(th), -np.sin(th)],
                    [np.sin(th),  np.cos(th)]])

        robot_circle.center = (x, y)
        robot_buffer.center = (x, y)

        dx = L_arrow * np.cos(th)
        dy = L_arrow * np.sin(th)

        heading_arrow.set_offsets([x, y])
        heading_arrow.set_UVC(dx, dy)


        path_line.set_data(x_history[:k+1], y_history[:k+1])
        # obs_plot.set_data([obs_x_hist[k]], [obs_y_hist[k]])

        plt.pause(0.04)
        
    
    

    plt.ioff()
    plt.show(block=True)



