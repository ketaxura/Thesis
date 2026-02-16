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
from matplotlib.patches import Rectangle
from params import *


""""""
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
    ax.set_xlim(-5, 24)
    ax.set_ylim(-8, 8)
    

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


        path_line.set_data(x_history[:k+1], y_history[:k+1])
        # obs_plot.set_data([obs_x_hist[k]], [obs_y_hist[k]])

        plt.pause(0.04)
        
    
    

    plt.ioff()
    plt.show(block=True)
