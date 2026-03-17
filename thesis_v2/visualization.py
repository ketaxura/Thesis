import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse

from params import *


def visualize(ref_traj, x_history, y_history, theta_history, static_rects, dyn_obs, dyn_obs_hist):

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))

    # Reference path
    ax.plot(
        ref_traj[0, :],
        ref_traj[1, :],
        "k--",
        linewidth=2,
        alpha=0.8,
        label="Reference path"
    )

    path_line, = ax.plot([], [], "b-", lw=2, label="Trajectory")

    robot_circle = plt.Circle(
        (x_history[0], y_history[0]),
        r_robot,
        edgecolor="b",
        facecolor="none",
        linewidth=2
    )
    ax.add_patch(robot_circle)


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

    # draw static obstacles
    for (cx, cy, hw, hh) in static_rects:
        rect = patches.Rectangle(
            (cx - hw, cy - hh),
            2 * hw,
            2 * hh,
            linewidth=1,
            edgecolor="black",
            facecolor="gray",
            alpha=0.5
        )
        ax.add_patch(rect)

    # dynamic obstacles
    dyn_ellipses = []

    init_snap = dyn_obs_hist[0] if dyn_obs_hist else [(obs.x, obs.y, 0.0) for obs in dyn_obs]

    for i, obs in enumerate(dyn_obs):
        ox0, oy0, oth0 = init_snap[i]
        ellipse = Ellipse(
            (ox0, oy0),
            width=2*obs.a,
            height=2*obs.b,
            angle=np.degrees(oth0),
            edgecolor='red',
            facecolor='salmon',
            alpha=0.5,
            linewidth=2
        )
        ax.add_patch(ellipse)
        dyn_ellipses.append(ellipse)

    L_arrow = 0.6

    heading_arrow = ax.quiver(
        x_history[0],
        y_history[0],
        L_arrow * np.cos(theta_history[0]),
        L_arrow * np.sin(theta_history[0]),
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.01,
        color="red"
    )

    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)

    ax.legend()
    ax.grid(True)

    T = len(x_history)

    for k in range(T):

        x = x_history[k]
        y = y_history[k]
        th = theta_history[k]

        robot_circle.center = (x, y)
        robot_buffer.center = (x, y)

        dx = L_arrow * np.cos(th)
        dy = L_arrow * np.sin(th)

        heading_arrow.set_offsets([x, y])
        heading_arrow.set_UVC(dx, dy)

        path_line.set_data(x_history[:k + 1], y_history[:k + 1])

        for i, obs in enumerate(dyn_obs):
            ox, oy, oth = dyn_obs_hist[k][i]
            dyn_ellipses[i].set_center((ox, oy))
            dyn_ellipses[i].set_angle(np.degrees(oth))

        plt.pause(0.04)

    plt.ioff()
    plt.show(block=True)