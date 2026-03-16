import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse

from params import *


def visualize(ref_traj, x_history, y_history, theta_history, static_rects, dyn_obs, dyn_obs_hist):

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))

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

    # Robot body circle
    robot_circle = plt.Circle(
        (x_history[0], y_history[0]),
        r_robot,
        edgecolor="blue",
        facecolor="lightblue",
        alpha=0.6,
        linewidth=2,
        label="Robot"
    )
    ax.add_patch(robot_circle)

    # Safety buffer ring — this is what the solver actually enforces clearance to
    robot_buffer = plt.Circle(
        (x_history[0], y_history[0]),
        r_robot + safety_buffer,
        edgecolor="blue",
        facecolor="none",
        linestyle="--",
        alpha=0.4,
        linewidth=1.5,
        label=f"Safety buffer (+{safety_buffer}m)"
    )
    ax.add_patch(robot_buffer)

    # Static obstacles
    for (cx, cy, hw, hh) in static_rects:
        rect = patches.Rectangle(
            (cx - hw, cy - hh),
            2 * hw,
            2 * hh,
            linewidth=1,
            edgecolor="black",
            facecolor="gray",
            alpha=0.6
        )
        ax.add_patch(rect)

        # Show the margin the solver uses around static obstacles
        margin = r_robot + safety_buffer
        rect_margin = patches.Rectangle(
            (cx - hw - margin, cy - hh - margin),
            2 * (hw + margin),
            2 * (hh + margin),
            linewidth=1,
            edgecolor="gray",
            facecolor="none",
            linestyle=":",
            alpha=0.4
        )
        ax.add_patch(rect_margin)

    # Dynamic obstacles — show both the real body and the solver's exclusion zone
    dyn_ellipses_body   = []
    dyn_ellipses_margin = []

    # Initialise at step-0 recorded position, NOT end-of-sim obs.x/obs.y
    init_positions = dyn_obs_hist[0] if dyn_obs_hist else [(obs.x, obs.y) for obs in dyn_obs]

    for i, obs in enumerate(dyn_obs):
        ox0, oy0 = init_positions[i]

        # Actual obstacle body
        body = Ellipse(
            (ox0, oy0),
            width=2 * obs.a,
            height=2 * obs.b,
            edgecolor="red",
            facecolor="salmon",
            alpha=0.5,
            linewidth=2,
            label="Dynamic obstacle" if i == 0 else "_nolegend_"
        )
        ax.add_patch(body)
        dyn_ellipses_body.append(body)

        # Solver exclusion zone: obs half-axes + r_robot + safety_buffer
        a_excl = obs.a + r_robot + safety_buffer
        b_excl = obs.b + r_robot + safety_buffer
        excl = Ellipse(
            (ox0, oy0),
            width=2 * a_excl,
            height=2 * b_excl,
            edgecolor="red",
            facecolor="none",
            linestyle="--",
            alpha=0.6,
            linewidth=1.5,
            label="Exclusion zone" if i == 0 else "_nolegend_"
        )
        ax.add_patch(excl)
        dyn_ellipses_margin.append(excl)

    # Heading arrow
    L_arrow = 0.4
    heading_arrow = ax.quiver(
        x_history[0],
        y_history[0],
        L_arrow * np.cos(theta_history[0]),
        L_arrow * np.sin(theta_history[0]),
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.012,
        color="darkblue",
        zorder=5
    )

    step_text = ax.text(
        0.02, 0.97, "step: 0",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    ax.set_xlim(-2, 23)
    ax.set_ylim(-4, 7)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("MPCC — Dynamic Obstacle Avoidance")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    T = min(len(x_history), len(dyn_obs_hist))

    for k in range(T):

        x  = x_history[k]
        y  = y_history[k]
        th = theta_history[k]

        robot_circle.center = (x, y)
        robot_buffer.center = (x, y)

        heading_arrow.set_offsets([x, y])
        heading_arrow.set_UVC(L_arrow * np.cos(th), L_arrow * np.sin(th))

        path_line.set_data(x_history[:k + 1], y_history[:k + 1])

        for i in range(len(dyn_obs)):
            ox, oy = dyn_obs_hist[k][i]
            dyn_ellipses_body[i].set_center((ox, oy))
            dyn_ellipses_margin[i].set_center((ox, oy))

        step_text.set_text(f"step: {k}")

        plt.pause(0.04)

    plt.ioff()
    plt.show(block=True)