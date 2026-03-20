import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Circle

from obs import Obstacle
from run_experiment_unified import MPCCController, dt, N


# ============================================================
# Config
# ============================================================
PATH_ID = 10         # 10 = map_corridor_structured, 11 = map_open_clutter
SEED_OFFSET = 0
USE_HARD = True      # no dyn obs anyway, but keep static constraints hard
MAX_STEPS = 800
GOAL_TOL = 0.5
SHOW_PREDICTIONS = False
PRED_MARKER_SIZE = 12
ROBOT_DRAW_RADIUS = 0.20

# ============================================================
# Helpers
# ============================================================

def rotated_ellipse_level(px, py, ox, oy, a, b, th):
    dx = px - ox
    dy = py - oy
    c = np.cos(th)
    s = np.sin(th)
    x_local = c * dx + s * dy
    y_local = -s * dx + c * dy
    return (x_local / a) ** 2 + (y_local / b) ** 2


def compute_plot_limits(ref_traj, static_rects, dyn_obs, margin=1.5):
    xs = [ref_traj[0, :].min(), ref_traj[0, :].max()]
    ys = [ref_traj[1, :].min(), ref_traj[1, :].max()]

    for (cx, cy, hw, hh) in static_rects:
        xs.extend([cx - hw, cx + hw])
        ys.extend([cy - hh, cy + hh])

    for obs in dyn_obs:
        xs.extend([obs.x - obs.a, obs.x + obs.a])
        ys.extend([obs.y - obs.b, obs.y + obs.b])

    return (
        min(xs) - margin,
        max(xs) + margin,
        min(ys) - margin,
        max(ys) + margin,
    )


def draw_scene(
    ax,
    ref_traj,
    static_rects,
    dyn_obs,
    robot_state,
    traj_xy,
    pred_sets,
    step_idx,
    status_text,
    controller,
):
    ax.clear()

    # Reference path
    ax.plot(ref_traj[0, :], ref_traj[1, :], "--", linewidth=1.5, label="reference")

    # Traveled path
    if len(traj_xy) > 1:
        traj_np = np.array(traj_xy)
        ax.plot(traj_np[:, 0], traj_np[:, 1], linewidth=2.0, label="robot path")

    # Static rectangles
    for i, (cx, cy, hw, hh) in enumerate(static_rects):
        rect = Rectangle(
            (cx - hw, cy - hh),
            2 * hw,
            2 * hh,
            fill=False,
            linewidth=2.0,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, f"S{i}", ha="center", va="center", fontsize=8)

    # Dynamic obstacles
    for i, obs in enumerate(dyn_obs):
        ell = Ellipse(
            (obs.x, obs.y),
            width=2 * obs.a,
            height=2 * obs.b,
            angle=np.degrees(obs.theta),
            fill=False,
            linewidth=2.0,
        )
        ax.add_patch(ell)

        ax.plot(obs.x, obs.y, "o", markersize=5)
        ax.text(obs.x, obs.y, f"D{i}", fontsize=8, ha="left", va="bottom")

        # Heading arrow
        hx = 0.5 * obs.a * np.cos(obs.theta)
        hy = 0.5 * obs.a * np.sin(obs.theta)
        ax.arrow(
            obs.x, obs.y, hx, hy,
            head_width=0.08, head_length=0.10,
            length_includes_head=True,
        )

    # Predicted obstacle horizons
    if SHOW_PREDICTIONS and pred_sets is not None:
        for i, (xs, ys, ths) in enumerate(pred_sets):
            ax.scatter(xs, ys, s=PRED_MARKER_SIZE, alpha=0.8)
            if len(xs) > 0:
                ax.plot(xs[0], ys[0], "x", markersize=8)

    # Robot
    rx, ry, rth = robot_state
    robot = Circle((rx, ry), ROBOT_DRAW_RADIUS, fill=False, linewidth=2.0)
    ax.add_patch(robot)

    hx = ROBOT_DRAW_RADIUS * 1.6 * np.cos(rth)
    hy = ROBOT_DRAW_RADIUS * 1.6 * np.sin(rth)
    ax.arrow(
        rx, ry, hx, hy,
        head_width=0.08, head_length=0.10,
        length_includes_head=True,
    )

    ax.plot(rx, ry, "o", markersize=6)

    # Current MPCC progress point
    k_ref = max(int(np.floor(controller.mu)), 0)
    k_ref = min(k_ref, ref_traj.shape[1] - 1)
    ax.plot(ref_traj[0, k_ref], ref_traj[1, k_ref], "*", markersize=12)

    ax.set_title(f"MPCC Debug | path_id={PATH_ID} | no dyn obs | step={step_idx}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8)

    xmin, xmax, ymin, ymax = compute_plot_limits(ref_traj, static_rects, dyn_obs)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.text(
        0.01, 0.99, status_text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


# ============================================================
# Main
# ============================================================

def main():
    env = Obstacle(PATH_ID)
    ref_traj = env.path_selector(PATH_ID)
    static_rects = env.static_obs()
    dyn_obs = env.dynamic_obs_seeded(SEED_OFFSET)

    p0 = ref_traj[:, 0]
    p1 = ref_traj[:, 1]
    th0 = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    x_current = np.array([p0[0], p0[1], th0], dtype=float)
    x_goal = np.array([ref_traj[0, -1], ref_traj[1, -1], 0.0], dtype=float)

    controller = MPCCController(
        static_rects=static_rects,
        dyn_obs=dyn_obs,
        ref_traj=ref_traj,
        use_hard=USE_HARD,
    )

    traj_xy = [x_current[:2].copy()]

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    for k in range(MAX_STEPS):
        # Horizon predictions before solve
        pred_sets = []
        for obs in dyn_obs:
            if hasattr(obs, "predict_horizon_with_rects"):
                pred_sets.append(obs.predict_horizon_with_rects(N, dt, static_rects))
            else:
                pred_sets.append(obs.predict_horizon(N, dt))

        step_result = controller.solve(x_current, dyn_obs, x_goal)

        status_text = (
            f"solver_status: {step_result.status}\n"
            f"ok: {step_result.ok}\n"
            f"solve_time_ms: {1000.0 * step_result.solve_time_s:.2f}\n"
            f"mu: {controller.mu:.3f}\n"
            f"progress_index: {step_result.progress_index}\n"
        )

        if not step_result.ok:
            draw_scene(
                ax=ax,
                ref_traj=ref_traj,
                static_rects=static_rects,
                dyn_obs=dyn_obs,
                robot_state=x_current,
                traj_xy=traj_xy,
                pred_sets=pred_sets,
                step_idx=k,
                status_text=status_text + "\nFAILED",
                controller=controller,
            )
            plt.ioff()
            plt.show()
            print(f"\nSolver failed at step {k}: {step_result.status}")
            return

        v, omega = step_result.u

        x_current = np.array([
            x_current[0] + dt * v * np.cos(x_current[2]),
            x_current[1] + dt * v * np.sin(x_current[2]),
            x_current[2] + dt * omega,
        ], dtype=float)

        traj_xy.append(x_current[:2].copy())

        for obs in dyn_obs:
            obs.step(dt, static_rects)

        # Quick collision diagnostics
        dyn_hits = 0
        if dyn_obs:
            for obs in dyn_obs:
                lvl = rotated_ellipse_level(
                    x_current[0], x_current[1],
                    obs.x, obs.y,
                    obs.a, obs.b,
                    obs.theta,
                )
                if lvl <= 1.0:
                    dyn_hits += 1

        goal_dist = np.linalg.norm(x_current[:2] - x_goal[:2])

        status_text += (
            f"v: {v:.3f}\n"
            f"omega: {omega:.3f}\n"
            f"goal_dist: {goal_dist:.3f}\n"
            f"dyn_hits: {dyn_hits}\n"
        )

        draw_scene(
            ax=ax,
            ref_traj=ref_traj,
            static_rects=static_rects,
            dyn_obs=dyn_obs,
            robot_state=x_current,
            traj_xy=traj_xy,
            pred_sets=pred_sets,
            step_idx=k,
            status_text=status_text,
            controller=controller,
        )

        plt.pause(0.05)

        if goal_dist < GOAL_TOL and step_result.progress_index >= ref_traj.shape[1] - 2:
            print(f"\nGoal reached at step {k}.")
            break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()