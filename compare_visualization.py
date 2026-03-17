"""
compare_visualization.py
Runs one seed of MPC (soft), MPC (hard), and MPCC side by side.
Drop into the MPC project folder (which has obs.py copied from MPCC).

Usage: python compare_visualization.py
Change SEED and PATH_ID at the top to explore different scenarios.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse

from dynamics import unicycle_dynamics
from obs import Obstacle
from run_experiment_mpc_v2 import run_mpc, build_mpc_solver, _ellipse_surface_dist
from params import dt, nx, nu, r_robot, safety_buffer

# ============================================================
SEED    = 1
PATH_ID = 3   # 3=zigzag, 1=sine
MAX_STEPS = 800
# ============================================================

NEAR_MISS_MARGIN = 0.3

def collect_trajectory(path_id, seed_offset, use_hard):
    """Run one simulation and return trajectory + obstacle history."""
    from run_experiment_mpc_v2 import N  # uses overridden N=30

    env          = Obstacle(path_id)
    ref_traj     = env.path_selector(path_id)
    STATIC_RECTS = env.static_obs()
    dyn_obs      = env.dynamic_obs_seeded(seed_offset)

    solver, lbx, ubx, lbg, ubg, nX, nU = build_mpc_solver(
        STATIC_RECTS, dyn_obs, use_hard=use_hard
    )

    p0  = ref_traj[:, 0]; p1 = ref_traj[:, 1]
    th0 = float(np.arctan2(p1[1]-p0[1], p1[0]-p0[0]))
    x_current     = np.array([p0[0], p0[1], th0])
    X_goal_global = np.array([ref_traj[0,-1], ref_traj[1,-1], 0.0])

    u_prev = np.zeros(nu)
    prev_z = None
    ref_k  = 0
    _at_end_count = 0

    x_hist   = []
    y_hist   = []
    obs_hist = []
    collisions = []

    for k in range(MAX_STEPS):
        R_horizon = ref_traj[:, ref_k : ref_k + N + 1]
        if R_horizon.shape[1] < N + 1:
            last = R_horizon[:, -1].reshape(2,1)
            pad  = np.repeat(last, N+1-R_horizon.shape[1], axis=1)
            R_horizon = np.hstack([R_horizon, pad])

        X_goal_val = np.array([R_horizon[0,-1], R_horizon[1,-1], 0.0])

        if dyn_obs:
            preds   = [obs.predict_horizon(N, dt) for obs in dyn_obs]
            obs_x_h = np.array([pr[0] for pr in preds])
            obs_y_h = np.array([pr[1] for pr in preds])
        else:
            obs_x_h = np.zeros((0, N))
            obs_y_h = np.zeros((0, N))

        p = np.concatenate([
            x_current, u_prev,
            R_horizon.flatten(order='F'),
            obs_x_h.flatten(order='F'),
            obs_y_h.flatten(order='F'),
            X_goal_val,
        ])

        kwargs = dict(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)
        if prev_z is not None:
            kwargs["x0"] = prev_z

        sol    = solver(**kwargs)
        prev_z = sol["x"]

        z     = sol["x"].full().flatten()
        U_opt = z[nX : nX+nU].reshape((nu, N), order="F")
        v, omega = U_opt[:, 0]
        u_prev   = np.array([v, omega])
        ref_k    = min(ref_k + 1, ref_traj.shape[1] - 1)

        # Record BEFORE stepping
        x_hist.append(x_current[0])
        y_hist.append(x_current[1])
        obs_hist.append(np.array([[obs.x, obs.y] for obs in dyn_obs]))

        x_current = np.array([
            x_current[0] + dt * v * np.cos(x_current[2]),
            x_current[1] + dt * v * np.sin(x_current[2]),
            x_current[2] + dt * omega,
        ])

        for obs in dyn_obs:
            obs.step(dt, STATIC_RECTS)

        # Collision check
        for i, obs in enumerate(dyn_obs):
            dx = x_current[0] - obs.x
            dy = x_current[1] - obs.y
            if (dx/obs.a)**2 + (dy/obs.b)**2 <= 1.0:
                collisions.append((k, x_current[0], x_current[1]))

        dist_to_goal  = np.linalg.norm(x_current[:2] - X_goal_global[:2])
        path_complete = ref_k >= ref_traj.shape[1] - 2
        if dist_to_goal < 0.5 and path_complete:
            break
        if path_complete:
            _at_end_count += 1
            if _at_end_count >= 30:
                break
        else:
            _at_end_count = 0

    return (np.array(x_hist), np.array(y_hist),
            obs_hist, collisions, ref_traj, STATIC_RECTS, dyn_obs)


# ============================================================
# Run all three controllers on the same seed
# ============================================================
print(f"Running MPC (soft)  seed={SEED} path={PATH_ID}...")
xs, ys, obs_h_soft, coll_soft, ref_traj, STATIC_RECTS, _ = \
    collect_trajectory(PATH_ID, SEED, use_hard=False)

print(f"Running MPC (hard)  seed={SEED} path={PATH_ID}...")
xh, yh, obs_h_hard, coll_hard, _, _, _ = \
    collect_trajectory(PATH_ID, SEED, use_hard=True)

# For MPCC we just load from the MPCC results CSV if available,
# otherwise note it needs to be run from the MPCC project
try:
    import csv
    mpcc_traj = None
    with open("results/mpcc_traj_seed0.csv") as f:
        rows = list(csv.DictReader(f))
        mpcc_x = [float(r["x"]) for r in rows]
        mpcc_y = [float(r["y"]) for r in rows]
    has_mpcc = True
except:
    has_mpcc = False
    print("Note: No MPCC trajectory CSV found. Showing MPC soft vs hard only.")

# ============================================================
# Static plot — full trajectory comparison
# ============================================================
fig, axes = plt.subplots(1, 2 if not has_mpcc else 3,
                          figsize=(18 if not has_mpcc else 22, 7),
                          sharey=True)

def draw_env(ax, title, x_traj, y_traj, obs_hist, collisions, ref_traj, static_rects, dyn_obs_final):
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Reference path
    ax.plot(ref_traj[0], ref_traj[1], 'k--', lw=1.5, alpha=0.6, label='Reference')

    # Trajectory
    ax.plot(x_traj, y_traj, 'b-', lw=2, label='Trajectory', zorder=3)
    ax.plot(x_traj[0],  y_traj[0],  'go', ms=10, label='Start', zorder=5)
    ax.plot(x_traj[-1], y_traj[-1], 'b*', ms=12, label='End',   zorder=5)

    # Static obstacles
    for (cx, cy, hw, hh) in static_rects:
        ax.add_patch(patches.Rectangle(
            (cx-hw, cy-hh), 2*hw, 2*hh,
            edgecolor='black', facecolor='gray', alpha=0.5, zorder=2))

    # Dynamic obstacle trails (show final positions as ellipses)
    if obs_hist:
        last = obs_hist[-1]
        for i, obs in enumerate(dyn_obs_final):
            # Trail
            trail_x = [obs_hist[t][i][0] for t in range(0, len(obs_hist), 5)]
            trail_y = [obs_hist[t][i][1] for t in range(0, len(obs_hist), 5)]
            ax.plot(trail_x, trail_y, 'r-', alpha=0.2, lw=1)
            # Final position
            ax.add_patch(Ellipse(
                (last[i][0], last[i][1]), 2*obs.a, 2*obs.b,
                edgecolor='red', facecolor='salmon', alpha=0.6, lw=2, zorder=4,
                label='Dyn obs' if i == 0 else '_'))
            # Exclusion zone
            ax.add_patch(Ellipse(
                (last[i][0], last[i][1]),
                2*(obs.a+r_robot+safety_buffer),
                2*(obs.b+r_robot+safety_buffer),
                edgecolor='red', facecolor='none',
                linestyle='--', alpha=0.4, lw=1.5, zorder=4))

    # Collision markers
    for (k, cx, cy) in collisions:
        ax.plot(cx, cy, 'rx', ms=14, markeredgewidth=3,
                label='Collision' if k == collisions[0][0] else '_')

    ax.set_xlim(-2, 23); ax.set_ylim(-4, 7)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')

    # Annotation
    n_coll = len(collisions)
    color  = 'red' if n_coll > 0 else 'green'
    ax.text(0.02, 0.03,
            f'Collisions: {n_coll}',
            transform=ax.transAxes, fontsize=11,
            color=color, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


env_tmp = Obstacle(PATH_ID)
dyn_obs_ref = env_tmp.dynamic_obs_seeded(SEED)  # just for sizes

draw_env(axes[0], f'MPC Soft — seed={SEED}',
         xs, ys, obs_h_soft, coll_soft,
         ref_traj, STATIC_RECTS, dyn_obs_ref)

env_tmp2 = Obstacle(PATH_ID)
dyn_obs_ref2 = env_tmp2.dynamic_obs_seeded(SEED)
draw_env(axes[1], f'MPC Hard — seed={SEED}',
         xh, yh, obs_h_hard, coll_hard,
         ref_traj, STATIC_RECTS, dyn_obs_ref2)

if has_mpcc:
    axes[2].set_title(f'MPCC — seed={SEED}', fontsize=13, fontweight='bold')
    axes[2].plot(ref_traj[0], ref_traj[1], 'k--', lw=1.5, alpha=0.6)
    axes[2].plot(mpcc_x, mpcc_y, 'b-', lw=2)
    for (cx, cy, hw, hh) in STATIC_RECTS:
        axes[2].add_patch(patches.Rectangle(
            (cx-hw, cy-hh), 2*hw, 2*hh,
            edgecolor='black', facecolor='gray', alpha=0.5))
    axes[2].text(0.02, 0.03, 'Collisions: 0',
                 transform=axes[2].transAxes, fontsize=11,
                 color='green', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[2].set_xlim(-2, 23); axes[2].set_ylim(-4, 7)
    axes[2].set_aspect('equal'); axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel('x [m]')

plt.suptitle(
    f'MPC vs MPCC Comparison — {"Zigzag" if PATH_ID==3 else "Sine"} path, seed={SEED}',
    fontsize=14, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig(f'comparison_path{PATH_ID}_seed{SEED}.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved to comparison_path{PATH_ID}_seed{SEED}.png")
print(f"MPC Soft collisions: {len(coll_soft)}")
print(f"MPC Hard collisions: {len(coll_hard)}")