"""
debug.py  –  thesis-clean single-run debugger for MPC / MPCC
=============================================================
Architecture
------------
  1. ROLLOUT PHASE   : call controller.solve() every step, store StepRecord per step
  2. PLAYBACK PHASE  : replay stored records for visualization (no solving)

Collision taxonomy
------------------
  static_body_collision      : robot centre overlaps the *actual* obstacle (margin = 0)
  static_zone_violation      : robot centre enters the inflated forbidden zone
                               (margin = r_robot + safety_buffer).
                               For MPCC hard-mode this *should* never happen.
  dyn_body_collision         : robot centre overlaps the *actual* dynamic ellipse (margin = 0)
  dyn_exclusion_violation    : robot centre enters the inflated dynamic exclusion zone
                               (margin = r_robot + safety_buffer)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse

from dynamics import unicycle_dynamics  # noqa: F401  (kept for caller convenience)
from obs import Obstacle, DynamicObstacle
from run_experiment_unified import MPCCController, MPCController, dt, N
from params import r_robot, safety_buffer

# ============================================================
# Run configuration
# ============================================================
PATH_ID            = 11        # 10 = map_corridor_structured, 11 = map_open_clutter
SEED_OFFSET        = 0
USE_HARD           = False     # hard vs soft dynamic obstacle mode
MAX_STEPS          = 2000
GOAL_TOL           = 0.35      # metres
END_PROGRESS_TOL   = 1.0       # steps from end of ref_traj
SHOW_PREDICTIONS   = False
PRED_MARKER_SIZE   = 12
ROBOT_DOT_SIZE      = 12
ROBOT_HEADING_LEN   = 0.22

ALGO               = "mpc"    # "mpc" or "mpcc"

ENABLE_DEBUG_DYN_OBS = True
DEBUG_DYN_OBS_MODE   = "manual"   # "manual" or "env"

# p-norm used for static-rectangle level-set evaluation (must match mpcc.py)
P_NORM = 6

VIOLATION_EPS = 1e-6
VIOLATION_THRESHOLD = 1.0 - VIOLATION_EPS


# ============================================================
# Data record for one time step
# ============================================================

@dataclass
class StepRecord:
    step_idx:   int
    state:      np.ndarray          # [x, y, theta]
    u:          np.ndarray          # [v, omega]
    status:     str
    ok:         bool
    progress_index: float           # controller-agnostic progress along ref

    # Obstacle snapshots (positions at *this* step, post-move)
    dyn_obs_snapshot: list          # list of (x, y, theta, a, b)
    pred_sets:        list | None   # list of (xs, ys, thetas) arrays, or None

    # Collision flags  ── filled by check_collisions()
    static_body_collision:   bool = False
    static_zone_violation:   bool = False
    static_body_ids:         list = field(default_factory=list)
    static_zone_ids:         list = field(default_factory=list)

    dyn_body_collision:      bool = False
    dyn_exclusion_violation: bool = False
    dyn_body_ids:            list = field(default_factory=list)
    dyn_exclusion_ids:       list = field(default_factory=list)


    dyn_exclusion_levels: list = field(default_factory=list)
    dyn_exclusion_depths: list = field(default_factory=list)
    worst_dyn_excl_id: int | None = None
    worst_dyn_excl_level: float | None = None
    worst_dyn_excl_depth: float = 0.0


    dyn_body_levels: list = field(default_factory=list)
    dyn_body_clearances: list = field(default_factory=list)   # positive = outside body, negative = inside body
    worst_dyn_body_id: int | None = None
    worst_dyn_body_level: float | None = None
    worst_dyn_body_clearance: float | None = None


    solve_time_s: float = 0.0
    select_time_s: float = 0.0
    predict_time_s: float = 0.0
    pack_time_s: float = 0.0
    post_time_s: float = 0.0
    active_dyn_count: int = 0

        


# ============================================================
# Collision checkers  (controller-agnostic, pure geometry)
# ============================================================

def _rect_level(px: float, py: float, cx: float, cy: float,
                hw: float, hh: float, margin: float, p_norm: int = P_NORM) -> float:
    """
    Super-ellipse level value for point (px, py) w.r.t. rectangle (cx, cy, hw, hh)
    inflated by *margin*.  value < 1.0  →  inside forbidden zone.
    """
    dx = px - cx
    dy = py - cy
    return (abs(dx) / (hw + margin)) ** p_norm + (abs(dy) / (hh + margin)) ** p_norm


def _ellipse_level(px: float, py: float, ox: float, oy: float,
                   a: float, b: float, theta: float) -> float:
    """Rotated-ellipse level value.  value <= 1.0  →  inside ellipse."""
    dx = px - ox
    dy = py - oy
    c, s = np.cos(theta), np.sin(theta)
    x_local =  c * dx + s * dy
    y_local = -s * dx + c * dy
    return (x_local / a) ** 2 + (y_local / b) ** 2


def check_static_body_collision(
    xy: np.ndarray,
    static_rects: list,
    p_norm: int = P_NORM,
) -> tuple[bool, list[int]]:
    """
    True collision with the *actual* obstacle body (margin = 0).
    Returns (collided, list_of_obstacle_indices).
    """
    px, py = float(xy[0]), float(xy[1])
    ids = []
    for i, (cx, cy, hw, hh) in enumerate(static_rects):
        level = _rect_level(px, py, cx, cy, hw, hh, margin=0.0, p_norm=p_norm)
        if level < VIOLATION_THRESHOLD:
            ids.append(i)
    return (len(ids) > 0), ids


def check_static_zone_violation(
    xy: np.ndarray,
    static_rects: list,
    margin: float | None = None,
    p_norm: int = P_NORM,
) -> tuple[bool, list[int]]:
    """
    Penetration of the inflated forbidden zone used by the controller
    (margin = r_robot + safety_buffer).  For MPCC hard-mode this must be 0.
    Returns (violated, list_of_obstacle_indices).
    """
    if margin is None:
        margin = r_robot + safety_buffer
    px, py = float(xy[0]), float(xy[1])
    ids = []
    for i, (cx, cy, hw, hh) in enumerate(static_rects):
        level = _rect_level(px, py, cx, cy, hw, hh, margin=margin, p_norm=p_norm)
        if level < VIOLATION_THRESHOLD:
            ids.append(i)
    return (len(ids) > 0), ids


def check_dyn_body_collision(
    xy: np.ndarray,
    dyn_obs: list,
) -> tuple[bool, list[int]]:
    """
    True collision with the *actual* dynamic obstacle body (margin = 0).
    Returns (collided, list_of_obstacle_indices).
    """
    px, py = float(xy[0]), float(xy[1])
    ids = []
    for i, obs in enumerate(dyn_obs):
        level = _ellipse_level(px, py, obs.x, obs.y, obs.a, obs.b, obs.theta)
        if level < VIOLATION_THRESHOLD:
            ids.append(i)
    return (len(ids) > 0), ids


def check_dyn_exclusion_violation(
    xy: np.ndarray,
    dyn_obs: list,
    margin: float | None = None,
) -> tuple[bool, list[int]]:
    if margin is None:
        margin = r_robot + safety_buffer
    px, py = float(xy[0]), float(xy[1])
    ids = []
    for i, obs in enumerate(dyn_obs):
        a_eff = obs.a + margin
        b_eff = obs.b + margin
        level = _ellipse_level(px, py, obs.x, obs.y, a_eff, b_eff, obs.theta)
        if level < VIOLATION_THRESHOLD:
            ids.append(i)
    return (len(ids) > 0), ids

def check_collisions(record: StepRecord, static_rects: list, dyn_obs_live: list) -> None:
    """
    Fill all four collision fields on *record* in-place.
    dyn_obs_live should be the obstacles *after* their step() call for this step.
    """
    xy = record.state[:2]

    record.static_body_collision, record.static_body_ids = \
        check_static_body_collision(xy, static_rects)

    record.static_zone_violation, record.static_zone_ids = \
        check_static_zone_violation(xy, static_rects)

    record.dyn_body_collision, record.dyn_body_ids = \
        check_dyn_body_collision(xy, dyn_obs_live)

    record.dyn_exclusion_violation, record.dyn_exclusion_ids = \
        check_dyn_exclusion_violation(xy, dyn_obs_live)
    
    (
        record.dyn_exclusion_levels,
        record.dyn_exclusion_depths,
        record.worst_dyn_excl_id,
        record.worst_dyn_excl_level,
        record.worst_dyn_excl_depth,
    ) = evaluate_dyn_exclusion_depths(xy, dyn_obs_live)


    (
        record.dyn_body_levels,
        record.dyn_body_clearances,
        record.worst_dyn_body_id,
        record.worst_dyn_body_level,
        record.worst_dyn_body_clearance,
    ) = evaluate_dyn_body_proximity(xy, dyn_obs_live)
        


def evaluate_dyn_exclusion_depths(
    xy: np.ndarray,
    dyn_obs: list,
    margin: float | None = None,
) -> tuple[list[float], list[float], int | None, float | None, float]:
    """
    Returns:
      levels[i] = exclusion-zone level for obstacle i
      depths[i] = max(0, VIOLATION_THRESHOLD - levels[i])
      worst_id
      worst_level
      worst_depth
    """
    if margin is None:
        margin = r_robot + safety_buffer

    px, py = float(xy[0]), float(xy[1])
    levels = []
    depths = []

    for obs in dyn_obs:
        a_eff = obs.a + margin
        b_eff = obs.b + margin
        level = _ellipse_level(px, py, obs.x, obs.y, a_eff, b_eff, obs.theta)
        depth = max(0.0, VIOLATION_THRESHOLD - level)
        levels.append(float(level))
        depths.append(float(depth))

    if len(depths) == 0:
        return levels, depths, None, None, 0.0

    worst_id = int(np.argmax(depths))
    worst_depth = float(depths[worst_id])
    worst_level = float(levels[worst_id])

    if worst_depth <= 0.0:
        return levels, depths, None, None, 0.0

    return levels, depths, worst_id, worst_level, worst_depth



def evaluate_dyn_body_proximity(
    xy: np.ndarray,
    dyn_obs: list,
) -> tuple[list[float], list[float], int | None, float | None, float | None]:
    """
    Returns, for each dynamic obstacle:
      levels[i]      = actual-body ellipse level
      clearances[i]  = sqrt(levels[i]) - 1.0

    Interpretation:
      clearance > 0   -> outside actual body
      clearance = 0   -> on actual body boundary
      clearance < 0   -> inside actual body (true collision)

    The clearance is dimensionless but much more interpretable than raw level:
      level = 1.00  -> clearance = 0
      level = 1.21  -> clearance = 0.10
      level = 0.81  -> clearance = -0.10
    """
    px, py = float(xy[0]), float(xy[1])
    levels = []
    clearances = []

    for obs in dyn_obs:
        level = _ellipse_level(px, py, obs.x, obs.y, obs.a, obs.b, obs.theta)
        clearance = np.sqrt(max(level, 0.0)) - 1.0
        levels.append(float(level))
        clearances.append(float(clearance))

    if len(clearances) == 0:
        return levels, clearances, None, None, None

    worst_id = int(np.argmin(clearances))   # smallest clearance = closest / worst
    worst_level = float(levels[worst_id])
    worst_clearance = float(clearances[worst_id])

    return levels, clearances, worst_id, worst_level, worst_clearance


# ============================================================
# Pre-run feasibility checks  (printed once before rollout)
# ============================================================

def check_reference_prefix_feasibility(ref_traj: np.ndarray, static_rects: list,
                                        num_pts: int = 40) -> bool:
    margin = r_robot + safety_buffer
    n = min(num_pts, ref_traj.shape[1])
    print(f"\n── Reference-prefix feasibility (first {n} pts, zone margin={margin:.3f}) ──")
    any_violation = False
    for k in range(n):
        x, y = ref_traj[0, k], ref_traj[1, k]
        violated, ids = check_static_zone_violation(np.array([x, y]), static_rects)
        tag = f"  ZONE VIOLATION  obs={ids}" if violated else "  ok"
        print(f"  k={k:03d}  ({x:.3f}, {y:.3f}){tag}")
        any_violation |= violated
    print(f"  → prefix feasible: {not any_violation}")
    return not any_violation


def check_initial_state_feasibility(x0: np.ndarray, static_rects: list) -> bool:
    margin = r_robot + safety_buffer
    print(f"\n── Initial-state feasibility  state={x0[:2]}  margin={margin:.3f} ──")
    body_hit, body_ids = check_static_body_collision(x0[:2], static_rects)
    zone_hit, zone_ids = check_static_zone_violation(x0[:2], static_rects)
    print(f"  body collision : {body_hit}  obs={body_ids}")
    print(f"  zone violation : {zone_hit}  obs={zone_ids}")
    return not body_hit


# ============================================================
# Progress extraction  (controller-agnostic)
# ============================================================

def extract_progress(controller, step_result) -> tuple[float, str]:
    """Return (progress_value, label_string) regardless of controller type."""
    if hasattr(controller, "mu"):
        v = float(controller.mu)
        return v, f"mu={v:.3f}"
    if hasattr(controller, "ref_k"):
        v = float(controller.ref_k)
        return v, f"ref_k={int(v)}"
    v = float(step_result.progress_index)
    return v, f"progress_index={int(v)}"


# ============================================================
# Rollout phase  →  returns list[StepRecord]
# ============================================================

def run_rollout(
    controller,
    ref_traj: np.ndarray,
    static_rects: list,
    dyn_obs: list,
    x_start: np.ndarray,
    x_goal: np.ndarray,
) -> list[StepRecord]:
    """
    Pure solve loop.  No matplotlib calls.
    Returns a list of StepRecord objects (one per step executed).
    Dynamic obstacles in *dyn_obs* are stepped in-place; the function
    snapshots their state into each record before stepping.
    """
    records: list[StepRecord] = []
    x_current = x_start.copy()

    for k in range(MAX_STEPS):
        # ── Horizon predictions (before step, for display) ──────────────
        if SHOW_PREDICTIONS:
            pred_sets = []
            for obs in dyn_obs:
                if hasattr(obs, "predict_horizon_with_rects"):
                    pred_sets.append(obs.predict_horizon_with_rects(N, dt, static_rects))
                else:
                    pred_sets.append(obs.predict_horizon(N, dt))
        else:
            pred_sets = None

        # ── Solve ────────────────────────────────────────────────────────
        step_result = controller.solve(x_current, dyn_obs, x_goal)
        progress_value, progress_label = extract_progress(controller, step_result)

        if k % 50 == 0:
            print(
                f"  step {k:4d}  v={step_result.u[0]:.3f}  ω={step_result.u[1]:.3f}"
                f"  {progress_label}  status={step_result.status}"
                f"  active_dyn={step_result.active_dyn_count}"
                f"  solve={1e3*step_result.solve_time_s:.1f}ms"
                f"  pred={1e3*step_result.predict_time_s:.1f}ms"
                f"  pack={1e3*step_result.pack_time_s:.1f}ms"
            )

        # ── Propagate state ──────────────────────────────────────────────
        v, omega = step_result.u
        x_next = np.array([
            x_current[0] + dt * v * np.cos(x_current[2]),
            x_current[1] + dt * v * np.sin(x_current[2]),
            x_current[2] + dt * omega,
        ], dtype=float)

        # ── Step dynamic obstacles ───────────────────────────────────────
        for obs in dyn_obs:
            obs.step(dt, static_rects)

        # ── Snapshot dynamic obstacle state after step ───────────────────
        dyn_snapshot = [(obs.x, obs.y, obs.theta, obs.a, obs.b) for obs in dyn_obs]

        # ── Build record ─────────────────────────────────────────────────
        record = StepRecord(
            step_idx=k,
            state=x_next.copy(),
            u=step_result.u.copy(),
            status=step_result.status,
            ok=step_result.ok,
            progress_index=progress_value,
            dyn_obs_snapshot=dyn_snapshot,
            pred_sets=pred_sets,
            solve_time_s=step_result.solve_time_s,
            select_time_s=step_result.select_time_s,
            predict_time_s=step_result.predict_time_s,
            pack_time_s=step_result.pack_time_s,
            post_time_s=step_result.post_time_s,
            active_dyn_count=step_result.active_dyn_count,
        )

        # ── Collision diagnostics (against live dyn_obs after step) ─────
        check_collisions(record, static_rects, dyn_obs)

        records.append(record)

        if not step_result.ok:
            print(f"  ✗  Solver FAILED at step {k}: {step_result.status}")
            break

        x_current = x_next

        # ── Termination ──────────────────────────────────────────────────
        goal_dist = float(np.linalg.norm(x_current[:2] - x_goal[:2]))
        at_end = progress_value >= (ref_traj.shape[1] - END_PROGRESS_TOL)

        if goal_dist <= GOAL_TOL or at_end:
            print(f"  ✓  Reached goal/end at step {k}  "
                  f"goal_dist={goal_dist:.4f}  {progress_label}")
            break

    return records


# ============================================================
# Rollout summary  (printed after rollout, before playback)
# ============================================================

def print_rollout_summary(records: list[StepRecord]) -> None:
    n = len(records)
    n_static_body  = sum(1 for r in records if r.static_body_collision)
    n_static_zone  = sum(1 for r in records if r.static_zone_violation)
    n_dyn_body     = sum(1 for r in records if r.dyn_body_collision)
    n_dyn_excl     = sum(1 for r in records if r.dyn_exclusion_violation)
    solver_fails   = sum(1 for r in records if not r.ok)
    ended_ok       = records[-1].ok if records else False

    print("\n" + "=" * 60)
    print("ROLLOUT SUMMARY")
    print("=" * 60)
    print(f"  Total steps          : {n}")
    print(f"  Solver failures      : {solver_fails}")
    print(f"  Ended successfully   : {ended_ok}")
    print()
    print("  Static obstacle diagnostics")
    print(f"    body collision steps  : {n_static_body:4d}  "
          f"(robot overlaps *actual* rect body)")
    print(f"    zone violation steps  : {n_static_zone:4d}  "
          f"(robot inside inflated forbidden zone, margin={r_robot+safety_buffer:.3f} m)")
    print()
    print("  Dynamic obstacle diagnostics")
    print(f"    body collision steps  : {n_dyn_body:4d}  "
          f"(robot overlaps *actual* ellipse body)")
    print(f"    exclusion viol. steps : {n_dyn_excl:4d}  "
          f"(robot inside inflated exclusion zone, margin={r_robot+safety_buffer:.3f} m)")
    print("=" * 60 + "\n")


def print_timing_summary(records: list[StepRecord]) -> None:
    if not records:
        print("No timing data available.")
        return

    solve_total   = sum(r.solve_time_s for r in records)
    select_total  = sum(r.select_time_s for r in records)
    predict_total = sum(r.predict_time_s for r in records)
    pack_total    = sum(r.pack_time_s for r in records)
    post_total    = sum(r.post_time_s for r in records)

    n = len(records)
    active_counts = [r.active_dyn_count for r in records]

    print("── Timing summary ──")
    print(f"   avg active dyn obs : {np.mean(active_counts):.2f}")
    print(f"   max active dyn obs : {np.max(active_counts)}")
    print()
    print(f"   solve total   : {solve_total:.3f} s   ({1e3*solve_total/n:.2f} ms/step)")
    print(f"   select total  : {select_total:.3f} s   ({1e3*select_total/n:.2f} ms/step)")
    print(f"   predict total : {predict_total:.3f} s   ({1e3*predict_total/n:.2f} ms/step)")
    print(f"   pack total    : {pack_total:.3f} s   ({1e3*pack_total/n:.2f} ms/step)")
    print(f"   post total    : {post_total:.3f} s   ({1e3*post_total/n:.2f} ms/step)")


def print_dyn_exclusion_depth_summary(records: list[StepRecord]) -> None:
    flagged = [r for r in records if r.worst_dyn_excl_depth > 0.0]
    if not flagged:
        print("No dynamic exclusion violations to diagnose.")
        return

    print(f"── Dynamic exclusion violation depth  ({len(flagged)} flagged steps) ──")
    print(f"   margin={r_robot+safety_buffer:.4f}  (threshold = {VIOLATION_THRESHOLD:.9f})")
    print(f"{'step':>7}  {'obs':>4}  {'level':>12}  {'depth':>12}  status")
    print("   " + "-" * 60)

    worst = None
    for r in flagged:
        print(f"{r.step_idx:7d}  "
              f"D{r.worst_dyn_excl_id:<3d}  "
              f"{r.worst_dyn_excl_level:12.9f}  "
              f"{r.worst_dyn_excl_depth:12.3e}  "
              f"{r.status}")
        if worst is None or r.worst_dyn_excl_depth > worst.worst_dyn_excl_depth:
            worst = r

    print()
    print(f"   Worst: step={worst.step_idx}  obs=D{worst.worst_dyn_excl_id}  "
          f"depth={worst.worst_dyn_excl_depth:.3e}")
    


def print_dyn_body_proximity_summary(records: list[StepRecord]) -> None:
    if not records:
        print("No records available for dynamic body proximity summary.")
        return

    worst = min(
        (r for r in records if r.worst_dyn_body_clearance is not None),
        key=lambda r: r.worst_dyn_body_clearance,
        default=None,
    )

    if worst is None:
        print("No dynamic-body proximity data available.")
        return

    print(f"── Dynamic body proximity summary ──")
    print(f"   Worst step : {worst.step_idx}")
    print(f"   Worst obs  : D{worst.worst_dyn_body_id}")
    print(f"   body level : {worst.worst_dyn_body_level:.9f}")
    print(f"   clearance  : {worst.worst_dyn_body_clearance:.6f}")
    print()
    print("   INTERPRETATION")
    print("   ──────────────────────────────────────────────────────")
    print("   clearance > 0   → outside actual dynamic body")
    print("   clearance = 0   → on body boundary")
    print("   clearance < 0   → true body collision")


# ============================================================
# Visualization helpers
# ============================================================

def _make_status_text(record: StepRecord) -> str:
    """Thesis-clean status string shown in the plot annotation box."""

    dyn_depth_str = (
        f"D{record.worst_dyn_excl_id}  level={record.worst_dyn_excl_level:.6f}  "
        f"depth={record.worst_dyn_excl_depth:.3e}"
        if record.worst_dyn_excl_id is not None else "none"
    )

    dyn_body_str = (
        f"D{record.worst_dyn_body_id}  level={record.worst_dyn_body_level:.6f}  "
        f"clr={record.worst_dyn_body_clearance:.3e}"
        if record.worst_dyn_body_id is not None else "none"
    )

    lines = [
        f"step: {record.step_idx}",
        f"progress: {record.progress_index:.2f}",
        f"v: {record.u[0]:.3f}   ω: {record.u[1]:.3f}",
        f"solver: {record.status}",
        "",
        f"static body collision : {'YES  ' + str(record.static_body_ids) if record.static_body_collision else 'no'}",
        f"static zone violation : {'YES  ' + str(record.static_zone_ids) if record.static_zone_violation else 'no'}",
        f"dyn body collision    : {'YES  ' + str(record.dyn_body_ids)    if record.dyn_body_collision    else 'no'}",
        f"dyn exclusion viol.   : {'YES  ' + str(record.dyn_exclusion_ids) if record.dyn_exclusion_violation else 'no'}",
        f"worst dyn excl depth : {dyn_depth_str}",
        f"worst dyn body prox  : {dyn_body_str}",
    ]
    return "\n".join(lines)


def _annotation_color(record: StepRecord) -> str:
    if record.static_body_collision or record.dyn_body_collision:
        return "#ffcccc"   # red tint → true collision
    if record.static_zone_violation or record.dyn_exclusion_violation:
        return "#fff3cc"   # yellow tint → zone penetration
    return "white"


def compute_plot_limits(ref_traj, static_rects, dyn_obs, margin=1.5):
    xs = [ref_traj[0, :].min(), ref_traj[0, :].max()]
    ys = [ref_traj[1, :].min(), ref_traj[1, :].max()]
    for (cx, cy, hw, hh) in static_rects:
        xs += [cx - hw, cx + hw]
        ys += [cy - hh, cy + hh]
    for obs in dyn_obs:
        xs += [obs.x - obs.a, obs.x + obs.a]
        ys += [obs.y - obs.b, obs.y + obs.b]
    return min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin


def draw_frame(
    ax,
    ref_traj: np.ndarray,
    static_rects: list,
    record: StepRecord,
    traj_xy: list[np.ndarray],   # all positions up to this step
    algo: str,
    path_id: int,
) -> None:
    """Render one frame from a StepRecord.  No solver calls."""
    ax.clear()

    # ── Reference path ───────────────────────────────────────────────
    ax.plot(ref_traj[0, :], ref_traj[1, :], "--", lw=1.5, label="reference", color="steelblue")

    # ── Travelled path ───────────────────────────────────────────────
    if len(traj_xy) > 1:
        traj_np = np.array(traj_xy)
        ax.plot(traj_np[:, 0], traj_np[:, 1], lw=2.0, label="robot path", color="tab:orange")

    # ── Progress anchor on reference ─────────────────────────────────
    k_ref = int(np.clip(record.progress_index, 0, ref_traj.shape[1] - 1))
    ax.plot(
        ref_traj[0, k_ref], ref_traj[1, k_ref],
        marker="o",
        markersize=7,
        linestyle="None",
        color="yellow",
        markeredgecolor="black",
        markeredgewidth=0.8,
        zorder=8,
        label="progress anchor",
    )

    # ── Static obstacles (body + inflated zone) ──────────────────────
    zone_margin = r_robot + safety_buffer
    for i, (cx, cy, hw, hh) in enumerate(static_rects):
        body_color = "red" if i in record.static_body_ids else "black"
        zone_color = "orange" if i in record.static_zone_ids else "gray"

        ax.add_patch(Rectangle(
            (cx - hw, cy - hh), 2 * hw, 2 * hh,
            fill=True, facecolor="#e0e0e0", edgecolor=body_color, lw=2.0))
        ax.text(cx, cy, f"S{i}", ha="center", va="center", fontsize=8)

        ax.add_patch(Rectangle(
            (cx - hw - zone_margin, cy - hh - zone_margin),
            2 * (hw + zone_margin), 2 * (hh + zone_margin),
            fill=False, edgecolor=zone_color, lw=1.0, ls=":", alpha=0.6))

    # ── Dynamic obstacles (body + exclusion zone) ────────────────────
    for i, (ox, oy, oth, oa, ob_) in enumerate(record.dyn_obs_snapshot):
        body_color = "red" if i in record.dyn_body_ids else "tab:blue"
        excl_color = "orange" if i in record.dyn_exclusion_ids else "cornflowerblue"

        ax.add_patch(Ellipse(
            (ox, oy), 2 * oa, 2 * ob_,
            angle=np.degrees(oth),
            fill=True, facecolor="#cce5ff", edgecolor=body_color, lw=2.0))

        ax.add_patch(Ellipse(
            (ox, oy),
            2 * (oa + zone_margin), 2 * (ob_ + zone_margin),
            angle=np.degrees(oth),
            fill=False, edgecolor=excl_color, lw=1.0, ls=":", alpha=0.6))

        ax.text(ox + 0.12, oy + 0.10, f"D{i}", fontsize=8, ha="left", va="bottom")
        hx = 0.5 * oa * np.cos(oth)
        hy = 0.5 * oa * np.sin(oth)
        ax.arrow(ox, oy, hx, hy, head_width=0.08, head_length=0.10,
                 length_includes_head=True, color=body_color)

    # ── Horizon predictions ──────────────────────────────────────────
    if SHOW_PREDICTIONS and record.pred_sets is not None:
        for i, (xs, ys, _) in enumerate(record.pred_sets):
            ax.scatter(xs, ys, s=PRED_MARKER_SIZE, alpha=0.8, zorder=4)

    # ── Robot center-state (point robot in inflated-obstacle model) ──
    rx, ry, rth = record.state
    robot_color = "red" if (record.static_body_collision or record.dyn_body_collision) else "tab:green"
    ax.scatter([rx], [ry], s=ROBOT_DOT_SIZE, color=robot_color, zorder=6, label="robot center")
    hx = ROBOT_HEADING_LEN * np.cos(rth)
    hy = ROBOT_HEADING_LEN * np.sin(rth)
    ax.arrow(rx, ry, hx, hy, head_width=0.08, head_length=0.10,
             length_includes_head=True, color=robot_color, zorder=6)

    # ── Annotation box ───────────────────────────────────────────────
    status_text = _make_status_text(record)
    box_color   = _annotation_color(record)
    ax.text(0.01, 0.99, status_text,
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.88))

    ax.set_title(
        f"{algo.upper()} Debug  │  path_id={path_id}  │  step={record.step_idx}",
        fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8)


# ============================================================
# Playback phase
# ============================================================

def run_playback(
    records: list[StepRecord],
    ref_traj: np.ndarray,
    static_rects: list,
    algo: str,
    path_id: int,
    x_start: np.ndarray,
    pause_sec: float = 0.001,
) -> None:
    """Replay stored StepRecords frame-by-frame.  No solver calls."""
    # Pre-compute axis limits from the full trajectory
    all_x = [r.state[0] for r in records]
    all_y = [r.state[1] for r in records]
    dummy_obs = [
        type("_O", (), {"x": ox, "y": oy, "a": oa, "b": ob_})()
        for r in records[:1]
        for (ox, oy, _, oa, ob_) in r.dyn_obs_snapshot
    ]
    xmin, xmax, ymin, ymax = compute_plot_limits(ref_traj, static_rects, dummy_obs)
    xmin = min(xmin, min(all_x) - 1.5)
    xmax = max(xmax, max(all_x) + 1.5)
    ymin = min(ymin, min(all_y) - 1.5)
    ymax = max(ymax, max(all_y) + 1.5)

    plt.ion()
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.22, right=0.98, top=0.93, bottom=0.08)
    maximize_figure_window(fig)


    traj_xy: list[np.ndarray] = [x_start[:2].copy()]

    for record in records:
        draw_frame(ax, ref_traj, static_rects, record, traj_xy, algo, path_id)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        traj_xy.append(record.state[:2].copy())
        plt.pause(pause_sec)

    plt.ioff()
    plt.show()


# ============================================================
# Dynamic obstacle factory
# ============================================================

def make_debug_dyn_obs(path_id: int, seed_offset: int = 0) -> list:
    dyn_obs = []

    if path_id == 10:
        pts = [
            (11, 33), (16, 24), (21, 23), (18, 19), (28, 16),
            (32, 15), (33, 13), (35, 10), (35, 8),  (25, 21),
        ]
        for i, (x, y) in enumerate(pts):
            dyn_obs.append(DynamicObstacle(
                x=float(x), y=float(y), a=0.6, b=0.4,
                seed=100 + seed_offset * 100 + i,
                v_max=0.30, change_interval=25,
            ))

    elif path_id == 11:
        pts = [
            (9, 2), (11, 4), (13, 7), (15, 8), (19, 9), 
            (20, 13), (17, 15), (20, 19), (25, 22),  (23, 26), (28, 30), (32, 32),
        ]
        for i, (x, y) in enumerate(pts):
            dyn_obs.append(DynamicObstacle(
                x=float(x), y=float(y), a=0.6, b=0.4,
                seed=100 + seed_offset * 100 + i,
                v_max=0.30, change_interval=25,
            ))

    else:
        dyn_obs = Obstacle(path_id).dynamic_obs_seeded(seed_offset)

    return dyn_obs


def maximize_figure_window(fig):
    manager = plt.get_current_fig_manager()
    try:
        manager.window.state('zoomed')          # TkAgg on Windows
        return
    except Exception:
        pass
    try:
        manager.window.showMaximized()          # Qt backend
        return
    except Exception:
        pass
    try:
        manager.frame.Maximize(True)            # wx backend
        return
    except Exception:
        pass




# ============================================================
# Main
# ============================================================

def main() -> None:
    # ── Environment setup ────────────────────────────────────────────
    env        = Obstacle(PATH_ID)
    ref_traj   = env.path_selector(PATH_ID)
    START_TRIM = 8
    ref_traj   = ref_traj[:, START_TRIM:]

    static_rects = env.static_obs()

    if ENABLE_DEBUG_DYN_OBS:
        dyn_obs = (env.dynamic_obs_seeded(SEED_OFFSET)
                   if DEBUG_DYN_OBS_MODE == "env"
                   else make_debug_dyn_obs(PATH_ID, SEED_OFFSET))
    else:
        dyn_obs = []

    print("=" * 60)
    print(f"ALGO={ALGO.upper()}  USE_HARD={USE_HARD}  path_id={PATH_ID}")
    print(f"ref_traj shape: {ref_traj.shape}")
    print(f"static rects  : {len(static_rects)}")
    print(f"dynamic obs   : {len(dyn_obs)}")
    print(f"r_robot={r_robot}  safety_buffer={safety_buffer}"
          f"  total_margin={r_robot+safety_buffer:.3f}")

    # ── Initial state ────────────────────────────────────────────────
    p0      = ref_traj[:, 0]
    look_k  = min(10, ref_traj.shape[1] - 1)
    th0     = float(np.arctan2(ref_traj[1, look_k] - p0[1],
                               ref_traj[0, look_k] - p0[0]))
    x_start = np.array([p0[0], p0[1], th0], dtype=float)
    x_goal  = np.array([ref_traj[0, -1], ref_traj[1, -1], 0.0], dtype=float)
    print(f"x_start={x_start}  x_goal={x_goal[:2]}")

    # ── Optional pre-run sanity checks ───────────────────────────────
    check_initial_state_feasibility(x_start, static_rects)
    # check_reference_prefix_feasibility(ref_traj, static_rects, num_pts=60)

    # ── Build controller ─────────────────────────────────────────────
    algo = ALGO.lower()
    if algo == "mpc":
        controller = MPCController(
            static_rects=static_rects,
            dyn_obs=dyn_obs,
            ref_traj=ref_traj,
            use_hard=USE_HARD,
        )
    elif algo == "mpcc":
        controller = MPCCController(
            static_rects=static_rects,
            dyn_obs=dyn_obs,
            ref_traj=ref_traj,
            use_hard=USE_HARD,
        )
    else:
        raise ValueError(f"Unknown ALGO='{ALGO}'")

    # ================================================================
    # PHASE 1 : ROLLOUT  (solve-only, no plotting)
    # ================================================================
    print("\n── Phase 1: rollout ──")
    records = run_rollout(
        controller=controller,
        ref_traj=ref_traj,
        static_rects=static_rects,
        dyn_obs=dyn_obs,
        x_start=x_start,
        x_goal=x_goal,
    )

    print_rollout_summary(records)
    print_dyn_exclusion_depth_summary(records)
    print_dyn_body_proximity_summary(records)

    # ================================================================
    # PHASE 2 : PLAYBACK  (visualise stored records, no solving)
    # ================================================================
    print("── Phase 2: playback ──")
    
    run_playback(
        records=records,
        ref_traj=ref_traj,
        static_rects=static_rects,
        algo=ALGO,
        path_id=PATH_ID,
        x_start=x_start,
        pause_sec=0.001,
    )


if __name__ == "__main__":
    main()