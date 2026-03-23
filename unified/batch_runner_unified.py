from __future__ import annotations

import csv
import math
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


from dataclasses import asdict
from statistics import mean, pstdev
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed




import numpy as np

from obs import Obstacle
from params import dt, r_robot, safety_buffer
from run_experiment_unified import (
    MPCCController,
    MPCController,
    _compute_common_tracking_errors,
)
from debug import (
    GOAL_TOL,
    END_PROGRESS_TOL,
    VIOLATION_THRESHOLD,
    check_collisions,
    extract_progress,
    make_debug_dyn_obs,
    StepRecord,
)

# ============================================================
# Batch configuration
# ============================================================
RESULTS_DIR = "results"
N_SEEDS = 10
MAX_WORKERS = 14
MAX_STEPS = 800

# Match the thesis-clean debug workflow by default.
SCENARIOS = [
    {"path_id": 10, "label": "CorridorStructured"},
    {"path_id": 11, "label": "OpenClutter"},
]

USE_DEBUG_DYN_OBS = True
DEBUG_DYN_OBS_MODE = "manual"  # "manual" or "env"
START_TRIM = 8
INITIAL_HEADING_LOOKAHEAD = 10

RAW_FIELDS = [
    "scenario",
    "controller",
    "path_id",
    "seed_offset",
    "goal_reached",
    "steps_taken",
    "completion_time_s",
    "path_length",
    "path_efficiency",
    "mean_speed",
    "std_speed",
    "smoothness",
    "mean_cont_err",
    "rms_cont_err",
    "max_cont_err",
    "std_cont_err",
    "mean_lag_err",
    "rms_lag_err",
    "max_lag_err",
    "std_lag_err",
    "mean_solve_ms",
    "max_solve_ms",
    "std_solve_ms",
    "solver_fail_count",
    "static_body_collision_steps",
    "static_zone_violation_steps",
    "dyn_body_collision_steps",
    "dyn_exclusion_violation_steps",
    "total_body_collision_steps",
    "worst_dyn_excl_depth",
    "worst_dyn_excl_step",
    "worst_dyn_excl_obs",
    "min_dyn_body_clearance",
    "min_dyn_body_clearance_step",
    "min_dyn_body_clearance_obs",
    "mean_active_dyn_obs",
    "max_active_dyn_obs",
    "mean_state_mismatch",
    "max_state_mismatch",
    "min_clearance_m",
    "error",
]

SUMMARY_METRICS = [
    "goal_reached",
    "completion_time_s",
    "path_efficiency",
    "mean_speed",
    "smoothness",
    "mean_cont_err",
    "rms_cont_err",
    "max_cont_err",
    "mean_lag_err",
    "rms_lag_err",
    "mean_solve_ms",
    "max_solve_ms",
    "solver_fail_count",
    "static_body_collision_steps",
    "static_zone_violation_steps",
    "dyn_body_collision_steps",
    "dyn_exclusion_violation_steps",
    "total_body_collision_steps",
    "worst_dyn_excl_depth",
    "min_dyn_body_clearance",
    "min_clearance_m",
]


# ============================================================
# Environment / rollout helpers
# ============================================================
def make_dyn_obs(path_id: int, seed_offset: int) -> list:
    env = Obstacle(path_id)
    if USE_DEBUG_DYN_OBS:
        if DEBUG_DYN_OBS_MODE == "env":
            return env.dynamic_obs_seeded(seed_offset)
        return make_debug_dyn_obs(path_id, seed_offset)
    return env.dynamic_obs_seeded(seed_offset)


def build_env(path_id: int, seed_offset: int) -> tuple[np.ndarray, list, list, np.ndarray, np.ndarray]:
    env = Obstacle(path_id)
    ref_traj = env.path_selector(path_id)
    if START_TRIM > 0:
        ref_traj = ref_traj[:, START_TRIM:]

    static_rects = env.static_obs()
    dyn_obs = make_dyn_obs(path_id, seed_offset)

    p0 = ref_traj[:, 0]
    look_k = min(INITIAL_HEADING_LOOKAHEAD, ref_traj.shape[1] - 1)
    th0 = float(np.arctan2(ref_traj[1, look_k] - p0[1], ref_traj[0, look_k] - p0[0]))
    x_start = np.array([p0[0], p0[1], th0], dtype=float)
    x_goal = np.array([ref_traj[0, -1], ref_traj[1, -1], 0.0], dtype=float)
    return ref_traj, static_rects, dyn_obs, x_start, x_goal


def build_controller(controller_type: str, static_rects: list, dyn_obs: list, ref_traj: np.ndarray, use_hard: bool):
    controller_type = controller_type.lower()
    if controller_type == "mpc":
        return MPCController(static_rects=static_rects, dyn_obs=dyn_obs, ref_traj=ref_traj, use_hard=use_hard)
    if controller_type == "mpcc":
        return MPCCController(static_rects=static_rects, dyn_obs=dyn_obs, ref_traj=ref_traj, use_hard=use_hard)
    raise ValueError(f"Unknown controller_type={controller_type!r}")


# ============================================================
# Metrics aggregation
# ============================================================
def _safe_mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _safe_std(xs: list[float]) -> float:
    return float(np.std(xs)) if xs else 0.0


def run_single_experiment(
    controller_type: str,
    path_id: int,
    seed_offset: int,
    use_hard: bool,
    max_steps: int = MAX_STEPS,
) -> dict[str, Any]:
    ref_traj, static_rects, dyn_obs, x_start, x_goal = build_env(path_id, seed_offset)
    controller = build_controller(controller_type, static_rects, dyn_obs, ref_traj, use_hard)

    x_current = x_start.copy()
    prev_pos = x_current[:2].copy()
    straight_line_dist = float(np.linalg.norm(ref_traj[:, -1] - ref_traj[:, 0]))

    solve_times_ms: list[float] = []
    speeds: list[float] = []
    omegas: list[float] = []
    cont_errors: list[float] = []
    lag_errors: list[float] = []
    active_dyn_counts: list[int] = []
    state_mismatches: list[float] = []

    path_length = 0.0
    solver_fail_count = 0
    steps_taken = 0

    static_body_collision_steps = 0
    static_zone_violation_steps = 0
    dyn_body_collision_steps = 0
    dyn_exclusion_violation_steps = 0

    worst_dyn_excl_depth = 0.0
    worst_dyn_excl_step: int | None = None
    worst_dyn_excl_obs: int | None = None

    min_dyn_body_clearance = math.inf
    min_dyn_body_clearance_step: int | None = None
    min_dyn_body_clearance_obs: int | None = None

    min_clearance_m = math.inf
    goal_reached = False

    for k in range(max_steps):
        step_result = controller.solve(x_current, dyn_obs, x_goal)
        progress_value, _ = extract_progress(controller, step_result)

        solve_times_ms.append(1000.0 * float(step_result.solve_time_s))
        active_dyn_counts.append(int(getattr(step_result, "active_dyn_count", 0)))
        state_mismatches.append(float(getattr(step_result, "state_mismatch_norm", 0.0)))

        if step_result.status != "Solve_Succeeded":
            solver_fail_count += 1

        if not step_result.ok:
            break

        v, omega = float(step_result.u[0]), float(step_result.u[1])
        speeds.append(v)
        omegas.append(abs(omega))

        x_next = np.array([
            x_current[0] + dt * v * np.cos(x_current[2]),
            x_current[1] + dt * v * np.sin(x_current[2]),
            x_current[2] + dt * omega,
        ], dtype=float)

        for obs in dyn_obs:
            obs.step(dt, static_rects)

        record = StepRecord(
            step_idx=k,
            state=x_next.copy(),
            u=np.array([v, omega], dtype=float),
            status=step_result.status,
            ok=step_result.ok,
            progress_index=float(progress_value),
            dyn_obs_snapshot=[(obs.x, obs.y, obs.theta, obs.a, obs.b) for obs in dyn_obs],
            pred_sets=None,
            solve_time_s=float(step_result.solve_time_s),
            select_time_s=float(getattr(step_result, "select_time_s", 0.0)),
            predict_time_s=float(getattr(step_result, "predict_time_s", 0.0)),
            pack_time_s=float(getattr(step_result, "pack_time_s", 0.0)),
            post_time_s=float(getattr(step_result, "post_time_s", 0.0)),
            active_dyn_count=int(getattr(step_result, "active_dyn_count", 0)),
        )
        check_collisions(record, static_rects, dyn_obs)

        static_body_collision_steps += int(record.static_body_collision)
        static_zone_violation_steps += int(record.static_zone_violation)
        dyn_body_collision_steps += int(record.dyn_body_collision)
        dyn_exclusion_violation_steps += int(record.dyn_exclusion_violation)

        if record.worst_dyn_excl_depth > worst_dyn_excl_depth:
            worst_dyn_excl_depth = float(record.worst_dyn_excl_depth)
            worst_dyn_excl_step = k
            worst_dyn_excl_obs = record.worst_dyn_excl_id

        if record.worst_dyn_body_clearance is not None and record.worst_dyn_body_clearance < min_dyn_body_clearance:
            min_dyn_body_clearance = float(record.worst_dyn_body_clearance)
            min_dyn_body_clearance_step = k
            min_dyn_body_clearance_obs = record.worst_dyn_body_id

        # Distance-style minimum clearance metric, matching run_experiment_unified semantics.
        for obs in dyn_obs:
            dx = x_next[0] - obs.x
            dy = x_next[1] - obs.y
            c, s = np.cos(obs.theta), np.sin(obs.theta)
            x_body = c * dx + s * dy
            y_body = -s * dx + c * dy
            d = np.sqrt((x_body / obs.a) ** 2 + (y_body / obs.b) ** 2)
            clearance_dyn = (d - 1.0) * np.sqrt(obs.a * obs.b)
            min_clearance_m = min(min_clearance_m, float(clearance_dyn))

        for (cx, cy, hw, hh) in static_rects:
            dx_s = max(abs(x_next[0] - cx) - hw, 0.0)
            dy_s = max(abs(x_next[1] - cy) - hh, 0.0)
            clearance_s = np.sqrt(dx_s ** 2 + dy_s ** 2) - r_robot
            min_clearance_m = min(min_clearance_m, float(clearance_s))

        cont_err, lag_err = _compute_common_tracking_errors(x_next, ref_traj)
        cont_errors.append(float(abs(cont_err)))
        lag_errors.append(float(lag_err))

        steps_taken += 1
        path_length += float(np.linalg.norm(x_next[:2] - prev_pos))
        prev_pos = x_next[:2].copy()
        x_current = x_next

        goal_dist = float(np.linalg.norm(x_current[:2] - x_goal[:2]))
        at_end = float(progress_value) >= (ref_traj.shape[1] - END_PROGRESS_TOL)
        if goal_dist <= GOAL_TOL or at_end:
            goal_reached = True
            break

    cont_arr = np.asarray(cont_errors, dtype=float) if cont_errors else np.zeros(1)
    lag_arr = np.asarray(lag_errors, dtype=float) if lag_errors else np.zeros(1)
    solve_arr = np.asarray(solve_times_ms, dtype=float) if solve_times_ms else np.zeros(1)
    speed_arr = np.asarray(speeds, dtype=float) if speeds else np.zeros(1)
    omega_arr = np.asarray(omegas, dtype=float) if omegas else np.zeros(1)

    controller_label = f"{controller_type.lower()}_{'hard' if use_hard else 'soft'}"
    total_body_collision_steps = static_body_collision_steps + dyn_body_collision_steps
    path_efficiency = float(straight_line_dist / path_length) if path_length > 1e-12 else 0.0

    return {
        "controller": controller_label,
        "path_id": path_id,
        "seed_offset": seed_offset,
        "goal_reached": int(goal_reached),
        "steps_taken": int(steps_taken),
        "completion_time_s": round(float(steps_taken * dt), 4),
        "path_length": round(float(path_length), 4),
        "path_efficiency": round(float(path_efficiency), 4),
        "mean_speed": round(float(np.mean(speed_arr)), 4),
        "std_speed": round(float(np.std(speed_arr)), 4),
        "smoothness": round(float(np.sum(omega_arr) * dt), 4),
        "mean_cont_err": round(float(np.mean(cont_arr)), 4),
        "rms_cont_err": round(float(np.sqrt(np.mean(cont_arr ** 2))), 4),
        "max_cont_err": round(float(np.max(cont_arr)), 4),
        "std_cont_err": round(float(np.std(cont_arr)), 4),
        "mean_lag_err": round(float(np.mean(lag_arr)), 4),
        "rms_lag_err": round(float(np.sqrt(np.mean(lag_arr ** 2))), 4),
        "max_lag_err": round(float(np.max(np.abs(lag_arr))), 4),
        "std_lag_err": round(float(np.std(lag_arr)), 4),
        "mean_solve_ms": round(float(np.mean(solve_arr)), 2),
        "max_solve_ms": round(float(np.max(solve_arr)), 2),
        "std_solve_ms": round(float(np.std(solve_arr)), 2),
        "solver_fail_count": int(solver_fail_count),
        "static_body_collision_steps": int(static_body_collision_steps),
        "static_zone_violation_steps": int(static_zone_violation_steps),
        "dyn_body_collision_steps": int(dyn_body_collision_steps),
        "dyn_exclusion_violation_steps": int(dyn_exclusion_violation_steps),
        "total_body_collision_steps": int(total_body_collision_steps),
        "worst_dyn_excl_depth": round(float(worst_dyn_excl_depth), 6),
        "worst_dyn_excl_step": worst_dyn_excl_step,
        "worst_dyn_excl_obs": worst_dyn_excl_obs,
        "min_dyn_body_clearance": round(float(min_dyn_body_clearance), 6) if min_dyn_body_clearance < math.inf else None,
        "min_dyn_body_clearance_step": min_dyn_body_clearance_step,
        "min_dyn_body_clearance_obs": min_dyn_body_clearance_obs,
        "mean_active_dyn_obs": round(float(np.mean(active_dyn_counts)) if active_dyn_counts else 0.0, 4),
        "max_active_dyn_obs": int(max(active_dyn_counts)) if active_dyn_counts else 0,
        "mean_state_mismatch": round(float(np.mean(state_mismatches)) if state_mismatches else 0.0, 10),
        "max_state_mismatch": round(float(np.max(state_mismatches)) if state_mismatches else 0.0, 10),
        "min_clearance_m": round(float(min_clearance_m), 6) if min_clearance_m < math.inf else None,
        "error": "",
    }


# ============================================================
# CSV helpers
# ============================================================
def _write_csv(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarize_rows(rows: list[dict[str, Any]], controller_label: str) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        label = scenario["label"]
        runs = [r for r in rows if r.get("scenario") == label and not r.get("error")]
        if not runs:
            continue

        row: dict[str, Any] = {
            "scenario": label,
            "controller": controller_label,
            "n_runs": len(runs),
        }
        for metric in SUMMARY_METRICS:
            vals = [float(r[metric]) for r in runs if r.get(metric) is not None and r.get(metric) != ""]
            if not vals:
                continue
            row[f"{metric}_mean"] = round(float(np.mean(vals)), 6)
            row[f"{metric}_std"] = round(float(np.std(vals)), 6)
        summary_rows.append(row)
    return summary_rows


def _run_job(job: tuple[str, bool, dict[str, Any], int]) -> dict[str, Any]:
    controller_type, use_hard, scenario, seed = job
    path_id = int(scenario["path_id"])
    label = str(scenario["label"])

    try:
        result = run_single_experiment(
            controller_type=controller_type,
            path_id=path_id,
            seed_offset=seed,
            use_hard=use_hard,
            max_steps=MAX_STEPS,
        )
        result["scenario"] = label
        return result
    except Exception as exc:
        return {
            "scenario": label,
            "controller": f"{controller_type}_{'hard' if use_hard else 'soft'}",
            "path_id": path_id,
            "seed_offset": seed,
            "error": str(exc),
        }



# ============================================================
# Batch execution
# ============================================================
def run_batch(controller_type: str, use_hard: bool) -> tuple[str, str]:
    controller_type = controller_type.lower()
    controller_label = f"{controller_type}_{'hard' if use_hard else 'soft'}"
    raw_csv = os.path.join(RESULTS_DIR, f"{controller_label}_results.csv")
    summary_csv = os.path.join(RESULTS_DIR, f"{controller_label}_summary.csv")

    jobs: list[tuple[str, bool, dict[str, Any], int]] = []
    for scenario in SCENARIOS:
        for seed in range(N_SEEDS):
            jobs.append((controller_type, use_hard, scenario, seed))

    print("\n" + "=" * 68)
    print(f"  {controller_label.upper()}   launching {len(jobs)} runs with {MAX_WORKERS} workers")
    print("=" * 68)

    rows: list[dict[str, Any]] = []
    completed = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_run_job, job): job for job in jobs}

        for fut in as_completed(futures):
            result = fut.result()
            rows.append(result)
            completed += 1

            if result.get("error"):
                print(
                    f"[{completed:3d}/{len(jobs):3d}] "
                    f"{result.get('scenario','?')} seed={result.get('seed_offset','?')} "
                    f"ERROR: {result['error']}"
                )
            else:
                print(
                    f"[{completed:3d}/{len(jobs):3d}] "
                    f"{result['scenario']} seed={result['seed_offset']} "
                    f"goal={result['goal_reached']} steps={result['steps_taken']} "
                    f"solve={result['mean_solve_ms']:.1f}ms "
                    f"static_zone={result['static_zone_violation_steps']} "
                    f"dyn_excl={result['dyn_exclusion_violation_steps']} "
                    f"dyn_body={result['dyn_body_collision_steps']}"
                )

    _write_csv(raw_csv, rows, RAW_FIELDS)
    summary_rows = _summarize_rows(rows, controller_label)
    if summary_rows:
        summary_fields = sorted(
            {k for row in summary_rows for k in row.keys()},
            key=lambda x: (x not in {"scenario", "controller", "n_runs"}, x),
        )
        _write_csv(summary_csv, summary_rows, summary_fields)
    else:
        _write_csv(summary_csv, [], ["scenario", "controller", "n_runs"])

    print(f"\nRaw results saved to   {raw_csv}")
    print(f"Summary saved to      {summary_csv}")
    return raw_csv, summary_csv



def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for controller_type in ("mpcc", "mpc"):
        for use_hard in (False, True):
            run_batch(controller_type, use_hard)


if __name__ == "__main__":
    main()
