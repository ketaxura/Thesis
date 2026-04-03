"""
sensitivity_study.py
=====================
Sensitivity sweep for MPCC soft/hard controllers.

Varied weights (one at a time, others at default):
  q_cont  : [1.5, 3.0, 6.0]
  q_vs    : [5.0, 10.0, 20.0]
  rho_dyn : [5e4, 1e5, 2e5]

Design:
  - 2 controllers  (mpcc_soft, mpcc_hard)
  - 3 weight groups × 3 values each  = 9 weight configurations
  - 2 maps  (CorridorStructured path_id=10, OpenClutter path_id=11)
  - 3 congestion tiers  (low, mid, high)
  - 10 seed runs each

Total runs: 9 configs × 2 controllers × 2 maps × 3 tiers × 10 seeds = 1 080
(matches 720 "per weight set" if you count only the MPCC variant, 1080 total with both)

Output:
  sensitivity_results/sensitivity_raw.csv
  sensitivity_results/sensitivity_summary.csv
"""

from __future__ import annotations

import csv
import math
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Project imports (same assumptions as batch_runner_unified.py)
# ---------------------------------------------------------------------------
from obs import Obstacle
from params import dt, r_robot, safety_buffer
from run_experiment_unified import (
    MPCCController,
    _compute_common_tracking_errors,
)
from mpcc import MPCCConfig, build_mpcc_solver
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
# Study configuration
# ============================================================
RESULTS_DIR = "sensitivity_results"
N_SEEDS = 10
MAX_WORKERS = 16
MAX_STEPS = 800

SCENARIOS = [
    {"path_id": 10, "label": "CorridorStructured"},
    {"path_id": 11, "label": "OpenClutter"},
]

CONGESTION_TIERS = ["low", "mid", "high"]

USE_DEBUG_DYN_OBS = True
DYN_OBS_MODE = "path_tier"
START_TRIM = 8
INITIAL_HEADING_LOOKAHEAD = 10

# ---------------------------------------------------------------------------
# Default MPCC config (used as the baseline; only the swept param changes)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = MPCCConfig()  # q_cont=3.0, q_vs=10.0, rho_dyn=1e5, etc.

# ---------------------------------------------------------------------------
# Sensitivity sweep definitions
# ---------------------------------------------------------------------------
# Each entry: (group_name, param_name, value)
SWEEP_CONFIGS: list[tuple[str, str, float]] = []

for v in [1.5, 3.0, 6.0]:
    SWEEP_CONFIGS.append(("q_cont", "q_cont", v))

for v in [5.0, 10.0, 20.0]:
    SWEEP_CONFIGS.append(("q_vs", "q_vs", v))

for v in [5e4, 1e5, 2e5]:
    SWEEP_CONFIGS.append(("rho_dyn", "rho_dyn", v))

# ============================================================
# CSV field definitions
# ============================================================
RAW_FIELDS = [
    "sweep_group",
    "sweep_param",
    "sweep_value",
    "controller",
    "scenario",
    "path_id",
    "seed_offset",
    "congestion_tier",
    "termination_reason",
    "spawned_dyn_obs_count",
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
    "dyn_body_collision_steps",
    "dyn_exclusion_violation_steps",
    "total_body_collision_steps",
    "worst_dyn_excl_depth",
    "min_dyn_body_clearance",
    "min_clearance_m",
]

# ============================================================
# Environment helpers (mirrored from batch_runner_unified.py)
# ============================================================

def _make_dyn_obs(
    path_id, seed_offset, ref_traj, static_rects, x_start, x_goal, congestion_tier
):
    if USE_DEBUG_DYN_OBS:
        return make_debug_dyn_obs(
            path_id,
            seed_offset,
            mode=DYN_OBS_MODE,
            ref_traj=ref_traj,
            static_rects=static_rects,
            x_start=x_start,
            x_goal=x_goal,
            tier=congestion_tier,
        )
    return Obstacle(path_id).dynamic_obs_seeded(seed_offset)


def _build_env(path_id: int, seed_offset: int, congestion_tier: str):
    env = Obstacle(path_id)
    ref_traj = env.path_selector(path_id)
    if START_TRIM > 0:
        ref_traj = ref_traj[:, START_TRIM:]

    static_rects = env.static_obs()

    p0 = ref_traj[:, 0]
    look_k = min(INITIAL_HEADING_LOOKAHEAD, ref_traj.shape[1] - 1)
    th0 = float(np.arctan2(ref_traj[1, look_k] - p0[1], ref_traj[0, look_k] - p0[0]))
    x_start = np.array([p0[0], p0[1], th0], dtype=float)
    x_goal = np.array([ref_traj[0, -1], ref_traj[1, -1], 0.0], dtype=float)

    dyn_obs = _make_dyn_obs(
        path_id=path_id,
        seed_offset=seed_offset,
        ref_traj=ref_traj,
        static_rects=static_rects,
        x_start=x_start,
        x_goal=x_goal,
        congestion_tier=congestion_tier,
    )

    return ref_traj, static_rects, dyn_obs, x_start, x_goal


# ============================================================
# Build a custom MPCCController with an overridden MPCCConfig
# ============================================================

class _ConfiguredMPCCController(MPCCController):
    """
    Thin subclass that injects a custom MPCCConfig into build_mpcc_solver
    while reusing all of MPCCController's solve() logic untouched.
    """

    def __init__(self, static_rects, dyn_obs, ref_traj, use_hard: bool, config: MPCCConfig):
        from run_experiment_unified import (
            _make_solver_dyn_templates,
            _effective_observation_radius,
            MAX_ACTIVE_DYN_OBS,
            CONTROLLER_EXPERIMENT_CONFIG,
        )
        from params import nu

        # Replicate MPCCController.__init__ but pass config to build_mpcc_solver
        self.ref_traj = ref_traj
        self.use_hard = use_hard
        self.static_rects = static_rects
        self.max_active_dyn = min(MAX_ACTIVE_DYN_OBS, len(dyn_obs))
        self.observation_radius = _effective_observation_radius(dyn_obs)
        self.observation_noise_std = float(CONTROLLER_EXPERIMENT_CONFIG.get("observation_noise_std", 0.0))
        self.obs_rng = np.random.default_rng(7000)
        self.solver_dyn_templates = _make_solver_dyn_templates(dyn_obs, self.max_active_dyn)

        self.mpcc_data = build_mpcc_solver(
            ref_traj,
            static_rects,
            self.solver_dyn_templates,
            use_hard=use_hard,
            config=config,
        )

        self.solver = self.mpcc_data.solver
        self.lbx = self.mpcc_data.lbx
        self.ubx = self.mpcc_data.ubx
        self.lbg = self.mpcc_data.lbg
        self.ubg = self.mpcc_data.ubg
        self.mu = 0.0
        self.u_prev = np.zeros(nu)
        self.prev_z = None
        # suppress the class-level warning flag inherited from MPCCController
        _ConfiguredMPCCController._warned_no_true_hard_soft = True


def _build_mpcc_controller(
    static_rects, dyn_obs, ref_traj, use_hard: bool, config: MPCCConfig
) -> _ConfiguredMPCCController:
    return _ConfiguredMPCCController(static_rects, dyn_obs, ref_traj, use_hard, config)


# ============================================================
# Single experiment rollout  (mirrors run_experiment_unified logic)
# ============================================================

def _run_single(
    path_id: int,
    use_hard: bool,
    seed_offset: int,
    congestion_tier: str,
    config: MPCCConfig,
) -> dict[str, Any]:
    from params import dt, N, nx, nu, r_robot, safety_buffer
    from run_experiment_unified import (
        _select_active_dyn_obs,
        _make_observed_dyn_obs,
        _predict_and_pad_dyn_obs,
        REF_SLOWDOWN,
        SolveStep,
    )

    ref_traj, static_rects, dyn_obs, x_start, x_goal = _build_env(
        path_id, seed_offset, congestion_tier
    )

    ctrl = _build_mpcc_controller(static_rects, dyn_obs, ref_traj, use_hard, config)

    x_current = x_start.copy()
    records: list[StepRecord] = []

    cont_errs, lag_errs, speeds, omegas = [], [], [], []
    solve_times_ms = []
    solver_fail_count = 0
    state_mismatches = []
    active_dyn_counts = []

    goal_reached = False
    termination_reason = "max_steps"

    for step in range(MAX_STEPS):
        dist_to_goal = float(np.linalg.norm(x_current[:2] - x_goal[:2]))
        progress_idx = extract_progress(x_current, ref_traj)
        end_tol = END_PROGRESS_TOL * ref_traj.shape[1]

        if dist_to_goal < GOAL_TOL or progress_idx >= ref_traj.shape[1] - 1 - end_tol:
            goal_reached = True
            termination_reason = "goal"
            break

        try:
            step_result = ctrl.solve(x_current, dyn_obs, x_goal)
        except Exception as exc:
            termination_reason = f"solver_exception:{exc}"
            break

        if not step_result.ok:
            solver_fail_count += 1
            u = np.zeros(nu)
        else:
            u = step_result.u

        v, omega = float(u[0]), float(u[1])

        # Advance ground-truth simulation
        x_next = np.array([
            x_current[0] + dt * v * np.cos(x_current[2]),
            x_current[1] + dt * v * np.sin(x_current[2]),
            x_current[2] + dt * omega,
        ], dtype=float)

        # Step dynamic obstacles
        for obs in dyn_obs:
            obs.step(dt, static_rects)

        # Record tracking errors
        cont_err, lag_err = _compute_common_tracking_errors(x_current, ref_traj)
        cont_errs.append(abs(cont_err))
        lag_errs.append(lag_err)
        speeds.append(abs(v))
        omegas.append(abs(omega))
        solve_times_ms.append(step_result.solve_time_s * 1000.0)

        if step_result.state_mismatch_norm is not None:
            state_mismatches.append(step_result.state_mismatch_norm)
        if hasattr(step_result, "active_dyn_count") and step_result.active_dyn_count is not None:
            active_dyn_counts.append(step_result.active_dyn_count)

        # Build StepRecord for collision checker
        records.append(StepRecord(
            step=step,
            x=float(x_current[0]),
            y=float(x_current[1]),
            theta=float(x_current[2]),
            v=v,
            omega=omega,
            dyn_obs_states=[
                {"x": obs.x, "y": obs.y, "theta": obs.theta, "a": obs.a, "b": obs.b}
                for obs in dyn_obs
            ],
        ))

        x_current = x_next

    steps_taken = len(records)

    # --- Path length ---
    path_length = 0.0
    for i in range(1, len(records)):
        dx = records[i].x - records[i - 1].x
        dy = records[i].y - records[i - 1].y
        path_length += math.hypot(dx, dy)

    straight_line_dist = float(np.linalg.norm(x_goal[:2] - x_start[:2]))
    path_efficiency = float(straight_line_dist / path_length) if path_length > 1e-12 else 0.0

    # --- Collision metrics ---
    collision_info = check_collisions(
        records=records,
        static_rects=static_rects,
        r_robot=r_robot,
        safety_buffer=safety_buffer,
        violation_threshold=VIOLATION_THRESHOLD,
    )

    static_body_collision_steps   = int(collision_info.get("static_body_collision_steps", 0))
    static_zone_violation_steps   = int(collision_info.get("static_zone_violation_steps", 0))
    dyn_body_collision_steps      = int(collision_info.get("dyn_body_collision_steps", 0))
    dyn_exclusion_violation_steps = int(collision_info.get("dyn_exclusion_violation_steps", 0))
    worst_dyn_excl_depth          = float(collision_info.get("worst_dyn_excl_depth", 0.0))
    worst_dyn_excl_step           = collision_info.get("worst_dyn_excl_step", None)
    worst_dyn_excl_obs            = collision_info.get("worst_dyn_excl_obs", None)
    min_dyn_body_clearance        = float(collision_info.get("min_dyn_body_clearance", math.inf))
    min_dyn_body_clearance_step   = collision_info.get("min_dyn_body_clearance_step", None)
    min_dyn_body_clearance_obs    = collision_info.get("min_dyn_body_clearance_obs", None)
    min_clearance_m               = float(collision_info.get("min_clearance_m", math.inf))

    cont_arr  = np.asarray(cont_errs,    dtype=float) if cont_errs  else np.zeros(1)
    lag_arr   = np.asarray(lag_errs,     dtype=float) if lag_errs   else np.zeros(1)
    speed_arr = np.asarray(speeds,       dtype=float) if speeds     else np.zeros(1)
    omega_arr = np.asarray(omegas,       dtype=float) if omegas     else np.zeros(1)
    solve_arr = np.asarray(solve_times_ms, dtype=float) if solve_times_ms else np.zeros(1)

    return {
        "goal_reached":                   int(goal_reached),
        "steps_taken":                    int(steps_taken),
        "termination_reason":             termination_reason,
        "spawned_dyn_obs_count":          int(len(dyn_obs)),
        "completion_time_s":              round(float(steps_taken * dt), 4),
        "path_length":                    round(float(path_length), 4),
        "path_efficiency":                round(float(path_efficiency), 4),
        "mean_speed":                     round(float(np.mean(speed_arr)), 4),
        "std_speed":                      round(float(np.std(speed_arr)), 4),
        "smoothness":                     round(float(np.sum(omega_arr) * dt), 4),
        "mean_cont_err":                  round(float(np.mean(cont_arr)), 4),
        "rms_cont_err":                   round(float(np.sqrt(np.mean(cont_arr ** 2))), 4),
        "max_cont_err":                   round(float(np.max(cont_arr)), 4),
        "std_cont_err":                   round(float(np.std(cont_arr)), 4),
        "mean_lag_err":                   round(float(np.mean(lag_arr)), 4),
        "rms_lag_err":                    round(float(np.sqrt(np.mean(lag_arr ** 2))), 4),
        "max_lag_err":                    round(float(np.max(np.abs(lag_arr))), 4),
        "std_lag_err":                    round(float(np.std(lag_arr)), 4),
        "mean_solve_ms":                  round(float(np.mean(solve_arr)), 2),
        "max_solve_ms":                   round(float(np.max(solve_arr)), 2),
        "std_solve_ms":                   round(float(np.std(solve_arr)), 2),
        "solver_fail_count":              int(solver_fail_count),
        "static_body_collision_steps":    int(static_body_collision_steps),
        "static_zone_violation_steps":    int(static_zone_violation_steps),
        "dyn_body_collision_steps":       int(dyn_body_collision_steps),
        "dyn_exclusion_violation_steps":  int(dyn_exclusion_violation_steps),
        "total_body_collision_steps":     int(static_body_collision_steps + dyn_body_collision_steps),
        "worst_dyn_excl_depth":           round(float(worst_dyn_excl_depth), 6),
        "worst_dyn_excl_step":            worst_dyn_excl_step,
        "worst_dyn_excl_obs":             worst_dyn_excl_obs,
        "min_dyn_body_clearance":         round(float(min_dyn_body_clearance), 6) if min_dyn_body_clearance < math.inf else None,
        "min_dyn_body_clearance_step":    min_dyn_body_clearance_step,
        "min_dyn_body_clearance_obs":     min_dyn_body_clearance_obs,
        "mean_active_dyn_obs":            round(float(np.mean(active_dyn_counts)) if active_dyn_counts else 0.0, 4),
        "max_active_dyn_obs":             int(max(active_dyn_counts)) if active_dyn_counts else 0,
        "mean_state_mismatch":            round(float(np.mean(state_mismatches)) if state_mismatches else 0.0, 10),
        "max_state_mismatch":             round(float(np.max(state_mismatches)) if state_mismatches else 0.0, 10),
        "min_clearance_m":                round(float(min_clearance_m), 6) if min_clearance_m < math.inf else None,
        "error":                          "",
    }


# ============================================================
# Job runner (top-level so it's picklable for ProcessPoolExecutor)
# ============================================================

def _run_job(job: dict) -> dict[str, Any]:
    """
    job keys:
        sweep_group, sweep_param, sweep_value,
        use_hard, scenario_label, path_id,
        seed_offset, congestion_tier
    """
    sweep_group    = job["sweep_group"]
    sweep_param    = job["sweep_param"]
    sweep_value    = job["sweep_value"]
    use_hard       = job["use_hard"]
    path_id        = job["path_id"]
    scenario_label = job["scenario_label"]
    seed_offset    = job["seed_offset"]
    congestion_tier = job["congestion_tier"]

    # Build config by overriding the single swept parameter
    cfg_kwargs = asdict(DEFAULT_CONFIG)
    cfg_kwargs[sweep_param] = sweep_value
    config = MPCCConfig(**cfg_kwargs)

    controller_label = f"mpcc_{'hard' if use_hard else 'soft'}"

    base = {
        "sweep_group":    sweep_group,
        "sweep_param":    sweep_param,
        "sweep_value":    sweep_value,
        "controller":     controller_label,
        "scenario":       scenario_label,
        "path_id":        path_id,
        "seed_offset":    seed_offset,
        "congestion_tier": congestion_tier,
    }

    try:
        result = _run_single(
            path_id=path_id,
            use_hard=use_hard,
            seed_offset=seed_offset,
            congestion_tier=congestion_tier,
            config=config,
        )
        return {**base, **result}
    except Exception as exc:
        empty = {f: None for f in RAW_FIELDS}
        empty.update(base)
        empty["goal_reached"] = 0
        empty["steps_taken"] = 0
        empty["termination_reason"] = "error"
        empty["error"] = str(exc)
        return empty


# ============================================================
# CSV helpers
# ============================================================

def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict]) -> list[dict]:
    """Aggregate by (sweep_group, sweep_param, sweep_value, controller, scenario)."""
    from itertools import groupby

    def key(r):
        return (
            r.get("sweep_group", ""),
            r.get("sweep_param", ""),
            r.get("sweep_value", ""),
            r.get("controller", ""),
            r.get("scenario", ""),
        )

    valid_rows = [r for r in rows if not r.get("error")]
    valid_rows.sort(key=key)

    summary = []
    for grp_key, grp_iter in groupby(valid_rows, key=key):
        grp = list(grp_iter)
        row: dict = {
            "sweep_group":  grp_key[0],
            "sweep_param":  grp_key[1],
            "sweep_value":  grp_key[2],
            "controller":   grp_key[3],
            "scenario":     grp_key[4],
            "n_runs":       len(grp),
        }
        for metric in SUMMARY_METRICS:
            vals = [float(r[metric]) for r in grp if r.get(metric) is not None and r.get(metric) != ""]
            if vals:
                row[f"{metric}_mean"] = round(float(np.mean(vals)), 6)
                row[f"{metric}_std"]  = round(float(np.std(vals)), 6)
        summary.append(row)

    return summary


# ============================================================
# Main
# ============================================================

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Build all jobs
    jobs: list[dict] = []
    for sweep_group, sweep_param, sweep_value in SWEEP_CONFIGS:
        for use_hard in (False, True):
            for scenario in SCENARIOS:
                for congestion_tier in CONGESTION_TIERS:
                    for seed in range(N_SEEDS):
                        jobs.append({
                            "sweep_group":    sweep_group,
                            "sweep_param":    sweep_param,
                            "sweep_value":    sweep_value,
                            "use_hard":       use_hard,
                            "path_id":        scenario["path_id"],
                            "scenario_label": scenario["label"],
                            "seed_offset":    seed,
                            "congestion_tier": congestion_tier,
                        })

    total = len(jobs)
    print(f"\n{'='*70}")
    print(f"  MPCC SENSITIVITY STUDY  —  {total} total runs  ({MAX_WORKERS} workers)")
    print(f"  Parameters swept: q_cont × q_vs × rho_dyn (one at a time)")
    print(f"  Controllers: mpcc_soft, mpcc_hard")
    print(f"  Scenarios:   {[s['label'] for s in SCENARIOS]}")
    print(f"  Tiers:       {CONGESTION_TIERS}")
    print(f"  Seeds:       {N_SEEDS}")
    print(f"{'='*70}\n")

    rows: list[dict] = []
    completed = 0
    t_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_run_job, job): job for job in jobs}

        for fut in as_completed(futures):
            result = fut.result()
            rows.append(result)
            completed += 1

            elapsed = time.perf_counter() - t_start
            eta_s = (elapsed / completed) * (total - completed) if completed else 0.0
            eta_str = f"{eta_s/60:.1f}min" if eta_s >= 60 else f"{eta_s:.0f}s"

            if result.get("error"):
                print(
                    f"[{completed:4d}/{total}] "
                    f"{result.get('sweep_group')}={result.get('sweep_value')} "
                    f"{result.get('controller')} {result.get('scenario')} "
                    f"seed={result.get('seed_offset')} tier={result.get('congestion_tier')} "
                    f"  ERROR: {result['error']}"
                )
            else:
                print(
                    f"[{completed:4d}/{total}] "
                    f"{result['sweep_group']}={result['sweep_value']:.4g} "
                    f"{result['controller']:15s} "
                    f"{result['scenario']:20s} "
                    f"tier={result['congestion_tier']:4s} seed={result['seed_offset']} "
                    f"goal={result['goal_reached']} steps={result['steps_taken']:4d} "
                    f"solve={result['mean_solve_ms']:.1f}ms "
                    f"dyn_excl={result['dyn_exclusion_violation_steps']} "
                    f"ETA {eta_str}"
                )

    raw_csv = os.path.join(RESULTS_DIR, "sensitivity_raw.csv")
    _write_csv(raw_csv, rows, RAW_FIELDS)
    print(f"\nRaw results  → {raw_csv}  ({len(rows)} rows)")

    summary_rows = _summarize(rows)
    if summary_rows:
        summary_fields = sorted(
            {k for r in summary_rows for k in r},
            key=lambda x: (x not in {"sweep_group", "sweep_param", "sweep_value", "controller", "scenario", "n_runs"}, x),
        )
        summary_csv = os.path.join(RESULTS_DIR, "sensitivity_summary.csv")
        _write_csv(summary_csv, summary_rows, summary_fields)
        print(f"Summary      → {summary_csv}  ({len(summary_rows)} rows)")
    else:
        print("No valid rows to summarize.")


if __name__ == "__main__":
    main()