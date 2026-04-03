from __future__ import annotations

import csv
import math
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import inspect
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any

import numpy as np

from debug import (
    GOAL_TOL,
    StepRecord,
    check_collisions,
    extract_progress,
    make_debug_dyn_obs,
)
from mpcc import MPCCConfig, build_mpcc_solver
from obs import Obstacle
from params import dt, N, nx, nu, r_robot
from run_experiment_unified import (
    CONTROLLER_EXPERIMENT_CONFIG,
    MAX_ACTIVE_DYN_OBS,
    SolveStep,
    _compute_common_tracking_errors,
    _effective_observation_radius,
    _make_observed_dyn_obs,
    _make_solver_dyn_templates,
    _predict_and_pad_dyn_obs,
    _select_active_dyn_obs,
)

# ============================================================
# Batch configuration
# ============================================================
RESULTS_DIR = "results_sensitivity"
N_SEEDS = 10
MAX_WORKERS = 12
MAX_STEPS = 800

SCENARIOS = [
    {"path_id": 10, "label": "CorridorStructured"},
    {"path_id": 11, "label": "OpenClutter"},
]

USE_DEBUG_DYN_OBS = True
START_TRIM = 8
INITIAL_HEADING_LOOKAHEAD = 10
DYN_OBS_MODE = "path_tier"
CONGESTION_TIERS = ["low", "mid", "high"]

WEIGHT_FACTORS = [0.5, 1.0, 2.0]
WEIGHT_NAMES = ["q_cont", "q_vs", "rho_dyn"]
BASE_CONFIG = MPCCConfig()

RAW_FIELDS = [
    "scenario",
    "controller",
    "weight_name",
    "weight_factor",
    "weight_value",
    "path_id",
    "seed_offset",
    "termination_reason",
    "congestion_tier",
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
    "static_body_collision_steps",
    "static_zone_violation_steps",
    "dyn_body_collision_steps",
    "dyn_exclusion_violation_steps",
    "total_body_collision_steps",
    "worst_dyn_excl_depth",
    "min_dyn_body_clearance",
    "min_clearance_m",
]


class MPCCControllerWithConfig:
    _warned_no_true_hard_soft = False

    def __init__(
        self,
        static_rects,
        dyn_obs,
        ref_traj,
        use_hard: bool,
        mpcc_config: MPCCConfig,
        controller_seed: int = 0,
    ):
        self.ref_traj = ref_traj
        self.use_hard = use_hard
        self.static_rects = static_rects
        self.max_active_dyn = min(MAX_ACTIVE_DYN_OBS, len(dyn_obs))
        self.observation_radius = _effective_observation_radius(dyn_obs)
        self.observation_noise_std = float(CONTROLLER_EXPERIMENT_CONFIG.get("observation_noise_std", 0.0))
        self.obs_rng = np.random.default_rng(7000 + int(controller_seed))
        self.solver_dyn_templates = _make_solver_dyn_templates(dyn_obs, self.max_active_dyn)
        self.mpcc_config = mpcc_config

        sig = inspect.signature(build_mpcc_solver)
        if "use_hard" in sig.parameters:
            self.mpcc_data = build_mpcc_solver(
                ref_traj,
                static_rects,
                self.solver_dyn_templates,
                use_hard=use_hard,
                config=mpcc_config,
            )
        else:
            if not MPCCControllerWithConfig._warned_no_true_hard_soft:
                print(
                    "WARNING: build_mpcc_solver() does not accept use_hard. "
                    "MPCC SOFT/HARD sensitivity runs will both use the same underlying MPCC solver."
                )
                MPCCControllerWithConfig._warned_no_true_hard_soft = True
            self.mpcc_data = build_mpcc_solver(
                ref_traj,
                static_rects,
                self.solver_dyn_templates,
                config=mpcc_config,
            )

        self.solver = self.mpcc_data.solver
        self.lbx = self.mpcc_data.lbx
        self.ubx = self.mpcc_data.ubx
        self.lbg = self.mpcc_data.lbg
        self.ubg = self.mpcc_data.ubg

        self.mu = 0.0
        self.u_prev = np.zeros(nu)
        self.prev_z = None

    def solve(self, x_current: np.ndarray, dyn_obs, goal_global: np.ndarray) -> SolveStep:
        k_ref = max(int(np.floor(self.mu)), 0)
        s_local = self.mu - k_ref

        R_horizon = self.ref_traj[:, k_ref:k_ref + N + 1]
        if R_horizon.shape[1] < N + 1:
            last = R_horizon[:, -1].reshape(-1, 1)
            pad = np.repeat(last, N + 1 - R_horizon.shape[1], axis=1)
            R_horizon = np.hstack((R_horizon, pad))

        active_dyn_obs = _select_active_dyn_obs(
            x_current, dyn_obs, self.max_active_dyn, self.observation_radius
        )
        observed_dyn_obs = _make_observed_dyn_obs(active_dyn_obs, self.obs_rng, self.observation_noise_std)
        obs_x_h, obs_y_h, obs_th_h = _predict_and_pad_dyn_obs(
            observed_dyn_obs, self.max_active_dyn, self.static_rects
        )

        X_goal_val = np.array([R_horizon[0, -1], R_horizon[1, -1], 0.0], dtype=float)

        p = np.concatenate([
            x_current,
            self.u_prev,
            np.array([s_local]),
            R_horizon.flatten(order="F"),
            X_goal_val,
            obs_x_h.flatten(order="F"),
            obs_y_h.flatten(order="F"),
            obs_th_h.flatten(order="F"),
        ])

        kwargs = dict(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)

        X_guess = np.tile(x_current.reshape(-1, 1), (1, N + 1))
        U_guess = np.zeros((nu, N))
        s_guess = np.zeros(N + 1)
        parts = [
            X_guess.flatten(order="F"),
            U_guess.flatten(order="F"),
            s_guess.flatten(order="F"),
        ]
        if not self.use_hard:
            n_dyn = self.max_active_dyn
            Sdyn_guess = np.zeros(n_dyn * N)
            parts.append(Sdyn_guess)
        z0 = np.concatenate(parts)

        expected_n = len(self.lbx)
        if self.prev_z is None or self.prev_z.shape[0] != expected_n:
            kwargs["x0"] = z0
        else:
            kwargs["x0"] = self.prev_z

        t0 = time.perf_counter()
        sol = self.solver(**kwargs)
        solve_time_s = time.perf_counter() - t0

        stats = self.solver.stats()
        status = stats["return_status"]
        acceptable_statuses = {
            "Solve_Succeeded",
            "Solved_To_Acceptable_Level",
            "Maximum_Iterations_Exceeded",
        }
        if status not in acceptable_statuses:
            return SolveStep(
                ok=False,
                status=status,
                solve_time_s=solve_time_s,
                u=np.zeros(nu),
                progress_index=int(np.floor(self.mu)),
            )

        self.prev_z = sol["x"]
        z = sol["x"].full().flatten()
        offset = 0
        X_opt = z[offset:offset + self.mpcc_data.nX].reshape((nx, N + 1), order="F")
        offset += self.mpcc_data.nX
        U_opt = z[offset:offset + self.mpcc_data.nU].reshape((nu, N), order="F")
        offset += self.mpcc_data.nU
        _s_opt = z[offset:offset + self.mpcc_data.nProg].reshape((N + 1,), order="F")

        v, omega = U_opt[:, 0]
        self.u_prev = np.array([v, omega])

        solver_next_state = np.array(X_opt[:, 1], dtype=float)
        rollout_next_state = np.array([
            x_current[0] + dt * v * np.cos(x_current[2]),
            x_current[1] + dt * v * np.sin(x_current[2]),
            x_current[2] + dt * omega,
        ], dtype=float)
        state_mismatch_norm = float(np.linalg.norm(solver_next_state - rollout_next_state))

        search_radius = 25
        k_center = int(np.clip(np.floor(self.mu), 0, self.ref_traj.shape[1] - 1))
        k0 = max(0, k_center - search_radius)
        k1 = min(self.ref_traj.shape[1], k_center + search_radius + 1)
        ref_slice = self.ref_traj[:, k0:k1].T
        dists = np.linalg.norm(ref_slice - rollout_next_state[:2], axis=1)
        local_idx = int(np.argmin(dists))
        self.mu = float(k0 + local_idx)

        return SolveStep(
            ok=True,
            status=status,
            solve_time_s=solve_time_s,
            u=np.array([v, omega]),
            progress_index=int(np.floor(self.mu)),
            solver_next_state=solver_next_state,
            rollout_next_state=rollout_next_state,
            state_mismatch_norm=state_mismatch_norm,
            active_dyn_count=len(active_dyn_obs),
        )


# ============================================================
# Environment helpers
# ============================================================
def make_dyn_obs(
    path_id: int,
    seed_offset: int,
    ref_traj: np.ndarray,
    static_rects: list,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    congestion_tier: str,
) -> list:
    env = Obstacle(path_id)
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
    return env.dynamic_obs_seeded(seed_offset)


def build_env(path_id: int, seed_offset: int, congestion_tier: str):
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
    dyn_obs = make_dyn_obs(path_id, seed_offset, ref_traj, static_rects, x_start, x_goal, congestion_tier)
    return ref_traj, static_rects, dyn_obs, x_start, x_goal


def make_config(weight_name: str, weight_factor: float) -> MPCCConfig:
    cfg = MPCCConfig(**asdict(BASE_CONFIG))
    if not hasattr(cfg, weight_name):
        raise ValueError(f"Unknown MPCC weight: {weight_name}")
    setattr(cfg, weight_name, float(getattr(BASE_CONFIG, weight_name)) * float(weight_factor))
    return cfg


def run_single_experiment(
    path_id: int,
    use_hard: bool,
    seed_offset: int,
    congestion_tier: str,
    weight_name: str,
    weight_factor: float,
) -> dict[str, Any]:
    ref_traj, static_rects, dyn_obs, x_start, x_goal = build_env(path_id, seed_offset, congestion_tier)
    mpcc_config = make_config(weight_name, weight_factor)
    controller = MPCCControllerWithConfig(
        static_rects=static_rects,
        dyn_obs=dyn_obs,
        ref_traj=ref_traj,
        use_hard=use_hard,
        mpcc_config=mpcc_config,
        controller_seed=seed_offset,
    )

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
    termination_reason = "max_steps"

    for k in range(MAX_STEPS):
        step_result = controller.solve(x_current, dyn_obs, x_goal)
        progress_value, _ = extract_progress(controller, step_result)

        solve_times_ms.append(1000.0 * float(step_result.solve_time_s))
        active_dyn_counts.append(int(getattr(step_result, "active_dyn_count", 0)))
        state_mismatches.append(float(getattr(step_result, "state_mismatch_norm", 0.0)))

        if step_result.status != "Solve_Succeeded":
            solver_fail_count += 1

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
        check_collisions(record, static_rects, dyn_obs, prev_xy=x_current[:2].copy())

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

        if not step_result.ok:
            termination_reason = f"solver_fail:{step_result.status}"
            break
        if record.static_body_collision:
            termination_reason = "static_body_collision"
            goal_reached = False
            break
        if record.dyn_body_collision:
            termination_reason = "dyn_body_collision"
            goal_reached = False
            break
        goal_dist = float(np.linalg.norm(x_current[:2] - x_goal[:2]))
        if goal_dist <= GOAL_TOL:
            termination_reason = "goal_reached"
            goal_reached = True
            break

    cont_arr = np.asarray(cont_errors, dtype=float) if cont_errors else np.zeros(1)
    lag_arr = np.asarray(lag_errors, dtype=float) if lag_errors else np.zeros(1)
    solve_arr = np.asarray(solve_times_ms, dtype=float) if solve_times_ms else np.zeros(1)
    speed_arr = np.asarray(speeds, dtype=float) if speeds else np.zeros(1)
    omega_arr = np.asarray(omegas, dtype=float) if omegas else np.zeros(1)

    controller_label = f"mpcc_{'hard' if use_hard else 'soft'}"
    total_body_collision_steps = static_body_collision_steps + dyn_body_collision_steps
    path_efficiency = float(straight_line_dist / path_length) if path_length > 1e-12 else 0.0

    return {
        "controller": controller_label,
        "weight_name": weight_name,
        "weight_factor": float(weight_factor),
        "weight_value": float(getattr(mpcc_config, weight_name)),
        "path_id": path_id,
        "seed_offset": seed_offset,
        "termination_reason": termination_reason,
        "congestion_tier": congestion_tier,
        "spawned_dyn_obs_count": int(len(dyn_obs)),
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


def _write_csv(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        label = scenario["label"]
        for controller in ("mpcc_soft", "mpcc_hard"):
            for weight_name in WEIGHT_NAMES:
                for weight_factor in WEIGHT_FACTORS:
                    runs = [
                        r for r in rows
                        if r.get("scenario") == label
                        and r.get("controller") == controller
                        and r.get("weight_name") == weight_name
                        and float(r.get("weight_factor", -999)) == float(weight_factor)
                        and not r.get("error")
                    ]
                    if not runs:
                        continue
                    row: dict[str, Any] = {
                        "scenario": label,
                        "controller": controller,
                        "weight_name": weight_name,
                        "weight_factor": float(weight_factor),
                        "weight_value": float(runs[0]["weight_value"]),
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


def _run_job(job: tuple[bool, dict[str, Any], int, str, str, float]) -> dict[str, Any]:
    use_hard, scenario, seed, congestion_tier, weight_name, weight_factor = job
    path_id = int(scenario["path_id"])
    label = str(scenario["label"])
    try:
        result = run_single_experiment(
            path_id=path_id,
            use_hard=use_hard,
            seed_offset=seed,
            congestion_tier=congestion_tier,
            weight_name=weight_name,
            weight_factor=weight_factor,
        )
        result["scenario"] = label
        return result
    except Exception as exc:
        return {
            "scenario": label,
            "controller": f"mpcc_{'hard' if use_hard else 'soft'}",
            "weight_name": weight_name,
            "weight_factor": float(weight_factor),
            "weight_value": None,
            "path_id": path_id,
            "seed_offset": seed,
            "termination_reason": "exception",
            "congestion_tier": congestion_tier,
            "spawned_dyn_obs_count": None,
            "goal_reached": 0,
            "steps_taken": 0,
            "completion_time_s": 0.0,
            "path_length": 0.0,
            "path_efficiency": 0.0,
            "mean_speed": 0.0,
            "std_speed": 0.0,
            "smoothness": 0.0,
            "mean_cont_err": None,
            "rms_cont_err": None,
            "max_cont_err": None,
            "std_cont_err": None,
            "mean_lag_err": None,
            "rms_lag_err": None,
            "max_lag_err": None,
            "std_lag_err": None,
            "mean_solve_ms": None,
            "max_solve_ms": None,
            "std_solve_ms": None,
            "solver_fail_count": None,
            "static_body_collision_steps": None,
            "static_zone_violation_steps": None,
            "dyn_body_collision_steps": None,
            "dyn_exclusion_violation_steps": None,
            "total_body_collision_steps": None,
            "worst_dyn_excl_depth": None,
            "worst_dyn_excl_step": None,
            "worst_dyn_excl_obs": None,
            "min_dyn_body_clearance": None,
            "min_dyn_body_clearance_step": None,
            "min_dyn_body_clearance_obs": None,
            "mean_active_dyn_obs": None,
            "max_active_dyn_obs": None,
            "mean_state_mismatch": None,
            "max_state_mismatch": None,
            "min_clearance_m": None,
            "error": str(exc),
        }


def run_batch() -> tuple[str, str]:
    raw_csv = os.path.join(RESULTS_DIR, "mpcc_sensitivity_results.csv")
    summary_csv = os.path.join(RESULTS_DIR, "mpcc_sensitivity_summary.csv")

    jobs: list[tuple[bool, dict[str, Any], int, str, str, float]] = []
    for use_hard in (False, True):
        for scenario in SCENARIOS:
            for congestion_tier in CONGESTION_TIERS:
                for weight_name in WEIGHT_NAMES:
                    for weight_factor in WEIGHT_FACTORS:
                        for seed in range(N_SEEDS):
                            jobs.append((use_hard, scenario, seed, congestion_tier, weight_name, weight_factor))

    print("\n" + "=" * 78)
    print(f"  MPCC SENSITIVITY launching {len(jobs)} runs with {MAX_WORKERS} workers")
    print("=" * 78)

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
                    f"[{completed:3d}/{len(jobs):3d}] {result.get('controller','?')} {result.get('weight_name','?')} "
                    f"x{result.get('weight_factor','?')} {result.get('scenario','?')} seed={result.get('seed_offset','?')} "
                    f"ERROR: {result['error']}"
                )
            else:
                print(
                    f"[{completed:3d}/{len(jobs):3d}] {result['controller']} {result['weight_name']} x{result['weight_factor']:.1f} "
                    f"{result['scenario']} tier={result['congestion_tier']} seed={result['seed_offset']} "
                    f"goal={result['goal_reached']} solve={result['mean_solve_ms']:.1f}ms "
                    f"static_zone={result['static_zone_violation_steps']} dyn_excl={result['dyn_exclusion_violation_steps']}"
                )

    _write_csv(raw_csv, rows, RAW_FIELDS)
    summary_rows = _summarize_rows(rows)
    if summary_rows:
        summary_fields = sorted(
            {k for row in summary_rows for k in row.keys()},
            key=lambda x: (x not in {"scenario", "controller", "weight_name", "weight_factor", "weight_value", "n_runs"}, x),
        )
        _write_csv(summary_csv, summary_rows, summary_fields)
    else:
        _write_csv(summary_csv, [], ["scenario", "controller", "weight_name", "weight_factor", "weight_value", "n_runs"])

    print(f"\nRaw results saved to   {raw_csv}")
    print(f"Summary saved to      {summary_csv}")
    return raw_csv, summary_csv


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_batch()


if __name__ == "__main__":
    main()
