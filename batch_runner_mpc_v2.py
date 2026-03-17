"""
batch_runner_mpc_v2.py
Runs MPC (soft) and MPC (hard) on the identical MPCC environment.

Instructions:
1. Copy obs.py from your MPCC project into this folder
2. Run: python batch_runner_mpc_v2.py

Outputs:
    results/mpc_soft_results.csv
    results/mpc_soft_summary.csv
    results/mpc_hard_results.csv
    results/mpc_hard_summary.csv
"""

import csv
import os
import numpy as np
from run_experiment_mpc_v2 import run_mpc

SCENARIOS = [
    {"path_id": 3, "label": "Zigzag"},
    {"path_id": 1, "label": "Sine"},
]
N_SEEDS   = 10
MAX_STEPS = 800

os.makedirs("results", exist_ok=True)

METRICS = [
    "goal_reached", "steps_taken", "completion_time_s",
    "path_length", "path_efficiency",
    "mean_speed", "std_speed", "smoothness",
    "mean_cont_err", "rms_cont_err", "max_cont_err", "std_cont_err",
    "mean_lag_err",  "rms_lag_err",  "max_lag_err",  "std_lag_err",
    "mean_solve_ms", "max_solve_ms", "std_solve_ms", "solver_fail_count",
    "dyn_body_collisions", "static_body_collisions", "total_body_collisions",
    "dyn_exclusion_violations", "near_miss_count",
    "danger_zone_steps", "min_clearance_m",
]

SUMMARY_METRICS = [
    "goal_reached", "completion_time_s", "path_efficiency",
    "mean_speed", "smoothness",
    "mean_cont_err", "rms_cont_err", "max_cont_err",
    "mean_lag_err", "rms_lag_err",
    "mean_solve_ms", "max_solve_ms", "solver_fail_count",
    "total_body_collisions", "dyn_exclusion_violations",
    "near_miss_count", "danger_zone_steps", "min_clearance_m",
]


def run_batch(use_hard: bool):
    label_suffix = "HARD" if use_hard else "SOFT"
    csv_results  = f"results/mpc_{label_suffix.lower()}_results.csv"
    csv_summary  = f"results/mpc_{label_suffix.lower()}_summary.csv"

    all_results = []

    for scenario in SCENARIOS:
        path_id = scenario["path_id"]
        label   = scenario["label"]

        print(f"\n{'='*60}")
        print(f"  MPC ({label_suffix})  {label} (path_id={path_id})")
        print(f"{'='*60}")

        for seed in range(N_SEEDS):
            print(f"  seed {seed:2d} / {N_SEEDS-1} ...", end=" ", flush=True)
            try:
                result = run_mpc(path_id=path_id, seed_offset=seed,
                                 max_steps=MAX_STEPS, use_hard=use_hard)
                result["scenario"] = label
                all_results.append(result)
                status = "✓" if result["goal_reached"] else "✗"
                print(f"{status}  cont={result['mean_cont_err']:.3f}  "
                    f"solve={result['mean_solve_ms']:.1f}ms  "
                    f"near_miss={result['near_miss_count']}  "
                    f"min_clear={result['min_clearance_m']:.2f}m  "
                    f"collisions={result['total_body_collisions']} "
                    f"(dyn={result['dyn_body_collisions']}, "
                    f"static={result['static_body_collisions']})")
            except Exception as e:
                print(f"ERROR: {e}")
                all_results.append({"scenario": label, "path_id": path_id,
                                     "seed_offset": seed, "ERROR": str(e)})

    # Raw CSV
    with open(csv_results, "w", newline="") as f:
        writer = csv.DictWriter(f,
            fieldnames=["scenario", "path_id", "seed_offset"] + METRICS,
            extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nRaw results saved to {csv_results}")

    # Summary
    summary_rows = []
    print(f"\n{'='*60}")
    print(f"  MPC ({label_suffix}) SUMMARY")
    print(f"{'='*60}")

    for scenario in SCENARIOS:
        label = scenario["label"]
        runs  = [r for r in all_results
                 if r.get("scenario") == label and "ERROR" not in r]
        if not runs:
            continue

        row = {"scenario": label, "n_runs": len(runs)}
        print(f"\n  {label}  ({len(runs)} runs)")
        print(f"  {'Metric':<30} {'Mean':>10}  {'Std':>10}")
        print(f"  {'-'*54}")

        for m in SUMMARY_METRICS:
            vals = np.array([r[m] for r in runs if m in r], dtype=float)
            if not len(vals):
                continue
            mean_v, std_v = vals.mean(), vals.std()
            row[f"{m}_mean"] = round(mean_v, 4)
            row[f"{m}_std"]  = round(std_v,  4)
            if m == "goal_reached":
                print(f"  {'goal_reached':<30} {mean_v*100:>9.1f}%")
            else:
                print(f"  {m:<30} {mean_v:>10.3f}  ±{std_v:>9.3f}")

        summary_rows.append(row)

    if summary_rows:
        with open(csv_summary, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()),
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary saved to {csv_summary}")


if __name__ == "__main__":
    print("Running MPC (SOFT) ...")
    run_batch(use_hard=False)

    print("\n\nRunning MPC (HARD) ...")
    run_batch(use_hard=True)