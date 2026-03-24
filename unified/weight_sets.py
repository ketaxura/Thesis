"""
weight_sets.py
==============
Three named MPCC weight presets and a thin wrapper that rebuilds an
MPCCConfig from a preset index chosen by the RL policy each step.

Weight set philosophy
---------------------
  0  CONSERVATIVE  — robot safety first
       high contouring penalty  → stay close to path
       low progress reward      → do not rush through crowds
       high dynamic rho         → strongly avoid dyn obs zone

  1  BALANCED      — default operating mode
       default weights from MPCCConfig

  2  AGGRESSIVE    — throughput first
       low contouring penalty   → allow wider deviations to go faster
       high progress reward     → maximise path speed
       low dynamic rho          → tolerate soft dyn obs intrusions

These are purposely coarse — three qualitatively distinct behaviours.
Do not micro-tune them before validating the switching logic first.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from mpcc import MPCCConfig


# ── The three presets ──────────────────────────────────────────────────────

CONSERVATIVE = MPCCConfig(
    q_cont          = 6.0,    # strong lateral tracking (was 3.0)
    q_lag           = 1.0,    # penalise falling behind (was 0.5)
    q_theta         = 5.0,
    q_goal          = 100.0,
    rho_obs         = 1e4,
    rho_dyn         = 2e5,    # strong soft dynamic penalty (was 1e5)
    q_vs            = 20.0,   # reduced speed incentive (was 40.0)
    q_s_terminal    = 5.0,    # less aggressive terminal progress (was 10.0)
    r_v             = 0.3,    # penalise high speed (was 0.1)
    r_w             = 0.4,    # penalise turning (was 0.2)
    r_dv            = 3.0,
    r_dw            = 5.0,
)

BALANCED = MPCCConfig()       # exact defaults from MPCCConfig dataclass

AGGRESSIVE = MPCCConfig(
    q_cont          = 1.5,    # relaxed lateral tracking
    q_lag           = 0.2,
    q_theta         = 3.0,
    q_goal          = 100.0,
    rho_obs         = 1e4,
    rho_dyn         = 3e4,    # tolerate more soft dynamic intrusion (was 1e5)
    q_vs            = 70.0,   # strong speed incentive (was 40.0)
    q_s_terminal    = 20.0,   # push hard for terminal progress
    r_v             = 0.05,
    r_w             = 0.1,
    r_dv            = 1.0,
    r_dw            = 2.0,
)

WEIGHT_SETS: list[MPCCConfig] = [CONSERVATIVE, BALANCED, AGGRESSIVE]
WEIGHT_SET_NAMES: list[str]   = ["conservative", "balanced", "aggressive"]

N_ACTIONS = len(WEIGHT_SETS)   # 3 — imported by rl_agent.py


def get_config(action: int) -> MPCCConfig:
    """Return the MPCCConfig for a given RL action index (0, 1, or 2)."""
    if action not in range(N_ACTIONS):
        raise ValueError(f"action must be 0, 1, or 2 — got {action}")
    return WEIGHT_SETS[action]


# ── Rule-based baseline policy ────────────────────────────────────────────
#
# Used before RL training to validate that weight-switching actually
# changes controller behaviour.  No learning, no neural net.
#
# Logic (thresholds are purposely coarse):
#   - If any obstacle within CLOSE_THRESH → conservative
#   - If obstacle count within radius > COUNT_THRESH → conservative
#   - If path is clear → aggressive
#   - Otherwise → balanced

CLOSE_THRESH = 2.0   # metres — nearest obstacle triggers conservative
COUNT_THRESH = 2     # n obstacles within SCAN_RADIUS triggers conservative
SCAN_RADIUS  = 4.0   # metres for count check


def rule_based_action(congestion_obs: "np.ndarray") -> int:
    """
    Map a 6-D congestion observation to a weight-set action index.

    congestion_obs layout (from lidar.congestion_obs):
        [n_hits_norm, min_range_norm, mean_hit_norm,
         front_density, side_density, rear_density]

    Returns
    -------
    0  conservative
    1  balanced
    2  aggressive
    """
    import numpy as np
    from lidar import MAX_RANGE

    n_hits_norm    = float(congestion_obs[0])
    min_range_norm = float(congestion_obs[1])
    front_density  = float(congestion_obs[3])

    min_range_m = min_range_norm * MAX_RANGE

    # Conservative: something is close in front, or dense
    if min_range_m < CLOSE_THRESH or front_density > 0.25:
        return 0

    # Aggressive: nothing detected
    if n_hits_norm < 0.05:
        return 2

    # Default
    return 1