"""
rl_train.py
===========
Training loop and evaluation harness for the RL weight-switching policy.

Two phases
----------
Phase 1 — Rule-based validation
    Run N_VALIDATE episodes with the hand-coded rule_based_action() policy.
    Confirm that switching weight sets actually changes controller behaviour
    (different exclusion counts, solve times, path lengths).
    Nothing is trained in this phase — it is a sanity check.

Phase 2 — DQN training
    For each episode:
      - Reset environment (new seeded dynamic obstacles)
      - At each step:
          1. Lidar scan → congestion obs
          2. Agent selects action (ε-greedy)
          3. Apply weight set → run MPCC solve step
          4. Compute reward
          5. Store transition, train step
      - Log episode metrics
    Save checkpoints every CHECKPOINT_EVERY episodes.

Reward function
---------------
    r_t = w_prog  * Δμ
        − w_excl  * (1 − λ) * worst_dyn_excl_depth
        − w_stat  * (1 − λ) * static_zone_flag
        − w_speed * λ       * (1 − v / v_max)
        + w_goal  * goal_bonus   (sparse, end of episode)

    λ = agent.lam  (0 = safety mode, 1 = speed mode)

Run
---
    # Validate rule-based policy first
    python3 rl_train.py --phase validate --lam 0.0

    # Train safety-mode policy
    python3 rl_train.py --phase train --lam 0.0 --episodes 500

    # Train speed-mode policy
    python3 rl_train.py --phase train --lam 1.0 --episodes 500

    # Evaluate a saved checkpoint
    python3 rl_train.py --phase eval --lam 0.0 --load checkpoints/lam0.0_ep500.pt
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

# ── Suppress threading before heavy imports ───────────────────────────────
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from obs import Obstacle, DynamicObstacle
from run_experiment_unified import MPCCController, dt, N
from params import r_robot, safety_buffer, v_max
from mpcc import build_mpcc_solver, MPCCConfig
from lidar import scan, congestion_obs
from weight_sets import (
    WEIGHT_SETS, WEIGHT_SET_NAMES, N_ACTIONS,
    get_config, rule_based_action,
)
from rl_agent import DQNAgent
from debug import (
    check_static_zone_violation,
    check_dyn_exclusion_violation,
    evaluate_dyn_exclusion_depths,
    extract_progress,
    VIOLATION_THRESHOLD,
)

try:
    from dyn_obs_generator import make_seeded_dyn_obs, MAP_BOUNDS, PATH_ID_TO_MAP_KEY
    HAS_GENERATOR = True
except ImportError:
    HAS_GENERATOR = False

# ════════════════════════════════════════════════════════════════════════════
# Training configuration
# ════════════════════════════════════════════════════════════════════════════

N_VALIDATE       = 20        # episodes for rule-based sanity check
N_TRAIN_EPISODES = 600       # DQN training episodes
MAX_STEPS        = 1500      # steps per episode
START_TRIM       = 8         # trim ref_traj prefix (same as debug.py)
GOAL_TOL         = 0.35      # metres
END_PROGRESS_TOL = 1.0

PATH_IDS         = [10, 11]  # maps used for training
DENSITY_TIERS    = ["low", "medium", "high"]
CHECKPOINT_EVERY = 50        # save checkpoint every N episodes
CHECKPOINT_DIR   = Path("checkpoints")
LOG_DIR          = Path("logs")

# Reward weights
W_PROG  = 1.0    # path progress (Δμ per step)
W_EXCL  = 5.0    # dynamic exclusion depth penalty
W_STAT  = 10.0   # static zone violation (per step)
W_SPEED = 2.0    # speed penalty (1 − v/v_max) in speed mode
W_GOAL  = 50.0   # sparse goal bonus


# ════════════════════════════════════════════════════════════════════════════
# Environment reset
# ════════════════════════════════════════════════════════════════════════════

def _make_env(path_id: int, seed: int, density_tier: str = "medium"):
    """Create ref_traj, static_rects, dyn_obs, x_start, x_goal."""
    env          = Obstacle(path_id)
    ref_traj     = env.path_selector(path_id)[:, START_TRIM:]
    static_rects = env.static_obs()

    p0     = ref_traj[:, 0]
    look_k = min(10, ref_traj.shape[1] - 1)
    th0    = float(np.arctan2(ref_traj[1, look_k] - p0[1],
                              ref_traj[0, look_k] - p0[0]))
    x_start = np.array([p0[0], p0[1], th0], dtype=float)
    x_goal  = np.array([ref_traj[0, -1], ref_traj[1, -1], 0.0], dtype=float)

    if HAS_GENERATOR:
        map_key = PATH_ID_TO_MAP_KEY.get(path_id)
        bounds  = MAP_BOUNDS.get(map_key, (0.0, 42.0, 0.0, 42.0))
        dyn_obs = make_seeded_dyn_obs(
            static_rects=static_rects,
            map_bounds=bounds,
            robot_start=x_start[:2],
            robot_goal=x_goal[:2],
            density_tier=density_tier,
            seed=seed,
        )
    else:
        # Fallback: use obs.py manual factory
        from debug import make_debug_dyn_obs
        dyn_obs = make_debug_dyn_obs(path_id, seed_offset=seed % 10)

    return ref_traj, static_rects, dyn_obs, x_start, x_goal


# ════════════════════════════════════════════════════════════════════════════
# Reward function
# ════════════════════════════════════════════════════════════════════════════

def compute_reward(
    lam:             float,
    delta_mu:        float,
    excl_depth:      float,
    static_violated: bool,
    v:               float,
    goal_reached:    bool,
) -> float:
    r  = W_PROG  * delta_mu
    r -= W_EXCL  * (1.0 - lam) * excl_depth
    r -= W_STAT  * (1.0 - lam) * float(static_violated)
    r -= W_SPEED * lam * max(0.0, 1.0 - v / v_max)
    if goal_reached:
        r += W_GOAL
    return float(r)


# ════════════════════════════════════════════════════════════════════════════
# Build an MPCCController with a given MPCCConfig
# ════════════════════════════════════════════════════════════════════════════

def _build_controller(
    config:       MPCCConfig,
    ref_traj:     np.ndarray,
    static_rects: list,
    dyn_obs:      list,
    use_hard:     bool = False,
) -> MPCCController:
    """
    Rebuild an MPCCController using a specific MPCCConfig weight set.
    We call build_mpcc_solver directly with the config so the NLP uses
    the right cost weights.
    """
    from mpcc import build_mpcc_solver, MPCCArtifacts

    data = build_mpcc_solver(
        ref_traj=ref_traj,
        static_rects=static_rects,
        dyn_obs=dyn_obs,
        use_hard=use_hard,
        config=config,
    )
    from run_experiment_unified import (
        MAX_ACTIVE_DYN_OBS,
        _default_observation_radius,
        _make_solver_dyn_templates,
    )

    ctrl = MPCCController.__new__(MPCCController)
    ctrl.ref_traj     = ref_traj
    ctrl.use_hard     = use_hard
    ctrl.static_rects = static_rects

    ctrl.max_active_dyn    = min(MAX_ACTIVE_DYN_OBS, len(dyn_obs))
    ctrl.observation_radius = _default_observation_radius(dyn_obs)
    ctrl.solver_dyn_templates = _make_solver_dyn_templates(
        dyn_obs, ctrl.max_active_dyn
    )

    ctrl.mpcc_data    = data
    ctrl.solver       = data.solver
    ctrl.lbx          = data.lbx
    ctrl.ubx          = data.ubx
    ctrl.lbg          = data.lbg
    ctrl.ubg          = data.ubg
    ctrl.mu           = 0.0
    ctrl.u_prev       = np.zeros(2)
    ctrl.prev_z       = None
    return ctrl


# ════════════════════════════════════════════════════════════════════════════
# Single episode rollout
# ════════════════════════════════════════════════════════════════════════════

def run_episode(
    agent_or_rule,       # DQNAgent | "rule"
    lam:         float,
    path_id:     int,
    seed:        int,
    density_tier:str    = "medium",
    use_hard:    bool   = False,
    train:       bool   = True,
    verbose:     bool   = False,
) -> dict:
    """
    Run one full episode.

    The controller is rebuilt when the selected weight set changes.
    This is necessary because the MPCC NLP embeds the weights at build
    time.  The rebuild cost (~1 s) happens at most 3 times per episode
    and is amortised over hundreds of steps.

    Returns a metrics dict.
    """
    ref_traj, static_rects, dyn_obs, x_start, x_goal = _make_env(
        path_id, seed, density_tier
    )

    # Build all three controllers up-front so switching is fast
    controllers = [
        _build_controller(cfg, ref_traj, static_rects, dyn_obs, use_hard)
        for cfg in WEIGHT_SETS
    ]

    x_current    = x_start.copy()
    prev_mu      = 0.0
    active_action = 1          # start with balanced
    controller   = controllers[active_action]

    total_reward    = 0.0
    goal_reached    = False
    solver_failures = 0
    steps_taken     = 0
    action_hist     = []
    loss_hist       = []
    excl_depth_hist = []
    v_hist          = []

    obs_vec = congestion_obs(
        scan(x_current[:2], x_current[2], static_rects, dyn_obs)
    )

    for k in range(MAX_STEPS):
        # ── Select action ─────────────────────────────────────────────
        if agent_or_rule == "rule":
            action = rule_based_action(obs_vec)
        else:
            action = agent_or_rule.select_action(obs_vec, greedy=(not train))

        # ── Switch weight set if needed ───────────────────────────────
        if action != active_action:
            active_action = action
            controller    = controllers[active_action]
            # Reset warm-start so the new cost landscape is explored cleanly
            controller.prev_z = None

        action_hist.append(action)

        # ── Solve ─────────────────────────────────────────────────────
        step_result = controller.solve(x_current, dyn_obs, x_goal)

        if not step_result.ok:
            solver_failures += 1
            # Try falling back to balanced
            if active_action != 1:
                active_action = 1
                controller    = controllers[1]
                controller.prev_z = None
                step_result = controller.solve(x_current, dyn_obs, x_goal)

            if not step_result.ok:
                break

        v, omega = step_result.u
        v_hist.append(float(v))

        # ── Propagate ─────────────────────────────────────────────────
        if (hasattr(step_result, "solver_next_state")
                and step_result.solver_next_state is not None):
            x_next = step_result.solver_next_state
        else:
            x_next = np.array([
                x_current[0] + dt * v * np.cos(x_current[2]),
                x_current[1] + dt * v * np.sin(x_current[2]),
                x_current[2] + dt * omega,
            ], dtype=float)

        for obs in dyn_obs:
            obs.step(dt, static_rects)

        steps_taken += 1

        # ── Compute reward ingredients ────────────────────────────────
        progress_value, _ = extract_progress(controller, step_result)
        delta_mu = float(progress_value) - prev_mu
        prev_mu  = float(progress_value)

        xy = x_next[:2]
        static_viol, _  = check_static_zone_violation(xy, static_rects)
        excl_viol,   _  = check_dyn_exclusion_violation(xy, dyn_obs)
        _, depths, _, _, worst_depth = evaluate_dyn_exclusion_depths(xy, dyn_obs)
        excl_depth_hist.append(worst_depth)

        goal_dist = float(np.linalg.norm(xy - x_goal[:2]))
        at_end    = progress_value >= (ref_traj.shape[1] - END_PROGRESS_TOL)
        goal_reached = (goal_dist <= GOAL_TOL or at_end)

        reward = compute_reward(lam, delta_mu, worst_depth,
                                static_viol, v, goal_reached)
        total_reward += reward

        # ── Get next observation ──────────────────────────────────────
        next_obs_vec = congestion_obs(
            scan(x_next[:2], x_next[2], static_rects, dyn_obs)
        )

        # ── RL update ─────────────────────────────────────────────────
        if train and agent_or_rule != "rule":
            agent_or_rule.store(obs_vec, action, reward, next_obs_vec, goal_reached)
            loss = agent_or_rule.train_step()
            if loss is not None:
                loss_hist.append(loss)

        obs_vec   = next_obs_vec
        x_current = x_next

        if verbose and k % 100 == 0:
            print(f"  step {k:4d}  action={WEIGHT_SET_NAMES[action]:<12}  "
                  f"Δμ={delta_mu:.3f}  excl_depth={worst_depth:.4f}  "
                  f"r={reward:.3f}  ε={getattr(agent_or_rule, 'epsilon', 0):.3f}")

        if goal_reached:
            break

    action_counts = [action_hist.count(i) for i in range(N_ACTIONS)]

    return {
        "goal_reached":      int(goal_reached),
        "steps_taken":       steps_taken,
        "total_reward":      round(total_reward, 3),
        "solver_failures":   solver_failures,
        "mean_excl_depth":   round(float(np.mean(excl_depth_hist)) if excl_depth_hist else 0.0, 5),
        "worst_excl_depth":  round(float(max(excl_depth_hist, default=0.0)), 5),
        "mean_speed":        round(float(np.mean(v_hist)) if v_hist else 0.0, 4),
        "action_counts":     action_counts,
        "mean_loss":         round(float(np.mean(loss_hist)) if loss_hist else 0.0, 5),
        "epsilon":           getattr(agent_or_rule, "epsilon", 0.0),
    }


# ════════════════════════════════════════════════════════════════════════════
# Phase 1 — rule-based validation
# ════════════════════════════════════════════════════════════════════════════

def validate_rule_based(lam: float = 0.0, n_episodes: int = N_VALIDATE) -> None:
    print("\n" + "=" * 64)
    print(f"PHASE 1: Rule-based policy validation  λ={lam}")
    print("=" * 64)

    results = []
    for ep in range(n_episodes):
        path_id = PATH_IDS[ep % len(PATH_IDS)]
        tier    = DENSITY_TIERS[ep % len(DENSITY_TIERS)]
        seed    = ep

        t0 = time.perf_counter()
        m  = run_episode("rule", lam, path_id, seed, tier, train=False)
        elapsed = time.perf_counter() - t0

        results.append(m)
        ac = m["action_counts"]
        print(
            f"  ep {ep:3d}  path={path_id}  {tier:<8}  "
            f"goal={m['goal_reached']}  steps={m['steps_taken']:4d}  "
            f"r={m['total_reward']:7.1f}  "
            f"excl={m['worst_excl_depth']:.4f}  "
            f"actions=[C:{ac[0]:3d} B:{ac[1]:3d} A:{ac[2]:3d}]  "
            f"{elapsed:.1f}s"
        )

    goal_rate = sum(r["goal_reached"] for r in results) / len(results)
    mean_excl = np.mean([r["worst_excl_depth"] for r in results])
    total_switches = sum(
        sum(abs(r["action_counts"][i] - r["action_counts"][i-1])
            for i in range(1, N_ACTIONS))
        for r in results
    )

    print(f"\n  Summary: goal_rate={goal_rate:.1%}  "
          f"mean_worst_excl_depth={mean_excl:.4f}  "
          f"total_weight_switches≈{total_switches}")
    print("\n  ✓  Rule-based validation complete.")
    print("     Check that action_counts shows meaningful switching")
    print("     (not always the same action) before proceeding to DQN training.")


# ════════════════════════════════════════════════════════════════════════════
# Phase 2 — DQN training
# ════════════════════════════════════════════════════════════════════════════

def train_dqn(
    lam:          float = 0.0,
    n_episodes:   int   = N_TRAIN_EPISODES,
    load_path:    str | None = None,
    verbose:      bool  = False,
) -> DQNAgent:
    print("\n" + "=" * 64)
    print(f"PHASE 2: DQN training  λ={lam}  episodes={n_episodes}")
    print("=" * 64)

    agent = DQNAgent(lam=lam)
    if load_path:
        agent.load(load_path)
        print(f"  Loaded checkpoint: {load_path}")

    LOG_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"train_lam{lam:.1f}.csv"

    with open(log_path, "w") as f:
        f.write("episode,path_id,tier,goal,steps,reward,"
                "mean_excl,worst_excl,mean_speed,"
                "act_C,act_B,act_A,mean_loss,epsilon\n")

    t_start = time.perf_counter()

    for ep in range(n_episodes):
        path_id = PATH_IDS[ep % len(PATH_IDS)]
        tier    = DENSITY_TIERS[ep % len(DENSITY_TIERS)]
        seed    = ep

        m = run_episode(agent, lam, path_id, seed, tier, train=True,
                        verbose=(verbose and ep % 50 == 0))

        ac = m["action_counts"]

        # Logging
        with open(log_path, "a") as f:
            f.write(f"{ep},{path_id},{tier},{m['goal_reached']},"
                    f"{m['steps_taken']},{m['total_reward']:.3f},"
                    f"{m['mean_excl_depth']},{m['worst_excl_depth']},"
                    f"{m['mean_speed']},{ac[0]},{ac[1]},{ac[2]},"
                    f"{m['mean_loss']},{m['epsilon']:.4f}\n")

        elapsed = time.perf_counter() - t_start
        eta = (elapsed / (ep + 1)) * (n_episodes - ep - 1)

        print(f"  ep {ep:4d}/{n_episodes}  path={path_id}  {tier:<8}  "
              f"goal={m['goal_reached']}  r={m['total_reward']:7.1f}  "
              f"excl={m['worst_excl_depth']:.4f}  "
              f"ε={m['epsilon']:.3f}  "
              f"[C:{ac[0]:3d} B:{ac[1]:3d} A:{ac[2]:3d}]  "
              f"ETA {eta/60:.1f}min")

        if (ep + 1) % CHECKPOINT_EVERY == 0:
            ck = CHECKPOINT_DIR / f"lam{lam:.1f}_ep{ep+1:04d}.pt"
            agent.save(ck)
            print(f"    → saved {ck}")

    final_ck = CHECKPOINT_DIR / f"lam{lam:.1f}_final.pt"
    agent.save(final_ck)
    print(f"\n  Training complete.  Final checkpoint: {final_ck}")
    print(f"  Log: {log_path}")
    return agent


# ════════════════════════════════════════════════════════════════════════════
# Phase 3 — evaluation
# ════════════════════════════════════════════════════════════════════════════

def evaluate(
    lam:         float,
    load_path:   str,
    n_episodes:  int = 30,
    density_tier:str = "medium",
) -> None:
    print("\n" + "=" * 64)
    print(f"EVALUATION  λ={lam}  checkpoint={load_path}")
    print("=" * 64)

    agent = DQNAgent(lam=lam)
    agent.load(load_path)

    results = []
    for ep in range(n_episodes):
        path_id = PATH_IDS[ep % len(PATH_IDS)]
        seed    = 1000 + ep   # use different seeds from training

        m = run_episode(agent, lam, path_id, seed, density_tier,
                        train=False)
        results.append(m)

        ac = m["action_counts"]
        print(
            f"  ep {ep:3d}  path={path_id}  "
            f"goal={m['goal_reached']}  steps={m['steps_taken']:4d}  "
            f"r={m['total_reward']:7.1f}  excl={m['worst_excl_depth']:.4f}  "
            f"[C:{ac[0]:3d} B:{ac[1]:3d} A:{ac[2]:3d}]"
        )

    goal_rate  = np.mean([r["goal_reached"]    for r in results])
    mean_excl  = np.mean([r["worst_excl_depth"] for r in results])
    mean_speed = np.mean([r["mean_speed"]       for r in results])

    print(f"\n  goal_rate={goal_rate:.1%}  "
          f"mean_worst_excl_depth={mean_excl:.4f}  "
          f"mean_speed={mean_speed:.3f} m/s")


# ════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="RL weight-switching for MPCC")
    parser.add_argument("--phase",    choices=["validate", "train", "eval"],
                        default="validate")
    parser.add_argument("--lam",      type=float, default=0.0,
                        help="Mode parameter λ (0=safety, 1=speed)")
    parser.add_argument("--episodes", type=int,   default=N_TRAIN_EPISODES)
    parser.add_argument("--load",     type=str,   default=None,
                        help="Path to checkpoint to resume from or evaluate")
    parser.add_argument("--verbose",  action="store_true")
    parser.add_argument("--tier",     default="medium",
                        choices=["low", "medium", "high"])
    args = parser.parse_args()

    if args.phase == "validate":
        validate_rule_based(lam=args.lam)

    elif args.phase == "train":
        train_dqn(lam=args.lam, n_episodes=args.episodes,
                  load_path=args.load, verbose=args.verbose)

    elif args.phase == "eval":
        if not args.load:
            parser.error("--load is required for eval phase")
        evaluate(lam=args.lam, load_path=args.load, density_tier=args.tier)


if __name__ == "__main__":
    main()