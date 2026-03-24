"""
lidar.py
========
Simulated 2-D lidar sensor and congestion feature extractor.

The sensor casts N_RAYS evenly-spaced rays from the robot centre
outward to MAX_RANGE, checking against static rectangles and
dynamic ellipses.  It returns the hit-range per ray (MAX_RANGE
if no hit).

The congestion feature extractor compresses the full scan into a
small observation vector used as input to the RL policy:

    obs = [
        n_hits_norm,          # fraction of rays that hit something
        min_range_norm,       # closest hit / MAX_RANGE  (0 = right next to it)
        mean_range_norm,      # mean hit range among HITS / MAX_RANGE
        front_density,        # hit fraction in the forward 90-deg sector
        side_density,         # hit fraction in left+right 90-deg sectors
        rear_density,         # hit fraction in rear 90-deg sector
    ]

All values ∈ [0, 1], making them safe to feed directly into a neural
network or a simple threshold rule-based policy.
"""

from __future__ import annotations

import numpy as np


# ── Sensor defaults ────────────────────────────────────────────────────────
N_RAYS    = 36       # number of rays (10-degree spacing)
MAX_RANGE = 6.0      # metres — beyond this we treat as free space
P_NORM    = 6        # must match the super-ellipse p-norm in mpcc.py


def _ray_hit_rect(
    ox: float, oy: float,
    dx: float, dy: float,
    cx: float, cy: float,
    hw: float, hh: float,
    max_t: float,
) -> float:
    """
    Slab-method ray–AABB intersection.
    Returns hit distance along the ray, or max_t if no hit.
    """
    inv_dx = 1.0 / dx if abs(dx) > 1e-12 else 1e12 * np.sign(dx + 1e-30)
    inv_dy = 1.0 / dy if abs(dy) > 1e-12 else 1e12 * np.sign(dy + 1e-30)

    tx1 = (cx - hw - ox) * inv_dx
    tx2 = (cx + hw - ox) * inv_dx
    ty1 = (cy - hh - oy) * inv_dy
    ty2 = (cy + hh - oy) * inv_dy

    t_min = max(min(tx1, tx2), min(ty1, ty2))
    t_max = min(max(tx1, tx2), max(ty1, ty2))

    if t_max < 0 or t_min > t_max or t_min > max_t:
        return max_t
    return max(0.0, t_min)


def _ray_hit_ellipse(
    ox: float, oy: float,
    dx: float, dy: float,
    ex: float, ey: float,
    a: float, b: float,
    theta: float,
    max_t: float,
) -> float:
    """
    Ray–rotated-ellipse intersection (analytic).
    Returns hit distance or max_t.
    """
    c, s = np.cos(theta), np.sin(theta)

    # Transform ray origin and direction into ellipse local frame
    rx = ox - ex
    ry = oy - ey
    # Rotate into local frame
    lox =  c * rx + s * ry
    loy = -s * rx + c * ry
    ldx =  c * dx + s * dy
    ldy = -s * dx + c * dy

    # Normalise by semiaxes
    lox /= a;  ldx /= a
    loy /= b;  ldy /= b

    A = ldx * ldx + ldy * ldy
    B = 2.0 * (lox * ldx + loy * ldy)
    C = lox * lox + loy * loy - 1.0

    disc = B * B - 4.0 * A * C
    if disc < 0 or A < 1e-12:
        return max_t

    sq = np.sqrt(disc)
    t1 = (-B - sq) / (2.0 * A)
    t2 = (-B + sq) / (2.0 * A)

    for t in sorted([t1, t2]):
        if 0.0 < t < max_t:
            return t
    return max_t


def scan(
    robot_xy: np.ndarray,
    robot_theta: float,
    static_rects: list,
    dyn_obs: list,
    n_rays: int = N_RAYS,
    max_range: float = MAX_RANGE,
) -> np.ndarray:
    """
    Cast n_rays evenly-spaced rays from the robot.

    Parameters
    ----------
    robot_xy     : np.ndarray shape (2,)
    robot_theta  : float — heading in radians
    static_rects : list of (cx, cy, hw, hh)
    dyn_obs      : list of DynamicObstacle with attributes x, y, a, b, theta

    Returns
    -------
    ranges : np.ndarray shape (n_rays,)
        Hit range per ray in [0, max_range].
        Ray 0 points straight ahead (robot_theta), then CCW.
    """
    ox, oy = float(robot_xy[0]), float(robot_xy[1])
    angles = robot_theta + np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
    ranges = np.full(n_rays, max_range, dtype=float)

    for i, ang in enumerate(angles):
        dx = np.cos(ang)
        dy = np.sin(ang)
        t  = max_range

        for (cx, cy, hw, hh) in static_rects:
            t = min(t, _ray_hit_rect(ox, oy, dx, dy, cx, cy, hw, hh, t))

        for obs in dyn_obs:
            t = min(t, _ray_hit_ellipse(
                ox, oy, dx, dy,
                obs.x, obs.y, obs.a, obs.b, obs.theta, t,
            ))

        ranges[i] = t

    return ranges


def congestion_obs(
    ranges: np.ndarray,
    max_range: float = MAX_RANGE,
) -> np.ndarray:
    """
    Compress a full lidar scan into a 6-D congestion observation.

    All values are in [0, 1].

    Layout of rays (ray 0 = front):
        front   :  rays 0 .. n//4-1  and  3n//4 .. n-1   (±45 deg each side)
        sides   :  rays n//4 .. 3n//4-1
        rear    :  middle quarter of sides (approx ±90 from front)

    Returns
    -------
    obs : np.ndarray shape (6,)
        [n_hits_norm, min_range_norm, mean_hit_range_norm,
         front_density, side_density, rear_density]
    """
    n   = len(ranges)
    eps = 1e-6

    hits      = ranges < max_range - eps
    n_hits    = hits.sum()

    n_hits_norm     = float(n_hits) / n
    min_range_norm  = float(ranges.min()) / max_range
    mean_hit_norm   = float(ranges[hits].mean() / max_range) if n_hits > 0 else 1.0

    # Sector boundaries (number of rays per sector)
    q = n // 4

    # Front: first q/2 and last q/2 rays
    front_idx = list(range(q // 2)) + list(range(n - q // 2, n))
    side_idx  = list(range(q // 2, n - q // 2))
    rear_idx  = list(range(n // 2 - q // 4, n // 2 + q // 4))

    front_density = float(hits[front_idx].mean()) if front_idx else 0.0
    side_density  = float(hits[side_idx].mean())  if side_idx  else 0.0
    rear_density  = float(hits[rear_idx].mean())  if rear_idx  else 0.0

    return np.array([
        n_hits_norm,
        min_range_norm,
        mean_hit_norm,
        front_density,
        side_density,
        rear_density,
    ], dtype=np.float32)


OBS_DIM = 6   # dimension of congestion_obs output — imported by rl_agent.py