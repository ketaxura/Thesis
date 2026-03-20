
import heapq
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import CubicSpline
from scipy.interpolate import splprep, splev

# ============================================================
# Two thesis map candidates with A* path generation.
# Static rectangles use (cx, cy, half_width, half_height).
# ============================================================

MAPS = {
    "map_corridor_structured": {
        "title": "Inspired Map A - Structured Corridor",
        "start": (0.8, 8.4),
        "goal": (9.2, 0.8),
        "bounds": (0.0, 10.0, 0.0, 9.0),
        "rects": [
            (5.0, 8.85, 5.0, 0.15),   # top wall
            (5.0, 0.15, 5.0, 0.15),   # bottom wall
            (0.15, 4.5, 0.15, 4.5),   # left wall
            (9.85, 4.5, 0.15, 4.5),   # right wall

            (2.2, 7.5, 1.6, 0.45),
            (3.35, 6.0, 0.45, 1.55),
            (5.1, 7.6, 0.55, 1.35),
            (7.8, 7.5, 1.2, 0.45),
            (7.1, 6.0, 0.45, 1.55),
            (1.1, 4.0, 0.45, 1.55),
            (4.0, 3.2, 1.55, 0.45),
            (5.7, 4.2, 0.55, 0.75),
            (7.3, 2.3, 0.45, 1.45),
            (8.0, 1.25, 0.95, 0.45),
            (3.25, 1.25, 2.45, 0.45),
        ],
    },
    "map_open_clutter": {
        "title": "Inspired Map B - Moderate Open Clutter",
        "start": (0.7, 0.8),
        "goal": (8.8, 8.0),
        "bounds": (0.0, 10.0, 0.0, 9.0),
        "rects": [
            (5.0, 8.85, 5.0, 0.15),   # top wall
            (5.0, 0.15, 5.0, 0.15),   # bottom wall
        (0.15, 4.5, 0.15, 4.5),   # left wall
        (9.85, 4.5, 0.15, 4.5),   # right wall

(1.15, 5.2, 0.55, 3.2),
(3.3, 7.0, 1.0, 0.45),
(2.55, 6.05, 0.45, 0.95),
            (5.1, 7.45, 0.35, 0.65),
            (7.35, 6.9, 1.0, 0.45),
            (8.25, 6.1, 0.45, 0.8),
            (3.1, 4.9, 0.45, 0.95),
            (4.95, 5.45, 0.75, 0.45),
            (7.55, 5.0, 0.8, 0.45),
            (3.35, 2.65, 1.05, 0.45),
            (2.55, 1.95, 0.45, 0.7),
            (5.8, 2.95, 0.45, 1.0),
            (7.7, 2.45, 0.45, 1.4),
            (7.7, 1.55, 0.9, 0.45),
            (5.0, 1.15, 1.15, 0.45),
            (1.5, 1.2, 0.5, 0.5),
            (4.7, 6.0, 0.35, 0.35),
        ],
    },
}

# ============================================================
# A* settings
# ============================================================

MAP_SCALE = 4.0
GRID_RES = 0.10
INFLATION_RADIUS = 0.75   # = r_robot + safety_buffer
ALLOW_DIAGONALS = True
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def scale_rect(rect, s: float):
    cx, cy, hw, hh = rect
    return (s * cx, s * cy, s * hw, s * hh)


def scale_map_cfg(cfg: dict, s: float) -> dict:
    xmin, xmax, ymin, ymax = cfg["bounds"]
    sx, sy = cfg["start"]
    gx, gy = cfg["goal"]

    return {
        "title": f'{cfg["title"]} (scale={s:.1f}x)',
        "start": (s * sx, s * sy),
        "goal": (s * gx, s * gy),
        "bounds": (s * xmin, s * xmax, s * ymin, s * ymax),
        "rects": [scale_rect(r, s) for r in cfg["rects"]],
    }


SCALED_MAPS = {
    name: scale_map_cfg(cfg, MAP_SCALE)
    for name, cfg in MAPS.items()
}


def world_to_grid(x: float, y: float, bounds, res: float) -> tuple[int, int]:
    xmin, xmax, ymin, ymax = bounds
    col = int(round((x - xmin) / res))
    row = int(round((y - ymin) / res))
    return row, col


def grid_to_world(row: int, col: int, bounds, res: float) -> tuple[float, float]:
    xmin, xmax, ymin, ymax = bounds
    x = xmin + col * res
    y = ymin + row * res
    return float(x), float(y)


def build_occupancy(bounds, rects, res: float, inflation: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = bounds
    xs = np.arange(xmin, xmax + 0.5 * res, res)
    ys = np.arange(ymin, ymax + 0.5 * res, res)
    occ = np.zeros((len(ys), len(xs)), dtype=np.uint8)

    X, Y = np.meshgrid(xs, ys)

    for (cx, cy, hw, hh) in rects:
        mask = (
            (np.abs(X - cx) <= (hw + inflation))
            & (np.abs(Y - cy) <= (hh + inflation))
        )
        occ[mask] = 1

    return occ, xs, ys


def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def neighbors(node: tuple[int, int], occ: np.ndarray):
    r, c = node
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if ALLOW_DIAGONALS:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    nrows, ncols = occ.shape
    for dr, dc in steps:
        rr = r + dr
        cc = c + dc
        if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
            continue
        if occ[rr, cc] != 0:
            continue

        # Prevent corner cutting through inflated blocks
        if dr != 0 and dc != 0:
            if occ[r + dr, c] != 0 or occ[r, c + dc] != 0:
                continue

        cost = float(np.hypot(dr, dc))
        yield (rr, cc), cost


def astar(occ: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    if occ[start] != 0:
        raise ValueError(f"Start lies inside occupied space: {start}")
    if occ[goal] != 0:
        raise ValueError(f"Goal lies inside occupied space: {goal}")

    pq = []
    heapq.heappush(pq, (0.0, start))

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_cost = {start: 0.0}
    closed = set()

    while pq:
        _, current = heapq.heappop(pq)

        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for nb, step_cost in neighbors(current, occ):
            tentative = g_cost[current] + step_cost
            if tentative < g_cost.get(nb, float("inf")):
                g_cost[nb] = tentative
                came_from[nb] = current
                f = tentative + heuristic(nb, goal)
                heapq.heappush(pq, (f, nb))

    raise RuntimeError("A* failed to find a path.")


def prune_collinear(points: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    if len(points) <= 2:
        return points.copy()

    kept = [points[0]]
    for i in range(1, len(points) - 1):
        a = kept[-1]
        b = points[i]
        c = points[i + 1]

        ab = b - a
        bc = c - b
        cross = ab[0] * bc[1] - ab[1] * bc[0]
        if abs(cross) > eps:
            kept.append(b)

    kept.append(points[-1])
    return np.array(kept, dtype=float)


def point_is_in_inflated_rect(px, py, rect, margin):
    cx, cy, hw, hh = rect
    return (abs(px - cx) <= hw + margin) and (abs(py - cy) <= hh + margin)


def segment_is_collision_free(p0, p1, static_rects, margin, ds_check=0.02):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    seg = p1 - p0
    L = np.linalg.norm(seg)
    if L < 1e-12:
        return True

    n = max(2, int(np.ceil(L / ds_check)) + 1)
    for a in np.linspace(0.0, 1.0, n):
        pt = (1.0 - a) * p0 + a * p1
        px, py = float(pt[0]), float(pt[1])

        for rect in static_rects:
            if point_is_in_inflated_rect(px, py, rect, margin):
                return False

    return True


def shortcut_prune_path(path_xy, static_rects, margin, ds_check=0.02):
    path_xy = np.asarray(path_xy, dtype=float)
    if len(path_xy) <= 2:
        return path_xy.copy()

    out = [path_xy[0]]
    i = 0
    n = len(path_xy)

    while i < n - 1:
        j_best = i + 1
        for j in range(n - 1, i, -1):
            if segment_is_collision_free(path_xy[i], path_xy[j], static_rects, margin, ds_check=ds_check):
                j_best = j
                break
        out.append(path_xy[j_best])
        i = j_best

    return np.asarray(out, dtype=float)




def path_min_clearance(path_xy: np.ndarray, rects, inflation: float) -> float:
    """
    Conservative clearance estimate from sampled path points to inflated axis-aligned rectangles.
    Positive means outside, negative means inside.
    """
    min_clear = float("inf")

    for px, py in path_xy:
        for (cx, cy, hw, hh) in rects:
            dx = abs(px - cx) - (hw + inflation)
            dy = abs(py - cy) - (hh + inflation)

            # outside distance to inflated rectangle
            ox = max(dx, 0.0)
            oy = max(dy, 0.0)
            outside_dist = np.hypot(ox, oy)

            # if inside inflated rectangle, use negative penetration-like measure
            if dx <= 0.0 and dy <= 0.0:
                inside_depth = -min(-dx, -dy)
                clear = inside_depth
            else:
                clear = outside_dist

            min_clear = min(min_clear, clear)

    return float(min_clear)


def densify_polyline(points: np.ndarray, ds: float = 0.05) -> np.ndarray:
    if len(points) <= 1:
        return points.copy()

    out = [points[0]]
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        seg = p1 - p0
        L = np.linalg.norm(seg)
        if L < 1e-12:
            continue
        n = max(1, int(np.ceil(L / ds)))
        for k in range(1, n + 1):
            alpha = k / n
            out.append((1 - alpha) * p0 + alpha * p1)
    return np.array(out, dtype=float)


def save_path_csv(path_xy: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = "x,y"
    np.savetxt(out_path, path_xy, delimiter=",", header=header, comments="")

def draw_map_with_path(ax, cfg, occ, xs, ys, raw_path_xy, ref_path_xy):
    xmin, xmax, ymin, ymax = cfg["bounds"]

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(cfg["title"])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.25)

    # Occupancy image for debug
    ax.imshow(
        occ,
        origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        alpha=0.12,
        interpolation="nearest",
    )

    # Obstacles
    for (cx, cy, hw, hh) in cfg["rects"]:
        rect = Rectangle((cx - hw, cy - hh), 2 * hw, 2 * hh)
        ax.add_patch(rect)

    sx, sy = cfg["start"]
    gx, gy = cfg["goal"]
    ax.plot(sx, sy, marker="o", markersize=8, label="start")
    ax.plot(gx, gy, marker="*", markersize=12, label="goal")

    if raw_path_xy is not None:
        ax.plot(raw_path_xy[:, 0], raw_path_xy[:, 1], "--", linewidth=1.2, label="raw A*")

    if ref_path_xy is not None:
        ax.plot(ref_path_xy[:, 0], ref_path_xy[:, 1], linewidth=2.5, label="saved ref path")

    ax.legend(loc="upper right", fontsize=8)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    saved = []

    for ax, (map_name, cfg) in zip(axes, SCALED_MAPS.items()):
        occ, xs, ys = build_occupancy(cfg["bounds"], cfg["rects"], GRID_RES, INFLATION_RADIUS)

        start_rc = world_to_grid(cfg["start"][0], cfg["start"][1], cfg["bounds"], GRID_RES)
        goal_rc = world_to_grid(cfg["goal"][0], cfg["goal"][1], cfg["bounds"], GRID_RES)

        grid_path = astar(occ, start_rc, goal_rc)

        raw_path_xy = np.array(
            [grid_to_world(r, c, cfg["bounds"], GRID_RES) for (r, c) in grid_path],
            dtype=float,
        )


        print(f"{map_name}: raw_path start = {raw_path_xy[0]}")
        print(f"{map_name}: raw_path end   = {raw_path_xy[-1]}")
        print(f"{map_name}: raw_path points = {len(raw_path_xy)}")

        pruned_path_xy = prune_collinear(raw_path_xy)

        shortcut_path_xy = shortcut_prune_path(
            pruned_path_xy,
            static_rects=cfg["rects"],
            margin=INFLATION_RADIUS,
            ds_check=0.02,
        )

        print(f"{map_name}: pruned points = {len(pruned_path_xy)}")
        print(f"{map_name}: shortcut points = {len(shortcut_path_xy)}")

        ref_path_xy = densify_polyline(shortcut_path_xy, ds=0.1)


        raw_min_clear = path_min_clearance(ref_path_xy, cfg["rects"], INFLATION_RADIUS)
        print(f"{map_name}: pruned+dense path min clearance to inflated rects = {raw_min_clear:.4f} m")

        raw_csv = OUTPUT_DIR / f"{map_name}_raw_astar_path.csv"
        ref_csv = OUTPUT_DIR / f"{map_name}_ref_path.csv"
        save_path_csv(raw_path_xy, raw_csv)
        save_path_csv(ref_path_xy, ref_csv)

        saved.append((raw_csv, ref_csv))
        draw_map_with_path(ax, cfg, occ, xs, ys, raw_path_xy, ref_path_xy)

    fig.suptitle(
        f"Inspired Thesis Maps with A* and Saved Reference Paths | scale={MAP_SCALE:.1f}x | inflation={INFLATION_RADIUS:.2f}",
        fontsize=14,
    )
    plt.tight_layout()

    png_path = OUTPUT_DIR / "map_inspired_astar_preview.png"
    plt.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.show()

    print("Saved files:")
    print(png_path)
    for raw_csv, ref_csv in saved:
        print(raw_csv)
        print(ref_csv)


if __name__ == "__main__":
    main()
