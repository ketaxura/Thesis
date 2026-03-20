
import heapq
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

GRID_RES = 0.10
INFLATION_RADIUS = 0.18
ALLOW_DIAGONALS = True

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

    for ax, (map_name, cfg) in zip(axes, MAPS.items()):
        occ, xs, ys = build_occupancy(cfg["bounds"], cfg["rects"], GRID_RES, INFLATION_RADIUS)

        start_rc = world_to_grid(cfg["start"][0], cfg["start"][1], cfg["bounds"], GRID_RES)
        goal_rc = world_to_grid(cfg["goal"][0], cfg["goal"][1], cfg["bounds"], GRID_RES)

        grid_path = astar(occ, start_rc, goal_rc)

        raw_path_xy = np.array(
            [grid_to_world(r, c, cfg["bounds"], GRID_RES) for (r, c) in grid_path],
            dtype=float,
        )

        pruned_path_xy = prune_collinear(raw_path_xy)
        ref_path_xy = densify_polyline(pruned_path_xy, ds=0.05)

        raw_csv = OUTPUT_DIR / f"{map_name}_raw_astar_path.csv"
        ref_csv = OUTPUT_DIR / f"{map_name}_ref_path.csv"
        save_path_csv(raw_path_xy, raw_csv)
        save_path_csv(ref_path_xy, ref_csv)

        saved.append((raw_csv, ref_csv))
        draw_map_with_path(ax, cfg, occ, xs, ys, raw_path_xy, ref_path_xy)

    fig.suptitle("Inspired Thesis Maps with A* and Saved Reference Paths", fontsize=14)
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
