
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================
# Two static maps inspired by the user's reference images.
# Obstacles are axis-aligned rectangles: (cx, cy, half_width, half_height)
# ============================================================

MAPS = {
    "map_corridor_structured": {
        "title": "Inspired Map A - Structured Corridor",
        "start": (0.6, 8.2),
        "goal": (9.2, 0.6),
        "bounds": (0.0, 10.0, 0.0, 9.0),
        "rects": [
            
            
            (5.0, 8.85, 5.0, 0.15),   # top wall
            (5.0, 0.15, 5.0, 0.15),   # bottom wall
            (0.15, 4.5, 0.15, 4.5),   # left wall
            (9.85, 4.5, 0.15, 4.5),   # right wall
            (2.2, 7.5, 1.6, 0.45),   # top-left long bar
            (3.35, 6.0, 0.45, 1.55), # left vertical drop
            (5.1, 7.6, 0.55, 1.35),  # center top vertical
            (7.8, 7.5, 1.2, 0.45),   # top-right cap
            (7.1, 6.0, 0.45, 1.55),  # right vertical drop
            (1.1, 4.0, 0.45, 1.55),  # left lower vertical
            (4.0, 3.2, 1.55, 0.45),  # center lower long bar
            (5.7, 4.2, 0.55, 0.75),  # center stub
            (7.3, 2.3, 0.45, 1.45),  # right lower vertical
            (8.0, 1.25, 0.95, 0.45), # right bottom foot
            (3.25, 1.25, 2.45, 0.45) # long bottom shelf
        ]
    },
    "map_open_clutter": {
        "title": "Inspired Map B - Moderate Open Clutter",
        "start": (0.8, 0.8),
        "goal": (8.6, 8.2),
        "bounds": (0.0, 10.0, 0.0, 9.0),
        "rects": [
            
            (5.0, 8.85, 5.0, 0.15),   # top wall
            (5.0, 0.15, 5.0, 0.15),   # bottom wall
            (0.15, 4.5, 0.15, 4.5),   # left wall
            (9.85, 4.5, 0.15, 4.5),   # right wall
            (1.15, 5.2, 0.55, 3.2),  # tall left wall
            (3.3, 7.0, 1.0, 0.45),   # top-left L horizontal
            (2.55, 6.05, 0.45, 0.95),# top-left L vertical
            (5.1, 7.45, 0.35, 0.65), # top center short vertical
            (7.35, 6.9, 1.0, 0.45),  # top-right L horizontal
            (8.25, 6.1, 0.45, 0.8),  # top-right L vertical
            (3.1, 4.9, 0.45, 0.95),  # mid-left vertical
            (4.95, 5.45, 0.75, 0.45),# mid bar
            (7.55, 5.0, 0.8, 0.45),  # mid-right bar
            (3.35, 2.65, 1.05, 0.45),# lower-left L horizontal
            (2.55, 1.95, 0.45, 0.7), # lower-left L vertical
            (5.8, 2.95, 0.45, 1.0),  # lower center vertical
            (7.7, 2.45, 0.45, 1.4),  # lower-right T stem
            (7.7, 1.55, 0.9, 0.45),  # lower-right T cap
            (5.0, 1.15, 1.15, 0.45), # bottom center bar
            (1.5, 1.2, 0.5, 0.5),    # small bottom-left block
            (4.7, 6.0, 0.35, 0.35),  # small central block
        ]
    }
}


def draw_map(ax, cfg):
    xmin, xmax, ymin, ymax = cfg["bounds"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(cfg["title"])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)

    # Outer boundary
    boundary = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                         fill=False, linewidth=2.2)
    ax.add_patch(boundary)

    # Obstacles
    for i, (cx, cy, hw, hh) in enumerate(cfg["rects"]):
        rect = Rectangle((cx - hw, cy - hh), 2*hw, 2*hh)
        ax.add_patch(rect)
        # Uncomment if you want labels:
        # ax.text(cx, cy, f"S{i}", ha="center", va="center", fontsize=8, color="white")

    sx, sy = cfg["start"]
    gx, gy = cfg["goal"]
    ax.plot(sx, sy, marker="o", markersize=8, label="start")
    ax.plot(gx, gy, marker="*", markersize=12, label="goal")
    ax.legend(loc="upper right", fontsize=8)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    draw_map(axes[0], MAPS["map_corridor_structured"])
    draw_map(axes[1], MAPS["map_open_clutter"])
    fig.suptitle("Two Thesis Map Candidates Inspired by the Reference Images", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
