import matplotlib.pyplot as plt
immport pandas as pd


fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, reverse_state in zip(axes, ["Enabled", "Disabled"]):
    sub = success_df[success_df["reverse"] == reverse_state].sort_values("dyn_speed")
    for ctrl in controllers:
        ax.plot(sub["dyn_speed"], sub[ctrl], marker="o", linewidth=2, label=labels[ctrl])
    ax.set_title(f"Reverse {reverse_state}")
    ax.set_xlabel("Dynamic obstacle speed")
    ax.set_xticks(sorted(sub["dyn_speed"].unique()))
    ax.set_ylim(40, 100)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Success rate (%)")

handles, legend_labels = axes[0].get_legend_handles_labels()

fig.suptitle("Success Rate vs Dynamic Obstacle Speed", y=0.98, fontsize=14)

fig.legend(
    handles,
    legend_labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=4,
    frameon=True
)

plt.tight_layout(rect=[0, 0, 1, 0.88])