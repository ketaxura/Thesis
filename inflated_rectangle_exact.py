import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

cx, cy = 0.0, 0.0
hw, hh = 2.5, 1.5
margin = 0.8

x = np.linspace(-5, 5, 600)
y = np.linspace(-4, 4, 500)
X, Y = np.meshgrid(x, y)

QX = np.maximum(np.abs(X - cx) - hw, 0.0)
QY = np.maximum(np.abs(Y - cy) - hh, 0.0)
Z = np.sqrt(QX**2 + QY**2)

fig, ax = plt.subplots(figsize=(8, 6))

rect = Rectangle((cx - hw, cy - hh), 2*hw, 2*hh, fill=False, linewidth=2)
ax.add_patch(rect)

ax.contour(X, Y, Z, levels=[margin], linewidths=2)

corners = [
    (cx - hw, cy - hh),
    (cx - hw, cy + hh),
    (cx + hw, cy - hh),
    (cx + hw, cy + hh),
]
for (px, py) in corners:
    ax.add_patch(Circle((px, py), margin, fill=False, linestyle='--', linewidth=1.4))

ax.plot([cx - hw, cx + hw], [cy + hh + margin, cy + hh + margin], linestyle='--', linewidth=1.2)
ax.plot([cx - hw, cx + hw], [cy - hh - margin, cy - hh - margin], linestyle='--', linewidth=1.2)
ax.plot([cx - hw - margin, cx - hw - margin], [cy - hh, cy + hh], linestyle='--', linewidth=1.2)
ax.plot([cx + hw + margin, cx + hw + margin], [cy - hh, cy + hh], linestyle='--', linewidth=1.2)

ax.set_aspect('equal')
ax.set_xlim(-5, 5)
ax.set_ylim(-4, 4)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Rectangle and its exact inflated boundary (rectangle ⊕ disk)")
ax.grid(True)
plt.show()
