import numpy as np
import matplotlib.pyplot as plt

# Parameters
rho = 1.0  # minimum turning radius

# Angle range
phi = np.linspace(0, 2*np.pi, 500)

# Circle center (robot starts at origin, turning left)
cx, cy = 0, rho

# Circle points
circle_x = cx + rho * np.sin(phi)
circle_y = cy - rho * np.cos(phi)

# Plot
plt.figure(figsize=(8, 6))

# Original path (x-axis)
x_path = np.linspace(-2, 4, 100)
y_path = np.zeros_like(x_path)
plt.plot(x_path, y_path, '--', label='Original Path (y=0)')

# Circle
plt.plot(circle_x, circle_y, label='Turning Circle (radius = rho)')

# Trajectory (same as circle from phi=0 to pi)
phi_traj = np.linspace(0, np.pi, 200)
traj_x = rho * np.sin(phi_traj)
traj_y = rho * (1 - np.cos(phi_traj))
plt.plot(traj_x, traj_y, 'r', linewidth=2, label='Robot Trajectory')

# Quarter circle point (phi = pi/2)
phi_q = np.pi / 2
x_q = rho * np.sin(phi_q)
y_q = rho * (1 - np.cos(phi_q))
plt.scatter(x_q, y_q, color='green', s=100, label='Quarter Circle (π/2)')
plt.text(x_q, y_q + 0.1, f'y = {y_q:.2f}')

# Half circle point (phi = pi)
phi_h = np.pi
x_h = rho * np.sin(phi_h)
y_h = rho * (1 - np.cos(phi_h))
plt.scatter(x_h, y_h, color='purple', s=100, label='Half Circle (π)')
plt.text(x_h, y_h + 0.1, f'y = {y_h:.2f}')

# Start point
plt.scatter(0, 0, color='black', s=100, label='Start')

# Formatting
plt.axhline(0)
plt.axvline(0)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Lateral Offset from Circular Motion')
plt.xlabel('x')
plt.ylabel('y (lateral offset)')
plt.legend()
plt.grid()

plt.show()