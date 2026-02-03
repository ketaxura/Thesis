import numpy as np

def generate_s_curve_ref(
    x_start=0.0,
    x_end=10.0,
    num_points=501,
    amplitude=2.0,
    frequency=0.6
):
    """
    Generates an S-shaped reference trajectory.
    Start: (x_start, 0)
    Goal:  (x_end, 0)
    """

    x = np.linspace(x_start, x_end, num_points)
    y = amplitude * np.sin(frequency * x)

    ref_traj = np.vstack([x, y])  # shape (2, T)
    return ref_traj
