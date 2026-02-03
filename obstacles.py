import numpy as np

def generate_dynamic_obstacles(x0, y0, v_max, dt, N, seed, accel_std=0.2):
    rng = np.random.default_rng(seed)

    x = np.zeros(N + 1)
    y = np.zeros(N + 1)

    x[0] = x0
    y[0] = y0

    # rng.uniform(a, b): sample uniformly in [a, b)
    vx = rng.uniform(-v_max, v_max)
    vy = rng.uniform(-v_max, v_max)

    for k in range(N):
        # random acceleration / velocity perturbation
        vx += rng.normal(0.0, accel_std)
        vy += rng.normal(0.0, accel_std)

        vx = np.clip(vx, -v_max, v_max)
        vy = np.clip(vy, -v_max, v_max)

        x[k + 1] = x[k] + vx * dt
        y[k + 1] = y[k] + vy * dt

    return x, y
