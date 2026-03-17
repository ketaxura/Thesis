import numpy as np

class Obstacle:
        def __init__(self, path_id_):
            self.path_id = path_id_

        def static_obs(self):

            STATIC_RECTS = []

            if self.path_id == 3:
                STATIC_RECTS = [
                    (7.0, 1.0, 0.6, 1.2),
                    (10.0, 0.0, 0.5, 0.5),
                ]
                STATIC_RECTS.append((10.0,  5.0, 20.0, 0.2))   # top wall
                STATIC_RECTS.append((10.0, -2.5, 20.0, 0.2))   # bottom wall

            elif self.path_id == 1:
                STATIC_RECTS = [
                    (7.0,  0.0, 0.5, 0.5),
                    (13.0, 0.0, 0.5, 0.5),
                ]
                STATIC_RECTS.append((10.0,  3.5, 20.0, 0.2))   # top wall
                STATIC_RECTS.append((10.0, -3.5, 20.0, 0.2))   # bottom wall

            return STATIC_RECTS
        

        def dynamic_obs(self):
            return self.dynamic_obs_seeded(seed_offset=0)

        def dynamic_obs_seeded(self, seed_offset: int = 0):
            """
            Returns dynamic obstacles with seeds shifted by seed_offset.
            This gives each experimental run a different obstacle trajectory
            while keeping the layout (start positions, sizes) the same.
            """
            dyn_obs = []

            if self.path_id == 3:
                dyn_obs.append(DynamicObstacle(
                    x0=2.0, y0=3.0, a=0.6, b=0.4,
                    seed=0 + seed_offset * 10, v_max=0.5, change_interval=20
                ))
                dyn_obs.append(DynamicObstacle(
                    x0=18.0, y0=2.0, a=0.6, b=0.4,
                    seed=1 + seed_offset * 10, v_max=0.5, change_interval=20
                ))
                dyn_obs.append(DynamicObstacle(
                    x0=14.0, y0=2.0, a=0.6, b=0.4,
                    seed=2 + seed_offset * 10, v_max=0.5, change_interval=20
                ))
                dyn_obs.append(DynamicObstacle(
                    x0=5.5, y0=3.5, a=0.6, b=0.4,
                    seed=3 + seed_offset * 10, v_max=0.0, change_interval=20
                ))
                dyn_obs.append(DynamicObstacle(
                    x0=10.0, y0=-1.0, a=0.6, b=0.4,
                    seed=4 + seed_offset * 10, v_max=0.5, change_interval=20
                ))

            elif self.path_id == 1:
                # Sine wave environment obstacles
                dyn_obs.append(DynamicObstacle(
                    x0=5.0, y0=1.0, a=0.6, b=0.4,
                    seed=0 + seed_offset * 10, v_max=0.5, change_interval=20
                ))
                dyn_obs.append(DynamicObstacle(
                    x0=10.0, y0=-1.0, a=0.6, b=0.4,
                    seed=1 + seed_offset * 10, v_max=0.5, change_interval=20
                ))
                dyn_obs.append(DynamicObstacle(
                    x0=15.0, y0=1.5, a=0.6, b=0.4,
                    seed=2 + seed_offset * 10, v_max=0.5, change_interval=20
                ))

            return dyn_obs
                
        def path_selector(self, path_id):
            if path_id == 0:
                # Straight Diagonal line
                ref_x = np.linspace(0, -10, 100)
                ref_y = np.linspace(0, 0, 100)
                ref_traj = np.vstack((ref_x, ref_y))   # shape (2, 100)

            elif path_id == 1:
                # Sine wave from x=0 to x=20
                t = np.linspace(0, 20, 300)
                ref_x = t
                ref_y = 2.0 * np.sin(0.5 * t)   # amplitude=2, frequency=0.5
                ref_traj = np.vstack((ref_x, ref_y))

            elif path_id == 2:
                # Figure-8 (requires negative x, direction reversal at crossing)
                t = np.linspace(0, 2*np.pi, 400)
                ref_x = 10 * np.sin(t)
                ref_y = 5 * np.sin(2*t)
                ref_traj = np.vstack((ref_x, ref_y))

            elif path_id == 3:
                # Sharp zigzag (tests corner handling)
                ref_x = np.array([0,5,10,15,20], dtype=float)
                ref_y = np.array([0,4,0,4,0], dtype=float)
                ref_x = np.interp(np.linspace(0,4,300), np.arange(5), ref_x)
                ref_y = np.interp(np.linspace(0,4,300), np.arange(5), ref_y)
                ref_traj = np.vstack((ref_x, ref_y))

            elif path_id == 4:
                # Tight spiral (continuously increasing curvature)
                t = np.linspace(0, 4*np.pi, 400)
                ref_x = t * np.cos(t)
                ref_y = t * np.sin(t)
                ref_traj = np.vstack((ref_x, ref_y))

            else:
                raise ValueError(f"Invalid path_id: {path_id}")

            return ref_traj



class DynamicObstacle:

    def __init__(self, x0, y0, a, b, seed=42, v_max=0.5, change_interval=20,
                 x_bounds=(-1.0, 21.0), y_bounds=(-2.3, 4.8),
                 slew_rate=0.05):
        """
        x_bounds, y_bounds : walls the obstacle bounces off
        slew_rate          : max change in vx or vy per step (0-1 scale of v_max)
                             lower = smoother turns, higher = snappier
        change_interval    : steps between random velocity target resamples
        """
        self.x = x0
        self.y = y0
        self.a = a
        self.b = b
        self.v_max = v_max
        self.change_interval = change_interval
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.slew_rate = slew_rate

        self.rng = np.random.default_rng(seed)
        # Current actual velocity
        self.vx, self.vy = self._new_velocity()
        # Target velocity the slew filter is moving toward
        self._vx_target, self._vy_target = self.vx, self.vy
        self._steps_since_change = 0
        # Heading angle — rotates smoothly toward velocity direction
        self.theta = np.arctan2(self.vy, self.vx) if v_max > 0 else 0.0

    def _new_velocity(self):
        vx = self.rng.uniform(-self.v_max, self.v_max)
        vy = self.rng.uniform(-self.v_max, self.v_max)
        return float(vx), float(vy)

    def step(self, dt, static_rects=None):
        """
        Advance one step:
        1. Resample target velocity every change_interval steps
        2. Slew actual velocity toward target
        3. Move
        4. Bounce off y_bounds / x_bounds walls
        5. Bounce off static rectangular obstacles
        """
        # 1. Resample target
        self._steps_since_change += 1
        if self._steps_since_change >= self.change_interval:
            self._vx_target, self._vy_target = self._new_velocity()
            self._steps_since_change = 0

        # 2. Slew — move actual velocity toward target by at most slew_rate * v_max per step
        max_delta = self.slew_rate * self.v_max
        self.vx += np.clip(self._vx_target - self.vx, -max_delta, max_delta)
        self.vy += np.clip(self._vy_target - self.vy, -max_delta, max_delta)

        # Update heading — rotate toward current velocity direction
        speed = np.hypot(self.vx, self.vy)
        if speed > 1e-3:
            target_theta = np.arctan2(self.vy, self.vx)
            # Wrap difference to [-pi, pi] then slew
            dtheta = (target_theta - self.theta + np.pi) % (2 * np.pi) - np.pi
            self.theta += np.clip(dtheta, -0.3, 0.3)  # max ~17 deg/step

        # 3. Move
        self.x += self.vx * dt
        self.y += self.vy * dt

        # 4. Bounce off world walls
        x_lo, x_hi = self.x_bounds
        y_lo, y_hi = self.y_bounds

        if self.x - self.a < x_lo:
            self.x = x_lo + self.a
            self.vx = abs(self.vx)
            self._vx_target = abs(self._vx_target)

        if self.x + self.a > x_hi:
            self.x = x_hi - self.a
            self.vx = -abs(self.vx)
            self._vx_target = -abs(self._vx_target)

        if self.y - self.b < y_lo:
            self.y = y_lo + self.b
            self.vy = abs(self.vy)
            self._vy_target = abs(self._vy_target)

        if self.y + self.b > y_hi:
            self.y = y_hi - self.b
            self.vy = -abs(self.vy)
            self._vy_target = -abs(self._vy_target)

        # 5. Bounce off static rectangular obstacles
        if static_rects:
            for (cx, cy, hw, hh) in static_rects:
                # Expand rect by obstacle half-axes for centre-point collision
                ex = hw + self.a
                ey = hh + self.b
                dx = self.x - cx
                dy = self.y - cy

                if abs(dx) < ex and abs(dy) < ey:
                    # Push out on the axis of least penetration and flip velocity
                    pen_x = ex - abs(dx)
                    pen_y = ey - abs(dy)
                    if pen_x < pen_y:
                        self.x += np.sign(dx) * pen_x
                        self.vx = np.sign(dx) * abs(self.vx)
                        self._vx_target = np.sign(dx) * abs(self._vx_target)
                    else:
                        self.y += np.sign(dy) * pen_y
                        self.vy = np.sign(dy) * abs(self.vy)
                        self._vy_target = np.sign(dy) * abs(self._vy_target)

    def predict(self, k, dt):
        """Predict k steps ahead assuming constant current velocity."""
        x = self.x + self.vx * dt * k
        y = self.y + self.vy * dt * k
        return x, y

    def predict_horizon(self, N, dt):
        """
        Return obs_x (N,), obs_y (N,), obs_theta (N,) for horizon steps 1..N.
        Position uses constant-velocity extrapolation.
        Angle is held constant at current theta (orientation changes slowly
        relative to dt, so this is a reasonable short-horizon approximation).
        """
        obs_x = np.array([self.x + self.vx * dt * k for k in range(1, N + 1)])
        obs_y = np.array([self.y + self.vy * dt * k for k in range(1, N + 1)])
        obs_theta = np.full(N, self.theta)
        return obs_x, obs_y, obs_theta