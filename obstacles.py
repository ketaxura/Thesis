import numpy as np

class ObstacleManager:
    def __init__(
        self,
        num_dyn_obs,
        dt,
        N,
        rng,
        v_max,
        a_max,
        omega_max,
        init_positions,
        wall_y_min,
        wall_y_max,
        y_margin=0.0,
    ):
        self.num_dyn_obs = num_dyn_obs
        self.dt = dt
        self.N = N

        self.v_max = v_max
        self.a_max = a_max
        self.omega_max = omega_max

        self.wall_y_min = float(wall_y_min)
        self.wall_y_max = float(wall_y_max)
        self.y_margin = float(y_margin)

        self.rng = rng

        self.pos = init_positions.copy()
        self.theta = rng.uniform(0, 2*np.pi, size=num_dyn_obs)
        self.speed = np.zeros(num_dyn_obs)
        self.vel = np.zeros((num_dyn_obs, 2))

        # keep initial positions inside corridor
        y_lo = self.wall_y_min + self.y_margin
        y_hi = self.wall_y_max - self.y_margin
        self.pos[:, 1] = np.clip(self.pos[:, 1], y_lo, y_hi)

    def _reflect_y_inplace(self, i: int):
        """Reflect obstacle i off the corridor walls in y, updating pos/vel/theta."""
        y_lo = self.wall_y_min + self.y_margin
        y_hi = self.wall_y_max - self.y_margin

        # Allow for the case where dt is big enough to cross more than once:
        for _ in range(2):
            y = self.pos[i, 1]
            if y < y_lo:
                self.pos[i, 1] = y_lo + (y_lo - y)   # mirror back inside
                self.vel[i, 1] *= -1.0              # reflect vy
            elif y > y_hi:
                self.pos[i, 1] = y_hi - (y - y_hi)
                self.vel[i, 1] *= -1.0
            else:
                break

        vx, vy = self.vel[i]
        if abs(vx) + abs(vy) > 0:
            self.theta[i] = np.arctan2(vy, vx)

    def step(self):
        """Propagate true obstacle states by one dt, reflecting off corridor walls."""
        for i in range(self.num_dyn_obs):
            v_des = self.rng.uniform(0.2, self.v_max)
            theta_des = self.theta[i] + self.rng.uniform(
                -self.omega_max * self.dt,
                 self.omega_max * self.dt
            )

            dv = np.clip(v_des - self.speed[i],
                         -self.a_max * self.dt,
                          self.a_max * self.dt)
            self.speed[i] += dv

            dtheta = np.clip(theta_des - self.theta[i],
                             -self.omega_max * self.dt,
                              self.omega_max * self.dt)
            self.theta[i] += dtheta

            self.vel[i] = self.speed[i] * np.array([
                np.cos(self.theta[i]),
                np.sin(self.theta[i])
            ])

            # integrate
            self.pos[i] += self.vel[i] * self.dt

            # reflect off walls
            self._reflect_y_inplace(i)

    def predict_horizon(self):
        """Return obsx_h, obsy_h shaped (num_dyn_obs, N), with wall reflections."""
        obsx_h = np.zeros((self.num_dyn_obs, self.N))
        obsy_h = np.zeros((self.num_dyn_obs, self.N))

        y_lo = self.wall_y_min + self.y_margin
        y_hi = self.wall_y_max - self.y_margin

        # dynamic: constant-velocity + reflection against walls
        for i in range(self.num_dyn_obs):
            x, y = self.pos[i]
            vx, vy = self.vel[i]

            for k in range(self.N):
                x_next = x + vx * self.dt
                y_next = y + vy * self.dt

                # reflect predicted step if it crosses walls
                if y_next < y_lo:
                    y_next = y_lo + (y_lo - y_next)
                    vy = -vy
                elif y_next > y_hi:
                    y_next = y_hi - (y_next - y_hi)
                    vy = -vy

                x, y = x_next, y_next
                obsx_h[i, k] = x
                obsy_h[i, k] = y

        return obsx_h, obsy_h
