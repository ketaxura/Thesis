import numpy as np
from pathlib import Path
from map_inspired_with_astar import MAPS

def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi



THIS_DIR = Path(__file__).resolve().parent

ASTAR_PATH_FILES = {
    10: "map_corridor_structured_ref_path.csv",
    11: "map_open_clutter_ref_path.csv",
}

ASTAR_MAP_KEYS = {
    10: "map_corridor_structured",
    11: "map_open_clutter",
}

class Obstacle:
    """
    Environment wrapper used by the unified experiment runner.

    Keeps the old API:
        - path_selector(path_id)
        - static_obs()
        - dynamic_obs()
        - dynamic_obs_seeded(seed_offset)

    but now returns the new bounded-slew DynamicObstacle objects.
    """

    def __init__(self, path_id_):
        self.path_id = path_id_

    def static_obs(self):
        if self.path_id in ASTAR_MAP_KEYS:
            return MAPS[ASTAR_MAP_KEYS[self.path_id]]["rects"]

        STATIC_RECTS = []

        if self.path_id == 3:
            STATIC_RECTS = [
                (7.0, 1.0, 0.6, 1.2),
                (10.0, 0.0, 0.5, 0.5),
            ]
            STATIC_RECTS.append((10.0, 5.0, 20.0, 0.2))    # top wall
            STATIC_RECTS.append((10.0, -2.5, 20.0, 0.2))   # bottom wall

        elif self.path_id == 1:
            STATIC_RECTS = [
                (7.0, 0.0, 0.5, 0.5),
                (13.0, 0.0, 0.5, 0.5),
            ]
            STATIC_RECTS.append((10.0, 3.5, 20.0, 0.2))    # top wall
            STATIC_RECTS.append((10.0, -3.5, 20.0, 0.2))   # bottom wall

        return STATIC_RECTS

    def dynamic_obs(self):
        return self.dynamic_obs_seeded(seed_offset=0)

    def dynamic_obs_seeded(self, seed_offset: int = 0):
        """
        Returns dynamic obstacles with seeds shifted by seed_offset.
        For A*-map debug runs, return [] so we can test static map + ref path first.
        """
        if self.path_id in ASTAR_MAP_KEYS:
            return []

        dyn_obs = []

        if self.path_id == 3:
            dyn_obs.append(DynamicObstacle(
                x=2.0, y=3.0, a=0.6, b=0.4,
                seed=0 + seed_offset * 10, v_max=0.5, change_interval=20
            ))
            dyn_obs.append(DynamicObstacle(
                x=18.0, y=2.0, a=0.6, b=0.4,
                seed=1 + seed_offset * 10, v_max=0.5, change_interval=20
            ))
            dyn_obs.append(DynamicObstacle(
                x=14.0, y=2.0, a=0.6, b=0.4,
                seed=2 + seed_offset * 10, v_max=0.5, change_interval=20
            ))
            dyn_obs.append(DynamicObstacle(
                x=5.5, y=3.5, a=0.6, b=0.4,
                seed=3 + seed_offset * 10, v_max=0.0, change_interval=20
            ))
            dyn_obs.append(DynamicObstacle(
                x=10.0, y=-1.0, a=0.6, b=0.4,
                seed=4 + seed_offset * 10, v_max=0.5, change_interval=20
            ))

        elif self.path_id == 1:
            dyn_obs.append(DynamicObstacle(
                x=5.0, y=1.0, a=0.6, b=0.4,
                seed=0 + seed_offset * 10, v_max=0.5, change_interval=20
            ))
            dyn_obs.append(DynamicObstacle(
                x=10.0, y=-1.0, a=0.6, b=0.4,
                seed=1 + seed_offset * 10, v_max=0.5, change_interval=20
            ))
            dyn_obs.append(DynamicObstacle(
                x=15.0, y=1.5, a=0.6, b=0.4,
                seed=2 + seed_offset * 10, v_max=0.5, change_interval=20
            ))

        return dyn_obs

    def path_selector(self, path_id):
        if path_id in ASTAR_PATH_FILES:
            csv_path = THIS_DIR / ASTAR_PATH_FILES[path_id]
            path_xy = np.loadtxt(csv_path, delimiter=",", skiprows=1)

            if path_xy.ndim != 2 or path_xy.shape[1] != 2:
                raise ValueError(f"Expected Nx2 CSV in {csv_path}, got shape {path_xy.shape}")

            return path_xy.T

        if path_id == 0:
            ref_x = np.linspace(0, -10, 100)
            ref_y = np.linspace(0, 0, 100)
            ref_traj = np.vstack((ref_x, ref_y))

        elif path_id == 1:
            t = np.linspace(0, 20, 300)
            ref_x = t
            ref_y = 2.0 * np.sin(0.5 * t)
            ref_traj = np.vstack((ref_x, ref_y))

        elif path_id == 2:
            t = np.linspace(0, 2 * np.pi, 400)
            ref_x = 10 * np.sin(t)
            ref_y = 5 * np.sin(2 * t)
            ref_traj = np.vstack((ref_x, ref_y))

        elif path_id == 3:
            ref_x = np.array([0, 5, 10, 15, 20], dtype=float)
            ref_y = np.array([0, 4, 0, 4, 0], dtype=float)
            ref_x = np.interp(np.linspace(0, 4, 300), np.arange(5), ref_x)
            ref_y = np.interp(np.linspace(0, 4, 300), np.arange(5), ref_y)
            ref_traj = np.vstack((ref_x, ref_y))

        elif path_id == 4:
            t = np.linspace(0, 4 * np.pi, 400)
            ref_x = t * np.cos(t)
            ref_y = t * np.sin(t)
            ref_traj = np.vstack((ref_x, ref_y))

        else:
            raise ValueError(f"Invalid path_id: {path_id}")

        return ref_traj


class DynamicObstacle:
    """
    Dynamic elliptical obstacle with:
      - bounded heading slew
      - bounded speed slew
      - optional random desired-heading updates at fixed intervals
      - rectangle bounce behavior
      - horizon prediction compatible with your runner

    Compatibility notes:
      - accepts x/y OR x0/y0
      - keeps seed/change_interval API from your older obstacle generator
      - preserves fields x, y, theta, a, b used elsewhere in the framework
    """

    def __init__(
        self,
        x: float | None = None,
        y: float | None = None,
        a: float = 0.6,
        b: float = 0.4,
        vx: float = 0.0,
        vy: float = 0.0,
        theta: float | None = None,
        v_nominal: float | None = None,
        v_max: float = 0.9,
        a_obs: float = 1.2,
        omega_obs: float = 1.8,
        seed: int | None = None,
        change_interval: int = 20,
        x0: float | None = None,
        y0: float | None = None,
    ):
        # Backward compatibility for old constructor style
        if x is None:
            x = x0
        if y is None:
            y = y0
        if x is None or y is None:
            raise ValueError("DynamicObstacle requires x/y or x0/y0.")

        self.x = float(x)
        self.y = float(y)

        self.a = float(a)
        self.b = float(b)

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.change_interval = int(change_interval)
        self.step_count = 0

        speed0 = float(np.hypot(vx, vy))
        if theta is None:
            if speed0 > 1e-9:
                theta = float(np.arctan2(vy, vx))
            else:
                theta = float(self.rng.uniform(-np.pi, np.pi)) if seed is not None else 0.0

        self.theta = float(theta)

        if v_nominal is None:
            if speed0 > 1e-9:
                v_nominal = speed0
            else:
                v_nominal = float(v_max)

        self.v = float(np.clip(v_nominal, 0.0, v_max))
        self.v_nominal = float(np.clip(v_nominal, 0.0, v_max))

        self.v_max = float(v_max)
        self.a_obs = float(a_obs)
        self.omega_obs = float(omega_obs)

        self.v_des = float(self.v_nominal)
        self.theta_des = float(self.theta)

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    def _maybe_refresh_targets(self):
        """
        Randomly refresh desired heading at intervals.
        Speed target stays at v_nominal unless v_max == 0.
        """
        if self.v_max <= 1e-12:
            self.v_des = 0.0
            return

        if self.change_interval > 0 and (self.step_count % self.change_interval == 0):
            self.theta_des = float(self.rng.uniform(-np.pi, np.pi))
            self.v_des = float(self.v_nominal)

    def _slew_scalar(self, current: float, target: float, max_delta: float) -> float:
        delta = target - current
        delta = np.clip(delta, -max_delta, max_delta)
        return current + delta

    def _slew_angle(self, current: float, target: float, max_delta: float) -> float:
        err = wrap_angle(target - current)
        err = np.clip(err, -max_delta, max_delta)
        return wrap_angle(current + err)

    def _rect_hit(self, x_next: float, y_next: float, static_rects):
        """
        Check whether obstacle center enters a rectangle inflated by obstacle size.
        Returns:
            hit, nx, ny
        where (nx, ny) is a simple outward normal for reflection.
        """
        pad = max(self.a, self.b)

        for (cx, cy, hw, hh) in static_rects:
            dx = x_next - cx
            dy = y_next - cy
            lim_x = hw + pad
            lim_y = hh + pad

            if abs(dx) <= lim_x and abs(dy) <= lim_y:
                pen_x = lim_x - abs(dx)
                pen_y = lim_y - abs(dy)

                if pen_x < pen_y:
                    nx = 1.0 if dx >= 0.0 else -1.0
                    ny = 0.0
                else:
                    nx = 0.0
                    ny = 1.0 if dy >= 0.0 else -1.0

                return True, nx, ny

        return False, 0.0, 0.0

    def _reflected_heading(self, theta_in: float, nx: float, ny: float) -> float:
        vx = np.cos(theta_in)
        vy = np.sin(theta_in)

        dot = vx * nx + vy * ny
        rx = vx - 2.0 * dot * nx
        ry = vy - 2.0 * dot * ny

        return float(np.arctan2(ry, rx))

    def _advance_once(self, dt: float, static_rects) -> None:
        max_dv = self.a_obs * dt
        max_dtheta = self.omega_obs * dt

        self._maybe_refresh_targets()

        self.v = self._slew_scalar(self.v, self.v_des, max_dv)
        self.v = float(np.clip(self.v, 0.0, self.v_max))

        self.theta = self._slew_angle(self.theta, self.theta_des, max_dtheta)

        x_next = self.x + dt * self.v * np.cos(self.theta)
        y_next = self.y + dt * self.v * np.sin(self.theta)

        hit, nx, ny = self._rect_hit(x_next, y_next, static_rects)

        if hit:
            self.theta_des = self._reflected_heading(self.theta_des, nx, ny)
            self.theta = self._slew_angle(self.theta, self.theta_des, max_dtheta)

            x_next = self.x + dt * self.v * np.cos(self.theta)
            y_next = self.y + dt * self.v * np.sin(self.theta)

            hit2, _, _ = self._rect_hit(x_next, y_next, static_rects)
            if hit2:
                self.step_count += 1
                return

        self.x = float(x_next)
        self.y = float(y_next)
        self.step_count += 1

    # --------------------------------------------------------
    # Public interface used by runners
    # --------------------------------------------------------

    def step(self, dt: float, static_rects) -> None:
        self._advance_once(dt, static_rects)

    def predict_horizon(self, N: int, dt: float):
        """
        Smooth bounded-slew prediction without rectangle bounce.
        Kept for backward compatibility.
        """
        x = float(self.x)
        y = float(self.y)
        theta = float(self.theta)
        v = float(self.v)
        theta_des = float(self.theta_des)
        v_des = float(self.v_des)

        max_dv = self.a_obs * dt
        max_dtheta = self.omega_obs * dt

        xs = np.zeros(N)
        ys = np.zeros(N)
        thetas = np.zeros(N)

        for k in range(N):
            dv = np.clip(v_des - v, -max_dv, max_dv)
            v = float(np.clip(v + dv, 0.0, self.v_max))

            err = wrap_angle(theta_des - theta)
            err = np.clip(err, -max_dtheta, max_dtheta)
            theta = wrap_angle(theta + err)

            x = x + dt * v * np.cos(theta)
            y = y + dt * v * np.sin(theta)

            xs[k] = x
            ys[k] = y
            thetas[k] = theta

        return xs, ys, thetas

    def predict_horizon_with_rects(self, N: int, dt: float, static_rects):
        """
        Bounce-aware bounded-slew prediction.
        This is the one your unified runner should use.
        """
        x = float(self.x)
        y = float(self.y)
        theta = float(self.theta)
        v = float(self.v)
        theta_des = float(self.theta_des)
        v_des = float(self.v_des)

        max_dv = self.a_obs * dt
        max_dtheta = self.omega_obs * dt

        xs = np.zeros(N)
        ys = np.zeros(N)
        thetas = np.zeros(N)

        for k in range(N):
            dv = np.clip(v_des - v, -max_dv, max_dv)
            v = float(np.clip(v + dv, 0.0, self.v_max))

            err = wrap_angle(theta_des - theta)
            err = np.clip(err, -max_dtheta, max_dtheta)
            theta = wrap_angle(theta + err)

            x_next = x + dt * v * np.cos(theta)
            y_next = y + dt * v * np.sin(theta)

            hit, nx, ny = self._rect_hit(x_next, y_next, static_rects)
            if hit:
                theta_des = self._reflected_heading(theta_des, nx, ny)

                err = wrap_angle(theta_des - theta)
                err = np.clip(err, -max_dtheta, max_dtheta)
                theta = wrap_angle(theta + err)

                x_next = x + dt * v * np.cos(theta)
                y_next = y + dt * v * np.sin(theta)

                hit2, _, _ = self._rect_hit(x_next, y_next, static_rects)
                if hit2:
                    x_next, y_next = x, y

            x = x_next
            y = y_next

            xs[k] = x
            ys[k] = y
            thetas[k] = theta

        return xs, ys, thetas