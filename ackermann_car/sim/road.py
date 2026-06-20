from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

Waypoint = Tuple[float, float, float]


def _bezier_cubic(p0, p1, p2, p3, t: float) -> np.ndarray:

    t = np.clip(t, 0.0, 1.0)
    u = 1.0 - t
    return u**3 * p0 + 3 * u**2 * t * p1 + 3 * u * t**2 * p2 + t**3 * p3


def _bezier_cubic_derivative(p0, p1, p2, p3, t: float) -> np.ndarray:

    t = np.clip(t, 0.0, 1.0)
    u = 1.0 - t
    return 3 * u**2 * (p1 - p0) + 6 * u * t * (p2 - p1) + 3 * t**2 * (p3 - p2)


class Road:

    def __init__(
        self,
        control_points: List[Tuple[float, float]] | None = None,
        samples_per_segment: int = 50,
    ) -> None:
        if control_points is None:
            control_points = _default_track()
        self._ctrl = [np.array(p, dtype=float) for p in control_points]
        self._sps = samples_per_segment
        self._waypoints: List[Waypoint] = []
        self._arc_lengths: np.ndarray = np.array([])
        self._build()

    def _build(self) -> None:

        pts = self._ctrl
        n = len(pts)
        if n < 4:
            raise ValueError("Need at least 4 control points for a cubic Bézier.")

        waypoints: List[Waypoint] = []

        seg_starts = list(range(0, n - 1, 3))

        for s in seg_starts:
            if s + 3 >= n:
                break
            p0, p1, p2, p3 = pts[s], pts[s + 1], pts[s + 2], pts[s + 3]
            for i in range(self._sps):
                t = i / self._sps
                xy = _bezier_cubic(p0, p1, p2, p3, t)
                dxy = _bezier_cubic_derivative(p0, p1, p2, p3, t)
                heading = math.atan2(dxy[1], dxy[0])
                waypoints.append((float(xy[0]), float(xy[1]), float(heading)))

        p0, p1, p2, p3 = pts[-4], pts[-3], pts[-2], pts[-1]
        xy = _bezier_cubic(p0, p1, p2, p3, 1.0)
        dxy = _bezier_cubic_derivative(p0, p1, p2, p3, 1.0)
        waypoints.append(
            (float(xy[0]), float(xy[1]), float(math.atan2(dxy[1], dxy[0])))
        )

        self._waypoints = waypoints

        coords = np.array([(w[0], w[1]) for w in waypoints])
        diffs = np.diff(coords, axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        self._arc_lengths = np.concatenate([[0.0], np.cumsum(seg_lens)])

    @property
    def waypoints(self) -> List[Waypoint]:

        return self._waypoints

    @property
    def total_length(self) -> float:

        return float(self._arc_lengths[-1])

    def nearest_waypoint_index(self, x: float, y: float) -> int:

        coords = np.array([(w[0], w[1]) for w in self._waypoints])
        dists = np.hypot(coords[:, 0] - x, coords[:, 1] - y)
        return int(np.argmin(dists))

    def lateral_error(self, x: float, y: float) -> float:

        idx = self.nearest_waypoint_index(x, y)
        wx, wy, wh = self._waypoints[idx]
        dx, dy = x - wx, y - wy
        nx, ny = -math.sin(wh), math.cos(wh)
        return float(dx * nx + dy * ny)

    def heading_error(self, x: float, y: float, car_yaw: float) -> float:

        idx = self.nearest_waypoint_index(x, y)
        _, _, road_heading = self._waypoints[idx]
        err = car_yaw - road_heading
        return float((err + math.pi) % (2 * math.pi) - math.pi)

    def lookahead_waypoints(
        self, x: float, y: float, n: int, step: int = 3
    ) -> List[Waypoint]:

        idx = self.nearest_waypoint_index(x, y)
        result = []
        total = len(self._waypoints)
        for k in range(1, n + 1):
            future_idx = min(idx + k * step, total - 1)
            result.append(self._waypoints[future_idx])
        return result

    def xy_array(self) -> np.ndarray:

        return np.array([(w[0], w[1]) for w in self._waypoints])


def _default_track() -> List[Tuple[float, float]]:

    return [
        (0.0, 0.0),  # anchor 0
        (4.0, 0.0),  # cp
        (8.0, 3.0),  # cp
        (12.0, 3.0),  # anchor 1
        (16.0, 3.0),  # cp
        (20.0, 0.0),  # cp
        (24.0, 0.0),  # anchor2
        (28.0, 0.0),  # cp
        (32.0, -3.0),  # cp
        (36.0, -3.0),  # anchor3
        (40.0, -3.0),  # cp
        (44.0, 0.0),  # cp
        (48.0, 0.0),  # anchor 4
        (52.0, 0.0),  # cp
        (56.0, 3.0),  # cp
        (60.0, 3.0),  # anchor5
        (64.0, 3.0),  # cp
        (68.0, 0.0),  # cp
        (72.0, 0.0),  # anchor6
    ]


if __name__ == "__main__":
    road = Road()
    print(f"Total waypoints : {len(road.waypoints)}")
    print(f"Total arc length: {road.total_length:.2f} m")
    idx = road.nearest_waypoint_index(0.0, 0.5)
    lat_err = road.lateral_error(0.0, 0.5)
    print(f"Nearest idx at (0, 0.5): {idx}, lateral_error={lat_err:.4f} m")
    ahead = road.lookahead_waypoints(0.0, 0.0, n=5, step=3)
    print(f"Lookahead waypoints: {ahead}")
