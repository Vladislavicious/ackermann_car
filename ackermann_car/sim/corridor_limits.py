from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .road import Road


class CorridorLimits:

    def __init__(self, road: Road, half_width: float = 1.75) -> None:
        self._road = road
        self.half_width = half_width

    def boundary_points(self, step: int = 1) -> Tuple[np.ndarray, np.ndarray]:

        waypoints = self._road.waypoints[::step]
        left_pts = []
        right_pts = []
        for x, y, heading in waypoints:
            # Normal vectors (perpendicular to heading)
            left_nx = -math.sin(heading)
            left_ny = math.cos(heading)
            right_nx = math.sin(heading)
            right_ny = -math.cos(heading)
            left_pts.append(
                (
                    x + self.half_width * left_nx,
                    y + self.half_width * left_ny,
                )
            )
            right_pts.append(
                (
                    x + self.half_width * right_nx,
                    y + self.half_width * right_ny,
                )
            )
        return np.array(left_pts), np.array(right_pts)

    def is_inside(self, x: float, y: float) -> bool:

        lat_err = abs(self._road.lateral_error(x, y))
        return lat_err <= self.half_width

    def distance_to_boundary(self, x: float, y: float) -> float:

        lat_err = abs(self._road.lateral_error(x, y))
        return self.half_width - lat_err

    def corridor_violation_penalty(self, x: float, y: float) -> float:

        overshoot = -self.distance_to_boundary(x, y)
        if overshoot <= 0:
            return 0.0
        return float(overshoot**2) * 100.0  # large coefficient → hard constraint


if __name__ == "__main__":
    road = Road()
    corridor = CorridorLimits(road, half_width=1.75)
    left, right = corridor.boundary_points()
    print(f"Left boundary shape : {left.shape}")
    print(f"Right boundary shape: {right.shape}")
    print(f"Is (0, 0) inside?   : {corridor.is_inside(0.0, 0.0)}")
    print(f"Is (0, 5) inside?   : {corridor.is_inside(0.0, 5.0)}")
    print(f"Distance at (0, 1.5): {corridor.distance_to_boundary(0.0, 1.5):.3f} m")
