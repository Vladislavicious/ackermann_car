from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from ackermann_car.sim.road import Road, Waypoint

# (x_ref, y_ref, heading_ref, speed_ref)
RefPoint = Tuple[float, float, float, float]


class ReferencePath:

    def __init__(
        self,
        road: Road | None = None,
        ref_speed: float = 2.5,
        lookahead_step: int = 3,
    ) -> None:
        self._road = road or Road()
        self.ref_speed = ref_speed
        self.lookahead_step = lookahead_step

    def get_reference(
        self,
        x: float,
        y: float,
        yaw: float,
        horizon: int = 10,
    ) -> Tuple[float, float, List[RefPoint]]:

        lat_err = self._road.lateral_error(x, y)
        heading_err = self._road.heading_error(x, y, yaw)

        lookahead = self._road.lookahead_waypoints(
            x, y, n=horizon, step=self.lookahead_step
        )
        ref_points: List[RefPoint] = [
            (wx, wy, wh, self.ref_speed) for (wx, wy, wh) in lookahead
        ]
        # Pad if we hit the end of the road
        while len(ref_points) < horizon:
            ref_points.append(ref_points[-1])

        return lat_err, heading_err, ref_points

    def nearest_curvature(self, x: float, y: float) -> float:

        idx = self._road.nearest_waypoint_index(x, y)
        total = len(self._road.waypoints)
        # Use finite differences of heading angle
        i0 = max(0, idx - 2)
        i1 = min(total - 1, idx + 2)
        h0 = self._road.waypoints[i0][2]
        h1 = self._road.waypoints[i1][2]
        # Arc between i0 and i1
        coords = np.array([(w[0], w[1]) for w in self._road.waypoints[i0 : i1 + 1]])
        arc = float(np.sum(np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1]))))
        if arc < 1e-6:
            return 0.0
        delta_h = (h1 - h0 + math.pi) % (2 * math.pi) - math.pi
        return float(delta_h / arc)

    def speed_reference(
        self, x: float, y: float, base_speed: float | None = None
    ) -> float:

        if base_speed is None:
            base_speed = self.ref_speed
        kappa = abs(self.nearest_curvature(x, y))
        # speed ~ 1/sqrt(1 + C * kappa)
        C = 20.0
        return float(base_speed / math.sqrt(1.0 + C * kappa))

    def xy_array(self) -> np.ndarray:
        """Return Nx2 centre-line for plotting."""
        return self._road.xy_array()
