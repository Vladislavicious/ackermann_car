

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from ackermann_car.control.vehicle_model import (
    KinematicBicycleModel, VehicleControl, VehicleParams, VehicleState,
)
from ackermann_car.control.reference_path import ReferencePath



@dataclass
class MPCParams:
    horizon       : int   = 10      # N prediction steps
    dt            : float = 0.1     # timestep per step [s]

    # Cost weights
    w_lat         : float = 8.0     # lateral error
    w_heading     : float = 5.0     # heading error
    w_speed       : float = 2.0     # speed tracking
    w_steer       : float = 0.5     # steering effort
    w_steer_rate  : float = 1.0     # steering smoothness (δ_{k+1} - δ_k)
    w_accel       : float = 0.2     # acceleration effort

    # Vehicle limits (must match VehicleParams)
    max_steer     : float = 0.5     # [rad]
    max_accel     : float = 2.0     # [m/s²]
    min_accel     : float = -3.0
    max_speed     : float = 5.0
    min_speed     : float = 0.0



class MPCController:


    def __init__(
        self,
        mpc_params     : MPCParams      | None = None,
        vehicle_params : VehicleParams  | None = None,
        ref_path       : ReferencePath  | None = None,
    ) -> None:
        self.mp   = mpc_params      or MPCParams()
        self.vp   = vehicle_params  or VehicleParams()
        self.rp   = ref_path        or ReferencePath()
        self._model = KinematicBicycleModel(self.vp)

        self._u_prev: np.ndarray = np.zeros(2 * self.mp.horizon)
        self._prev_steer: float = 0.0

    def _cost(
        self,
        u_flat    : np.ndarray,
        state     : VehicleState,
        ref_points: list,
    ) -> float:

        N   = self.mp.horizon
        mp  = self.mp
        dt  = mp.dt

        total = 0.0
        s     = state
        prev_delta = self._prev_steer

        for k in range(N):
            a_k     = float(u_flat[2 * k])
            delta_k = float(u_flat[2 * k + 1])

            ctrl = VehicleControl(acceleration=a_k, steering=delta_k)
            s    = self._model.step(s, ctrl, dt=dt)

            rx, ry, rh, rv = ref_points[k]

            # Project vehicle onto road: signed cross-track error
            dx   = s.x - rx
            dy   = s.y - ry
            # Road normal direction (perpendicular to heading)
            n_x  = -math.sin(rh)
            n_y  =  math.cos(rh)
            lat  = dx * n_x + dy * n_y
            total += mp.w_lat * lat ** 2

            hdg_err = (s.yaw - rh + math.pi) % (2 * math.pi) - math.pi

            total  += mp.w_heading * hdg_err ** 2
            total  += mp.w_speed * (s.v - rv) ** 2
            total  += mp.w_steer * delta_k ** 2
            total  += mp.w_steer_rate * (delta_k - prev_delta) ** 2

            prev_delta = delta_k
            total  += mp.w_accel * a_k ** 2

        return total

    def solve(
        self,
        state_dict : dict,
    ) -> Tuple[float, float, dict]:

        t_start = time.perf_counter()

        state = VehicleState.from_dict(state_dict)

        lat_err, heading_err, ref_points = self.rp.get_reference(
            x=state.x, y=state.y, yaw=state.yaw, horizon=self.mp.horizon
        )

        v_ref = self.rp.speed_reference(state.x, state.y)
        ref_points = [(rx, ry, rh, v_ref) for (rx, ry, rh, _) in ref_points]

        N = self.mp.horizon

        # Bounds on [a_0, δ_0, a_1, δ_1, ...]
        lb = []
        ub = []
        for _ in range(N):
            lb += [self.mp.min_accel,  -self.mp.max_steer]
            ub += [self.mp.max_accel,   self.mp.max_steer]
        bounds = Bounds(lb=lb, ub=ub)

        u0 = np.roll(self._u_prev, -2)
        u0[-2] = 0.0
        u0[-1] = 0.0

        result = minimize(
            fun=self._cost,
            x0=u0,
            args=(state, ref_points),
            method="SLSQP",
            bounds=bounds,
            options={
                "maxiter": 50,
                "ftol"   : 1e-4,
            },
        )

        self._u_prev = result.x

        a_opt     = float(np.clip(result.x[0], self.mp.min_accel, self.mp.max_accel))
        delta_opt = float(np.clip(result.x[1], -self.mp.max_steer, self.mp.max_steer))

        velocity_cmd = float(np.clip(
            state.v + a_opt * self.mp.dt,
            self.mp.min_speed,
            self.mp.max_speed,
        ))

        self._prev_steer = delta_opt

        solve_ms = (time.perf_counter() - t_start) * 1000

        info = {
            "lat_err"      : lat_err,
            "heading_err"  : heading_err,
            "accel_opt"    : a_opt,
            "cost"         : float(result.fun),
            "solver_ms"    : solve_ms,
            "success"      : bool(result.success),
            "v_ref"        : v_ref,
        }

        return velocity_cmd, delta_opt, info


if __name__ == "__main__":
    mpc = MPCController()
    sample_state = {
        "x": 0.0, "y": 0.5, "yaw": 0.05, "velocity": 2.0, "timestamp": 0.0
    }
    v_cmd, steer, info = mpc.solve(sample_state)
    print(f"velocity_cmd : {v_cmd:.4f} m/s")
    print(f"steering     : {steer:.4f} rad ({math.degrees(steer):.2f}°)")
    print(f"lateral_err  : {info['lat_err']:.4f} m")
    print(f"heading_err  : {info['heading_err']:.4f} rad")
    print(f"cost         : {info['cost']:.4f}")
    print(f"solver_ms    : {info['solver_ms']:.1f} ms")
    print(f"success      : {info['success']}")
