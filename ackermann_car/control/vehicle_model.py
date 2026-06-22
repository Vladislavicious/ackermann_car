

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class VehicleParams:
    wheelbase   : float = 0.75    # L  total wheelbase [m]
    lf          : float = 0.375   # distance CoM → front axle [m]
    lr          : float = 0.375   # distance CoM → rear  axle [m]
    max_speed   : float = 5.0     # [m/s]
    min_speed   : float = 0.0     # [m/s]  (no reverse in this demo)
    max_steer   : float = 0.5     # [rad]  ~28.6°
    max_accel   : float = 2.0     # [m/s²]
    min_accel   : float = -3.0    # [m/s²]


@dataclass
class VehicleState:
    x   : float = 0.0
    y   : float = 0.0
    yaw : float = 0.0
    v   : float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw, self.v])

    @classmethod
    def from_dict(cls, d: dict) -> "VehicleState":
        return cls(x=d["x"], y=d["y"], yaw=d["yaw"], v=d["velocity"])


@dataclass
class VehicleControl:
    acceleration : float = 0.0   # [m/s²]
    steering     : float = 0.0   # [rad]



class KinematicBicycleModel:


    def __init__(self, params: VehicleParams | None = None) -> None:
        self.p = params or VehicleParams()

    def step(
        self,
        state: VehicleState,
        control: VehicleControl,
        dt: float = 0.1,
    ) -> VehicleState:

        x, y, psi, v = state.x, state.y, state.yaw, state.v
        a, delta      = control.acceleration, control.steering

        # Clamp inputs to physical limits
        a     = float(np.clip(a,     self.p.min_accel, self.p.max_accel))
        delta = float(np.clip(delta, -self.p.max_steer, self.p.max_steer))

        # Slip angle at centre of mass
        beta = math.atan2(self.p.lr * math.tan(delta), self.p.wheelbase)

        # Euler integration
        x_next   = x   + v * math.cos(psi + beta) * dt
        y_next   = y   + v * math.sin(psi + beta) * dt
        psi_next = psi + (v / self.p.wheelbase) * math.tan(delta) * math.cos(beta) * dt
        v_next   = float(np.clip(v + a * dt, self.p.min_speed, self.p.max_speed))

        # Wrap yaw to (-π, π]
        psi_next = (psi_next + math.pi) % (2 * math.pi) - math.pi

        return VehicleState(x=x_next, y=y_next, yaw=psi_next, v=v_next)

    def predict_trajectory(
        self,
        initial_state: VehicleState,
        controls: List[VehicleControl],
        dt: float = 0.1,
    ) -> List[VehicleState]:

        states = [initial_state]
        for ctrl in controls:
            states.append(self.step(states[-1], ctrl, dt=dt))
        return states


if __name__ == "__main__":
    model = KinematicBicycleModel()
    s0    = VehicleState(x=0.0, y=0.0, yaw=0.0, v=2.0)
    ctrl  = VehicleControl(acceleration=0.0, steering=0.15)
    traj  = model.predict_trajectory(s0, [ctrl] * 20, dt=0.1)
    print("Predicted trajectory (x, y, yaw, v):")
    for i, s in enumerate(traj):
        print(f"  t={i*0.1:.1f}s  x={s.x:.3f}  y={s.y:.3f}  yaw={s.yaw:.4f}  v={s.v:.2f}")
