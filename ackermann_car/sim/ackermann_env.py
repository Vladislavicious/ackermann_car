from __future__ import annotations

import math
from typing import Dict, Literal

from .disturbance_model import DisturbanceModel

BackendLiteral = Literal["kinematic"]


class _KinematicBicycleSim:

    DT = 0.01

    def __init__(self) -> None:
        self.reset()

    def reset(self, x=0.0, y=0.0, yaw=0.0, v=0.0) -> None:
        self.x, self.y, self.yaw, self.v = x, y, yaw, float(v)
        self.t = 0.0

    def step(
        self,
        velocity_cmd: float,
        steering_angle: float,
        disturbance: tuple = (0.0, 0.0, 0.0),
    ) -> None:
        self.x, self.y, self.yaw, self.v = _bicycle_step(
            self.x,
            self.y,
            self.yaw,
            self.v,
            velocity_cmd,
            steering_angle,
            self.DT,
            disturbance,
        )
        self.t += self.DT

    def get_state(self) -> Dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "yaw": self.yaw,
            "velocity": self.v,
            "timestamp": self.t,
        }


def _bicycle_step(
    x: float,
    y: float,
    yaw: float,
    v: float,
    velocity_cmd: float,
    steering_angle: float,
    dt: float,
    disturbance: tuple = (0.0, 0.0, 0.0),
    wheelbase: float = 0.75,
) -> tuple[float, float, float, float]:

    v_new = v + 0.5 * (velocity_cmd - v) * dt * 10

    beta = math.atan(0.5 * math.tan(steering_angle))
    x_new = x + v_new * math.cos(yaw + beta) * dt
    y_new = y + v_new * math.sin(yaw + beta) * dt
    yaw_new = yaw + (v_new / wheelbase) * math.tan(steering_angle) * dt

    fx, fy, tz = disturbance
    mass = 10.0
    x_new += 0.5 * (fx / mass) * dt**2
    y_new += 0.5 * (fy / mass) * dt**2
    yaw_new += 0.5 * (tz / 0.5) * dt**2

    yaw_new = (yaw_new + math.pi) % (2 * math.pi) - math.pi
    return x_new, y_new, yaw_new, v_new


class AckermannEnv:

    def __init__(
        self,
        backend: BackendLiteral = "kinematic",
        use_disturbances: bool = False,
    ) -> None:
        self.backend = backend

        if backend == "kinematic":
            self._sim = _KinematicBicycleSim()
        else:
            raise ValueError(f"Unknown backend '{backend}'")

        self._dist = DisturbanceModel(enabled=use_disturbances)
        print(f"backend={backend}  disturbances={use_disturbances}")

    def reset(self, x=0.0, y=0.0, yaw=0.0, v=0.0) -> Dict[str, float]:

        self._dist.reset()
        self._sim.reset(x=x, y=y, yaw=yaw, v=v)
        return self._sim.get_state()

    def step(
        self, velocity_cmd: float, steering_angle: float, sim_time: float | None = None
    ) -> Dict[str, float]:

        disturbance = self._dist.get_disturbance(t=sim_time)
        self._sim.step(velocity_cmd, steering_angle, disturbance=disturbance)
        return self._sim.get_state()

    def get_state(self) -> Dict[str, float]:
        return self._sim.get_state()


if __name__ == "__main__":
    print("kinematic backend")
    env = AckermannEnv(backend="kinematic")
    s = env.reset()
    for _ in range(5):
        s = env.step(2.0, 0.1)
    print(s)
