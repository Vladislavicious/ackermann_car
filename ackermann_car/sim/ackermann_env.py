from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import mujoco

from .disturbance_model import DisturbanceModel

XML_PATH = Path(__file__).parent / "mujoco_ackermann.xml"
XML_GEN_PATH = Path(__file__).parent / "mujoco_ackermann_generated.xml"

BackendLiteral = Literal["kinematic", "mujoco_kinematic"]


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


class _MuJoCoKinematicSim:

    DT = 0.01

    def __init__(self, xml_path: Path | None = None) -> None:

        path = xml_path or (XML_GEN_PATH if XML_GEN_PATH.exists() else XML_PATH)
        self._model = mujoco.MjModel.from_xml_path(str(path))
        self._data = mujoco.MjData(self._model)

        def _qadr(name):
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return int(self._model.jnt_qposadr[jid])

        self._qadr_x = _qadr("joint_x")
        self._qadr_y = _qadr("joint_y")
        self._qadr_yaw = _qadr("joint_yaw")

        self.reset()

    def reset(self, x=0.0, y=0.0, yaw=0.0, v=0.0) -> None:
        mujoco.mj_resetData(self._model, self._data)
        self.x, self.y, self.yaw, self.v = float(x), float(y), float(yaw), float(v)
        self.t = 0.0
        self._write_qpos_and_forward()

    def _write_qpos_and_forward(self) -> None:

        self._data.qpos[self._qadr_x] = self.x
        self._data.qpos[self._qadr_y] = self.y
        self._data.qpos[self._qadr_yaw] = self.yaw
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

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
        self._write_qpos_and_forward()

    def get_state(self) -> Dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "yaw": self.yaw,
            "velocity": self.v,
            "timestamp": self.t,
        }

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data


class AckermannEnv:

    def __init__(
        self,
        backend: BackendLiteral = "kinematic",
        use_disturbances: bool = False,
        xml_path: Optional[Path] = None,
    ) -> None:
        self.backend = backend

        if backend == "kinematic":
            self._sim = _KinematicBicycleSim()
        elif backend == "mujoco_kinematic":
            self._sim = _MuJoCoKinematicSim(xml_path=xml_path)
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

    @property
    def mj_model(self):
        if not hasattr(self._sim, "model"):
            raise AttributeError("mj_model is only available for MuJoCo backends.")
        return self._sim.model

    @property
    def mj_data(self):
        if not hasattr(self._sim, "data"):
            raise AttributeError("mj_data is only available for MuJoCo backends.")
        return self._sim.data


if __name__ == "__main__":
    print("kinematic backend")
    env = AckermannEnv(backend="kinematic")
    s = env.reset()
    for _ in range(5):
        s = env.step(2.0, 0.1)
    print(s)
