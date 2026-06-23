from __future__ import annotations

import math
import time
from typing import Tuple

import numpy as np


class DisturbanceModel:

    def __init__(
        self,
        wind_magnitude: float = 2.0,
        wind_frequency: float = 0.1,
        noise_std: float = 0.3,
        seed: int | None = 42,
        enabled: bool = True,
    ) -> None:
        self.wind_magnitude = wind_magnitude
        self.wind_frequency = wind_frequency
        self.noise_std = noise_std
        self.enabled = enabled
        self._rng = np.random.default_rng(seed)
        self._t0 = time.time()

    def get_disturbance(self, t: float | None = None) -> Tuple[float, float, float]:

        if not self.enabled:
            return 0.0, 0.0, 0.0

        if t is None:
            t = time.time() - self._t0

        # Sinusoidal cross-wind (lateral direction dominant)
        wind_y = self.wind_magnitude * math.sin(2 * math.pi * self.wind_frequency * t)
        wind_x = (
            0.2
            * self.wind_magnitude
            * math.cos(2 * math.pi * self.wind_frequency * 0.37 * t)
        )

        # Gaussian noise
        noise_y = self._rng.normal(0.0, self.noise_std)
        noise_x = self._rng.normal(0.0, self.noise_std * 0.3)

        # Small yaw moment from asymmetric wind
        torque_z = 0.1 * wind_y * self._rng.normal(1.0, 0.2)

        return (
            float(wind_x + noise_x),
            float(wind_y + noise_y),
            float(torque_z),
        )

    def reset(self) -> None:

        self._t0 = time.time()


if __name__ == "__main__":
    model = DisturbanceModel(wind_magnitude=2.0, enabled=True)
    print("Disturbance samples:")
    for i in range(5):
        fx, fy, tz = model.get_disturbance(t=float(i) * 0.5)
        print(f"  t={i*0.5:.1f}s  fx={fx:+.3f} N  fy={fy:+.3f} N  tz={tz:+.3f} Nm")
