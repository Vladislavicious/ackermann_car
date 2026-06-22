

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import zmq

SERVER_ADDR     = "tcp://localhost:5555"
RESULTS_DIR     = _REPO_ROOT / "ackermann_car" / "results"
LOG_FILE        = RESULTS_DIR / "trajectory_log.csv"
MAX_STEPS       = 430        # stop after this many control steps
CONTROL_HZ      = 10           # MPC runs at 10 Hz
CONTROL_DT      = 1.0 / CONTROL_HZ
SIM_STEPS_PER_CTRL = 10        # sim runs at 100 Hz internally, MPC at 10 Hz
REF_SPEED       = 2.5          # [m/s]

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ctx    = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.connect(SERVER_ADDR)
    print(f"Connected to {SERVER_ADDR}")

    socket.send(json.dumps({
        "cmd": "reset", "x": 0.0, "y": 0.0, "yaw": 0.0, "v": 0.0
    }).encode())
    init_state = json.loads(socket.recv())
    print(f"Initial state: {init_state}")