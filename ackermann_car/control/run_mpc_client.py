from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import zmq

from ackermann_car.control.mpc_controller import MPCController, MPCParams
from ackermann_car.control.reference_path import ReferencePath
from ackermann_car.sim.road import Road

SERVER_ADDR     = "tcp://localhost:5555"
RESULTS_DIR     = _REPO_ROOT / "ackermann_car" / "results"
LOG_FILE        = RESULTS_DIR / "trajectory_log.csv"
MAX_STEPS       = 430        # stop after this many control steps
CONTROL_HZ      = 10           # MPC runs at 10 Hz
CONTROL_DT      = 1.0 / CONTROL_HZ
SIM_STEPS_PER_CTRL = 10        # sim runs at 100 Hz internally, MPC at 10 Hz
REF_SPEED       = 2.5          # [m/s]

CSV_FIELDS = [
    "step", "time",
    "x", "y", "yaw", "velocity",
    "lateral_error", "heading_error",
    "velocity_cmd", "steering_cmd",
    "solver_ms", "mpc_cost",
    "inside_corridor",
]

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

    road    = Road()
    ref_path = ReferencePath(road=road, ref_speed=REF_SPEED, lookahead_step=3)
    mpc     = MPCController(
        mpc_params=MPCParams(
            horizon=10,
            dt=CONTROL_DT,
            w_lat=8.0,
            w_heading=5.0,
            w_speed=2.0,
            w_steer=0.5,
            w_steer_rate=1.0,
            w_accel=0.2,
        ),
        ref_path=ref_path,
    )

    log_fh  = open(LOG_FILE, "w", newline="")
    writer  = csv.DictWriter(log_fh, fieldnames=CSV_FIELDS)
    writer.writeheader()

    print(f"Logging to {LOG_FILE}")
    print(f"Running MPC for up to {MAX_STEPS} steps …")
    print()

    state        = init_state
    step         = 0
    t_control    = 0.0
    velocity_cmd = 0.0
    steering_cmd = 0.0

    try:
        while step < MAX_STEPS:
            t_wall_start = time.perf_counter()

            velocity_cmd, steering_cmd, info = mpc.solve(state)

            last_state = state
            for _ in range(SIM_STEPS_PER_CTRL):
                socket.send(json.dumps({
                    "cmd"      : "step",
                    "velocity" : velocity_cmd,
                    "steering" : steering_cmd,
                }).encode())
                last_state = json.loads(socket.recv())

            state = last_state

            writer.writerow({
                "step"          : step,
                "time"          : round(t_control, 4),
                "x"             : round(state.get("x", 0.0), 4),
                "y"             : round(state.get("y", 0.0), 4),
                "yaw"           : round(state.get("yaw", 0.0), 4),
                "velocity"      : round(state.get("velocity", 0.0), 4),
                "lateral_error" : round(info["lat_err"], 4),
                "heading_error" : round(info["heading_err"], 4),
                "velocity_cmd"  : round(velocity_cmd, 4),
                "steering_cmd"  : round(steering_cmd, 4),
                "solver_ms"     : round(info["solver_ms"], 2),
                "mpc_cost"      : round(info["cost"], 4),
                "inside_corridor": int(state.get("inside_corridor", True)),
            })

            t_control += CONTROL_DT
            step      += 1

            if step % 20 == 0:
                print(
                    f"step={step:4d}  t={t_control:6.2f}s  "
                    f"x={state['x']:6.2f}  y={state['y']:6.2f}  "
                    f"yaw={state['yaw']:+.3f}rad  "
                    f"v={state['velocity']:.2f}m/s  "
                    f"lat={info['lat_err']:+.3f}m  "
                    f"steer={math.degrees(steering_cmd):+.1f}°  "
                    f"solve={info['solver_ms']:.0f}ms"
                )

            elapsed = time.perf_counter() - t_wall_start
            sleep_t = CONTROL_DT - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        log_fh.close()
        socket.close()
        ctx.term()

    print()
    print(f"Done.  {step} control steps executed.")
    print(f"Log saved to: {LOG_FILE}")
