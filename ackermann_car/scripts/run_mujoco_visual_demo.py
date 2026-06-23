
from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path
import mujoco
import mujoco.viewer

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from ackermann_car.sim.ackermann_env import AckermannEnv
from ackermann_car.sim.road import Road
from ackermann_car.control.mpc_controller import MPCController, MPCParams
from ackermann_car.control.reference_path import ReferencePath

RESULTS_DIR = _REPO_ROOT / "ackermann_car" / "results"
LOG_FILE    = RESULTS_DIR / "mujoco_visual_log.csv"
GEN_XML     = _REPO_ROOT / "ackermann_car" / "sim" / "mujoco_ackermann_generated.xml"

CONTROL_HZ       = 10         # MPC runs at 10 Hz
SIM_STEPS_PER_MPC = 10        # sim steps (0.01 s each) per MPC step
REF_SPEED        = 2.5        # m/s
ROAD_END_MARGIN  = 3.0        # stop when this close to road end [m]
MAX_MPC_STEPS    = 3000       # hard cap

CSV_FIELDS = ["step", "time", "x", "y", "yaw", "velocity",
              "lateral_error", "heading_error", "velocity_cmd", "steering_cmd"]


def _setup_camera(viewer, road: Road) -> None:
    cam = viewer.cam
    cam.type       = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:]  = [5.0, 0.0, 0.0]   # initial look-at point
    cam.distance   = 18.0
    cam.elevation  = -45.0
    cam.azimuth    = 90.0


def _update_camera(viewer, x: float, y: float) -> None:
    alpha = 0.05
    viewer.cam.lookat[0] += alpha * (x - viewer.cam.lookat[0])
    viewer.cam.lookat[1] += alpha * (y - viewer.cam.lookat[1])


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not GEN_XML.exists():
        print(f"Generated XML not found: {GEN_XML}")
        print("Run first:  python ackermann_car/scripts/generate_mujoco_road_xml.py")
        sys.exit(1)

    env = AckermannEnv(backend="mujoco_kinematic", xml_path=GEN_XML)
    state = env.reset(x=0.0, y=0.0, yaw=0.0, v=0.0)

    road = Road()
    rp   = ReferencePath(road=road, ref_speed=REF_SPEED, lookahead_step=3)
    mpc  = MPCController(
        mpc_params=MPCParams(horizon=10, dt=1.0 / CONTROL_HZ,
                             w_lat=8.0, w_heading=5.0, w_speed=2.0,
                             w_steer=0.5, w_steer_rate=1.0, w_accel=0.2),
        ref_path=rp,
    )

    total_road_len = road.total_length
    print(f"Road length: {total_road_len:.2f} m")
    print(f"Will stop within {ROAD_END_MARGIN} m of road end.")

    log_fh = open(LOG_FILE, "w", newline="")
    writer = csv.DictWriter(log_fh, fieldnames=CSV_FIELDS)
    writer.writeheader()

    mj_model = env.mj_model
    mj_data  = env.mj_data

    step       = 0
    t_ctrl     = 0.0
    v_cmd      = 0.0
    steer_cmd  = 0.0

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        _setup_camera(viewer, road)

        while viewer.is_running() and step < MAX_MPC_STEPS:
            t_loop_start = time.perf_counter()

            state_dict = {
                "x": state["x"], "y": state["y"],
                "yaw": state["yaw"], "velocity": state["velocity"],
                "timestamp": t_ctrl,
            }
            v_cmd, steer_cmd, info = mpc.solve(state_dict)

            for _ in range(SIM_STEPS_PER_MPC):
                state = env.step(v_cmd, steer_cmd, sim_time=t_ctrl)
                viewer.sync()

            t_ctrl += 1.0 / CONTROL_HZ
            step   += 1

            lat_err = road.lateral_error(state["x"], state["y"])
            writer.writerow({
                "step"         : step,
                "time"         : round(t_ctrl, 3),
                "x"            : round(state["x"], 4),
                "y"            : round(state["y"], 4),
                "yaw"          : round(state["yaw"], 4),
                "velocity"     : round(state["velocity"], 4),
                "lateral_error": round(lat_err, 4),
                "heading_error": round(info["heading_err"], 4),
                "velocity_cmd" : round(v_cmd, 4),
                "steering_cmd" : round(steer_cmd, 4),
            })

            _update_camera(viewer, state["x"], state["y"])

            if step % 20 == 0:
                # Approx arc-length covered (nearest waypoint arc)
                idx = road.nearest_waypoint_index(state["x"], state["y"])
                arc = (idx / len(road.waypoints)) * total_road_len
                print(f"  step={step:4d}  t={t_ctrl:5.1f}s  "
                      f"x={state['x']:6.2f}  y={state['y']:5.2f}  "
                      f"lat={lat_err:+.3f}m  "
                      f"steer={math.degrees(steer_cmd):+.1f}°  "
                      f"road={arc:.1f}/{total_road_len:.1f}m"
                )

            idx     = road.nearest_waypoint_index(state["x"], state["y"])
            arc_pos = (idx / max(1, len(road.waypoints) - 1)) * total_road_len
            remaining = total_road_len - arc_pos
            if remaining < ROAD_END_MARGIN:
                print(f"\nReached road end at t={t_ctrl:.1f}s")
                break

            elapsed = time.perf_counter() - t_loop_start
            sleep_t = (1.0 / CONTROL_HZ) - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    log_fh.close()
    print(f"\nDone — {step} control steps.")
    print(f"Log saved: {LOG_FILE}")
