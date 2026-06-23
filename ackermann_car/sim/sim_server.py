from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import zmq

from ackermann_car.sim.ackermann_env import AckermannEnv
from ackermann_car.sim.road import Road
from ackermann_car.sim.corridor_limits import CorridorLimits

ZMQ_ADDR = "tcp://*:5555"
DT = 0.01
USE_DISTURB = False
TARGET_RT = True


def main() -> None:
    env = AckermannEnv(use_disturbances=USE_DISTURB)
    road = Road()
    corridor = CorridorLimits(road, half_width=1.75)
    state = env.reset(x=0.0, y=0.0, yaw=0.0, v=0.0)

    print(f"[sim_server] Backend  : {env.backend}")
    print(f"[sim_server] Road pts : {len(road.waypoints)}")
    print(f"[sim_server] Road len : {road.total_length:.1f} m")

    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(ZMQ_ADDR)
    print(f"[sim_server] Listening on {ZMQ_ADDR}")
    print("[sim_server] Waiting for MPC client …")

    step_count = 0
    t_sim = 0.0

    try:
        while True:
            try:
                raw = socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.001)
                continue

            msg = json.loads(raw.decode())
            cmd = msg.get("cmd", "get_state")

            if cmd == "get_state":
                reply = dict(state)

            elif cmd == "step":
                velocity_cmd = float(msg.get("velocity", 0.0))
                steering_angle = float(msg.get("steering", 0.0))

                t_wall_start = time.perf_counter()
                state = env.step(
                    velocity_cmd=velocity_cmd,
                    steering_angle=steering_angle,
                    sim_time=t_sim,
                )
                t_sim += DT
                step_count += 1

                if TARGET_RT:
                    elapsed = time.perf_counter() - t_wall_start
                    sleep_t = DT - elapsed
                    if sleep_t > 0:
                        time.sleep(sleep_t)

                lat_err = road.lateral_error(state["x"], state["y"])
                inside = corridor.is_inside(state["x"], state["y"])

                reply = dict(state)
                reply["lateral_error"] = lat_err
                reply["inside_corridor"] = inside

                if step_count % 100 == 0:
                    print(
                        f"[sim_server] t={t_sim:6.2f}s "
                        f"x={state['x']:6.2f} y={state['y']:6.2f} "
                        f"yaw={state['yaw']:5.3f}rad "
                        f"v={state['velocity']:.2f}m/s "
                        f"lat_err={lat_err:+.3f}m "
                        f"in_corridor={inside}"
                    )

            elif cmd == "reset":
                state = env.reset(
                    x=float(msg.get("x", 0.0)),
                    y=float(msg.get("y", 0.0)),
                    yaw=float(msg.get("yaw", 0.0)),
                    v=float(msg.get("v", 0.0)),
                )
                t_sim = 0.0
                step_count = 0
                reply = dict(state)
                print("[sim_server] Environment reset.")

            else:
                reply = {"error": f"Unknown command: {cmd}"}

            socket.send(json.dumps(reply).encode())

    except KeyboardInterrupt:
        print("\n[sim_server] Shutting down.")
    finally:
        socket.close()
        ctx.term()


if __name__ == "__main__":
    main()
