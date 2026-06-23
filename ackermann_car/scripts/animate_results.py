

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

matplotlib.rcParams.update({"figure.dpi": 100, "font.size": 9})

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from ackermann_car.sim.road import Road
from ackermann_car.sim.corridor_limits import CorridorLimits

RESULTS_DIR = _REPO_ROOT / "ackermann_car" / "results"
LOG_FILE    = RESULTS_DIR / "trajectory_log.csv"
GIF_OUT     = RESULTS_DIR / "trajectory_animation.gif"
MP4_OUT     = RESULTS_DIR / "trajectory_animation.mp4"

FRAME_STEP   = 3       # use every Nth log row as a frame (speeds up GIF)
INTERVAL_MS  = 80
SAVE_GIF     = True
SAVE_MP4     = False   # set True if ffmpeg is installed


# Front of car points in +x direction
_CAR_CORNERS = np.array([
    [ 0.45,  0.22],
    [ 0.45, -0.22],
    [-0.45, -0.22],
    [-0.45,  0.22],
])

def _car_polygon(x: float, y: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return (_CAR_CORNERS @ R.T) + np.array([x, y])


if __name__ == "__main__":
    if not LOG_FILE.exists():
        print(f"Log not found: {LOG_FILE}")
        print("Run the simulation first:")
        sys.exit(1)

    df = pd.read_csv(LOG_FILE)
    print(f"Loaded {len(df)} rows from {LOG_FILE}")

    # Downsample for animation speed
    df_anim = df.iloc[::FRAME_STEP].reset_index(drop=True)
    n_frames = len(df_anim)
    print(f"{n_frames} animation frames (step={FRAME_STEP})")

    road     = Road()
    corridor = CorridorLimits(road, half_width=1.75)
    cl_xy    = road.xy_array()
    left, right = corridor.boundary_points(step=1)

    fig = plt.figure(figsize=(13, 7))
    gs  = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], hspace=0.4, wspace=0.35)

    ax_road   = fig.add_subplot(gs[:, 0])
    ax_lat    = fig.add_subplot(gs[0, 1])
    ax_steer  = fig.add_subplot(gs[1, 1])

    ax_road.set_facecolor("#444444")
    ax_road.plot(cl_xy[:,0], cl_xy[:,1], "y--", lw=1.2, label="Centre line", zorder=3)
    ax_road.plot(left[:,0],  left[:,1],  "w-",  lw=1.0, zorder=3)
    ax_road.plot(right[:,0], right[:,1], "w-",  lw=1.0, zorder=3, label="Boundary")
    # Full trajectory ghost
    ax_road.plot(df["x"], df["y"], color="cyan", lw=0.6, alpha=0.35, zorder=2, label="Full path")

    ax_road.set_xlabel("X [m]")
    ax_road.set_ylabel("Y [m]")
    ax_road.set_title("MPC Road Tracking — Live Animation")
    ax_road.set_aspect("equal")
    ax_road.legend(loc="upper left", fontsize=7)

    init_pts = _car_polygon(df_anim["x"].iloc[0], df_anim["y"].iloc[0], df_anim["yaw"].iloc[0])
    car_poly = patches.Polygon(init_pts, closed=True,
                               fc="tomato", ec="darkred", lw=1.2, zorder=6)
    ax_road.add_patch(car_poly)

    vel_arrow = ax_road.annotate(
        "", xy=(0, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="lime", lw=2),
        zorder=7,
    )

    trace_x, trace_y = [], []
    trace_line, = ax_road.plot([], [], color="orange", lw=1.5, zorder=5, alpha=0.7)

    time_label = ax_road.text(
        0.02, 0.96, "", transform=ax_road.transAxes,
        fontsize=9, color="white", va="top",
        bbox=dict(fc="black", alpha=0.5, pad=2),
    )

    t_all   = df["time"].to_numpy()
    lat_all = df["lateral_error"].to_numpy()
    str_all = np.degrees(df["steering_cmd"].to_numpy())

    ax_lat.set_xlim(0, t_all[-1])
    ax_lat.set_ylim(-2.0, 2.0)
    ax_lat.axhline(0,    color="green",  lw=0.8, alpha=0.7)
    ax_lat.axhline(1.75, color="red",    lw=0.8, ls="--", alpha=0.8)
    ax_lat.axhline(-1.75, color="red",   lw=0.8, ls="--", alpha=0.8)
    ax_lat.set_xlabel("Time [s]")
    ax_lat.set_ylabel("Lateral error [m]")
    ax_lat.set_title("Lateral Error")
    ax_lat.grid(True, alpha=0.3)
    lat_line,     = ax_lat.plot([], [], color="steelblue", lw=1.4)
    lat_dot,      = ax_lat.plot([], [], "o", color="steelblue", ms=4)

    ax_steer.set_xlim(0, t_all[-1])
    ax_steer.set_ylim(-32, 32)
    ax_steer.axhline(0,   color="grey", lw=0.6, alpha=0.6)
    ax_steer.axhline(28.6, color="red", lw=0.8, ls="--", alpha=0.8)
    ax_steer.axhline(-28.6, color="red", lw=0.8, ls="--", alpha=0.8)
    ax_steer.set_xlabel("Time [s]")
    ax_steer.set_ylabel("Steering [°]")
    ax_steer.set_title("Steering Command")
    ax_steer.grid(True, alpha=0.3)
    steer_line,   = ax_steer.plot([], [], color="darkorange", lw=1.4)
    steer_dot,    = ax_steer.plot([], [], "o", color="darkorange", ms=4)

    VIEW_HALF = 15.0
    _view_cx = [df_anim["x"].iloc[0]]
    _view_cy = [df_anim["y"].iloc[0]]

    def _smooth_view(xi, yi):
        alpha = 0.08
        _view_cx[0] += alpha * (xi - _view_cx[0])
        _view_cy[0] += alpha * (yi - _view_cy[0])
        ax_road.set_xlim(_view_cx[0] - VIEW_HALF, _view_cx[0] + VIEW_HALF)
        ax_road.set_ylim(_view_cy[0] - VIEW_HALF, _view_cy[0] + VIEW_HALF)

    def _update(frame_idx: int):
        row = df_anim.iloc[frame_idx]
        x, y, yaw = float(row["x"]), float(row["y"]), float(row["yaw"])
        v, t      = float(row["velocity"]), float(row["time"])

        corners = _car_polygon(x, y, yaw)
        car_poly.set_xy(corners)

        arr_len = v * 0.5
        vel_arrow.xy         = (x + arr_len * math.cos(yaw),
                                y + arr_len * math.sin(yaw))
        vel_arrow.xytext     = (x, y)

        trace_x.append(x)
        trace_y.append(y)
        trace_line.set_data(trace_x, trace_y)

        time_label.set_text(
            f"t = {t:.1f}s\n"
            f"v = {v:.2f} m/s\n"
            f"lat = {row['lateral_error']:+.3f} m"
        )

        _smooth_view(x, y)

        mask = t_all <= t + 0.01
        if mask.any():
            lat_line.set_data(t_all[mask], lat_all[mask])
            lat_dot.set_data([t_all[mask][-1]], [lat_all[mask][-1]])
            steer_line.set_data(t_all[mask], str_all[mask])
            steer_dot.set_data([t_all[mask][-1]], [str_all[mask][-1]])

        return (car_poly, vel_arrow, trace_line, time_label,
                lat_line, lat_dot, steer_line, steer_dot)

    ani = animation.FuncAnimation(
        fig, _update,
        frames=n_frames,
        interval=INTERVAL_MS,
        blit=False,   # blit=True causes issues with Polygon patches
        repeat=False,
    )

    if SAVE_GIF:
        print(f"Saving GIF → {GIF_OUT}  (may take ~30 s) …")
        try:
            writer_gif = animation.PillowWriter(fps=1000 // INTERVAL_MS)
            ani.save(str(GIF_OUT), writer=writer_gif)
            print(f"Saved: {GIF_OUT.name}")
        except Exception as e:
            print(f"GIF save failed: {e}")
            print("  Install Pillow:  pip install Pillow")

    if SAVE_MP4:
        print(f"Saving MP4 → {MP4_OUT} …")
        try:
            writer_mp4 = animation.FFMpegWriter(fps=1000 // INTERVAL_MS)
            ani.save(str(MP4_OUT), writer=writer_mp4)
            print(f"Saved: {MP4_OUT.name}")
        except Exception as e:
            print(f"MP4 save failed: {e}  (is ffmpeg installed?)")

    plt.tight_layout()
    plt.show()
