

from __future__ import annotations

import sys
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from ackermann_car.sim.road import Road
from ackermann_car.sim.corridor_limits import CorridorLimits

LOG_FILE  = _REPO_ROOT / "ackermann_car" / "results" / "trajectory_log.csv"
PLOT_DIR  = _REPO_ROOT / "ackermann_car" / "results"

matplotlib.rcParams.update({
    "figure.dpi"       : 120,
    "font.size"        : 10,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "lines.linewidth"  : 1.8,
})


def load_log() -> pd.DataFrame:
    if not LOG_FILE.exists():
        print(f"Log file not found: {LOG_FILE}")
        print("  Run the simulation first")
        sys.exit(1)
    df = pd.read_csv(LOG_FILE)
    print(f"Loaded {len(df)} rows from {LOG_FILE}")
    return df


def plot_trajectory(df: pd.DataFrame, road: Road, corridor: CorridorLimits) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))

    left_pts, right_pts = corridor.boundary_points(step=1)
    road_xy = road.xy_array()
    ax.fill_betweenx(
        left_pts[:, 1],
        left_pts[:, 0], right_pts[:, 0],
        alpha=0.15, color="grey", label="Road surface",
    )

    ax.plot(left_pts[:, 0],  left_pts[:, 1],  "w-",  lw=1.5, label="Boundary")
    ax.plot(right_pts[:, 0], right_pts[:, 1], "w-",  lw=1.5)

    ax.plot(road_xy[:, 0], road_xy[:, 1], "y--", lw=1.5, label="Centre line", zorder=3)

    x_arr   = df["x"].to_numpy()
    y_arr   = df["y"].to_numpy()
    lat_arr = df["lateral_error"].to_numpy()

    sc = ax.scatter(
        x_arr, y_arr,
        c=np.abs(lat_arr), cmap="RdYlGn_r",
        vmin=0, vmax=1.75,
        s=4, zorder=5, label="Vehicle path",
    )
    plt.colorbar(sc, ax=ax, label="|Lateral error| [m]")

    ax.plot(x_arr[0],  y_arr[0],  "go", ms=8, zorder=6, label="Start")
    ax.plot(x_arr[-1], y_arr[-1], "rs", ms=8, zorder=6, label="End")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("MPC Road Tracking — Vehicle Trajectory vs Centre Line")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_lateral_error(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))

    t   = df["time"].to_numpy()
    lat = df["lateral_error"].to_numpy()

    ax.plot(t, lat, color="steelblue", label="Lateral error")
    ax.axhline(y= 1.75, color="red", ls="--", lw=1, label="Boundary ±1.75 m")
    ax.axhline(y=-1.75, color="red", ls="--", lw=1)
    ax.axhline(y=0,     color="green", ls="-", lw=0.8, alpha=0.6, label="Centre line")

    ax.fill_between(t, lat, 0, alpha=0.2, color="steelblue")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Lateral error [m]")
    ax.set_title("Lateral Error Over Time  (positive = left of centre)")
    ax.legend()

    # Stats annotation
    rms = float(np.sqrt(np.mean(lat**2)))
    ax.text(
        0.98, 0.95,
        f"RMS = {rms:.3f} m\nMax = {np.max(np.abs(lat)):.3f} m",
        transform=ax.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    return fig


def plot_steering(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))

    t     = df["time"].to_numpy()
    steer = np.degrees(df["steering_cmd"].to_numpy())

    ax.plot(t, steer, color="darkorange", label="Steering angle")
    ax.axhline(y= math.degrees(0.5), color="red", ls="--", lw=1, label="Limit ±28.6°")
    ax.axhline(y=-math.degrees(0.5), color="red", ls="--", lw=1)
    ax.fill_between(t, steer, 0, alpha=0.2, color="darkorange")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Steering angle [°]")
    ax.set_title("MPC Steering Command Over Time")
    ax.legend()

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    df       = load_log()
    road     = Road()
    corridor = CorridorLimits(road, half_width=1.75)

    fig1 = plot_trajectory(df, road, corridor)
    fig2 = plot_lateral_error(df)
    fig3 = plot_steering(df)

    out1 = PLOT_DIR / "trajectory.png"
    out2 = PLOT_DIR / "lateral_error.png"
    out3 = PLOT_DIR / "steering_cmd.png"

    fig1.savefig(out1, bbox_inches="tight")
    fig2.savefig(out2, bbox_inches="tight")
    fig3.savefig(out3, bbox_inches="tight")

    print(f"Saved: {out1.name}")
    print(f"Saved: {out2.name}")
    print(f"Saved: {out3.name}")

    plt.show()
