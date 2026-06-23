from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from ackermann_car.sim.road import Road
from ackermann_car.sim.corridor_limits import CorridorLimits

OUTPUT_PATH = _REPO_ROOT / "ackermann_car" / "sim" / "mujoco_ackermann_generated.xml"

SEG_HALF_LEN = 0.22  # half-length of each road stripe box [m]
SEG_HALF_W = 0.04  # half-width of stripe [m]
SEG_HALF_H = 0.001
SEG_STEP = 2  # sample every Nth waypoint (reduces geom count)
HALF_ROAD_W = 1.75  # corridor half-width [m]


def _box_geom(
    name: str,
    x: float,
    y: float,
    heading: float,
    rgba: str,
    sx: float,
    sy: float,
    sz: float,
) -> str:
    deg = math.degrees(heading)
    return (
        f'      <geom name="{name}" type="box" '
        f'pos="{x:.4f} {y:.4f} {sz:.4f}" '
        f'euler="0 0 {deg:.4f}" '
        f'size="{sx:.4f} {sy:.4f} {sz:.4f}" '
        f'rgba="{rgba}" contype="0" conaffinity="0"/>\n'
    )


def build_xml(road: Road, corridor: CorridorLimits) -> str:
    waypoints = road.waypoints

    centre_geoms = []
    left_geoms = []
    right_geoms = []

    for i, (wx, wy, wh) in enumerate(waypoints[::SEG_STEP]):
        tag = i * SEG_STEP

        if (i % 2) == 0:
            centre_geoms.append(
                _box_geom(
                    f"cl_{tag}",
                    wx,
                    wy,
                    wh,
                    "1.0 0.9 0.0 1",
                    SEG_HALF_LEN,
                    SEG_HALF_W,
                    SEG_HALF_H,
                )
            )

        # Boundary offsets
        lx = wx - math.sin(wh) * HALF_ROAD_W
        ly = wy + math.cos(wh) * HALF_ROAD_W
        rx = wx + math.sin(wh) * HALF_ROAD_W
        ry = wy - math.cos(wh) * HALF_ROAD_W

        left_geoms.append(
            _box_geom(
                f"lb_{tag}",
                lx,
                ly,
                wh,
                "1.0 1.0 1.0 1",
                SEG_HALF_LEN,
                SEG_HALF_W,
                SEG_HALF_H,
            )
        )
        right_geoms.append(
            _box_geom(
                f"rb_{tag}",
                rx,
                ry,
                wh,
                "1.0 1.0 1.0 1",
                SEG_HALF_LEN,
                SEG_HALF_W,
                SEG_HALF_H,
            )
        )

    centre_xml = "".join(centre_geoms)
    left_xml = "".join(left_geoms)
    right_xml = "".join(right_geoms)

    road_xy = road.xy_array()
    cx = float(road_xy[:, 0].mean())
    cy = float(road_xy[:, 1].mean())
    xrange = float(road_xy[:, 0].max() - road_xy[:, 0].min())
    yrange = float(road_xy[:, 1].max() - road_xy[:, 1].min())
    ground_sx = max(xrange / 2 + 5, 20)
    ground_sy = max(yrange / 2 + 5, 10)

    xml = f"""<!--
  Road: {len(waypoints)} waypoints, {road.total_length:.1f} m total length
  Corridor: ±{HALF_ROAD_W} m
  Geom count: centre={len(centre_geoms)}, left={len(left_geoms)}, right={len(right_geoms)}
-->
<mujoco model="ackermann_car_scurve">

  <compiler angle="radian" coordinate="local"/>
  <option gravity="0 0 -9.81" timestep="0.01" integrator="RK4"/>

  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
    <rgba haze="0.5 0.5 0.5 1"/>
  </visual>

  <worldbody>


    <geom name="ground" type="plane"
          pos="{cx:.2f} {cy:.2f} 0"
          size="{ground_sx:.1f} {ground_sy:.1f} 0.1"
          rgba="0.28 0.28 0.28 1"
          contype="1" conaffinity="1"/>

    <body name="centreline">
{centre_xml}
    </body>

    <body name="left_boundary">
{left_xml}
    </body>

    <body name="right_boundary">
{right_xml}
    </body>

    <body name="car" pos="0 0 0.15">

      <joint name="joint_x"   type="slide" axis="1 0 0"
             range="-200 200" damping="0.5"/>
      <joint name="joint_y"   type="slide" axis="0 1 0"
             range="-200 200" damping="0.5"/>
      <joint name="joint_yaw" type="hinge" axis="0 0 1"
             range="-6.28 6.28" damping="1.0"/>

      <geom name="chassis" type="box" size="0.45 0.22 0.10"
            rgba="0.8 0.15 0.15 1" mass="10"
            contype="1" conaffinity="1"/>

      <geom name="windshield" type="box" size="0.12 0.20 0.07"
            pos="0.18 0 0.12" rgba="0.3 0.6 0.9 0.55"
            contype="0" conaffinity="0"/>

      <body name="fl_wheel" pos="0.30  0.25 -0.08">
        <geom name="fl_geom" type="cylinder" size="0.08 0.05"
              euler="1.5708 0 0" rgba="0.1 0.1 0.1 1"
              mass="0.5" contype="1" conaffinity="1"/>
      </body>
      <body name="fr_wheel" pos="0.30 -0.25 -0.08">
        <geom name="fr_geom" type="cylinder" size="0.08 0.05"
              euler="1.5708 0 0" rgba="0.1 0.1 0.1 1"
              mass="0.5" contype="1" conaffinity="1"/>
      </body>
      <body name="rl_wheel" pos="-0.30  0.25 -0.08">
        <geom name="rl_geom" type="cylinder" size="0.08 0.05"
              euler="1.5708 0 0" rgba="0.1 0.1 0.1 1"
              mass="0.5" contype="1" conaffinity="1"/>
      </body>
      <body name="rr_wheel" pos="-0.30 -0.25 -0.08">
        <geom name="rr_geom" type="cylinder" size="0.08 0.05"
              euler="1.5708 0 0" rgba="0.1 0.1 0.1 1"
              mass="0.5" contype="1" conaffinity="1"/>
      </body>

      <site name="car_imu" pos="0 0 0" size="0.02"/>

    </body>

  </worldbody>

  <actuator>
    <motor name="drive" joint="joint_x"   gear="1" ctrllimited="true" ctrlrange="-20 20"/>
    <motor name="steer" joint="joint_yaw" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
  </actuator>

  <sensor>
    <jointpos name="sen_x"   joint="joint_x"/>
    <jointpos name="sen_y"   joint="joint_y"/>
    <jointpos name="sen_yaw" joint="joint_yaw"/>
    <jointvel name="sen_vx"  joint="joint_x"/>
    <jointvel name="sen_vy"  joint="joint_y"/>
  </sensor>

</mujoco>
"""
    return xml


if __name__ == "__main__":
    road = Road()
    corridor = CorridorLimits(road, half_width=HALF_ROAD_W)

    print(f"  waypoints    : {len(road.waypoints)}")
    print(f"  total length : {road.total_length:.2f} m")

    xml = build_xml(road, corridor)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(xml, encoding="utf-8")

    print(f"Saved → {OUTPUT_PATH.relative_to(_REPO_ROOT)}")

    try:
        import mujoco

        m = mujoco.MjModel.from_xml_path(str(OUTPUT_PATH))
        print(f"MuJoCo validation OK  " f"(nbody={m.nbody}, ngeom={m.ngeom})")
    except Exception as e:
        print(f"MuJoCo validation FAIL: {e}")
        raise
