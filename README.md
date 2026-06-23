# Issue #39 — Ackermann Car Road Centre-Line Tracking with MPC

## Task Description

Implement an autonomous Ackermann car that tracks the centre line of a road
using **Model Predictive Control (MPC)**. The system runs as two separate
processes communicating over **ZeroMQ**. The robot is constrained to planar
2D motion with three allowed DOF: **x**, **y**, and **yaw**, while **z**,
**roll**, and **pitch** are fixed.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Terminal 1                    Terminal 2                           │
│                                                                     │
│  sim_server.py                 run_mpc_client.py                    │
│  ┌────────────────────┐        ┌──────────────────────────────────┐ │
│  │  AckermannEnv      │        │  MPCController                   │ │
│  │  (MuJoCo / bicycle)│        │  ┌───────────────────────────┐   │ │
│  │                    │        │  │ KinematicBicycleModel     │   │ │
│  │  road.py           │        │  │ (prediction model)        │   │ │
│  │  corridor_limits.py│        │  └───────────────────────────┘   │ │
│  │  disturbance_model │        │  ┌───────────────────────────┐   │ │
│  │                    │        │  │ ReferencePath             │   │ │
│  │  ZMQ REP :5555     │◄──────►│  │ (road waypoints)          │   │ │
│  │                    │        │  └───────────────────────────┘   │ │
│  │  sends:            │  JSON  │  scipy SLSQP optimiser           │ │
│  │   x, y, yaw, v, t  │◄──────►│                                  │ │
│  │  receives:         │        │  sends: velocity_cmd, steering   │ │
│  │   velocity, steer  │        │  logs:  results/trajectory_log   │ │
│  └────────────────────┘        └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
ackermann_car/
 sim/
  mujoco_ackermann.xml           ← base MuJoCo scene
  mujoco_ackermann_generated.xml ← generated S-curve MuJoCo road scene
  ackermann_env.py               ← simulation environment and backend selection
  road.py                        ← Bézier centre-line generator
  corridor_limits.py             ← boundary checker
  sim_server.py                  ← ZeroMQ REP server (Terminal 1)
  disturbance_model.py           ← wind / noise disturbance injector
 control/
  vehicle_model.py               ← kinematic bicycle model
  reference_path.py              ← road-to-MPC reference generator
  mpc_controller.py              ← MPC with SLSQP (NOT PD/PID)
  run_mpc_client.py              ← ZeroMQ REQ client + logging
 scripts/
  plot_results.py                ← trajectory/error/steering plots
  animate_results.py             ← animated GIF from logs
  generate_mujoco_road_xml.py    ← S-curve road XML generator
  run_mujoco_visual_demo.py      ← MuJoCo visual demo
 results/
  .gitkeep
  trajectory_log.csv             ← generated at runtime
  trajectory.png                 ← generated plot
  lateral_error.png              ← generated plot
  steering_cmd.png               ← generated plot
  trajectory_animation.gif       ← generated animation
 README.md
requirements.txt
```

---

## Installation

```bash
# 1. Clone / fork the repo
git clone https://github.com/YOUR_FORK/simulator.git
cd simulator

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Mujoco via installation guide from:
https://github.com/google-deepmind/mujoco


---

## Run Instructions

Open **two terminals** from the repository root.

**Terminal 1 — Start the simulation server:**
```bash
python ackermann_car/sim/sim_server.py
```
Output: `[sim_server] Listening on tcp://*:5555`

**Terminal 2 — Start the MPC client:**
```bash
python ackermann_car/control/run_mpc_client.py
```
The MPC loop runs for 430 control steps (~43 s), then saves `results/trajectory_log.csv`.

---

## Full Run Commands

### 1. Main assignment — ZeroMQ MPC experiment
```bash
# Terminal 1
python ackermann_car/sim/sim_server.py

# Terminal 2
python ackermann_car/control/run_mpc_client.py

# After run completes:
python ackermann_car/scripts/plot_results.py
```

### 2. Generate S-curve road XML (required before MuJoCo demo)
```bash
python ackermann_car/scripts/generate_mujoco_road_xml.py
# writes ackermann_car/sim/mujoco_ackermann_generated.xml
```

### 3. MuJoCo visual demo (requires `pip install mujoco`)
```bash
python ackermann_car/scripts/run_mujoco_visual_demo.py
# Opens MuJoCo viewer; car follows S-curve road under MPC.
# Stops automatically when it reaches the road end.
# writes results/mujoco_visual_log.csv
```

### 4. Animate results from log (no MuJoCo needed)
```bash
# After run_mpc_client.py has produced trajectory_log.csv:
python ackermann_car/scripts/animate_results.py
# saves results/trajectory_animation.gif
# opens interactive Matplotlib window (close to exit)
```

The animation shows:
- Car (red box) following the S-curve road in real time
- Velocity arrow updating each frame
- Side panels: lateral error and steering command streamed live
- Camera follows the car with smooth lerp

To also export `.mp4` (requires `ffmpeg`), set `SAVE_MP4 = True` at the top
of `animate_results.py`.
