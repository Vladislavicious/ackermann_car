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
    mujoco_ackermann.xml            ← base MuJoCo scene
    mujoco_ackermann_generated.xml  ← generated S-curve MuJoCo road scene
    ackermann_env.py                ← simulation environment and backend selection
    road.py                         ← Bézier centre-line generator
    corridor_limits.py              ← boundary checker
    sim_server.py                   ← ZeroMQ REP server (Terminal 1)
    disturbance_model.py            ← wind / noise disturbance injector
  control/
  scripts/
    generate_mujoco_road_xml.py     ← S-curve road XML generator
  results/
    .gitkeep
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
source .venv/bin/activate      # Windows PowerShell: .\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Mujoco via installation guide from:
https://github.com/google-deepmind/mujoco


---
