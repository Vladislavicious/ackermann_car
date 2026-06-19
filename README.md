# Issue #39 — Ackermann Car Road Centre-Line Tracking with MPC

## Task Description

Implement an autonomous Ackermann car that tracks the centre line of a road
using **Model Predictive Control (MPC)**. The system runs as two separate
processes communicating over **ZeroMQ**. The robot is constrained to planar
2D motion with three allowed DOF: **x**, **y**, and **yaw**, while **z**,
**roll**, and **pitch** are fixed.

