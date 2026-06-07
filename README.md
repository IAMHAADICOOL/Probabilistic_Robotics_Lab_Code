# PR_LAB1 — Differential Drive Robot Simulation & Dead Reckoning Localization

**Course:** Probabilistic Robotics — IFROS / MIRS Masters, Universitat de Girona
**Topic:** Dead Reckoning using wheel encoder odometry for a 3-DOF differential drive robot

---

## Overview

This lab simulates a differential drive mobile robot moving through a 2D environment and estimates its pose using **dead reckoning** — integrating noisy wheel encoder readings over time to track position and heading, with no external corrections or landmarks used.

The simulation runs for 5000 time steps at dt = 0.1 s. The robot is driven along a trajectory defined by a constant forward velocity and angular rate. The ground-truth pose (from the physics simulation) and the dead-reckoned estimate are plotted together so drift can be observed.

---

## Repository Structure

```
PR_LAB1/
├── main.py                          # Entry point — wires up the simulation and runs the localization loop
├── DR_3DOFDifferentialDrive.py      # Dead reckoning algorithm (encoder odometry → pose estimate)
├── DifferentialDriveSimulatedRobot.py  # Ground-truth physics simulator + encoder/compass sensors
├── SimulatedRobot.py                # Abstract base class for all robot simulators
├── Pose3D.py                        # 3-DOF pose representation with ⊕ / ⊖ operators
├── Localization.py                  # Base class for the localization loop
├── IndexStruct.py                   # Named tuple for mapping state/simulation/observation indices
└── Lab0_compounding.ipynb           # Lab 0: Pose compounding notebook (prerequisite)
```

---

## File Descriptions

### `main.py`
Entry point. Creates the simulated environment:
- Defines a map of 6 2D point features (not used for localization in this lab — carried over for the lab framework).
- Instantiates `DifferentialDriveSimulatedRobot` and `DR_3DOFDifferentialDrive`.
- Calls `LocalizationLoop` with a constant input velocity `[0.5, 0, 0.03]^T` (forward speed, lateral speed, yaw rate) to run the simulation.

### `DR_3DOFDifferentialDrive.py`
Implements **dead reckoning** for a 3-DOF differential drive robot. Inherits from `Localization`.

- **`GetInput()`** — Reads left/right encoder pulse counts from the simulated robot.
- **`Localize(xk_1, uk)`** — The motion model (core of dead reckoning):
  1. Converts pulse counts → wheel angular displacements → left/right linear velocities.
  2. Computes forward velocity `v = (v_R + v_L) / 2` and angular velocity `w = (v_R - v_L) / wheelBase`.
  3. Propagates the previous pose estimate via the `oplus` pose compounding operator: `η_k = η_{k-1} ⊕ (ν_k · dt)`.
  4. Returns the updated state `[x, y, ψ, v, 0, w]^T`.

  If no encoder reading is available (rate limiting), the previous velocity is reused.

### `DifferentialDriveSimulatedRobot.py`
Ground-truth physics simulator for the differential drive robot. Inherits from `SimulatedRobot`.

- **`fs(xsk_1, usk)`** — Simulates true robot motion using a first-order velocity tracking model with additive Gaussian acceleration noise (`Q_sk`). Updates position via `oplus` and velocity via a gain-based approach: `ν_k = ν_{k-1} + K(ν_d − ν_{k-1}) + w_k·dt`.
- **`ReadEncoders()`** — Simulates wheel encoder readings. Converts the true wheel velocities to pulse counts using the encoder resolution (1024 pulses/turn), then adds Gaussian noise (`Re = diag(22², 22²)`). Rate-limited to `encoder_reading_frequency`.
- **`ReadCompass()`** — Simulates a noisy compass (heading sensor). Not used in this lab but available in the framework.
- **`PlotRobot()`** — Updates the live animation icon at the current simulated pose.

### `SimulatedRobot.py`
Abstract base class for all robot simulators in this lab series.
- Sets up the matplotlib animation figure, trajectory buffers, and the vehicle icon overlay.
- Defines the `fs()` interface (overridden by each robot type) and a helper `_PlotSample()` for drawing uncertainty samples.

### `Pose3D.py`
Represents a 3-DOF robot pose `[x, y, ψ]^T` as a NumPy ndarray subclass.

- **`oplus(AxB, BxC)`** — Pose composition (frame chaining). Applies the rotation of frame A→B to BxC's translation, then sums headings. Wraps the result to `[-π, π]`.
- **`ominus(AxB)`** — Inverse pose composition. Returns the pose of A expressed in B's frame.

These operators are the building blocks of the dead-reckoning integration step.

### `Localization.py`
Base class for the localization algorithm loop.

- **`LocalizationLoop(x0, usk)`** — Runs the simulation for `kSteps` iterations: simulates the robot (`fs`), reads input (`GetInput`), estimates the pose (`Localize`), and updates the trajectory plot.
- **`PlotTrajectory()`** — Live-plots the estimated (blue) trajectory every `visualizationInterval` steps.

### `IndexStruct.py`
A lightweight `NamedTuple` with fields `(state, simulation, observation)`. Maps elements of the estimated state vector to their counterparts in the simulation and observation vectors, used by the logging and plotting framework.

### `Lab0_compounding.ipynb`
A Jupyter notebook covering **Lab 0: Pose Compounding**. Walks through the mathematics and implementation of the `oplus` and `ominus` operators in 2D, serving as the prerequisite for understanding how poses are chained in the dead-reckoning integrator.

---

## Key Concepts

| Concept | Where it appears |
|---|---|
| Pose compounding (`⊕` / `⊖`) | `Pose3D.oplus`, `Pose3D.ominus` |
| Dead reckoning via encoder odometry | `DR_3DOFDifferentialDrive.Localize` |
| Encoder pulse → velocity conversion | `DR_3DOFDifferentialDrive.Localize`, `DifferentialDriveSimulatedRobot.ReadEncoders` |
| Simulated sensor noise | `DifferentialDriveSimulatedRobot.Re` (encoders), `Qsk` (motion) |
| Drift accumulation over time | Observable by comparing ground-truth vs estimated trajectory in the live plot |

---

## Running the Lab

```bash
pip install roboticstoolbox-python numpy matplotlib scipy
cd PR_LAB1
python main.py
```

A matplotlib window will open showing:
- The **orange** dots: ground-truth robot trajectory (from the physics simulation).
- The **blue** dots: dead-reckoned estimated trajectory.

Drift between the two grows over time, illustrating why dead reckoning alone is insufficient for long-horizon localization.

---

## Parameters

| Parameter | Value | Description |
|---|---|---|
| `wheelRadius` | 0.1 m | Radius of each wheel |
| `wheelBase` | 0.5 m | Distance between the two wheels |
| `pulse_x_wheelTurns` | 4096 (DR) / 1024 (sim) | Encoder resolution (pulses per full wheel revolution) |
| `encoder_reading_frequency` | 1 Hz | How often encoder readings are sampled |
| `Qsk` | diag(0.1², 0.01², 1°²) | Motion model noise covariance |
| `Re` | diag(22², 22²) | Encoder measurement noise covariance |
| `dt` | 0.1 s | Simulation time step |
| `kSteps` | 5000 | Total simulation steps |
