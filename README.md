# PR_LAB3 — EKF Localization with Constant Displacement Motion Model

**Course:** Probabilistic Robotics — IFROS / MIRS Masters, Universitat de Girona
**Topic:** Extended Kalman Filter (EKF) localization using an input displacement motion model, with compass-based measurement updates and a map-based localization framework

---

## Overview

This lab extends the dead reckoning work from PR_LAB1 by replacing the open-loop integrator with a full **Extended Kalman Filter (EKF)**. The key differences from LAB1 are:

- **Motion model input changes from velocity to displacement.** Instead of tracking velocity and integrating it, the EKF takes the body-frame displacement `u_k = [Δx, Δy, Δψ]^T` directly as input, computed from wheel encoder readings scaled by `dt`.
- **Covariance is propagated.** The EKF tracks not just the pose estimate but its uncertainty (covariance matrix `P_k`), linearizing the non-linear motion model via Jacobians to propagate it.
- **A compass sensor provides measurement updates.** At each step the filter reads a noisy yaw measurement from a simulated compass and uses the EKF update step to correct the heading estimate and reduce uncertainty.
- **A map-based localization framework is scaffolded.** The `FEKFMBL` and `MapFeature` classes lay the groundwork for fusing feature observations (2D Cartesian landmarks) into the filter, with several methods left as `TODO` stubs for the next lab.

The simulation runs for 5000 steps at dt = 0.1 s and produces trajectory plots and per-state estimation error plots at different compass reading frequencies.

---

## Repository Structure

```
PR_LAB3_EKF_CONSTANT_DISPLACEMENT/
│
│  ── Entry point ──
├── MBL_3DOFDDInputDisplacementMM_2DCartesianFeatureOM.py   # Top-level runner (map-based)
├── EKF_3DOFDifferentialDriveInputDisplacement.py           # Core EKF: motion + measurement model
│
│  ── EKF filter stack ──
├── EKF.py                  # EKF Prediction & Update equations
├── GaussianFilter.py       # Abstract Gaussian filter interface
├── GFLocalization.py       # Localization loop + logging + plotting for Gaussian filters
├── KF.py                   # Kalman filter base (linear case)
│
│  ── Map-based localization framework ──
├── FEKFMBL.py              # Feature EKF Map-Based Localization (partially stubbed)
├── MapFeature.py           # Feature observation/inverse models and Jacobians
├── Feature.py              # CartesianFeature: boxplus operator and Jacobians
│
│  ── Inherited from PR_LAB1 ──
├── DR_3DOFDifferentialDrive.py        # Dead reckoning (GetInput still used here)
├── DifferentialDriveSimulatedRobot.py # Ground-truth simulator + encoder/compass sensors
├── SimulatedRobot.py                  # Abstract robot base class
├── Pose3D.py                          # 3-DOF pose with ⊕ / ⊖ operators
├── Localization.py                    # Base localization loop
├── IndexStruct.py                     # State/simulation/observation index mapping
│
│  ── Utilities ──
├── conversions.py          # Coordinate conversion helpers
├── GetEllipse.py           # Uncertainty ellipse points for plotting
├── blockarray.py           # Block-structured matrix helpers
├── Pose.py                 # Pose base class
│
│  ── Results ──
├── Results_freq=*.png      # Per-state error plots at different compass frequencies
└── Trajectory_freq=*.png   # XY trajectory plots at different compass frequencies
```

---

## File Descriptions

### `MBL_3DOFDDInputDisplacementMM_2DCartesianFeatureOM.py`
Top-level entry point for the **map-based** version. Wires together three parent classes via multiple inheritance:
- `Cartesian2DMapFeature` — reads 2D Cartesian landmark observations from the robot.
- `FEKFMBL` — Feature EKF map-based localization framework.
- `EKF_3DOFDifferentialDriveInputDisplacement` — the core EKF with displacement motion model.

Sets up 6 landmark features in the world frame, creates the simulated robot, and calls `LocalizationLoop(x0, P0, usk)`.

### `EKF_3DOFDifferentialDriveInputDisplacement.py`
The core student implementation. Inherits from `GFLocalization`, `DR_3DOFDifferentialDrive`, and `EKF`.

**`GetInput()`**
- Reads left/right encoder pulses from the simulated robot.
- Converts pulses → wheel velocities → forward speed `v` and angular rate `w`.
- Computes the body-frame displacement: `u_k = [v·dt, 0, w·dt]^T`.
- Propagates encoder noise through the kinematic chain using the "magic table" (covariance propagation):
  `Q_pulses → Q_wheel_velocity → Q_body_velocity → Q_displacement`
- Returns `(u_k, Q_displacement)`.

**`f(xk_1, uk)`** — Motion model
```
x̂_k = x_{k-1} ⊕ u_k
```
The displacement `u_k` expressed in the body frame is compounded onto the previous pose using the `Pose3D.oplus` operator.

**`Jfx(xk_1, uk)`** — Jacobian of `f` w.r.t. the state:
```
∂f/∂x = [[1, 0, -Δx·sin(ψ) - Δy·cos(ψ)],
          [0, 1,  Δx·cos(ψ) - Δy·sin(ψ)],
          [0, 0,  1                      ]]
```

**`Jfw(xk_1, uk)`** — Jacobian of `f` w.r.t. the noise (rotation matrix from body to world):
```
∂f/∂w = [[cos(ψ), -sin(ψ), 0],
          [sin(ψ),  cos(ψ), 0],
          [0,       0,      1]]
```

**`h(xk)`** — Observation model: returns `xk[2]` (the yaw angle), since the compass directly observes heading.

**`GetMeasurements()`**
- Reads a compass yaw measurement from the robot.
- Returns `(z_k, R_k, H_k, V_k)` where `H_k = [[0, 0, 1]]` (the compass observes yaw only) and `V_k = [[1]]`.

### `EKF.py`
Implements the EKF **Prediction** and **Update** equations.

**`Prediction(uk, Qk)`**
```
x̂_k|k-1 = f(x_{k-1}, u_k)
P_k|k-1 = Jfx · P_{k-1} · Jfx^T + Jfw · Q_k · Jfw^T
```

**`Update(zk, Rk, xk_bar, Pk_bar, Hk, Vk)`**
```
K = P_k|k-1 · H^T · (H · P_k|k-1 · H^T + V · R_k · V^T)^-1
x_k = x̂_k|k-1 + K · wrap(z_k - h(x̂_k|k-1))
P_k = (I - K·H) · P_k|k-1
```
Angle wrapping is applied to the innovation to avoid discontinuities at ±π.

### `GFLocalization.py`
Extends `Localization` with a Gaussian-filter-specific loop:
- `LocalizationLoop` calls `fs` (simulate), then `Localize` (predict + update) for each step.
- `Localize` orchestrates `GetInput → Prediction → GetMeasurements → Update`.
- Logs ground truth, estimates, covariances, and predictions to arrays.
- `PlotState` produces per-DOF plots of: estimate + 3σ bounds vs. ground truth, estimation error + 3σ envelope, and an error histogram.
- `PlotXY` plots the XY trajectory.
- `PlotUncertainty` draws the live uncertainty ellipse around the robot pose every `visualizationInterval` steps.

### `GaussianFilter.py`
Minimal abstract base class defining the `Prediction` and `Update` interface. All filter variants (`KF`, `EKF`) implement these.

### `FEKFMBL.py`
Scaffold for **Feature EKF Map-Based Localization**. Extends `GFLocalization` and `MapFeature`. Most methods (`h`, `hm`, `ICNN`, `DataAssociation`, `StackMeasurementsAndFeatures`, `SplitFeatures`, `Localize`) are left as `TODO` stubs — implementing data association and the joint observation model is the task for the subsequent lab. Plotting helpers for feature observation ellipses and expected feature observation ellipses are fully implemented.

### `MapFeature.py`
Provides the mathematical interface for landmark observations:
- `hfj(xk, Fj)` — expected observation of feature `Fj` from pose `xk`: `s2o(⊖xk ⊕ M[Fj])`.
- `Jhfjx`, `Jhfv` — Jacobians of the feature observation function.
- `g(xk, BxFj)` — inverse observation model: `xk ⊕ o2s(BxFj)`.
- `Jgx`, `Jgv` — Jacobians of the inverse model.
- `Cartesian2DMapFeature` subclass: overrides `GetFeatures` to call the robot's Cartesian feature sensor.

### `Feature.py`
Defines the `Feature` interface (abstract `boxplus`, Jacobians, `ToCartesian`) and implements `CartesianFeature` as a NumPy ndarray subclass with:
- `boxplus(NxB)` — transforms a body-frame feature position to the world frame: `F · (NxB ⊕ BxF)` where `F` projects out the position dimensions.
- `J_1boxplus`, `J_2boxplus` — Jacobians w.r.t. robot pose and feature position.

### Utilities
| File | Purpose |
|---|---|
| `conversions.py` | Angle and coordinate conversion helpers |
| `GetEllipse.py` | Computes 2D uncertainty ellipse points from a mean and covariance |
| `blockarray.py` | Helpers for indexing block-structured stacked observation vectors |
| `Pose.py` | Pose base class used by the Feature framework |

---

## Key Concepts

| Concept | Where it appears |
|---|---|
| Input displacement motion model `u_k = ν·dt` | `EKF_3DOFDifferentialDriveInputDisplacement.GetInput`, `f` |
| EKF covariance prediction via Jacobians | `EKF.Prediction`, `EKF_3DOFDifferentialDriveInputDisplacement.Jfx`, `Jfw` |
| Compass-based yaw update | `EKF_3DOFDifferentialDriveInputDisplacement.h`, `GetMeasurements` |
| Kalman gain + innovation update | `EKF.Update` |
| Angle-wrapping in innovation | `EKF.Update` (`wrap_angle`) |
| Noise propagation through kinematic chain | `EKF_3DOFDifferentialDriveInputDisplacement.GetInput` (magic table) |
| Uncertainty ellipse visualization | `GFLocalization.PlotUncertainty`, `GetEllipse` |
| Feature observation model + Jacobians | `MapFeature.hfj`, `Jhfjx`, `g`, `Jgv` |
| Pose-feature compounding (boxplus) | `Feature.boxplus`, `CartesianFeature.boxplus` |

---

## Effect of Compass Reading Frequency

The lab includes result plots at four compass reading frequencies:

| Frequency | Behaviour |
|---|---|
| `1/500` Hz (very rare) | Essentially no compass updates; behaves close to dead reckoning, heading drifts |
| `0.1` Hz | Occasional corrections; visible yaw drift between updates |
| `1` Hz | Good heading tracking; position estimate tightens significantly |
| `10` Hz | Near-continuous correction; 3σ bounds remain narrow throughout |

These results illustrate how sensor update frequency directly controls uncertainty growth — a core insight of the EKF framework.

---

## Running the Lab

```bash
pip install roboticstoolbox-python numpy matplotlib scipy
cd PR_LAB3_EKF_CONSTANT_DISPLACEMENT

# Compass-corrected EKF only (no map features):
python EKF_3DOFDifferentialDriveInputDisplacement.py
```

---

## Class Hierarchy

```
GaussianFilter
├── KF
└── EKF
      └── (mixed into EKF_3DOFDifferentialDriveInputDisplacement)

Localization
└── GFLocalization
      ├── (mixed into EKF_3DOFDifferentialDriveInputDisplacement)
      └── FEKFMBL
            └── (mixed into MBL_3DOFDDInputDisplacementMM_2DCartesianFeatureOM)

DR_3DOFDifferentialDrive     ← from PR_LAB1
└── (mixed into EKF_3DOFDifferentialDriveInputDisplacement)

MapFeature
├── Cartesian2DMapFeature
│     └── (mixed into MBL_3DOFDDInputDisplacementMM_2DCartesianFeatureOM)
└── (provides feature obs model to FEKFMBL)
```

---

## Parameters

| Parameter | Value | Description |
|---|---|---|
| `wheelRadius` | 0.1 m | Radius of each drive wheel |
| `wheelBase` | 0.5 m | Track width between wheels |
| `pulse_x_wheelTurns` | 1024 (sim) / 4096 (DR) | Encoder resolution |
| `Re` | diag(22², 22²) | Encoder measurement noise covariance |
| `v_yaw_std` | 5° | Compass heading noise std deviation |
| `dt` | 0.1 s | Simulation time step |
| `kSteps` | 5000 | Total simulation steps |
| `alpha` | 0.95 | Chi-squared confidence level for data association |
