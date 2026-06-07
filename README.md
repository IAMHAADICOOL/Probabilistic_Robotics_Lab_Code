# PR_LAB3 — EKF Localization with Constant Velocity Motion Model

**Course:** Probabilistic Robotics — IFROS / MIRS Masters, Universitat de Girona
**Topic:** Extended Kalman Filter (EKF) localization using a constant velocity motion model, with multi-sensor fusion (compass + wheel encoders treated as velocity observations)

---

## Overview

This lab extends the dead reckoning work from PR_LAB1 using an Extended Kalman Filter, but with a fundamentally different approach from the constant displacement version:

**The state vector is expanded to 6 DOF:** `x_k = [x, y, ψ, v_x, v_y, ω]^T`  
The robot's body-frame velocities are now *part of the estimated state*, not computed separately from encoders.

**The motion model has no external input.** The constant velocity assumption means the robot's velocity is predicted to remain unchanged between steps. Pose is propagated by integrating the velocity in the current state. This is fundamentally different from the displacement model where encoder readings drove the prediction.

**Wheel encoders are fused as velocity measurements in the update step**, not used as motion model inputs. The filter corrects the velocity component of its state by comparing the encoder-derived velocity against the expected velocity from the current state estimate. The compass provides a yaw correction as a second, independent measurement.

The result is a 6-state EKF that simultaneously estimates pose and velocity, with four possible sensor fusion configurations per step depending on sensor availability.

---

## Repository Structure

```
PR_LAB3_CONSTANT_VELOCITY/
│
│  ── Entry points ──
├── EKF_3DOFDifferentialDriveCtVelocity.py    # Core EKF: constant velocity model + multi-sensor update
├── MBL_3DOFDDCtVelocityMM_2DCartesianFeatureOM.py  # Map-based localization entry point
│
│  ── EKF filter stack (shared framework) ──
├── EKF.py                  # EKF Prediction & Update (extended Update signature)
├── GaussianFilter.py       # Abstract Gaussian filter interface
├── GFLocalization.py       # Localization loop + logging + plotting
├── KF.py                   # Kalman filter base (linear case)
│
│  ── Map-based localization framework ──
├── FEKFMBL.py              # Feature EKF Map-Based Localization (partially stubbed)
├── MapFeature.py           # Feature observation/inverse models and Jacobians
├── Feature.py              # CartesianFeature: boxplus operator and Jacobians
│
│  ── Inherited from PR_LAB1 ──
├── DR_3DOFDifferentialDrive.py        # Dead reckoning (encoder conversion used in GetMeasurements)
├── DifferentialDriveSimulatedRobot.py # Ground-truth simulator + encoder/compass sensors
├── SimulatedRobot.py                  # Abstract robot base class
├── Pose3D.py                          # 3-DOF pose with ⊕ / ⊖ operators and Jacobians
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
├── Results_freq=*.png      # Per-state error plots at different compass/encoder frequencies
└── Trajectory_freq=*.png   # XY trajectory plots at different sensor frequencies
```

---

## The Constant Velocity Motion Model — Key Concepts

### State vector (6 DOF)
```
x_k = [x,  y,  ψ,  v_x,  v_y,  ω]^T
        ←pose (η)→  ←body velocities (ν)→
```

### Motion model `f(x_{k-1}, u_k)` — no external input
The velocity is assumed constant between steps; pose is integrated from velocity:
```
ν_k     = ν_{k-1}                       (velocity unchanged)
u_k     = ν_{k-1} · dt                  (effective displacement)
η_k     = η_{k-1} ⊕ u_k                (pose compounded)
x_k     = [η_k^T,  ν_k^T]^T
```
`GetInput()` returns `u_k = 0` (no control input) and a 3×3 acceleration noise covariance `Q_k`:

| σ_u_dot | σ_v_dot | σ_r_dot |
|---|---|---|
| 0.1 m/s² | 0.01 m/s² | 1°/s² |

### Jacobians

**`Jfx`** — how the full 6-state propagates (6×6):
```
       ∂η_k/∂η     ∂η_k/∂ν
Jfx = [J_1⊕(η,u)  J_2⊕(η) · dt ]
      [  0_{3×3}       I_{3×3}   ]
```
Top-left: how the previous pose affects the new pose (from the `oplus` Jacobian w.r.t. the left argument).  
Top-right: how the previous velocity affects the new pose (via `J_2⊕ · dt`, the `oplus` Jacobian w.r.t. the displacement, scaled by dt).  
Bottom: velocity state propagates as identity.

**`Jfw`** — how the 3D acceleration noise `w_k` maps into the 6-state (6×3):
```
       ∂η_k/∂w
Jfw = [J_2⊕(η) · dt²/2]   (acceleration → pose, second-order)
      [    I · dt        ]   (acceleration → velocity, first-order)
```

---

## Multi-Sensor Measurement Fusion

This is the defining feature of this lab. `GetMeasurements()` returns a dynamically sized observation vector depending on which sensors fired:

### Case 1 — Both compass and encoders available
```
z_k = [ψ_compass, v_x, v_y, ω]^T      (4×1)
h(x_k) = x_k[2:6]                      (yaw + all velocities)
H_k = [[0, 0, 1, 0, 0, 0],             (compass → yaw)
        [0, 0, 0, 1, 0, 0],             (encoder → v_x)
        [0, 0, 0, 0, 1, 0],             (encoder → v_y)
        [0, 0, 0, 0, 0, 1]]             (encoder → ω)
R_k = block_diag(R_compass, Q_velocity)  (4×4)
```

### Case 2 — Compass only (no encoder reading this step)
```
z_k = [ψ_compass]                      (1×1)
H_k = [[0, 0, 1, 0, 0, 0]]
R_k = R_compass                         (1×1)
```

### Case 3 — Encoders only (no compass reading this step)
```
z_k = [v_x, v_y, ω]^T                 (3×1)
H_k = [[0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]]
R_k = Q_velocity                        (3×3)
```

### Case 4 — Neither sensor fires
```
z_k = None  →  skip update, keep predicted state
```

The encoder-derived velocity covariance `Q_velocity` is propagated from pulse noise through the kinematic chain using the same "magic table" approach as the displacement lab:
`Q_pulses → Q_wheel_velocities → Q_body_velocities`

### `EKF.Update` — extended signature
Because `h(x_k)` depends on which sensors are active, the Update method takes two extra boolean flags:
```python
Update(zk, Rk, xk_bar, Pk_bar, Hk, Vk, received_encoder_readings, received_heading_data)
```
Angle wrapping on the compass innovation is applied only when heading data was received.

---

## Key Differences from the Constant Displacement Lab

| Aspect | Constant Displacement (LAB3-CD) | Constant Velocity (this lab) |
|---|---|---|
| State dimension | 3 (pose only) | 6 (pose + velocity) |
| Encoder role | Motion model input (`u_k`) | Measurement in update step |
| `GetInput` | Reads encoders, computes displacement | Returns zero input + acceleration noise |
| `GetMeasurements` | Reads compass only | Reads compass + encoders (velocity obs.) |
| Motion model | `x_k = x_{k-1} ⊕ u_k` | `η_k = η_{k-1} ⊕ ν_{k-1}·dt`, `ν_k = ν_{k-1}` |
| Process noise `Q_k` | 3×3 displacement noise | 3×3 acceleration noise |
| Observation `H_k` | Fixed 1×3 (compass → yaw) | Dynamic 1×6 / 3×6 / 4×6 |
| Velocity estimate | Not tracked | Estimated and corrected by EKF |

---

## File Descriptions

### `EKF_3DOFDifferentialDriveCtVelocity.py`
The core student implementation. Inherits from `GFLocalization`, `DR_3DOFDifferentialDrive`, and `EKF`.

**`f(xk_1, uk)`** — Constant velocity prediction:
extracts `ν_{k-1}` from state, computes displacement `u = ν·dt`, compounds pose with `oplus`, returns `[η_k^T, ν_{k-1}^T]^T`.

**`Jfx(xk_1, uk)`** — 6×6 state transition Jacobian. Uses `Pose3D.J_1oplus` and `Pose3D.J_2oplus` for the pose-pose and velocity-pose blocks respectively.

**`Jfw(xk_1, uk)`** — 6×3 noise Jacobian. Maps 3D acceleration noise to pose (via `J_2oplus · dt²/2`) and velocity (via `I · dt`).

**`h(xk, received_encoder_readings, received_heading_data)`** — Adaptive observation model returning the appropriate state subvector based on active sensors.

**`GetInput()`** — Returns `u_k = 0` (constant velocity needs no input) and hardcoded `Q_k = diag(0.1², 0.01², 1°²)` (acceleration noise).

**`GetMeasurements()`** — Reads compass and encoders, converts encoder pulses to `[v_x, v_y, ω]`, propagates noise through the kinematic chain, then assembles `(z_k, R_k, H_k, V_k)` for whichever combination of sensors fired.

### `MBL_3DOFDDCtVelocityMM_2DCartesianFeatureOM.py`
Top-level entry point combining `Cartesian2DMapFeature`, `FEKFMBL`, and `EKF_3DOFDifferentialDriveCtVelocity` via multiple inheritance. Sets up 6 landmarks and calls `LocalizationLoop`.

### `EKF.py`
Identical prediction step to the displacement lab. The **Update** method has an extended signature with `received_encoder_readings` and `received_heading_data` flags, forwarded to `h()` to select the correct observation subvector.

### `GFLocalization.py`
Same localization loop as the displacement lab, but unpacks 6 return values from `GetMeasurements()` (the extra two sensor-availability flags) and passes them through to `Update`.

### Shared framework files
`GaussianFilter.py`, `MapFeature.py`, `Feature.py`, `FEKFMBL.py`, `GetEllipse.py`, `blockarray.py`, `conversions.py`, `Pose.py` — identical in purpose to the displacement lab. See the lab3 (displacement) README for descriptions.

---

## Class Hierarchy

```
GaussianFilter
└── EKF
      └── (mixed into EKF_3DOFDifferentialDriveCtVelocity)

Localization
└── GFLocalization
      ├── (mixed into EKF_3DOFDifferentialDriveCtVelocity)
      └── FEKFMBL
            └── (mixed into MBL_3DOFDDCtVelocityMM_2DCartesianFeatureOM)

DR_3DOFDifferentialDrive     ← from PR_LAB1 (encoder conversion reused in GetMeasurements)
└── (mixed into EKF_3DOFDifferentialDriveCtVelocity)

MapFeature
└── Cartesian2DMapFeature
      └── (mixed into MBL_3DOFDDCtVelocityMM_2DCartesianFeatureOM)
```

---

## Running the Lab

```bash
pip install roboticstoolbox-python numpy matplotlib scipy
cd PR_LAB3_CONSTANT_VELOCITY

# 6-state constant velocity EKF (compass + encoder fusion):
python EKF_3DOFDifferentialDriveCtVelocity.py

# With Cartesian map-feature framework:
python MBL_3DOFDDCtVelocityMM_2DCartesianFeatureOM.py
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
| `σ_u_dot` | 0.1 m/s² | Process noise: forward acceleration |
| `σ_v_dot` | 0.01 m/s² | Process noise: lateral acceleration |
| `σ_r_dot` | 1°/s² | Process noise: angular acceleration |
| `x0` | `[0,0,0,0,0,0]^T` | Initial state (pose + zero velocity) |
| `P0` | diag(0, 0, 0, 0.5², 0², 0.05²) | Initial covariance (velocity uncertain) |
| `dt` | 0.1 s | Simulation time step |
| `kSteps` | 5000 | Total simulation steps |
