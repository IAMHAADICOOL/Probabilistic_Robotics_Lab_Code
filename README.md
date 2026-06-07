# PR_LAB4_SLAM — Feature EKF Simultaneous Localization and Mapping (FEKFSLAM)

**Course:** Probabilistic Robotics — IFROS / MIRS Masters, Universitat de Girona  
**Topic:** Simultaneous Localization and Mapping using the Feature Extended Kalman Filter (FEKFSLAM) with Cartesian features and a displacement motion model

---

## Overview

This lab implements **FEKFSLAM**: the robot starts with no prior map knowledge and simultaneously builds a map of Cartesian point features while localizing itself within it. In contrast to the map-based localization (MBL) labs — where a complete map was provided at initialization — here the map begins empty and grows dynamically as the robot observes previously unseen features.

The algorithm is built on top of the FEKFMBL (Feature EKF Map-Based Localization) stack. The key additions for SLAM are:

1. **Dynamic state vector growth** — each newly observed feature is appended to the state vector and its cross-correlations with the robot pose and existing features are properly initialized.
2. **SLAM-aware prediction** — the EKF prediction only propagates the robot pose block through the motion model Jacobians; feature blocks remain frozen.
3. **Inverse observation model** — when an observation cannot be matched to any known feature, the inverse model `g(x_B, z_i)` converts it to a world-frame position estimate that seeds the new feature's entry in the map.

---

## Key Concepts

### State vector — grows over time
```
x_k = [^Nx_B^T,  ^Nx_{F1}^T,  ...,  ^Nx_{Fn}^T]^T
```
- `^Nx_B` — robot pose (3 DOF: `[x, y, ψ]`)
- `^Nx_{Fi}` — world-frame position of the *i*-th mapped feature (2 DOF: `[x, y]`)
- The state dimension starts at 3 and grows by 2 for each newly discovered feature.

### Motion model — displacement input
```
u_k = [Δx, Δy, Δψ]^T    (from wheel encoders)
x_k = x_{k-1} ⊕ u_k     (pose compounding via Pose3D)
```

### Observation model `hfj`
```
z_{fi} = s2o( ⊖^Nx_B  ⊞  ^Nx_{Fj} ) + v_{fi}
```
Both `^Nx_B` and `^Nx_{Fj}` are read from the current state vector (unlike MBL where `^Nx_{Fj}` came from a static map). For Cartesian features `s2o` is the identity — storage and observation share the same representation.

### SLAM prediction (FEKFSLAM.Prediction)
Only the robot-pose block is propagated through `Jfx` and `Jfw`. Feature positions are unchanged:
```
P_RR_bar = Jfx @ P_RR @ Jfx^T + Jfw @ Qk @ Jfw^T
P_RM_bar = Jfx @ P_RM       # robot–feature cross-covariance
P_MM_bar = P_MM             # feature–feature covariance frozen
```

### Adding new features (FEKFSLAM.AddNewFeatures)
For each observation `z_i` not matched to any existing map feature:
1. Compute the world-frame position via the inverse model: `^Nx_{Fnew} = g(x_B, z_i)`
2. Append it to `self.M` and extend `xk` by 2 dimensions.
3. Compute and append the new rows/columns to `Pk`:
```
P_new_new = J1 @ P_B @ J1^T + J2 @ Ri @ J2^T
P_new_old = J1 @ P_{robot+all}
```
where `J1 = Jgx` (w.r.t. robot pose) and `J2 = Jgv` (w.r.t. observation noise).

### Data association — ICNN
Same individual-compatibility nearest-neighbour test as FEKFMBL, using Mahalanobis distance and a Chi-squared gate. SLAM uses `alpha = 0.99` (tighter than MBL's 0.95) to avoid adding spurious features prematurely.

---

## Class Hierarchy

```
GFLocalization ──── LocalizationLoop, Log, PlotState
    └── FEKFMBL ─── DataAssociation, Localize, hfj (from MapFeature)
          └── FEKFSLAM ── Prediction (SLAM-aware), AddNewFeatures, Localize (extends MBL + new features)
                └── FEKFSLAM_3DOFDD_InputVelocityMM_2DCartesianFeatureOM   ← entry point

FEKFSLAMFeature ─── hfj, Jhfjx  (reads features from state vector instead of static map)
    └── FEKFSLAM2DCartesianFeature (also inherits Cartesian2DMapFeature)
          └── FEKFSLAM_3DOFDD_InputVelocityMM_2DCartesianFeatureOM   ← entry point

EKF_3DOFDifferentialDriveInputDisplacement ── Prediction (base), f, Jfx, Jfw
    └── FEKFSLAM_3DOFDD_InputVelocityMM_2DCartesianFeatureOM   ← entry point
```

The entry-point class uses Python multiple inheritance (MRO) to compose the above three chains into a single runnable filter.

---

## Repository Structure

```
PR_LAB4_SLAM/
│
│  ── Entry point ──
├── FEKFSLAM_3DOFDD_InputVelocityMM_2DCartesianFeatureOM.py   # main file
│
│  ── SLAM core ──
├── FEKFSLAM.py                       # SLAM extensions: AddNewFeatures, SLAM-aware Prediction, Localize
├── FEKFSLAMFeature.py                # hfj / Jhfjx reading features from state vector
│
│  ── MBL base (shared with previous labs) ──
├── FEKFMBL.py                        # Data association + MBL localization loop
├── MapFeature.py                     # MapFeature base + Cartesian2DMapFeature
├── Feature.py                        # CartesianFeature (boxplus + Jacobians)
│
│  ── Motion model ──
├── EKF_3DOFDifferentialDriveInputDisplacement.py   # Displacement MM + compass measurement
│
│  ── EKF framework ──
├── EKF.py                            # Prediction & Update
├── GaussianFilter.py                 # Abstract Gaussian filter interface
├── GFLocalization.py                 # Localization loop + logging + plotting
├── KF.py                             # Kalman filter base
│
│  ── Robot simulation ──
├── DR_3DOFDifferentialDrive.py       # Dead reckoning (encoder conversion)
├── DifferentialDriveSimulatedRobot.py  # Ground-truth simulator + sensors
├── SimulatedRobot.py                 # Abstract robot base
├── Pose3D.py                         # 3-DOF pose with ⊕ / ⊖ operators and Jacobians
├── Localization.py                   # Base localization loop
├── IndexStruct.py                    # State/simulation/observation index mapping
│
│  ── Utilities ──
├── conversions.py                    # c2p, p2c, and related coordinate helpers
├── GetEllipse.py                     # Uncertainty ellipse for plotting
├── blockarray.py                     # Block-structured matrix helpers
├── Pose.py                           # Pose base class
```

---

## SLAM vs. MBL — Key Differences

| Aspect | FEKFMBL (previous labs) | FEKFSLAM (this lab) |
|---|---|---|
| Initial map | Full known map provided | Empty `[]` — no prior knowledge |
| State vector size | Fixed (3 DOF robot only) | Grows by 2 for each new feature |
| `hfj` / `Jhfjx` | Read feature from static `self.M` | Read feature from state vector |
| Prediction | Full `Pk` propagated | Only robot block propagated; features frozen |
| Unmatched observations | Discarded | Trigger `AddNewFeatures`, extending `xk` and `Pk` |
| `alpha` (ICNN gate) | 0.95 | 0.99 (tighter, to avoid spurious features) |
| Map landmarks in `__main__` | Ground-truth AND filter initialization | Ground-truth only (simulator); filter starts empty |

---

## Running the Lab

```bash
pip install roboticstoolbox-python numpy matplotlib scipy
cd PR_LAB4_SLAM

python FEKFSLAM_3DOFDD_InputVelocityMM_2DCartesianFeatureOM.py
```

---

## Parameters

| Parameter | Value | Description |
|---|---|---|
| `wheelRadius` | 0.1 m | Radius of each drive wheel |
| `wheelBase` | 0.5 m | Track width between wheels |
| `alpha` | 0.99 | Chi-squared confidence level for ICNN compatibility test |
| `x0` | `[0, 0, 0]^T` | Initial robot pose estimate |
| `P0` | `0_{3×3}` | Initial pose covariance (robot starts with known pose) |
| `dt` | 0.1 s | Simulation time step |
| `kSteps` | 5000 | Total simulation steps |
| `usk` | `[0.5, 0, 0.03]^T` | Constant velocity input to the simulated robot |
| Map `M` | 6 Cartesian landmarks | Used by the simulator for ground truth only — NOT passed to the filter |
