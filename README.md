# PR_LAB4 — Feature EKF Map-Based Localization (Displacement Motion Model, Cartesian Features)

**Course:** Probabilistic Robotics — IFROS / MIRS Masters, Universitat de Girona  
**Topic:** Feature EKF Map-Based Localization (FEKFMBL) using a 2D Cartesian feature map, Cartesian point observations, and an input displacement motion model

---

## Overview

This lab is the first full implementation of **map-based localization**: the robot knows a pre-built map of point landmarks (Cartesian `[x, y]` features) and uses EKF to correct its pose estimate whenever it observes those landmarks.

The three pillars of this lab:

1. **Motion model — Input Displacement:** Same as the displacement EKF from LAB3. Encoder readings are converted to a body-frame displacement `u_k = [Δx, Δy, Δψ]^T` which drives the EKF prediction. The 3-DOF state remains `x_k = [x, y, ψ]^T`.

2. **Feature observations — Cartesian points:** The robot sensor returns the `[x, y]` position of nearby landmarks in the robot body frame. Both storage (in the map) and observation are in Cartesian coordinates — no coordinate conversion needed.

3. **Data association — ICNN:** The Individual Compatibility Nearest Neighbor algorithm matches each raw observation to the most likely map feature using the Mahalanobis distance with a Chi-squared compatibility test.

The result is a filter that dead-reckons between feature sightings and pulls its pose back toward the true trajectory whenever it sees a known landmark.

---

## Repository Structure

```
PR_LAB4_MBL_DISPMM_CARTMM/
│
│  ── Entry point ──
├── MBL_3DOFDDInputDisplacementMM_2DCartesianFeatureOM.py  # Top-level: run this
│
│  ── Core localization stack ──
├── FEKFMBL.py                             # Feature EKF MBL: data association + localization loop
├── EKF_3DOFDifferentialDriveInputDisplacement.py  # Displacement motion model + compass measurement
├── MapFeature.py                          # Feature observation/inverse models and Jacobians
├── Feature.py                            # CartesianFeature: boxplus operator and its Jacobians
│
│  ── EKF filter framework ──
├── EKF.py                                # EKF Prediction & Update
├── GaussianFilter.py                     # Abstract Gaussian filter interface
├── GFLocalization.py                     # Localization loop + logging + plotting
├── KF.py                                 # Kalman filter base (linear case)
│
│  ── Inherited from PR_LAB1 ──
├── DR_3DOFDifferentialDrive.py            # Dead reckoning (encoder conversion)
├── DifferentialDriveSimulatedRobot.py     # Ground-truth simulator + sensors
├── SimulatedRobot.py                      # Abstract robot base class
├── Pose3D.py                             # 3-DOF pose with ⊕ / ⊖ operators and Jacobians
├── Localization.py                        # Base localization loop
├── IndexStruct.py                         # State/simulation/observation index mapping
│
│  ── Utilities ──
├── conversions.py                         # Coordinate conversion helpers
├── GetEllipse.py                          # Uncertainty ellipse points for plotting
├── blockarray.py                          # Block-structured matrix helpers
├── Pose.py                               # Pose base class
│
│  ── Results ──
├── Results_freq=*.png                     # Per-state error plots at different feature observation frequencies
└── Trajectory_freq=*.png                  # XY trajectory plots at different observation frequencies
```

---

## Key Concepts

### State vector (3 DOF)
```
x_k = [x,  y,  ψ]^T
```
Identical to the displacement EKF in LAB3 — the MBL extension does not change the state dimensionality.

### Motion model `f(x_{k-1}, u_k)` — displacement input
Encoder readings are converted to a body-frame displacement:
```
u_k = [Δx, Δy, Δψ]^T    (from wheel encoders via kinematic chain)
x_k = x_{k-1} ⊕ u_k     (pose compounding via Pose3D.oplus)
```
Jacobians `Jfx` (3×3) and `Jfw` (3×3, rotation matrix) are the same as in LAB3.  
Process noise `Q_k` is propagated from encoder pulse noise through the kinematic chain.

### Feature observation model `hfj(xk, Fj)`
For a known map feature `^Nx_{Fj}` and predicted robot pose `^Nx_B` inside `x_k`:
```
z_{fi} = hfj(x_k) = s2o( ⊖^Nx_B  ⊞  ^Nx_{Fj} )
```
In Cartesian coordinates `s2o` is the identity, so this simplifies to:
```
z_{fi} = ^Nx_{Fj}.boxplus( x_k.ominus() )
       = F · (x_k ⊕ [^Bx_{Fj}^T, 0]^T)
```
where `F = [[1,0,0],[0,1,0]]` extracts the translation from the compounded 3-DOF pose.

The Jacobian `Jhfjx` (2×3) gives how the expected observation changes with the robot pose:
```
Jhfjx = J_s2o · J_1boxplus(⊖^Nx_B, ^Nx_{Fj}) · J_ominus(^Nx_B)
```

### `boxplus` operator (CartesianFeature)
The pose-feature compounding `^Nx_B ⊞ ^Bx_F` computes the world-frame position of a feature seen at `^Bx_F` from pose `^Nx_B`:
```
^Nx_F = F · (^Nx_B ⊕ [^Bx_F^T, 0]^T)
```
Jacobians `J_1boxplus` (2×3, w.r.t. robot pose) and `J_2boxplus` (2×2, w.r.t. feature position) are used for covariance propagation and update.

---

## Data Association — ICNN

The **Individual Compatibility Nearest Neighbor** algorithm runs every step that has feature observations:

1. **Build expected observations:** for each map feature `Fj`, compute `hfj(x_k_bar, Fj)` and its covariance `Phfj = Jhfjx @ Pk_bar @ Jhfjx.T`.

2. **For each raw observation `zfi`:** find the map feature `Fj` that minimizes the squared Mahalanobis distance:
   ```
   D²_ij = (zfi − hfj)^T · (Phfj + Rfi)^{-1} · (zfi − hfj)
   ```
   Accept the pairing only if `D²_ij ≤ χ²_{dof, α}` (Chi-squared test at confidence `α = 0.95`).

3. **Output:** association hypothesis vector `H` where `H[i] = j` means raw observation `i` is paired with map feature `j`, or `None` if no compatible match was found.

Paired observations go into the EKF update; unmatched observations are discarded (in SLAM they would become new feature candidates).

---

## Localization Loop (`FEKFMBL.Localize`)

Each time step:

```
1. uk, Qk = GetInput()                          # encoder displacement + noise
2. xk_bar, Pk_bar = Prediction(uk, Qk)          # EKF predict
3. zm, Rm, Hm, Vm = GetMeasurements()           # compass yaw (may be None)
4. zf, Rf = GetFeatures(xk_bar)                 # Cartesian feature observations from sensor
5. H = DataAssociation(xk_bar, Pk_bar, zf, Rf)  # ICNN pairing
6. zk, Rk, Hk, Vk = StackMeasurementsAndFeatures(zm, Rm, Hm, Vm, zf, Rf, H)
   # stacks compass + paired feature obs into one joint observation vector
7. xk, Pk = Update(zk, Rk, xk_bar, Pk_bar, Hk, Vk)  # EKF update
```

The stacked observation combines:
- **Compass measurement** `zm = [ψ]^T` with `Hm = [[0, 0, 1]]`  
- **Paired feature observations** `zf_p` with their per-feature Jacobians stacked vertically into `Hk`

If neither compass nor any features are available, the predicted state is kept unchanged.

---

## Class Hierarchy

```
GaussianFilter
└── EKF
      └── (mixed into EKF_3DOFDifferentialDriveInputDisplacement)

Localization
└── GFLocalization
      ├── (mixed into EKF_3DOFDifferentialDriveInputDisplacement)
      └── FEKFMBL
            └── (mixed into MBL_3DOFDDInputDisplacementMM_2DCartesianFeatureOM)

DR_3DOFDifferentialDrive     ← from PR_LAB1 (encoder → displacement conversion)
└── (mixed into EKF_3DOFDifferentialDriveInputDisplacement)

MapFeature
└── Cartesian2DMapFeature
      └── (mixed into MBL_3DOFDDInputDisplacementMM_2DCartesianFeatureOM)
```

`MBL_3DOFDDInputDisplacementMM_2DCartesianFeatureOM` inherits from all three branches simultaneously via Python multiple inheritance, combining the displacement motion model, the FEKFMBL data association loop, and the Cartesian feature observation model.

---

## Running the Lab

```bash
pip install roboticstoolbox-python numpy matplotlib scipy
cd PR_LAB4_MBL_DISPMM_CARTMM

# Feature EKF Map-Based Localization (displacement MM, Cartesian features):
python MBL_3DOFDDInputDisplacementMM_2DCartesianFeatureOM.py
```

---

## Parameters

| Parameter | Value | Description |
|---|---|---|
| `wheelRadius` | 0.1 m | Radius of each drive wheel |
| `wheelBase` | 0.5 m | Track width between wheels |
| `pulse_x_wheelTurns` | 4096 | Encoder resolution |
| `v_yaw_std` | 5° | Compass heading noise std deviation |
| `alpha` | 0.95 | Chi-squared confidence level for ICNN compatibility test |
| `x0` | `[0, 0, 0]^T` | Initial pose estimate |
| `P0` | `0_{3×3}` | Initial covariance (pose known exactly) |
| `dt` | 0.1 s | Simulation time step |
| `kSteps` | 5000 | Total simulation steps |
| `usk` | `[0.5, 0, 0.03]^T` | Constant velocity input to the simulated robot |
| Map `M` | 6 landmarks | Cartesian positions of known features in the world frame |
