# PR_LAB4 — Feature EKF Map-Based Localization (Displacement Motion Model, Polar Observations)

**Course:** Probabilistic Robotics — IFROS / MIRS Masters, Universitat de Girona  
**Topic:** Feature EKF Map-Based Localization (FEKFMBL) with two variations: polar-stored features observed in polar, and Cartesian-stored features observed in polar

---

## Overview

This lab extends the FEKFMBL framework from the Cartesian observation lab to work with **polar coordinate observations** (`[ρ, θ]`). The displacement motion model and ICNN data association pipeline are identical — the only difference is how features are represented in the sensor, in the map, and in the observation model.

Two variations are implemented, each as a standalone entry point:

### Variation 1 — Polar Features, Polar Observations (`MBL_3DOFDDInputDisplacementMM_2DPolarFeatureOM.py`)

The map itself stores features in **polar coordinates** `[ρ, θ]`. The sensor also returns observations in polar coordinates. Since storage and observation share the same representation, `s2o` and `o2s` are both the identity — no coordinate conversion occurs in the observation pipeline.

The `PolarFeature.boxplus` operator converts the polar feature to Cartesian internally, applies the standard Cartesian compounding, then converts back to polar. The Jacobians chain through `J_c2p` accordingly.

### Variation 2 — Cartesian-Stored Features, Polar Observations (`MBL_3DOFDDInputDisplacementMM_2DCartesianFeaturePolarObservedOM.py`)

The map stores features in **Cartesian coordinates** `[x, y]` (same as the previous Cartesian lab), but the sensor returns observations in **polar coordinates** `[ρ, θ]`. The `Cartesian2DStoredPolarObservedMapFeature` class bridges the two representations:

- `s2o(v)` — converts a Cartesian map feature to polar for comparison with a polar observation (`c2p`)
- `o2s(v)` — converts a polar observation back to Cartesian for the inverse model (`p2c`)
- `J_s2o(v)` — Jacobian of `c2p`
- `J_o2s(v)` — Jacobian of `p2c`

The generic `hfj` formula `s2o(⊖^Nx_B ⊞ ^Nx_{Fj})` then automatically handles the conversion: the Cartesian map feature is compounded with the inverted robot pose (in Cartesian), then the result is converted to polar for the innovation.

The Jacobian chain for `Jhfjx` becomes:
```
Jhfjx = J_s2o(result) · J_1boxplus(⊖^Nx_B, ^Nx_{Fj}) · J_ominus(^Nx_B)
       = J_c2p · J_1boxplus_cartesian · J_ominus
```

---

## Comparison of the Two Variations

| Aspect | Variation 1 (Polar-Polar) | Variation 2 (Cartesian-Polar) |
|---|---|---|
| Entry point | `MBL_3DOFDDInputDisplacementMM_2DPolarFeatureOM.py` | `MBL_3DOFDDInputDisplacementMM_2DCartesianFeaturePolarObservedOM.py` |
| Map feature type | `PolarFeature` `[ρ, θ]` | `CartesianFeature` `[x, y]` |
| Sensor output | Polar `[ρ, θ]` | Polar `[ρ, θ]` |
| `s2o` | identity | `c2p` (Cartesian → Polar) |
| `o2s` | identity | `p2c` (Polar → Cartesian) |
| `MapFeature` subclass | `PolarMapFeature` | `Cartesian2DStoredPolarObservedMapFeature` |
| `Feature` class | `PolarFeature` | `CartesianFeature` |
| Map initialization | `c2p(CartesianFeature(...))` | `CartesianFeature(...)` |

In both cases the **motion model**, **FEKFMBL loop**, and **ICNN data association** are completely unchanged from the Cartesian lab.

---

## Repository Structure

```
PR_LAB4_MBL_DISPMM_POLAR_AND_CART_OM/
│
│  ── Entry points ──
├── MBL_3DOFDDInputDisplacementMM_2DPolarFeatureOM.py              # Variation 1: polar map, polar obs
├── MBL_3DOFDDInputDisplacementMM_2DCartesianFeaturePolarObservedOM.py  # Variation 2: Cartesian map, polar obs
│
│  ── Core localization stack ──
├── FEKFMBL.py                             # Feature EKF MBL: data association + localization loop
├── EKF_3DOFDifferentialDriveInputDisplacement.py  # Displacement motion model + compass measurement
├── MapFeature.py                          # MapFeature base + PolarMapFeature + Cartesian2DStoredPolarObservedMapFeature
├── Feature.py                            # CartesianFeature, PolarFeature (boxplus + Jacobians)
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
├── conversions.py                         # c2p, p2c, J_c2p, J_p2c and related helpers
├── GetEllipse.py                          # Uncertainty ellipse points for plotting
├── blockarray.py                          # Block-structured matrix helpers
├── Pose.py                               # Pose base class
```

---

## Key Concepts

### State vector (both variations) — 3 DOF
```
x_k = [x,  y,  ψ]^T
```

### Motion model — displacement input (both variations)
```
u_k = [Δx, Δy, Δψ]^T    (from wheel encoders)
x_k = x_{k-1} ⊕ u_k     (pose compounding)
```
Identical to the previous displacement labs.

### Feature observation model `hfj` — general form
```
z_{fi} = s2o( ⊖^Nx_B  ⊞  ^Nx_{Fj} ) + v_{fi}
```

- For **Variation 1**: `s2o` is identity, `^Nx_{Fj}` is a `PolarFeature`, and `boxplus` operates in polar space (via Cartesian internally).
- For **Variation 2**: `s2o = c2p` converts the Cartesian result to polar, `^Nx_{Fj}` is a `CartesianFeature`, and `boxplus` is the standard Cartesian operator.

### `PolarFeature.boxplus` implementation
Rather than deriving boxplus natively in polar space, the implementation converts to Cartesian first, applies the Cartesian boxplus, then converts the result back to polar:
```python
BxF_cartesian = p2c(BxF)
NxF_cartesian = BxF_cartesian.boxplus(NxB)
return c2p(NxF_cartesian)
```
The Jacobians follow the same chain rule: `J_c2p @ J_1boxplus_cartesian` and `J_c2p @ J_2boxplus_cartesian @ J_p2c`.

### `Cartesian2DStoredPolarObservedMapFeature` — coordinate bridge
The key additions over the base `MapFeature`:
```python
s2o(v)    = c2p(v)      # Cartesian feature → polar observation
o2s(v)    = p2c(v)      # polar observation → Cartesian for inverse model
J_s2o(v)  = J_c2p(v)
J_o2s(v)  = J_p2c(v)
GetFeatures() calls robot.ReadPolarFeature()
```
The rest of the FEKFMBL pipeline (`hfj`, `Jhfjx`, `DataAssociation`, `StackMeasurementsAndFeatures`) is inherited unchanged.

---

## Class Hierarchy

### Variation 1 — Polar-Polar
```
MapFeature
└── PolarMapFeature
      └── (mixed into MBL_3DOFDDInputDisplacementMM_2DPolarFeatureOM)

FEKFMBL
└── (mixed into MBL_3DOFDDInputDisplacementMM_2DPolarFeatureOM)

EKF_3DOFDifferentialDriveInputDisplacement
└── (mixed into MBL_3DOFDDInputDisplacementMM_2DPolarFeatureOM)
```

### Variation 2 — Cartesian-Polar
```
MapFeature
└── Cartesian2DStoredPolarObservedMapFeature
      └── (mixed into MBL_3DOFDDInputDisplacementMM_2DCartesianFeaturePolarObservedOM)

FEKFMBL
└── (mixed into MBL_3DOFDDInputDisplacementMM_2DCartesianFeaturePolarObservedOM)

EKF_3DOFDifferentialDriveInputDisplacement
└── (mixed into MBL_3DOFDDInputDisplacementMM_2DCartesianFeaturePolarObservedOM)
```

---

## Running the Lab

```bash
pip install roboticstoolbox-python numpy matplotlib scipy
cd PR_LAB4_MBL_DISPMM_POLAR_AND_CART_OM

# Variation 1: polar map features, polar observations
python MBL_3DOFDDInputDisplacementMM_2DPolarFeatureOM.py

# Variation 2: Cartesian map features, polar observations
python MBL_3DOFDDInputDisplacementMM_2DCartesianFeaturePolarObservedOM.py
```

---

## Parameters

| Parameter | Value | Description |
|---|---|---|
| `wheelRadius` | 0.1 m | Radius of each drive wheel |
| `wheelBase` | 0.5 m | Track width between wheels |
| `alpha` | 0.95 | Chi-squared confidence level for ICNN compatibility test |
| `x0` | `[0, 0, 0]^T` | Initial pose estimate |
| `P0` | `0_{3×3}` | Initial covariance |
| `dt` | 0.1 s | Simulation time step |
| `kSteps` | 5000 | Total simulation steps |
| `usk` | `[0.5, 0, 0.03]^T` | Constant velocity input to the simulated robot |
| Map `M` | 6 landmarks | World-frame positions of known features (Cartesian or Polar depending on variation) |
