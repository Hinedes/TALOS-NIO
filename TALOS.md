> [!NOTE]
> ## Changelog / Reality Check — March 17, 2026
> 1. **Overlord output semantics:** `SpectralMLP` predicts **mean local velocity (m/s)** plus a **log-variance head** (`head_cov`). Quaternion head is not present in the model.
> 2. **Scheduler update:** Training uses `ReduceLROnPlateau` (validation loss driven), not `OneCycleLR`.
> 3. **ZARU quarantine:** `update_zaru()` uses Joseph-form covariance update and only updates gyro bias states (9:12).
> 4. **Runtime guardrails:** `bulwark.py` hard-zeros implausible local velocity predictions before ESKF injection.

# TALOS NIO — Neural-Inertial Odometry Pipeline
### Code-Synced State (Current Repository)
**Ground Truth State: March 17, 2026**

---

## 1. Mission

TALOS NIO bounds inertial drift using a hybrid pipeline:
- Fast 100Hz ESKF propagation on IMU1.
- Slower learned velocity corrections from a spectral MLP.
- Physics/biomechanics guardrails (LAID checks, ZARU/CAU stillness updates, cage clamp).

This repository is currently an offline training + evaluation stack around Nymeria data.

---

## 2. Implemented Architecture

### Fast Loop (ESKF)
Implemented in `incremental_train.py` (`ESKF` class):
- 15-state error-state filter:
  - position `[0:3]`
  - velocity `[3:6]`
  - orientation error `[6:9]`
  - gyro bias `[9:12]`
  - accel bias `[12:15]`
- Predict step uses SO(3) integration and SVD re-orthogonalization.
- `update_velocity()` includes Mahalanobis innovation gating (“Slap Gate”, default threshold 5.0).
- Overlord quarantine in velocity update zeroes Kalman gain rows `[6:15]` (orientation + biases untouched by neural correction).

### Neural Loop (Overlord / SpectralMLP)
Implemented in `SMLP.py`:
- Input: raw IMU window `(B, 6, 64)`
- Internal FFT path:
  - `rfft` over time axis
  - `log1p(abs(.))`
  - flatten to 198 features
- MLP backbone:
  - Dropout(0.4)
  - 198→256→128→64 with BatchNorm + ReLU
- Heads:
  - `head_trans` (3) = mean local velocity (m/s)
  - `head_cov` (3) = log-variance per axis

### Bounded Fusion
In `evaluate_eskf()` (`incremental_train.py`):
- Neural update runs every 10 samples once window is full.
- Predicted local velocity is passed through `bulwark()` before use.
- Local velocity is rotated to world frame and fused via ESKF velocity update.
- Dynamic covariance shaping uses predicted log-variance for diagnostics; current fusion noise uses static `R_obs = 0.1 * I`.

---

## 3. Guardrails and Physical Constraints

### Slap Gate
`ESKF.update_velocity()`:
- Computes Mahalanobis distance on velocity innovation.
- Rejects update if above threshold (`5.0^2`).
- Reuses `S_inv` for Kalman gain.

### Bulwark Hard Limits
`bulwark.py`:
- If any local velocity axis exceeds:
  - X: `0.40 m/s`
  - Y: `1.60 m/s`
  - Z: `0.35 m/s`
  then prediction is zeroed.

### ZARU (Zero Angular Rate Update)
`ESKF.update_zaru()` with detection in `evaluate_eskf()`:
- Trigger window: `ZARU_WINDOW = 50` (~0.5s at 100Hz)
- Stillness requirements:
  - gyro variance sum `< 1e-4`
  - accel variance sum `< 5e-3` (dual-lock against false positives)
- Measurement targets gyro bias only (`H[:, 9:12] = -I`)
- Joseph-form covariance update to preserve PSD/symmetry.
- Companion zero-velocity update is applied during detected standstill.

### CAU (Continuous Attitude Update)
`ESKF.update_cau()`:
- Enabled only under same stillness condition as ZARU.
- Uses accelerometer gravity observation to correct orientation + accel bias.
- Quarantines position/velocity/gyro-bias from CAU correction.

### LAID
`laid.py` currently provides:
- Differential-acceleration consistency check (`LAIDBouncer.check`) using lever-arm physics.
- Per-sample check API (`check_sample`) and batch utilities.
- Yaw anchor helper exists, but in current evaluation path this yaw-anchor injection is explicitly disabled (commented as mathematically flawed in rotating frames).

### NPP + Cage
- `npp.py`: dynamic NPP estimation + EMA/Z lock.
- `halo.py`: orientation clamp observer implementation exists.
- In current `evaluate_eskf()` path:
  - NPP tracking is active.
  - HALO orientation cage is instantiated but orientation clamping is disabled.
  - Positional cage clamp is active at radius `0.50 m` around tracked cage center.

---

## 4. Data and Labels

### Dataset
Nymeria (Aria) via `nymeria_loader.py`:
- `imu-right` (`1202-1`) treated as IMU1 (primary)
- `imu-left` (`1202-2`) treated as IMU2 (reference for differential methods)
- Streams are extrinsic-corrected to device frame and resampled to 100Hz.

### Windowing / Supervision
- Window size: `64`
- Stride: `10`
- Labels:
  - `trans`: **mean local velocity** over window (not displacement)
  - `quat`: relative rotation delta in `[W, X, Y, Z]`
- Loader returns both `imu1_features` and `imu2_features`.

### Augmentation
Training loader applies:
- temporal roll shift
- accel/gyro noise + per-window bias injection
- gravity/DC component preserved (not mean-subtracted)

---

## 5. Training Pipeline (incremental_train.py)

### Round Procedure
Per sequence:
1. Acquire/locate sequence.
2. Load cached or raw Nymeria windows.
3. Accumulate into growing subject pool.
4. Train model (`train_round`).
5. If warmup passed, evaluate physical ATE on held-out Shelby stream.
6. Apply dual early stopping criteria.

### Core Training Settings
- Optimizer: `AdamW(lr=1e-3, weight_decay=1e-2)`
- Scheduler: `ReduceLROnPlateau(mode='min', factor=0.5, patience=3, min_lr=1e-5)`
- Batch size: `4096`
- Epochs per round: `50`
- Warmup threshold for physical eval: `WARMUP_LOSS_THRESHOLD = 1.0`

### Loss
Gaussian NLL with velocity-magnitude weighting:
- model predicts mean + log-variance
- NLL weighted by `1 + 10 * ||gt_velocity||`

### Early Stopping
- Physical overfitting: ATE degradation strikes (`PATIENCE = 15`)
- Neural stagnation: insufficient loss improvement (`LOSS_PATIENCE = 10`, `LOSS_MIN_DELTA = 1e-5`)

---

## 6. Outputs and Telemetry

Generated artifacts include:
- model checkpoints (`talos.pth`, best physical checkpoint)
- ESKF trajectory plots per round
- master telemetry dashboard (ATE + train loss)
- diagnostic dashboard (`telemetry.py`) with:
  - scale collapse lens
  - slap gate timeline
  - covariance shadowing lens

---

## 7. Repository Structure (Current)

```
TALOS/
├── bulwark.py
├── cache_builder.py
├── halo.py
├── incremental_train.py
├── laid.py
├── notion_logger.py
├── npp.py
├── nymeria_loader.py
├── plot_shelby.py
├── README.md
├── scan_dataset.py
├── SMLP.py
├── TALOS.md
├── telemetry.py
├── archive/
│   ├── eval_rte.py
│   ├── laid_aria_calibrated.py
│   ├── laid.py
│   └── test_zaru.py
└── __pycache__/
```

---

## 8. Current Status Summary

Implemented and active in the main pipeline:
- Spectral MLP velocity prediction + uncertainty head
- ESKF propagation + Slap Gate
- ZARU + CAU stillness corrections
- LAID differential veto check
- NPP-driven positional cage clamp
- Incremental Nymeria training with physical validation loop

Implemented but not fully active in current eval path:
- HALO orientation clamping (module exists; disabled in evaluator)
- LAID yaw-anchor correction path (helper exists; disabled in evaluator)

---

*TALOS NIO is an inertial drift-bounding stack under active iteration. This document intentionally reflects code behavior, not aspirational design.*
