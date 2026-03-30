# TALOS NIO

Code-synced state: March 26, 2026.

This document reflects the current repository behavior, not aspirational design.

## 1. Mission

TALOS is an offline training and evaluation stack for Nymeria dual-IMU sequences. The main pipeline combines:

- 100 Hz ESKF propagation on the primary IMU
- learned local-velocity corrections from a spectral MLP
- physics and biomechanics guardrails for drift bounding

The current repository is centered on incremental training, physical evaluation, telemetry, and parameter recovery.

## 2. Current Architecture

### Fast loop: ESKF in `incremental_train.py`

The filter is a 15-state error-state system:

- position `[0:3]`
- velocity `[3:6]`
- orientation error `[6:9]`
- gyro bias `[9:12]`
- accel bias `[12:15]`

Propagation integrates accel and gyro, then re-orthogonalizes orientation with SVD.

Current update paths:

- `update_velocity()` performs world-velocity fusion with a dual Mahalanobis Slap Gate.
- `update_local_velocity()` is the main neural fusion path used in evaluation.
- `update_zaru()` is a gyro-bias-only stillness update with Joseph-form covariance handling.
- `update_cau()` uses gravity to correct orientation and accel bias during stillness.

`update_local_velocity()` does the following:

- rotates world velocity into the body frame to compare against the neural prediction
- gates on both state-aware and measurement-only Mahalanobis distance
- quarantines accel bias from the Kalman gain
- projects the orientation correction onto yaw only
- applies gyro-bias correction through cross-covariance

### Neural loop: `SMLP.py`

`BigSpectralMLP` is an alias of `SpectralMLP`.

- Input shape: `(B, 6, 64)`
- FFT path: CPU-side `rfft` over time, then signed `log1p(abs(.))` on real and imaginary parts
- Spectral feature size: 396
- Model layout: separate translation and covariance trunks
- Outputs: mean local velocity `(3,)` and log-variance `(3,)`

There is no quaternion prediction head in the current model.

In `incremental_train.py`, the FFT stays eager while `model.npu_core` is compiled.

### Fusion and guardrails

In evaluation, neural updates happen every 10 samples once the 64-sample window is full.

- The predicted local velocity is multiplied by `PRED_VEL_GAIN`
- `bulwark()` clips implausible local velocity per axis; it does not zero the vector
- LAID checks compare IMU1 and IMU2 differential acceleration over a window
- If LAID vetoes a window, the neural update is skipped
- `USE_DYNAMIC_R_OBS` exists, but the default path uses a fixed observation covariance
- Hard safety checks reject updates that exceed configured world-speed or innovation limits

Guardrail status in the current evaluation path:

- NPP tracking is active
- HALO is instantiated, but orientation clamping is disabled
- Positional cage clamping is active at a radius of `0.30 m` around the tracked cage center
- LAID yaw anchor is present, but disabled by default
- LAID differential update and windowed LAID update are present, but disabled by default

## 3. Data and Labels

`nymeria_loader.py` handles the Nymeria Aria recordings.

- `imu-right` is the primary stream used as IMU1
- `imu-left` is the secondary stream used as IMU2 for differential checks
- Both streams are extrinsic-corrected to device frame and resampled to 100 Hz

Windowing and supervision:

- window size: `64`
- stride: `10`
- `trans` label: mean local velocity over the window, not displacement
- `quat` label: relative rotation delta in `[W, X, Y, Z]`

The loader returns both `imu1_features` and `imu2_features`.

Augmentation applies:

- temporal roll shift
- accel and gyro noise
- per-window accel and gyro bias injection
- preserved gravity and DC content

Current training uses the translation target only. `quat` is carried through the dataset, but it is not used by the present loss.

## 4. Training Pipeline

The incremental loop in `incremental_train.py` does the following per sequence:

1. Acquire or locate the sequence
2. Load cached or raw Nymeria windows
3. Accumulate windows into a rolling subject pool
4. Train the model with `train_round()`
5. After warmup, run physical ESKF evaluation on the held-out Shelby stream
6. Apply the early-stop rules for physical overfit and neural stagnation

Training settings:

- optimizer: `AdamW(lr=1e-3, weight_decay=1e-2)`
- scheduler: `ReduceLROnPlateau(mode="min", factor=0.5, patience=3, min_lr=1e-5)`
- batch size: `4096`
- epochs per round: `50`
- warmup threshold for physical evaluation: `WARMUP_LOSS_THRESHOLD = 1.0`

Loss:

- Gaussian NLL over velocity with a log-variance head
- motion-direction cosine penalty during motion
- loss weighting increases with ground-truth speed

`MegaBuffer` stores the rolling dataset in CPU memory so the subject pool can grow without repeated concatenation.

## 5. Recovery and Parameter Search

`darwin.py` provides a mutation-based recovery path when evaluation stagnates.

It mutates the fusion parameters passed into `evaluate_eskf()`, including:

- `SLAP_THRESHOLD`
- `R_OBS_FIXED_DIAG`
- `PRED_VEL_GAIN`
- `CAGE_RADIUS`
- `MAX_PRED_WORLD_SPEED_MPS`
- `MAX_INNOVATION_NORM_MPS`
- `USE_DYNAMIC_R_OBS`

The current code writes the winning configuration and a generation log to disk.

## 6. Outputs and Telemetry

Evaluation and training generate:

- run-level telemetry CSV files
- per-round ESKF plots
- diagnostic dashboards for scale collapse, Slap Gate tension, and covariance shadowing
- master training charts comparing ATE and loss across rounds

`reporting.py` can publish the final status to ntfy and optionally Notion.

## 7. Repository Map

Key top-level files in the current repository:

- `incremental_train.py` - main incremental training and physical evaluation loop
- `SMLP.py` - spectral neural model
- `nymeria_loader.py` - Nymeria loading, alignment, and windowing
- `bulwark.py` - per-axis prediction clamp
- `laid.py` - lever-arm differential consistency checks
- `npp.py` - neck pivot point tracker
- `halo.py` - orientation cage observer, currently disabled in evaluation
- `telemetry.py` - CSV append and diagnostic plotting helpers
- `reporting.py` - ntfy and Notion publishing helpers
- `darwin.py` - stagnation recovery and parameter search
- `cache_builder.py` - cache generation utility
- `scan_dataset.py` - dataset inspection utility
- `plot.py`, `plot_shelby.py` - plotting utilities
- `eval_best.py`, `cpu_optuna_eskf.py` - evaluation and search helpers
- `retroactive_vrs_cleanup.py` - storage cleanup utility
- `talos_controller.py`, `agent.py` - controller and agent layer
- `train.py` - alternate training entry point
- `archive/` - historical experiments and older variants

## 8. Current Status Summary

Active in the main pipeline:

- ESKF propagation and Slap Gate fusion
- spectral local-velocity prediction
- LAID window veto
- ZARU and CAU stillness updates
- NPP tracking and positional cage clamp
- incremental Nymeria training with physical validation

Present but disabled by default in the current evaluation path:

- HALO orientation clamping
- LAID yaw anchor
- LAID differential update
- LAID windowed update
- dynamic observation covariance

Not present in the current code:

- quaternion prediction head
- OneCycleLR training schedule
- hard-zero bulwark behavior

*TALOS NIO is a drift-bounding inertial stack under active iteration. This document tracks the code as it exists now.*
