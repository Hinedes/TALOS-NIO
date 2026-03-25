# TALOS NIO

TALOS is an offline incremental training and evaluation stack for Nymeria dual-IMU data.

The current pipeline combines:

- 100 Hz ESKF propagation on the primary IMU
- a spectral MLP that predicts mean local velocity and log-variance
- physics-based guardrails such as LAID, ZARU, CAU, NPP tracking, and a positional cage

## Core Files

- [incremental_train.py](incremental_train.py) - main training and physical ESKF evaluation loop
- [SMLP.py](SMLP.py) - spectral neural model used by the trainer
- [nymeria_loader.py](nymeria_loader.py) - Nymeria loading, alignment, windowing, and augmentation
- [bulwark.py](bulwark.py) - per-axis local-velocity clamp
- [laid.py](laid.py) - lever-arm differential consistency checks
- [npp.py](npp.py) - neck pivot point tracking
- [halo.py](halo.py) - orientation cage observer, currently instantiated but disabled in evaluation
- [telemetry.py](telemetry.py) - telemetry CSV and diagnostic plots
- [reporting.py](reporting.py) - ntfy / Notion publishing helpers
- [darwin.py](darwin.py) - mutation-based recovery for stagnant runs
- [cache_builder.py](cache_builder.py) - cache generation utility
- [plot_shelby.py](plot_shelby.py) - Shelby trajectory plotting helper
- [scan_dataset.py](scan_dataset.py) - dataset inspection utility

## Runtime Notes

- `incremental_train.py` is CUDA-only in the current repository state.
- The model is trained with `AdamW` and `ReduceLROnPlateau`.
- `trans` labels are mean local velocity over each window; `quat` labels are carried through the dataset but are not used by the current loss.
- `bulwark()` clips implausible predictions; it does not zero them.
- HALO orientation clamping, LAID yaw anchor, and the dynamic observation covariance path are present but disabled by default in the evaluation loop.

## Quick Start

```bash
python cache_builder.py
python incremental_train.py
python plot_shelby.py
```

If you only need the detailed technical state, read [TALOS.md](TALOS.md).

## Status

Active R&D / prototype codebase under rapid iteration.
