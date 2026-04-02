"""
cache_builder.py -- One-time VRS -> NPZ cache for TALOS NIO
Saves pre-windowing arrays (aligned IMU + GT) to golden/cache/<seq_id>.npz
Run once. Then incremental_train.py reads from cache, VRS never touched again.
"""
import json
import numpy as np
from pathlib import Path
from projectaria_tools.core import data_provider
from nymeria_loader import (
    load_imu_stream, align_imu_streams, load_gt_trajectory,
    interpolate_gt, SID_RIGHT, SID_LEFT, TARGET_HZ
)

ROOT      = Path('/mnt/c/TALOS/nymeria')
CACHE_DIR = Path('/home/iclab/TALOS/golden/cache')
MANIFEST  = Path('/mnt/c/TALOS/Nymeria_download_urls.json')

CACHE_DIR.mkdir(parents=True, exist_ok=True)

manifest = json.loads(MANIFEST.read_text())

for seq_id, entry in manifest["sequences"].items():
    out_path = CACHE_DIR / f"Nymeria_v0.0_{seq_id}_recording_head.npz"
    if out_path.exists():
        print(f"[SKIP] {seq_id[:40]}")
        continue

    seq_path = ROOT / f"Nymeria_v0.0_{seq_id}_recording_head" / "recording_head"
    vrs_path  = seq_path / 'data' / 'motion.vrs'
    traj_path = seq_path / 'mps' / 'slam' / 'closed_loop_trajectory.csv'

    if not vrs_path.exists():
        print(f"[MISSING] {seq_id[:40]} -- skipping")
        continue

    print(f"[CACHE] {seq_id[:40]}...")
    try:
        dp = data_provider.create_vrs_data_provider(str(vrs_path))
        device_calib = dp.get_device_calibration()

        T_r = device_calib.get_transform_device_sensor("imu-right")
        R_r = T_r.rotation().to_matrix().astype(np.float32)
        T_l = device_calib.get_transform_device_sensor("imu-left")
        R_l = T_l.rotation().to_matrix().astype(np.float32)

        ts_right, imu_right = load_imu_stream(dp, SID_RIGHT, R_r)
        ts_left,  imu_left  = load_imu_stream(dp, SID_LEFT,  R_l)

        grid_ns, imu1_reg, imu2_reg = align_imu_streams(
            ts_right, imu_right, ts_left, imu_left, TARGET_HZ)

        gt_ts, gt_pos, gt_quat = load_gt_trajectory(traj_path)
        pos_at_imu, quat_at_imu = interpolate_gt(gt_ts, gt_pos, gt_quat, grid_ns)

        mask        = (grid_ns >= gt_ts[0]) & (grid_ns <= gt_ts[-1])
        grid_ns     = grid_ns[mask]
        imu1_reg    = imu1_reg[mask]
        imu2_reg    = imu2_reg[mask]
        pos_at_imu  = pos_at_imu[mask]
        quat_at_imu = quat_at_imu[mask]

        np.savez_compressed(out_path,
            grid_ns=grid_ns,
            imu1=imu1_reg,
            imu2=imu2_reg,
            pos=pos_at_imu,
            quat=quat_at_imu,
        )
        duration_s = (grid_ns[-1] - grid_ns[0]) / 1e9
        print(f"  -> {len(grid_ns)} samples, {duration_s:.1f}s -- saved.")
    except Exception as e:
        print(f"  !! FAILED: {e}")

print("\nDone. Run incremental_train.py -- VRS never opens again.")
