"""
nymeria_loader.py (TALOS NIO)
Loads a single Nymeria recording_head sequence and produces a golden-compatible
numpy dict ready for build_golden_dataset.py ingestion.

IMU Strategy:
    imu-right (1202-1) : IMU1 equivalent. Right temple. Feeds Overlord (SpectralMLP).
    imu-left  (1202-2) : IMU2 equivalent. Left temple. Feeds LAID bouncer only.

    Both streams are kept completely independent. No averaging. No fusion.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d

from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId

# Stream IDs
SID_RIGHT = StreamId("1202-1")   # imu-right (primary)
SID_LEFT  = StreamId("1202-2")   # imu-left  (noise reference)

# Config
WINDOW_SIZE = 256     # samples
STRIDE      = 10      # samples
TARGET_HZ   = 100.0   # resample both IMUs to this rate before windowing


def load_imu_stream(dp, sid, R_extrinsic=None) -> tuple[np.ndarray, np.ndarray]:
    n = dp.get_num_data(sid)
    timestamps = np.empty(n, dtype=np.float64)
    imu_data   = np.empty((n, 6), dtype=np.float32)

    for i in range(n):
        s = dp.get_imu_data_by_index(sid, i)
        timestamps[i]  = s.capture_timestamp_ns
        
        accel = np.array(s.accel_msec2, dtype=np.float32)
        gyro  = np.array(s.gyro_radsec, dtype=np.float32)
        
        # The Extrinsic Fix: Rotate from raw Hardware Sensor Frame to SLAM Device Frame
        if R_extrinsic is not None:
            accel = R_extrinsic @ accel
            gyro  = R_extrinsic @ gyro
            
        imu_data[i, :3] = accel
        imu_data[i, 3:] = gyro

    return timestamps, imu_data

def align_imu_streams(ts_right, imu_right, ts_left, imu_left, target_hz):
    """Resample both streams to a shared uniform grid, keeping them strictly separate."""
    t_start = max(ts_right[0], ts_left[0])
    t_end   = min(ts_right[-1], ts_left[-1])
    dt_ns = 1e9 / target_hz
    grid  = np.arange(t_start, t_end, dt_ns)

    interp_r = interp1d(ts_right, imu_right, axis=0, kind='linear', bounds_error=False, fill_value=(imu_right[0], imu_right[-1]))
    interp_l = interp1d(ts_left, imu_left, axis=0, kind='linear', bounds_error=False, fill_value=(imu_left[0], imu_left[-1]))

    return grid, interp_r(grid).astype(np.float32), interp_l(grid).astype(np.float32)

def load_gt_trajectory(traj_path: Path):
    df = pd.read_csv(traj_path)
    timestamps_ns = df['tracking_timestamp_us'].values * 1e3
    positions = df[['tx_world_device', 'ty_world_device', 'tz_world_device']].values.astype(np.float32)
    quaternions = df[['qx_world_device', 'qy_world_device', 'qz_world_device', 'qw_world_device']].values.astype(np.float32)
    return timestamps_ns, positions, quaternions

def interpolate_gt(gt_ts, gt_pos, gt_quat, imu_ts):
    pos_interp = interp1d(gt_ts, gt_pos, axis=0, kind='linear', bounds_error=False, fill_value=(gt_pos[0], gt_pos[-1]))
    rots  = Rotation.from_quat(gt_quat)
    slerp = Slerp(gt_ts, rots)
    imu_clamped = np.clip(imu_ts, gt_ts[0], gt_ts[-1])
    return pos_interp(imu_clamped).astype(np.float32), slerp(imu_clamped).as_quat().astype(np.float32)

def _random_yaw_rotation():
    """Generate a random rotation matrix around the gravity (Z) axis."""
    angle = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float32)

def make_windows(imu1, imu2, pos, quat, window, stride, augment=True):
    N = len(imu1)
    starts = range(0, N - window, stride)

    imu1_windows = []
    imu2_windows = []
    trans_labels, quat_labels  = [], []

    for s in starts:
        e = s + window

        # 1. Translation Delta (Target Label)
        delta_p_global = pos[e] - pos[s]
        
        # --- SURGICAL FIX: Frame Alignment ---
        # The window covers samples [s : e]. The newest sample the ESKF has 
        # when running inference is at index e-1. We must target this frame.
        R_end          = Rotation.from_quat(quat[e-1])
        delta_p_local  = R_end.inv().apply(delta_p_global)

        # 2. Rotational Delta (Target Label)
        R_start = Rotation.from_quat(quat[s])  # Nymeria quat is [X,Y,Z,W]
        R_delta = R_start.inv() * R_end
        q_delta_xyzw = R_delta.as_quat()
        
        # 3. Enforce [W, X, Y, Z] format
        q_delta_wxyz = np.array([q_delta_xyzw[3], q_delta_xyzw[0], q_delta_xyzw[1], q_delta_xyzw[2]], dtype=np.float32)

        imu_window = imu1[s:e].copy()
        if augment:
            shift = np.random.randint(0, 10)
            imu_window = np.roll(imu_window, shift, axis=0)

        # --- Data Augmentation (training only) ---
        if augment:
            # A. Random yaw rotation REMOVED — With the DC (Gravity) component preserved, 
            #    randomly rotating the world frame "tilts" the earth dynamically and 
            #    destroys the physical learning correlation for pitch/roll.

            # B. Noise injection — domain gap bridging
            #    Gaussian noise + per-window constant bias to simulate different IMU sensors
            accel_noise = np.random.normal(0, 0.01, imu_window[:, :3].shape).astype(np.float32)
            gyro_noise  = np.random.normal(0, 0.001, imu_window[:, 3:].shape).astype(np.float32)
            accel_bias  = np.random.normal(0, 0.05, (1, 3)).astype(np.float32)
            gyro_bias   = np.random.normal(0, 0.005, (1, 3)).astype(np.float32)
            imu_window[:, :3] += accel_noise + accel_bias
            imu_window[:, 3:] += gyro_noise  + gyro_bias

        # Gravity/DC component is deliberately preserved for postural context.
        # imu_window[:, :3] = imu_window[:, :3] - np.mean(imu_window[:, :3], axis=0)

        imu2_window = imu2[s:e].copy()
        # imu2_window[:, :3] -= np.mean(imu2_window[:, :3], axis=0)
        imu1_windows.append(imu_window.astype(np.float32))
        imu2_windows.append(imu2_window.astype(np.float32))
        
        # Convert displacement to mean velocity (m/s)
        window_time = window / TARGET_HZ
        mean_velocity = delta_p_local / window_time
        trans_labels.append(mean_velocity.astype(np.float32))
        
        quat_labels.append(q_delta_wxyz)

    return {
        'imu1_features': np.stack(imu1_windows).astype(np.float32),
        'imu2_features': np.stack(imu2_windows).astype(np.float32),
        'trans':         np.stack(trans_labels).astype(np.float32),
        'quat':          np.stack(quat_labels).astype(np.float32),
    }

def _parse_vrs_to_arrays(sequence_root: str | Path, target_hz: float = TARGET_HZ):
    root = Path(sequence_root)
    vrs_path  = root / 'data' / 'motion.vrs'
    traj_path = root / 'mps' / 'slam' / 'closed_loop_trajectory.csv'

    print(f"[nymeria_loader] Opening {vrs_path.name}...")
    dp = data_provider.create_vrs_data_provider(str(vrs_path))

    # Extract Hardware Extrinsics (Sensor -> Device)
    print("[nymeria_loader] Extracting Aria factory extrinsics...")
    device_calib = dp.get_device_calibration()
    
    T_device_right = device_calib.get_transform_device_sensor("imu-right")
    R_device_imu_right = T_device_right.rotation().to_matrix().astype(np.float32)
    
    T_device_left = device_calib.get_transform_device_sensor("imu-left")
    R_device_imu_left = T_device_left.rotation().to_matrix().astype(np.float32)

    print("[nymeria_loader] Reading and calibrating imu-right (primary) and imu-left (reference)...")
    ts_right, imu_right = load_imu_stream(dp, SID_RIGHT, R_device_imu_right)
    ts_left,  imu_left  = load_imu_stream(dp, SID_LEFT, R_device_imu_left)

    print(f"[nymeria_loader] Aligning streams at {target_hz:.0f}Hz (No fusion)...")
    grid_ns, imu1_reg, imu2_reg = align_imu_streams(ts_right, imu_right, ts_left, imu_left, target_hz)

    print("[nymeria_loader] Loading closed_loop_trajectory.csv...")
    gt_ts, gt_pos, gt_quat = load_gt_trajectory(traj_path)

    print("[nymeria_loader] Interpolating GT onto IMU grid...")
    pos_at_imu, quat_at_imu = interpolate_gt(gt_ts, gt_pos, gt_quat, grid_ns)

    mask        = (grid_ns >= gt_ts[0]) & (grid_ns <= gt_ts[-1])
    grid_ns     = grid_ns[mask]
    imu1_reg    = imu1_reg[mask]
    imu2_reg    = imu2_reg[mask]
    pos_at_imu  = pos_at_imu[mask]
    quat_at_imu = quat_at_imu[mask]
    print(f"[nymeria_loader] Trimmed to GT coverage : {len(grid_ns)} samples remaining")

    return imu1_reg, imu2_reg, pos_at_imu, quat_at_imu, grid_ns


def load_sequence(sequence_root: str | Path, window: int = WINDOW_SIZE, stride: int = STRIDE, target_hz: float = TARGET_HZ, augment: bool = True) -> dict:
    imu1_reg, imu2_reg, pos_at_imu, quat_at_imu, grid_ns = _parse_vrs_to_arrays(sequence_root, target_hz)

    print(f"[nymeria_loader] Windowing (size={window}, stride={stride}, augment={augment})...")
    result = make_windows(imu1_reg, imu2_reg, pos_at_imu, quat_at_imu, window, stride, augment=augment)

    duration_s = (grid_ns[-1] - grid_ns[0]) / 1e9
    print(f"[nymeria_loader] Done : {len(result['trans'])} windows from {duration_s:.1f}s")
    return result

if __name__ == '__main__':
    import sys

    SEQ = (
        '/mnt/c/TALOS/nymeria/'
        'Nymeria_v0.0_20230607_s0_james_johnson_act0_e72nhq_recording_head/'
        'recording_head'
    )

    out = load_sequence(SEQ)

    print("\n:: Output shapes ::")
    for k, v in out.items():
        print(f"  {k:20s}: {v.shape}  dtype={v.dtype}")

    print("\n:: Sample window 0 ::")
    print(f"  imu1[0,0]: {out['imu1_features'][0, 0]}")
    print(f"  trans[0]:  {out['trans'][0]}")
    print(f"  quat[0]:   {out['quat'][0]}")
# ---------------------------------------------------------------------------
# Cached loader -- bypasses VRS entirely if golden/cache/<seq_id>.npz exists
# ---------------------------------------------------------------------------
CACHE_DIR = Path('/home/iclab/TALOS/golden/cache')

def load_sequence_cached(sequence_root: str | Path, window: int = WINDOW_SIZE,
                         stride: int = STRIDE, augment: bool = True) -> dict:
    root   = Path(sequence_root)
    # Handle paths ending in 'recording_head' or ending directly in sequence IDs
    seq_id = root.parent.name if root.name == "recording_head" else root.name
    cache  = CACHE_DIR / f"{seq_id}.npz"

    if cache.exists():
        print(f"[cache] HIT {seq_id[:40]}")
        d = np.load(cache)
        imu1_reg = d['imu1']
        imu2_reg = d['imu2']
        pos_at_imu = d['pos']
        quat_at_imu = d['quat']
    else:
        print(f"[cache] MISS {seq_id[:40]} -- compiling from VRS")
        imu1_reg, imu2_reg, pos_at_imu, quat_at_imu, grid_ns = _parse_vrs_to_arrays(sequence_root, target_hz=TARGET_HZ)
        
        # Save cache so it never misses again!
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, imu1=imu1_reg, imu2=imu2_reg, pos=pos_at_imu, quat=quat_at_imu)
        print(f"[cache] SAVED new sequence to disk!")

    return make_windows(imu1_reg, imu2_reg, pos_at_imu, quat_at_imu, window, stride, augment=augment)
