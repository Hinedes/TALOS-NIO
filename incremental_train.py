"""
incremental_train.py : TALOS NIO (Unified ESKF Pipeline)
Incremental training loop over Nymeria sequences.

Each round:
    1. Download next sequence into an isolated directory.
    2. Extract windowed IMU data.
    3. Accumulate into rolling train dataset.
    4. Train SpectralMLP for N epochs (with Gaussian NLL Loss).
    5. Evaluate physical drift (ATE) by running the ESKF on a continuous Val stream.
       - ESKF evaluation is skipped until neural loss drops below WARMUP_LOSS_THRESHOLD.
    6. Stop if ESKF ATE worsens over PATIENCE rounds (physical overfitting).
    7. Stop if neural loss stagnates over LOSS_PATIENCE rounds (dead model).
"""

from bulwark import bulwark
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Strict headless operation
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from projectaria_tools.core import data_provider

from SMLP import BigSpectralMLP as SpectralMLP
from laid import LAIDBouncer
from halo import HALOObserver
from npp import NPPTracker
from nymeria_loader import (load_sequence, load_sequence_cached, load_imu_stream, align_imu_streams,
                            load_gt_trajectory, interpolate_gt,
                            SID_RIGHT, SID_LEFT, TARGET_HZ)

# Configuration
PATIENCE               = 15      # ESKF ATE strikes before halting (physical overfitting)
LOSS_PATIENCE          = 20     # Loss stagnation strikes before halting (dead model)
LOSS_MIN_DELTA         = 1e-5   # Minimum loss improvement to count as progress
WARMUP_LOSS_THRESHOLD  = 0.010  # Don't run ESKF eval until loss drops below this
STORAGE_FLOOR_GB       = 30.0
EPOCHS_PER_ROUND       = 20
BATCH_SIZE             = 4096
VAL_SUBJECT            = 'shelby_arroyo'  # 63m locomotion stress test

# FFT & Nymeria Config
WINDOW_SIZE = 64
N_BINS      = 33
N_CHANNELS  = 6
INPUT_DIM   = N_BINS * N_CHANNELS  # 198
ESKF_DT     = 1.0 / TARGET_HZ

# ZARU Config
ZARU_WINDOW          = 50
ZARU_THRESHOLD       = 1e-4
ZARU_ACCEL_THRESHOLD = 5e-3  # Dual-sensor lock requirement

# ESKF Physics Engine
class ESKF:
    def __init__(self, dt=0.01, gravity=None):
        self.dt      = dt
        self.gravity = gravity if gravity is not None else np.array([0., 0., -9.81], dtype=np.float32)
        self.position    = np.zeros(3)
        self.velocity    = np.zeros(3)
        self.orientation = np.eye(3)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.P  = np.eye(15) * 0.1
        self.Q  = np.diag([1e-6]*3 + [1e-4]*3 + [1e-5]*3 + [1e-3]*3 + [1e-2]*3)

    @staticmethod
    def _skew(v):
        return np.array([[ 0,    -v[2],  v[1]],
                         [ v[2],  0,    -v[0]],
                         [-v[1],  v[0],  0   ]])

    def predict(self, accel, gyro):
        dt, R = self.dt, self.orientation
        u_a = accel - self.ba
        u_g = gyro  - self.bg
        aw  = R @ u_a + self.gravity
        self.position += self.velocity * dt + 0.5 * aw * dt**2
        self.velocity += aw * dt
        ang = np.linalg.norm(u_g) * dt
        if ang > 1e-9:
            self.orientation = R @ Rotation.from_rotvec(u_g * dt).as_matrix()
        F = np.eye(15)
        F[0:3, 3:6]   = np.eye(3) * dt
        F[3:6, 6:9]   = -R @ self._skew(u_a) * dt
        F[3:6, 12:15] = -R * dt
        F[6:9, 9:12]  = -R * dt
        self.P = F @ self.P @ F.T + self.Q

    def update_velocity(self, vel, R_obs, slap_threshold=3.0):
        """ESKF velocity update with Mahalanobis Slap Gate (threshold=3.0)."""
        if not np.all(np.isfinite(vel)): return
        H = np.zeros((3, 15))
        H[0,3] = H[1,4] = H[2,5] = 1.0

        S     = H @ self.P @ H.T + R_obs
        r     = vel - self.velocity
        S_inv = np.linalg.inv(S)

        # The Slap -- Mahalanobis innovation gate
        mahal_sq = float(r @ S_inv @ r)
        if mahal_sq > slap_threshold ** 2:
            return  # slapped -- update silently rejected

        # Reuse S_inv for Kalman gain (zero redundant computation)
        K = self.P @ H.T @ S_inv

        # OVERLORD QUARANTINE: orientation and biases zeroed at gain level
        K[6:15, :] = 0.0

        dx = np.clip(K @ r, -2.0, 2.0)
        self.position += dx[0:3]
        self.velocity += dx[3:6]
        self.P = (np.eye(15) - K @ H) @ self.P

    def update_zaru(self, gyro_raw):
        H = np.zeros((3, 15))
        H[0:3, 9:12] = -np.eye(3)  # CORRECTED: States 9:12 target the gyro bias
        
        R_z = np.eye(3) * 1e-4
        z   = -(gyro_raw - self.bg)
        S   = H @ self.P @ H.T + R_z
        
        # Robust matrix solve
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        dx  = K @ z
        
        self.position    += dx[0:3]
        self.velocity    += dx[3:6]
        
        # Orientation is untouched by ZARU. It is unobservable here.
        # The Alpha Gate: Dampen bias updates
        self.bg += dx[9:12] * 0.1
        self.ba += dx[12:15] * 0.1
        
        self.P  = (np.eye(15) - K @ H) @ self.P

# Data Pipeline
def accumulate(existing: dict | None, new: dict) -> dict:
    if existing is None: return {k: v.copy() for k, v in new.items()}
    return {k: np.concatenate([existing[k], new[k]], axis=0) for k in new}

def to_raw(imu_windows: np.ndarray) -> np.ndarray:
    # (N, 64, 6) -> (N, 6, 64) for Conv1d
    return imu_windows.transpose(0, 2, 1).astype(np.float32)

def make_tensors(data: dict, device: torch.device):
    X = torch.from_numpy(to_raw(data['imu1_features'])).to(device)
    T = torch.from_numpy(data['trans']).to(device)
    Q = torch.from_numpy(data['quat']).to(device)
    return X, T, Q

def load_continuous_val_stream(seq_root: Path):
    """Loads a pure, continuous DataFrame for ESKF evaluation."""
    vrs_path  = seq_root / 'data' / 'motion.vrs'
    traj_path = seq_root / 'mps' / 'slam' / 'closed_loop_trajectory.csv'

    dp = data_provider.create_vrs_data_provider(str(vrs_path))

    # PATCH 1: Extract and apply hardware extrinsics for the validation stream
    device_calib = dp.get_device_calibration()
    T_device_right = device_calib.get_transform_device_sensor("imu-right")
    R_device_imu_right = T_device_right.rotation().to_matrix().astype(np.float32)
    T_device_left = device_calib.get_transform_device_sensor("imu-left")
    R_device_imu_left = T_device_left.rotation().to_matrix().astype(np.float32)

    ts_right, imu_right = load_imu_stream(dp, SID_RIGHT, R_device_imu_right)
    ts_left,  imu_left  = load_imu_stream(dp, SID_LEFT,  R_device_imu_left)
    grid_ns, imu1_reg, imu2_reg = align_imu_streams(ts_right, imu_right, ts_left, imu_left, TARGET_HZ)
    gt_ts, gt_pos, gt_quat = load_gt_trajectory(traj_path)

    df_traj = pd.read_csv(traj_path)
    gt_vel    = df_traj[['device_linear_velocity_x_device',
                          'device_linear_velocity_y_device',
                          'device_linear_velocity_z_device']].values.astype(np.float32)
    gt_vel_ts = df_traj['tracking_timestamp_us'].values * 1e3

    pos_at_imu, quat_at_imu = interpolate_gt(gt_ts, gt_pos, gt_quat, grid_ns)

    from scipy.interpolate import interp1d
    vel_interp = interp1d(gt_vel_ts, gt_vel, axis=0, kind='linear',
                          bounds_error=False, fill_value=(gt_vel[0], gt_vel[-1]))
    vel_at_imu = vel_interp(grid_ns).astype(np.float32)

    true_gravity = df_traj[['gravity_x_world',
                             'gravity_y_world',
                             'gravity_z_world']].iloc[0].values.astype(np.float32)

    mask = (grid_ns >= gt_ts[0]) & (grid_ns <= gt_ts[-1])
    df = pd.DataFrame({
        'ax': imu1_reg[mask, 0], 'ay': imu1_reg[mask, 1], 'az': imu1_reg[mask, 2],
        'wx': imu1_reg[mask, 3], 'wy': imu1_reg[mask, 4], 'wz': imu1_reg[mask, 5],
        'px': pos_at_imu[mask, 0], 'py': pos_at_imu[mask, 1], 'pz': pos_at_imu[mask, 2],
        'vx': vel_at_imu[mask, 0], 'vy': vel_at_imu[mask, 1], 'vz': vel_at_imu[mask, 2],
        'qx': quat_at_imu[mask, 0], 'qy': quat_at_imu[mask, 1],
        'qz': quat_at_imu[mask, 2], 'qw': quat_at_imu[mask, 3],
        'ax2': imu2_reg[mask, 0], 'ay2': imu2_reg[mask, 1], 'az2': imu2_reg[mask, 2],
        'wx2': imu2_reg[mask, 3], 'wy2': imu2_reg[mask, 4], 'wz2': imu2_reg[mask, 5],
    })
    return df, true_gravity

def download_sequence(seq_id: str, entry: dict, root: Path) -> Path | None:
    target_bundle = entry.get('recording_head')
    if not target_bundle: return None

    filename    = target_bundle['filename']
    zip_path    = root / filename
    extract_dir = root / zip_path.stem

    if not zip_path.exists() and not extract_dir.exists():
        cmd = ['aria2c', '-c', '-x', '16', '-s', '16', '-o', filename, target_bundle['download_url']]
        if subprocess.run(cmd, cwd=str(root)).returncode != 0: return None

    if not extract_dir.exists():
        subprocess.run(['unzip', '-q', str(zip_path), '-d', str(extract_dir)])

    seq_path = extract_dir / 'recording_head'
    if not (seq_path / 'data' / 'motion.vrs').exists(): return None
    return seq_path

# Training
def train_round(model, opt, sched, train_data, val_data, device, epochs, checkpoint_path):
    X_tr, T_tr, Q_tr = make_tensors(train_data, device)
    X_va, T_va, Q_va = make_tensors(val_data,   device)
    loader = DataLoader(TensorDataset(X_tr, T_tr, Q_tr), batch_size=BATCH_SIZE, shuffle=True)

    best_val, t_losses = float('inf'), []

    def loss_fn(pt, pq, pcov, gt, gq):
        # Phase 2: Gaussian Negative Log-Likelihood for translation + aleatoric uncertainty
        pcov_c = torch.clamp(pcov, min=-4.0, max=4.0)
        nll_trans = 0.5 * (torch.exp(-pcov_c) * (gt - pt)**2 + pcov_c)
        lt = nll_trans.sum(dim=1).mean()
        
        # Standard cosine-like loss for quaternion orientation
        lq = 1.0 - (pq * gq).sum(dim=1).abs().mean()
        
        return lt + 0.1 * lq

    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, tb, qb in loader:
            opt.zero_grad()
            pt, pq, pcov = model(xb)
            loss = loss_fn(pt, pq, pcov, tb, qb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            ep_loss += loss.item()
        t_losses.append(ep_loss / len(loader))
        sched.step(t_losses[-1])

        model.eval()
        with torch.no_grad():
            pvt, pvq, pvcov = model(X_va)
            v_loss = loss_fn(pvt, pvq, pvcov, T_va, Q_va)

        if v_loss.item() < best_val:
            best_val = v_loss.item()
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    return t_losses[-1], best_val

# ESKF Evaluation
def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def evaluate_eskf(model, df: pd.DataFrame, true_gravity: np.ndarray,
                  device, round_idx, plot_dir: Path, max_seconds=9999) -> float:
    """Runs the trained model through the physical ESKF and returns mean ATE."""
    dt          = ESKF_DT
    max_samples = int(max_seconds / dt)
    df          = df.iloc[:max_samples].reset_index(drop=True)

    eskf_talos = ESKF(dt=dt, gravity=true_gravity)
    eskf_pure  = ESKF(dt=dt, gravity=true_gravity)
    model.eval()

    accel  = df[['ax','ay','az']].values.astype(np.float32)
    accel2 = df[['ax2','ay2','az2']].values.astype(np.float32)
    gyro2  = df[['wx2','wy2','wz2']].values.astype(np.float32)
    gyro   = df[['wx','wy','wz']].values.astype(np.float32)
    gt_pos = df[['px','py','pz']].values
    gt_pos = gt_pos - gt_pos[0]

    init_rot        = Rotation.from_quat(df[['qx','qy','qz','qw']].iloc[0].values).as_matrix()
    init_vel_device = df[['vx','vy','vz']].iloc[0].values.copy()
    init_vel_world  = init_rot @ init_vel_device

    for e in [eskf_talos, eskf_pure]:
        e.velocity    = init_vel_world.copy()
        e.orientation = init_rot.copy()

    laid_bouncer = LAIDBouncer()
    npp_tracker  = NPPTracker()
    halo = HALOObserver(init_rot)
    accel_buf, gyro_buf = [], []
    accel2_buf, gyro2_buf = [], []
    talos_positions, pure_positions = [], []
    window_time = WINDOW_SIZE * dt

    for step in range(len(df)):
        a, g = accel[step], gyro[step]

        eskf_talos.predict(a, g)
        eskf_pure.predict(a, g)
        # NPP update (tracking only, sphere clamp disabled)
        v_device = eskf_talos.orientation.T @ eskf_talos.velocity
        npp_tracker.update(g, v_device)
        # HALO orientation cage disabled -- requires torso reference frame

        talos_positions.append(eskf_talos.position.copy())
        pure_positions.append(eskf_pure.position.copy())

        accel_buf.append(a)
        gyro_buf.append(g)
        accel2_buf.append(accel2[step])
        gyro2_buf.append(gyro2[step])

        if len(accel_buf) > WINDOW_SIZE:
            accel_buf.pop(0)
            gyro_buf.pop(0)
            accel2_buf.pop(0)
            gyro2_buf.pop(0)

        # 10Hz Neural Correction (TALOS only)
        if len(accel_buf) == WINDOW_SIZE and step % 10 == 0:
            win_accel = np.array(accel_buf)
            win_gyro  = np.array(gyro_buf)
            
            # Zero out the gravity/DC component for inference
            win_accel_corrected = win_accel - np.mean(win_accel, axis=0)
            
            win = np.concatenate([win_accel_corrected, win_gyro], axis=-1)
            win_tensor = torch.tensor(win.T[np.newaxis], dtype=torch.float32)  # (1, 6, 64)

            with torch.no_grad():
                pred_delta, _, pred_cov = model(win_tensor.to(device))

            pred_delta_np = pred_delta.cpu().numpy()[0]
            pred_delta_np = bulwark(pred_delta_np)
            pred_cov_np   = pred_cov.cpu().numpy()[0]

            # LAID veto
            win2_accel = np.array(accel2_buf) - np.mean(np.array(accel2_buf), axis=0)
            win2_gyro  = np.array(gyro2_buf)
            win2       = np.concatenate([win2_accel, win2_gyro], axis=-1)
            win1       = np.concatenate([win_accel_corrected, win_gyro], axis=-1)
            laid_veto, laid_rms = laid_bouncer.check(win1, win2)
            if not laid_veto:
                v_world = eskf_talos.orientation @ (pred_delta_np / window_time)
                R_obs_dynamic = np.diag(np.clip(np.exp(pred_cov_np), 1e-3, None))
                eskf_talos.update_velocity(v_world, R_obs=R_obs_dynamic)

        # ZARU (TALOS only)
        if len(gyro_buf) >= ZARU_WINDOW and step % ZARU_WINDOW == 0:
            gyro_var = np.var(np.array(gyro_buf[-ZARU_WINDOW:]), axis=0).sum()
            accel_var = np.var(np.array(accel_buf[-ZARU_WINDOW:]), axis=0).sum()
            
            # Dual-sensor lock to prevent false positives during slow motion
            if gyro_var < ZARU_THRESHOLD and accel_var < ZARU_ACCEL_THRESHOLD:
                eskf_talos.update_zaru(g)
                # Hardcoded low noise for verified zero-velocity updates
                eskf_talos.update_velocity(np.zeros(3), R_obs=np.eye(3) * 1e-4)


        # ---------------------------------------------------------
        # THE CAGE: Biomechanical Positional Clamp (12cm radius)
        # ---------------------------------------------------------
        current_npp_world = eskf_talos.position + eskf_talos.orientation @ npp_tracker.npp

        if not hasattr(evaluate_eskf, '_cage_center'):
            evaluate_eskf._cage_center = current_npp_world.copy()
        else:
            evaluate_eskf._cage_center[0:2] = current_npp_world[0:2]
            evaluate_eskf._cage_center[2]   = 0.999 * evaluate_eskf._cage_center[2] + 0.001 * current_npp_world[2]

        head_vector = eskf_talos.position - evaluate_eskf._cage_center
        distance = np.linalg.norm(head_vector)

        if distance > 0.12:
            eskf_talos.position = evaluate_eskf._cage_center + (head_vector / distance) * 0.12

    talos_positions = np.array(talos_positions)
    pure_positions  = np.array(pure_positions)
    evaluate_eskf._last_talos_pos = talos_positions
    evaluate_eskf._last_gt_pos    = pure_positions
    talos_err       = np.linalg.norm(talos_positions - gt_pos, axis=1)
    mean_ate        = talos_err.mean()
    final_ate       = talos_err[-1]
    total_distance  = np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1))
    mean_rte        = (mean_ate / total_distance) * 100
    final_rte       = (final_ate / total_distance) * 100

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(
        f"Round {round_idx} | TALOS ATE: {mean_ate:.3f}m (RTE {mean_rte:.2f}%) "
        f"| Final Drift: {final_ate:.3f}m ({final_rte:.2f}%) "
        f"| Pure IMU: {np.linalg.norm(pure_positions - gt_pos, axis=1).mean():.3f}m",
        fontweight='bold'
    )

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(*pure_positions.T, color='red',   lw=1.5, ls='--', label='Pure IMU (Drift)')
    ax1.plot(*talos_positions.T, color='blue', lw=2.0,          label='TALOS')
    ax1.plot(*gt_pos.T,          color='black', alpha=0.5,       label='GT')
    ax1.set_title("Macro View: Bounding Unobservable Drift")
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(*talos_positions.T, color='blue', lw=2.0,    label='TALOS')
    ax2.plot(*gt_pos.T,          color='black', alpha=0.5, label='GT')
    ax2.set_title("Micro View: Neural-Inertial Trajectory vs Reality")
    ax2.legend()

    set_axes_equal(ax1)
    set_axes_equal(ax2)

    plt.tight_layout()
    plt.savefig(plot_dir / f'eskf_eval_round_{round_idx}.png', dpi=150)
    plt.close()

    return mean_ate

# Telemetry Dashboard
def update_master_dashboard(history: list[dict], plot_path: Path):
    if not history: return

    rounds = [h['round']      for h in history]
    ates   = [h.get('ate')    for h in history]
    losses = [h['train_loss'] for h in history]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Training Round (Sequence)')
    ax1.set_ylabel('ESKF ATE (meters)', color=color, fontweight='bold')
    ate_rounds  = [h['round'] for h in history if h.get('ate') is not None]
    ate_vals    = [h['ate']   for h in history if h.get('ate') is not None]
    if ate_vals:
        ax1.plot(ate_rounds, ate_vals, marker='o', color=color, linewidth=2, label='Physical Drift (ATE)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Neural Train Loss', color=color, fontweight='bold')
    ax2.plot(rounds, losses, marker='s', linestyle='--', color=color, alpha=0.7, label='Train Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('TALOS NIO : Incremental Training & Physical Veto Dashboard', fontweight='bold')
    fig.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='Nymeria_download_urls.json')
    parser.add_argument('--root',     default='/mnt/c/TALOS/nymeria')
    parser.add_argument('--golden',   default='/mnt/c/TALOS/golden')
    args = parser.parse_args()

    root, golden = Path(args.root), Path(args.golden)
    root.mkdir(parents=True, exist_ok=True)
    golden.mkdir(parents=True, exist_ok=True)
    run_dir = golden / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f":: Run directory: {run_dir.name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.manifest) as f:
        manifest = json.load(f)['sequences']

    import random
    train_seqs = [(sid, e) for sid, e in manifest.items() if VAL_SUBJECT not in sid]
    def _on_disk(item):
        sid, e = item
        bundle = e.get("recording_head", {})
        fn = bundle.get("filename", "")
        return 0 if (root / Path(fn).stem).exists() else 1
    train_seqs.sort(key=_on_disk)
    # Shuffle within each group independently
    on_disk  = [x for x in train_seqs if _on_disk(x) == 0]
    off_disk = [x for x in train_seqs if _on_disk(x) == 1]
    random.shuffle(on_disk)
    train_seqs = on_disk + off_disk
    val_seqs   = [(sid, e) for sid, e in manifest.items() if VAL_SUBJECT in sid]

    print(f"\n:: Pre-loading ESKF Validation Baseline ({VAL_SUBJECT}) ::")
    val_sid, val_entry = val_seqs[0]
    val_seq_path = download_sequence(val_sid, val_entry, root)
    import pickle
    _val_cache = Path("/mnt/c/TALOS/golden/cache") / f"{val_seq_path.parent.name}_val_stream.pkl"
    if _val_cache.exists():
        print(f"  [cache] HIT val_stream")
        val_df, val_gravity = pickle.load(open(_val_cache, "rb"))
    else:
        val_df, val_gravity = load_continuous_val_stream(val_seq_path)
        pickle.dump((val_df, val_gravity), open(_val_cache, "wb"))
        print(f"  [cache] val_stream saved.")
    val_data = load_sequence_cached(val_seq_path, augment=False)
    print(f"  Val Sequence loaded. Duration: {len(val_df)*ESKF_DT:.1f}s")

    model      = SpectralMLP().to(device)
    opt        = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched      = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-5)
    train_data = None
    history    = []

    bad_rounds    = 0
    best_ate_ever = float('inf')
    best_ate_round = -1

    best_loss_ever       = float('inf')
    loss_stagnant_rounds = 0

    print("\n:: Commencing Incremental Training Loop ::")
    for round_idx, (sid, entry) in enumerate(train_seqs, start=1):
        free = shutil.disk_usage(root).free / 1e9
        if free < STORAGE_FLOOR_GB:
            print(f"!! Storage below {STORAGE_FLOOR_GB}GB. Halting.")
            break

        print(f"\n:: Round {round_idx} : {sid[:15]}... :: (Free disk: {free:.1f} GB)")
        seq_path = download_sequence(sid, entry, root)
        if not seq_path: continue

        try:
            new_data   = load_sequence_cached(seq_path)
            train_data = accumulate(train_data, new_data)
        except Exception as e:
            print(f"  !! Load failed: {e}")
            continue

        print(f"  [Train] Pool size: {train_data['trans'].shape[0]:,} windows")
        train_final, _ = train_round(model, opt, sched, train_data, val_data, device,
                                     EPOCHS_PER_ROUND, golden / 'talos.pth')

        if best_loss_ever - train_final > LOSS_MIN_DELTA:
            best_loss_ever       = train_final
            loss_stagnant_rounds = 0
        else:
            loss_stagnant_rounds += 1
            print(f"  !! Loss stagnant : Strike {loss_stagnant_rounds}/{LOSS_PATIENCE}")

        if loss_stagnant_rounds >= LOSS_PATIENCE:
            print(f"\n!! NEURAL STAGNATION DETECTED. Loss flat for {LOSS_PATIENCE} rounds. Halting.")
            break

        if train_final > WARMUP_LOSS_THRESHOLD:
            print(f"  [ESKF]  Skipped : warming up (loss={train_final:.4f} > {WARMUP_LOSS_THRESHOLD})")
            history.append({'round': round_idx, 'ate': None, 'train_loss': train_final})
            update_master_dashboard(history, run_dir / 'master_telemetry.png')
            continue

        print("  [ESKF]  Integrating validation sequence...")
        # Skip stationary period (first 313s), evaluate on real walking only
        val_df_walk = val_df.iloc[313*100:].reset_index(drop=True)
        mean_ate = evaluate_eskf(model, val_df_walk, val_gravity, device, round_idx, run_dir, max_seconds=300)
        print(f"  [Result] Neural Loss: {train_final:.4f} | ESKF ATE: {mean_ate:.3f}m")

        history.append({'round': round_idx, 'ate': mean_ate, 'train_loss': train_final})
        update_master_dashboard(history, run_dir / 'master_telemetry.png')

        if mean_ate < best_ate_ever:
            best_ate_ever = mean_ate
            best_ate_round = round_idx
            bad_rounds    = 0
            shutil.copy(golden / 'talos.pth', run_dir / 'talos_best_physical.pth')
            print(f"  [Best]  New best ATE: {mean_ate:.3f}m : checkpoint saved.")
        else:
            bad_rounds += 1
            print(f"  !! ATE degrading : Strike {bad_rounds}/{PATIENCE}")

        if bad_rounds >= PATIENCE:
            print(f"\n!! PHYSICAL OVERFITTING DETECTED. ESKF drift worsened for {PATIENCE} rounds. Halting.")
            break

    print(f"\n:: Training Complete ::")
    print(f"   Best ATE : {best_ate_ever:.3f}m")
    print(f"   Achieved : Round {best_ate_round}")
    print(f"   Checkpoint : golden/talos_best_physical.pth")
    import subprocess
    subprocess.run(["curl", "-s", "-d",
        f"TALOS done. Best ATE: {best_ate_ever:.3f}m @ Round {best_ate_round}/{round_idx}",
        "ntfy.sh/talos-aman-lab"], capture_output=True)
    import subprocess
    subprocess.run(["python3", "notion_logger.py",
        "--ate",   str(round(best_ate_ever, 3)),
        "--round", str(best_ate_round),
        "--total", str(round_idx),
        "--run", run_dir.name],
        cwd="/mnt/c/TALOS")

if __name__ == '__main__':
    main()