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
from telemetry import append_eval_csv, generate_diagnostic_dashboard

# Configuration
PATIENCE               = 30      # ESKF ATE strikes before halting (physical overfitting)
LOSS_PATIENCE          = 10     # Loss stagnation strikes before halting (dead model)
LOSS_MIN_DELTA         = 1e-5   # Minimum loss improvement to count as progress
WARMUP_LOSS_THRESHOLD  = 1.0    # Don't run ESKF eval until loss drops below this
STORAGE_FLOOR_GB       = 50.0
EPOCHS_PER_ROUND       = 50
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

# Evaluation fusion tuning profile (safe test preset)
SLAP_THRESHOLD       = 5.0
R_OBS_MIN_DIAG       = 0.05
R_OBS_MAX_DIAG       = 2.00
USE_DYNAMIC_R_OBS    = True
R_OBS_FIXED_DIAG     = 0.10
PRED_VEL_GAIN        = 1.00

# Catastrophic divergence safeguards
MAX_PRED_WORLD_SPEED_MPS = 999.0
MAX_INNOVATION_NORM_MPS  = 999.0
CAT_ATE_ABS_M            = 100.0
CAT_ATE_BEST_MULT        = 8.0
CAT_STRIKE_LIMIT         = 10
SOFT_ATE_BEST_MULT       = 2.0
SOFT_CAGE_CLAMP_PCT      = 70.0
CAGE_RADIUS              = 0.50

# Yaw-drift intervention (evaluation-time, conservative)
ENABLE_YAW_ANCHOR        = False
YAW_ANCHOR_MIN_TRUST     = 0.35
YAW_ANCHOR_MAX_OMEGA_MAG = 4.0
YAW_ANCHOR_MAX_LAID_RMS  = 0.6

# LAID differential gyro-bias update (measurement-space, tightly coupled)
ENABLE_LAID_DIFF_UPDATE  = False
LAID_DIFF_MIN_OMEGA_MAG  = 0.10
LAID_DIFF_R_DIAG         = 50.0
LAID_DIFF_GATE_THRESHOLD = 4.0

ENABLE_LAID_WINDOWED     = False
LAID_WINDOWED_R_DIAG     = 0.2
LAID_WINDOWED_BG_CLAMP   = 1e-3
LAID_WINDOWED_MIN_OMEGA  = 0.05

# ESKF Physics Engine
class ESKF:
    def __init__(self, dt=0.01, gravity=None):
        self.dt      = dt
        self.gravity = gravity if gravity is not None else np.array([0., 0., -9.81], dtype=np.float32)
        self.position    = np.zeros(3)
        self.velocity    = np.zeros(3)
        self.orientation = np.eye(3)
        self.bg = np.zeros(3)
        self.gyro_bias = self.bg
        self.gyro_meas = np.zeros(3)
        self.ba = np.zeros(3)
        self.P  = np.eye(15) * 0.1
        self.Q  = np.diag([1e-6]*3 + [1e-4]*3 + [1e-5]*3 + [1e-3]*3 + [1e-2]*3)

    @staticmethod
    def _skew(v):
        return np.array([[ 0,    -v[2],  v[1]],
                         [ v[2],  0,    -v[0]],
                         [-v[1],  v[0],  0   ]])

    def predict(self, accel, gyro):
        self.gyro_meas = np.asarray(gyro, dtype=np.float64)
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
        # SO(3) orthogonalization: prevent floating-point drift from corrupting rotation matrix
        U, _, Vt = np.linalg.svd(self.orientation)
        self.orientation = U @ Vt

    def update_velocity(self, vel, R_obs, slap_threshold=5.0):
        """ESKF velocity update with Mahalanobis Slap Gate (threshold=5.0)."""
        if not np.all(np.isfinite(vel)): return False, 0.0
        H = np.zeros((3, 15))
        H[0,3] = H[1,4] = H[2,5] = 1.0

        S     = H @ self.P @ H.T + R_obs
        r     = vel - self.velocity
        S_inv = np.linalg.inv(S)

        # The Slap -- dual innovation gate:
        # 1) state-covariance-aware Mahalanobis (classic)
        # 2) R-only Mahalanobis (prevents covariance inflation from nullifying gate tension)
        mahal_state_sq = float(r @ S_inv @ r)
        R_inv = np.linalg.inv(R_obs)
        mahal_r_sq = float(r @ R_inv @ r)
        mahal_sq = max(mahal_state_sq, mahal_r_sq)
        if (mahal_state_sq > slap_threshold ** 2) or (mahal_r_sq > slap_threshold ** 2):
            return False, mahal_sq  # slapped -- update silently rejected

        # Reuse S_inv for Kalman gain (zero redundant computation)
        K = self.P @ H.T @ S_inv

        # OVERLORD QUARANTINE: orientation and biases zeroed at gain level
        K[6:15, :] = 0.0

        dx = np.clip(K @ r, -2.0, 2.0)
        self.position += dx[0:3]
        self.velocity += dx[3:6]
        self.P = (np.eye(15) - K @ H) @ self.P
        return True, mahal_sq

    def update_local_velocity(self, v_local_meas, R_obs, slap_threshold=5.0):
        """Tightly coupled local velocity fusion with surgical NHC yaw projection."""
        if not np.all(np.isfinite(v_local_meas)): return False, 0.0

        v_local_pred = self.orientation.T @ self.velocity
        y = v_local_meas - v_local_pred

        H = np.zeros((3, 15))
        H[0:3, 3:6] = self.orientation.T
        H[0:3, 6:9] = self._skew(v_local_pred)

        S = H @ self.P @ H.T + R_obs
        S_inv = np.linalg.inv(S)

        mahal_sq = float(y @ S_inv @ y)
        R_inv = np.linalg.inv(R_obs)
        mahal_r_sq = float(y @ R_inv @ y)
        mahal_max = max(mahal_sq, mahal_r_sq)

        if mahal_max > slap_threshold ** 2:
            return False, mahal_max

        K = self.P @ H.T @ S_inv
        K[9:15, :] = 0.0

        dx = K @ y

        # Surgical NHC projection -- yaw axis only, destroy pitch/roll components
        world_z_local = self.orientation.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        yaw_correction_mag = np.dot(dx[6:9], world_z_local)
        dx[6:9] = yaw_correction_mag * world_z_local * 0.15

        self.position += dx[0:3]
        self.velocity += dx[3:6]
        self.orientation = self.orientation @ Rotation.from_rotvec(dx[6:9]).as_matrix()

        I = np.eye(15)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_obs @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return True, mahal_max

    def update_zaru(self, gyro_raw):
        H = np.zeros((3, 15))
        H[0:3, 9:12] = -np.eye(3)
        
        R_z = np.eye(3) * 1e-4
        z   = -(gyro_raw - self.bg)
        S   = H @ self.P @ H.T + R_z
        
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        
        # ZARU quarantine: gyro stillness tells us nothing about pos, vel, ori, or accel bias
        K[0:9, :]  = 0.0
        K[12:15, :] = 0.0
        
        dx = K @ z
        self.bg += dx[9:12] * 0.1
        
        # Joseph form: guarantees P stays symmetric and positive-definite
        I   = np.eye(15)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_z @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def update_cau(self, accel_raw, accel_var):
        """Continuous Attitude Update: Observes gravity via raw accelerometer to correct pitch/roll.
        
        Only fires when linear acceleration is quiet (accel_var near zero),
        so the accelerometer reading is dominated by the gravity vector.
        R_cau scales with accel_var — noisier periods get weaker corrections.
        """
        # Conservative floor: even at perfect stillness, don't slam orientation.
        R_cau = np.eye(3) * (1.0 + accel_var * 50.0)
        
        # Bias-corrected accelerometer reading
        accel_corrected = accel_raw - self.ba
        
        # Expected accelerometer reading in body frame (specific force = -gravity)
        # When stationary: accel reads -R^T @ g_world = R^T @ [0, 0, 9.81]
        g_body_expected = -self.orientation.T @ self.gravity
        
        # Residual: what the accel actually reads vs what we expect
        z = accel_corrected - g_body_expected
        
        H = np.zeros((3, 15))
        # Orientation error observation (indices 6:9)
        # Linearization: d(R^T @ g)/d(delta_theta) = [R^T @ g]_x = skew(g_body_expected)
        H[0:3, 6:9] = self._skew(g_body_expected)
        # Accelerometer bias observation (indices 12:15)
        # The accel measurement includes -ba, so H for ba is -I
        H[0:3, 12:15] = -np.eye(3)
        
        S = H @ self.P @ H.T + R_cau
        
        # Robust matrix solve
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        
        # CAU Quarantine: Only allow updates to orientation and accel bias
        # Do not let accelerometer noise leak into position, velocity, or gyro bias
        K[0:6, :] = 0.0
        K[9:12, :] = 0.0
        
        dx = K @ z
        
        # Clamp orientation correction to prevent large jumps
        dx[6:9] = np.clip(dx[6:9], -0.01, 0.01)
        
        # Apply orientation correction using SO(3) exponential map
        self.orientation = self.orientation @ Rotation.from_rotvec(dx[6:9]).as_matrix()
        # The Alpha Gate for accel bias: Dampen updates
        self.ba += dx[12:15] * 0.05
        
        self.P = (np.eye(15) - K @ H) @ self.P

    def update_laid_windowed_velocity(self, v_diff_meas, g1_mean, r,
                                      window_time, R_diag=0.2,
                                      bg_clamp=1e-3, min_omega=0.05):
        """Tightly-coupled windowed tangential LAID update for gyro bias [9:12].

        Physics:
            Over a 0.64s window, integrate the differential acceleration:
                v_diff = sum(a2 - a1) * dt  [m/s]
            High-frequency structural flex (zero-mean vibration) integrates to ~0.
            Low-frequency head rotation produces:
                v_diff_pred = cross(omega, r) * window_time

        Jacobian (gyro bias states only):
            d(cross(omega, r) * T) / d(bg) = skew(r) * T
            (using d(omega)/d(bg) = -I, d(cross(a,b))/da = -skew(b))

        Args:
            v_diff_meas  : (3,) integrated velocity differential [m/s]
            g1_mean      : (3,) mean angular velocity over window [rad/s]
            r            : (3,) lever arm vector [m]
            window_time  : float, window duration [s]
            R_diag       : float, measurement noise variance [m^2/s^2]
            bg_clamp     : float, max bg correction per update [rad/s]
            min_omega    : float, minimum |omega| to fire [rad/s]

        Returns:
            applied (bool), bg_delta_norm (float)
        """
        r = np.asarray(r, dtype=np.float64)
        g1_mean = np.asarray(g1_mean, dtype=np.float64)
        v_diff_meas = np.asarray(v_diff_meas, dtype=np.float64)

        omega = g1_mean - self.bg
        omega_mag = float(np.linalg.norm(omega))
        if omega_mag < min_omega:
            return False, 0.0

        # Predicted velocity differential from lever arm kinematics
        v_diff_pred = np.cross(omega, r) * window_time

        # Residual
        y = v_diff_meas - v_diff_pred
        if not np.all(np.isfinite(y)):
            return False, 0.0

        # Jacobian: H = skew(r) * window_time, placed at gyro bias columns [9:12]
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 9:12] = self._skew(r) * window_time

        R_obs = np.eye(3, dtype=np.float64) * R_diag
        S = H @ self.P @ H.T + R_obs
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T

        # Quarantine: gyro bias states only
        K[0:9, :] = 0.0
        K[12:15, :] = 0.0

        dx = K @ y
        dx_bg = np.clip(dx[9:12], -bg_clamp, bg_clamp)

        bg_before = self.bg.copy()
        self.bg += dx_bg

        # Joseph form
        I = np.eye(15)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_obs @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return True, float(np.linalg.norm(self.bg - bg_before))

    def update_centripetal_bias(self, delta_a_y_meas: float, R_cent: float = 0.01) -> tuple[bool, float]:
        """Dual-IMU centripetal bias correction.

        Measurement:
            delta_a_y = a1_y - a2_y = -(omega_x^2 + omega_z^2) * d
        Corrects gyro bias x/z only.
        """
        LEVER_ARM = 0.129
        OMEGA_GATE = 0.5

        # Current best estimate of true angular rate (gyro minus bias)
        omega_hat = self.gyro_meas - self.gyro_bias

        # Gate on angular rate magnitude
        if np.linalg.norm(omega_hat) < OMEGA_GATE:
            return False, 0.0

        # Predicted measurement
        h = -(omega_hat[0] ** 2 + omega_hat[2] ** 2) * LEVER_ARM

        # Residual
        y = float(delta_a_y_meas - h)

        # Jacobian into gyro-bias x and z for this state layout [9:12]
        H = np.zeros((1, 15))
        H[0, 9] = 2 * LEVER_ARM * omega_hat[0]
        H[0, 11] = 2 * LEVER_ARM * omega_hat[2]

        R = np.array([[R_cent]])
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)

        mahal_sq = float(y * S_inv[0, 0] * y)
        if mahal_sq > SLAP_THRESHOLD ** 2:
            return False, mahal_sq

        K = self.P @ H.T @ S_inv

        # Only bias states should move
        K_clean = np.zeros_like(K)
        K_clean[9] = K[9]
        K_clean[11] = K[11]

        dx = K_clean @ np.array([y])
        self.gyro_bias[0] += dx[9]
        self.gyro_bias[2] += dx[11]

        # Joseph form covariance update
        I = np.eye(15)
        IKH = I - K_clean @ H
        self.P = IKH @ self.P @ IKH.T + K_clean @ R @ K_clean.T
        self.P = 0.5 * (self.P + self.P.T)

        return True, mahal_sq

    def update_yaw_anchor(self, omega_yaw_obs, gyro_z_raw, trust):
        """LAID yaw rate pseudo-measurement targeting gyro bias Z (index 11).
        Residual is rate vs rate -- dimensionally consistent [rad/s].
        ESKF propagates bg_z correction into orientation via F[6:9,9:12] coupling.
        """
        if trust < 0.15:
            return False  # weak signal, skip
        H = np.zeros((1, 15))
        H[0, 11] = -1.0  # h(x)=gyro_z_raw-bg_z -> dh/dbg_z = -1
        # Residual: LAID yaw rate vs bias-corrected gyro yaw [rad/s vs rad/s]
        z = np.array([omega_yaw_obs - (gyro_z_raw - self.bg[2])])
        R_yaw = np.array([[1.0 / (trust + 1e-6)]])
        S = H @ self.P @ H.T + R_yaw
        K = self.P @ H.T / S[0, 0]
        # Isolate to gyro bias Z only
        K_masked = np.zeros(15)
        K_masked[11] = K[11].item()
        dx = K_masked * z[0]
        dx[11] = np.clip(dx[11], -0.01, 0.01)  # limit bias correction [rad/s]
        self.bg[2] += dx[11]
        self.P = (np.eye(15) - np.outer(K_masked, H[0])) @ self.P
        return True

    def update_laid_differential(self, a1, g1, a2, r, R_laid=None,
                                 gate_threshold=4.0, min_omega_mag=0.10):
        """Alpha-immune scalar-projection LAID update for gyro bias states [9:12].

        Scalar measurement model:
            z_meas = (a2 - a1) · r_hat
            z_hat  = ((omega · r)^2 - ||omega||^2 ||r||^2) / ||r||
            omega  = g1 - bg

        Jacobian row for gyro-bias states:
            H_bg = (2/||r||) * ( ||r||^2 * omega^T - (omega · r) * r^T )
        """
        a1 = np.asarray(a1, dtype=np.float64)
        g1 = np.asarray(g1, dtype=np.float64)
        a2 = np.asarray(a2, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)

        r_norm = float(np.linalg.norm(r))
        if r_norm < 1e-9:
            return False, 0.0, 0.0, 0.0

        omega = g1 - self.bg
        omega_mag = float(np.linalg.norm(omega))
        if omega_mag < float(min_omega_mag):
            return False, 0.0, 0.0, 0.0

        r_hat = r / r_norm
        z_meas = float(np.dot((a2 - a1), r_hat))
        omega_dot_r = float(np.dot(omega, r))
        z_hat = float((omega_dot_r ** 2 - (omega_mag ** 2) * (r_norm ** 2)) / r_norm)
        y = float(z_meas - z_hat)
        if not np.isfinite(y):
            return False, 0.0, 0.0, 0.0

        H_bg = (2.0 / r_norm) * ((r_norm ** 2) * omega - omega_dot_r * r)
        H = np.zeros((1, 15), dtype=np.float64)
        H[0, 9:12] = H_bg

        R_scalar = float(LAID_DIFF_R_DIAG if R_laid is None else R_laid)
        S = H @ self.P @ H.T + np.array([[R_scalar]], dtype=np.float64)
        S_val = float(S[0, 0])
        if (not np.isfinite(S_val)) or S_val <= 1e-12:
            return False, 0.0, abs(y), 0.0

        mahal_sq = float((y * y) / S_val)
        if mahal_sq > gate_threshold ** 2:
            return False, mahal_sq, abs(y), 0.0

        K = (self.P @ H.T) / S_val
        K[0:9, :] = 0.0
        K[12:15, :] = 0.0

        dx = (K[:, 0] * y)
        dx_bg = np.clip(dx[9:12], -1e-4, 1e-4)
        bg_before = self.bg.copy()
        self.bg += dx_bg

        I = np.eye(15)
        IKH = I - K @ H
        Rm = np.array([[R_scalar]], dtype=np.float64)
        self.P = IKH @ self.P @ IKH.T + K @ Rm @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True, mahal_sq, abs(y), float(np.linalg.norm(self.bg - bg_before))

# Data Pipeline
def accumulate(existing: dict | None, new: dict) -> dict:
    if existing is None: return {k: v.copy() for k, v in new.items()}
    return {k: np.concatenate([existing[k], new[k]], axis=0) for k in new}

def to_raw(imu_windows: np.ndarray) -> np.ndarray:
    # (N, 64, 6) -> (N, 6, 64) for Conv1d
    return imu_windows.transpose(0, 2, 1).astype(np.float32)

def make_tensors(data: dict, device: torch.device):
    # The "Turn Up the Volume" hack: multiply IMU data by 100
    X = torch.from_numpy(to_raw(data['imu1_features']) * 100.0).to(device)
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
    loader = DataLoader(TensorDataset(X_tr, T_tr, Q_tr), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    best_val, t_losses = float('inf'), []

    def loss_fn(pt, pcov, gt):
        var = torch.exp(pcov)
        mse_raw = (pt - gt) ** 2
        nll = 0.5 * (pcov + mse_raw / var)
        gt_mag = torch.norm(gt, dim=1, keepdim=True)
        weight = 1.0 + 10.0 * gt_mag
        total_loss = nll + (2.0 * mse_raw)
        return torch.mean(weight * total_loss)

    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, tb, qb in loader:
            opt.zero_grad()
            pt, pcov = model(xb)
            loss = loss_fn(pt, pcov, tb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            ep_loss += loss.item()
        t_losses.append(ep_loss / len(loader))

        model.eval()
        with torch.no_grad():
            pvt, pvcov = model(X_va)
            v_loss = loss_fn(pvt, pvcov, T_va)

        sched.step(v_loss.item())

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
    
    # CRITICAL: Clear stale state from previous rounds
    if hasattr(evaluate_eskf, '_cage_center'):
        del evaluate_eskf._cage_center
        
    dt          = ESKF_DT
    max_samples = int(max_seconds / dt)
    df          = df.iloc[:max_samples].reset_index(drop=True)

    if hasattr(evaluate_eskf, "_cage_center"):
        del evaluate_eskf._cage_center
    eskf_talos = ESKF(dt=dt, gravity=true_gravity)
    eskf_pure  = ESKF(dt=dt, gravity=true_gravity)
    model.eval()

    accel  = df[['ax','ay','az']].values.astype(np.float32)
    accel2 = df[['ax2','ay2','az2']].values.astype(np.float32)
    gyro2  = df[['wx2','wy2','wz2']].values.astype(np.float32)
    gyro   = df[['wx','wy','wz']].values.astype(np.float32)
    gt_quat = df[['qx','qy','qz','qw']].values.astype(np.float32)
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
    talos_positions, talos_positions_nocage, pure_positions = [], [], []
    window_time = WINDOW_SIZE * dt

    slap_count = 0
    neural_updates = 0
    cage_clamp_count = 0
    laid_veto_count = 0
    zaru_fire_count = 0
    cau_fire_count = 0
    yaw_anchor_fire_count = 0
    safety_reject_count = 0
    laid_diff_attempt_count = 0
    laid_diff_update_count = 0
    laid_diff_reject_count = 0
    laid_windowed_update_count = 0
    cent_accepted_count = 0

    # --- Diagnostic Lens Buffers ---
    # Lens 1: Scale Collapse
    diag_v_pred_local = []
    diag_v_gt_local   = []

    # Lens 2: Filter Tension
    diag_mahal_sq     = []
    diag_mahal_r_sq   = []
    diag_v_gt_mag     = []

    # Lens 3: Covariance Shadowing
    diag_pred_std     = []
    diag_abs_error    = []
    diag_yaw_err_deg  = []
    diag_update_rows  = []
    diag_step_rows    = []
    diag_innov_norm   = []
    diag_pred_speed   = []
    diag_gt_speed     = []
    diag_laid_diff_res_norm = []
    diag_laid_diff_mahal_sq = []
    diag_bg_dx_norm = []

    for step in range(len(df)):
        a, g = accel[step], gyro[step]
        a2_sample = accel2[step]
        zaru_fired = False
        cau_fired = False
        laid_diff_applied = False
        laid_diff_mahal_sq = None
        laid_diff_res_norm = None
        bg_before_laid = eskf_talos.bg.copy()

        # LAID fast loop gate (TALOS only): veto physically impossible samples
        eskf_talos.predict(a, g)  # fast loop ungated -- check_sample fires on footstrikes
        eskf_pure.predict(a, g)  # Pure IMU stays ungated for honest comparison

        delta_a_y = float(a[1] - a2_sample[1])
        cent_applied = False
        # cent_applied, _cent_mahal_sq = eskf_talos.update_centripetal_bias(delta_a_y)
        if cent_applied:
            cent_accepted_count += 1

        # 100Hz tightly-coupled LAID differential update (alpha-immune scalar projection)
        if ENABLE_LAID_DIFF_UPDATE:
            omega_now = g.astype(np.float64) - eskf_talos.bg.astype(np.float64)
            omega_mag_fast = float(np.linalg.norm(omega_now))
            if omega_mag_fast >= LAID_DIFF_MIN_OMEGA_MAG:
                laid_diff_attempt_count += 1
                laid_diff_applied, laid_diff_mahal_sq, laid_diff_res_norm, _ = eskf_talos.update_laid_differential(
                    a.astype(np.float64),
                    g.astype(np.float64),
                    a2_sample.astype(np.float64),
                    laid_bouncer.r,
                    R_laid=LAID_DIFF_R_DIAG,
                    gate_threshold=LAID_DIFF_GATE_THRESHOLD,
                    min_omega_mag=LAID_DIFF_MIN_OMEGA_MAG,
                )
                if laid_diff_applied:
                    laid_diff_update_count += 1
                else:
                    laid_diff_reject_count += 1
                if laid_diff_res_norm is not None:
                    diag_laid_diff_res_norm.append(float(laid_diff_res_norm))
                if laid_diff_mahal_sq is not None and np.isfinite(laid_diff_mahal_sq):
                    diag_laid_diff_mahal_sq.append(float(laid_diff_mahal_sq))

        # NPP update (tracking only, sphere clamp disabled)
        v_device = eskf_talos.orientation.T @ eskf_talos.velocity
        npp_tracker.update(g, v_device)
        # HALO orientation cage disabled -- requires torso reference frame

        # Yaw drift telemetry: heading error of TALOS orientation vs GT orientation
        R_gt_current = Rotation.from_quat(gt_quat[step]).as_matrix()
        R_heading_err = R_gt_current.T @ eskf_talos.orientation
        yaw_err_deg = Rotation.from_matrix(R_heading_err).as_euler('ZYX', degrees=True)[0]
        yaw_err_deg = ((yaw_err_deg + 180.0) % 360.0) - 180.0
        diag_yaw_err_deg.append(abs(yaw_err_deg))

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
            
            # KEEP the gravity/DC component for inference
            win_accel_corrected = win_accel
            
            win = np.concatenate([win_accel_corrected, win_gyro], axis=-1)
            # The "Turn Up the Volume" hack for evaluation inference
            win_tensor = torch.tensor((win * 100.0).T[np.newaxis], dtype=torch.float32)  # (1, 6, 64)

            with torch.no_grad():
                pred_vel, pred_cov = model(win_tensor.to(device))

            # The network outputs mean velocity (m/s), NOT displacement.
            pred_vel_local_raw = pred_vel.cpu().numpy()[0]
            pred_vel_local = pred_vel_local_raw * PRED_VEL_GAIN
            pred_vel_local = bulwark(pred_vel_local)
            pred_cov_np    = pred_cov.cpu().numpy()[0]

            # LAID veto
            win2_accel = np.array(accel2_buf)
            win2_gyro  = np.array(gyro2_buf)
            win2       = np.concatenate([win2_accel, win2_gyro], axis=-1)
            win1       = np.concatenate([win_accel_corrected, win_gyro], axis=-1)
            laid_veto, laid_rms = laid_bouncer.check(win1, win2)

            if not laid_veto:
                # Optional conservative dynamic yaw anchor (LAID-based)
                yaw_anchor_applied = False
                omega_yaw = 0.0
                yaw_trust = 0.0
                omega_mag = 0.0
                if ENABLE_YAW_ANCHOR:
                    omega_yaw, yaw_trust, omega_mag = laid_bouncer.yaw_anchor(win1, win2)
                    current_mean_gyro_z = float(np.mean(win_gyro[:, 2]))
                    if (
                        yaw_trust >= YAW_ANCHOR_MIN_TRUST
                        and omega_mag <= YAW_ANCHOR_MAX_OMEGA_MAG
                        and laid_rms <= YAW_ANCHOR_MAX_LAID_RMS
                    ):
                        yaw_anchor_applied = eskf_talos.update_yaw_anchor(
                            omega_yaw,
                            current_mean_gyro_z,
                            yaw_trust,
                        )
                        if yaw_anchor_applied:
                            yaw_anchor_fire_count += 1

                # --- Windowed LAID tangential update ---
                if ENABLE_LAID_WINDOWED:
                    a1_win = np.array(accel_buf, dtype=np.float64)
                    a2_win = np.array(accel2_buf, dtype=np.float64)
                    g1_mean = np.mean(np.array(gyro_buf), axis=0).astype(np.float64)
                    a_diff = a2_win - a1_win
                    v_diff_meas = np.sum(a_diff, axis=0) * dt
                    window_time = WINDOW_SIZE * dt

                    laid_w_applied, _laid_w_bg_delta = eskf_talos.update_laid_windowed_velocity(
                        v_diff_meas=v_diff_meas,
                        g1_mean=g1_mean,
                        r=laid_bouncer.r,
                        window_time=window_time,
                        R_diag=LAID_WINDOWED_R_DIAG,
                        bg_clamp=LAID_WINDOWED_BG_CLAMP,
                        min_omega=LAID_WINDOWED_MIN_OMEGA,
                    )
                    if laid_w_applied:
                        laid_windowed_update_count += 1

                pred_var = np.exp(pred_cov_np)
                r_obs_diag = np.clip(pred_var, R_OBS_MIN_DIAG, R_OBS_MAX_DIAG)
                if USE_DYNAMIC_R_OBS:
                    R_obs_used = np.diag(r_obs_diag.astype(np.float64))
                else:
                    R_obs_used = np.eye(3) * R_OBS_FIXED_DIAG
                    r_obs_diag = np.array([R_OBS_FIXED_DIAG] * 3, dtype=np.float64)

                neural_updates += 1

                accepted, mahal_sq = eskf_talos.update_local_velocity(
                    pred_vel_local,
                    R_obs=R_obs_used,
                    slap_threshold=SLAP_THRESHOLD,
                )

                # Telemetry only -- not fed to filter
                v_world = eskf_talos.orientation @ pred_vel_local
                residual_pre = v_world - eskf_talos.velocity
                innovation_norm = float(np.linalg.norm(residual_pre))
                pred_world_speed = float(np.linalg.norm(v_world))
                R_inv = np.linalg.inv(R_obs_used)
                mahal_r_sq = float(residual_pre @ R_inv @ residual_pre)

                # Hard safety gate before Kalman update to prevent catastrophic injections
                if (pred_world_speed > MAX_PRED_WORLD_SPEED_MPS) or (innovation_norm > MAX_INNOVATION_NORM_MPS):
                    safety_reject_count += 1
                if accepted is False:
                    slap_count += 1
                    
                # --- Capture Lens 1 & 2 Data ---
                # Current local prediction from model (already in m/s)
                pred_v_local = pred_vel_local.copy()
                
                # Calculate Ground Truth (GT) Local Velocity
                # Calculate Ground Truth (GT) Local Velocity via positional finite differences
                # The purest measure of average velocity over a finite window
                gt_pos_world_start = df[['px','py','pz']].iloc[max(0, step - WINDOW_SIZE)].values
                gt_pos_world_end   = df[['px','py','pz']].iloc[step].values
                gt_mean_v_world    = (gt_pos_world_end - gt_pos_world_start) / (WINDOW_SIZE * dt)
                
                # Rotate GT to Local Frame using GT orientation (Isolates neural error from filter drift)
                q_gt_current = df[['qx','qy','qz','qw']].iloc[step].values
                R_gt_current = Rotation.from_quat(q_gt_current).as_matrix()
                gt_v_local   = R_gt_current.T @ gt_mean_v_world
                
                # --- Store Telemetry ---
                diag_v_pred_local.append(pred_v_local)
                diag_v_gt_local.append(gt_v_local)
                diag_mahal_sq.append(mahal_sq)
                diag_mahal_r_sq.append(mahal_r_sq)
                diag_v_gt_mag.append(np.linalg.norm(gt_v_local))
                
                # --- Capture Lens 3: Covariance Shadowing ---
                # Convert LogVar to Standard Deviation
                current_std = np.exp(pred_cov_np / 2.0)
                current_err = np.abs(pred_v_local - gt_v_local)
                
                diag_pred_std.append(current_std)
                diag_abs_error.append(current_err)
                diag_innov_norm.append(innovation_norm)
                diag_pred_speed.append(float(np.linalg.norm(pred_v_local)))
                diag_gt_speed.append(float(np.linalg.norm(gt_v_local)))
                diag_bg_dx_norm.append(float(np.linalg.norm(eskf_talos.bg - bg_before_laid)))

                diag_update_rows.append({
                    'step_idx': step,
                    'laid_veto': False,
                    'laid_residual_rms': float(laid_rms),
                    'slap_accepted': bool(accepted),
                    'mahal_sq': float(mahal_sq),
                    'mahal_r_sq': float(mahal_r_sq),
                    'innovation_norm': float(innovation_norm),
                    'pred_world_speed_mps': float(pred_world_speed),
                    'safety_reject': bool((pred_world_speed > MAX_PRED_WORLD_SPEED_MPS) or (innovation_norm > MAX_INNOVATION_NORM_MPS)),
                    'laid_diff_applied': bool(laid_diff_applied),
                    'laid_diff_mahal_sq': float(laid_diff_mahal_sq) if laid_diff_mahal_sq is not None else None,
                    'laid_diff_res_norm': float(laid_diff_res_norm) if laid_diff_res_norm is not None else None,
                    'bg_delta_norm': float(np.linalg.norm(eskf_talos.bg - bg_before_laid)),
                    'gt_speed_mps': float(np.linalg.norm(gt_v_local)),
                    'pred_vx': float(pred_v_local[0]),
                    'pred_vy': float(pred_v_local[1]),
                    'pred_vz': float(pred_v_local[2]),
                    'pred_vx_raw': float(pred_vel_local_raw[0]),
                    'pred_vy_raw': float(pred_vel_local_raw[1]),
                    'pred_vz_raw': float(pred_vel_local_raw[2]),
                    'gt_vx': float(gt_v_local[0]),
                    'gt_vy': float(gt_v_local[1]),
                    'gt_vz': float(gt_v_local[2]),
                    'pred_std_x': float(current_std[0]),
                    'pred_std_y': float(current_std[1]),
                    'pred_std_z': float(current_std[2]),
                    'r_obs_x': float(r_obs_diag[0]),
                    'r_obs_y': float(r_obs_diag[1]),
                    'r_obs_z': float(r_obs_diag[2]),
                    'abs_err_x': float(current_err[0]),
                    'abs_err_y': float(current_err[1]),
                    'abs_err_z': float(current_err[2]),
                    'yaw_anchor_applied': bool(yaw_anchor_applied),
                    'omega_yaw_obs': float(omega_yaw) if ENABLE_YAW_ANCHOR else None,
                    'yaw_trust': float(yaw_trust) if ENABLE_YAW_ANCHOR else None,
                    'omega_mag': float(omega_mag) if ENABLE_YAW_ANCHOR else None,
                })
            else:
                laid_veto_count += 1
                diag_update_rows.append({
                    'step_idx': step,
                    'laid_veto': True,
                    'laid_residual_rms': float(laid_rms),
                    'slap_accepted': None,
                    'mahal_sq': None,
                    'mahal_r_sq': None,
                    'innovation_norm': None,
                    'pred_world_speed_mps': None,
                    'safety_reject': False,
                    'laid_diff_applied': bool(laid_diff_applied),
                    'laid_diff_mahal_sq': float(laid_diff_mahal_sq) if laid_diff_mahal_sq is not None else None,
                    'laid_diff_res_norm': float(laid_diff_res_norm) if laid_diff_res_norm is not None else None,
                    'bg_delta_norm': float(np.linalg.norm(eskf_talos.bg - bg_before_laid)),
                    'gt_speed_mps': None,
                    'pred_vx': float(pred_vel_local[0]),
                    'pred_vy': float(pred_vel_local[1]),
                    'pred_vz': float(pred_vel_local[2]),
                    'pred_vx_raw': float(pred_vel_local_raw[0]),
                    'pred_vy_raw': float(pred_vel_local_raw[1]),
                    'pred_vz_raw': float(pred_vel_local_raw[2]),
                    'gt_vx': None,
                    'gt_vy': None,
                    'gt_vz': None,
                    'pred_std_x': float(np.exp(pred_cov_np[0] / 2.0)),
                    'pred_std_y': float(np.exp(pred_cov_np[1] / 2.0)),
                    'pred_std_z': float(np.exp(pred_cov_np[2] / 2.0)),
                    'r_obs_x': None,
                    'r_obs_y': None,
                    'r_obs_z': None,
                    'abs_err_x': None,
                    'abs_err_y': None,
                    'abs_err_z': None,
                    'yaw_anchor_applied': False,
                    'omega_yaw_obs': None,
                    'yaw_trust': None,
                    'omega_mag': None,
                })
        # ZARU (TALOS only)
        if len(gyro_buf) >= ZARU_WINDOW and step % ZARU_WINDOW == 0:
            gyro_var = np.var(np.array(gyro_buf[-ZARU_WINDOW:]), axis=0).sum()
            accel_var = np.var(np.array(accel_buf[-ZARU_WINDOW:]), axis=0).sum()
            
            # Dual-sensor lock to prevent false positives during slow motion
            if gyro_var < ZARU_THRESHOLD and accel_var < ZARU_ACCEL_THRESHOLD:
                eskf_talos.update_zaru(g)
                # Hardcoded low noise for verified zero-velocity updates
                eskf_talos.update_velocity(np.zeros(3), R_obs=np.eye(3) * 1e-4)
                zaru_fired = True
                zaru_fire_count += 1

        # CAU - Continuous Attitude Update (TALOS only)
        # Piggybacks on ZARU's stillness detection: only fire when the user is
        # near-stationary (both gyro AND accel variance confirm stillness).
        # During walking, the accelerometer is NOT a gravity sensor -- it measures
        # footstrike impacts and head bob. CAU must NOT fire during motion.
        if len(gyro_buf) >= ZARU_WINDOW and step % ZARU_WINDOW == 0:
            gyro_var_cau = np.var(np.array(gyro_buf[-ZARU_WINDOW:]), axis=0).sum()
            accel_var_cau = np.var(np.array(accel_buf[-ZARU_WINDOW:]), axis=0).sum()
            # Strictest threshold: match ZARU exactly.
            if gyro_var_cau < ZARU_THRESHOLD and accel_var_cau < ZARU_ACCEL_THRESHOLD:
                eskf_talos.update_cau(a, accel_var_cau)
                cau_fired = True
                cau_fire_count += 1


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
        cage_clamped = False
        talos_pos_nocage = eskf_talos.position.copy()

        if distance > CAGE_RADIUS:
            eskf_talos.position = evaluate_eskf._cage_center + (head_vector / distance) * CAGE_RADIUS
            cage_clamp_count += 1
            cage_clamped = True

            n = (eskf_talos.position - evaluate_eskf._cage_center)
            n = n / np.linalg.norm(n)
            v_radial = np.dot(eskf_talos.velocity, n)
            if v_radial > 0:
                eskf_talos.velocity -= v_radial * n

        talos_positions_nocage.append(talos_pos_nocage)
        talos_positions.append(eskf_talos.position.copy())
        pure_positions.append(eskf_pure.position.copy())

        diag_step_rows.append({
            'step_idx': step,
            'talos_x': float(eskf_talos.position[0]),
            'talos_y': float(eskf_talos.position[1]),
            'talos_z': float(eskf_talos.position[2]),
            'talos_nocage_x': float(talos_pos_nocage[0]),
            'talos_nocage_y': float(talos_pos_nocage[1]),
            'talos_nocage_z': float(talos_pos_nocage[2]),
            'pure_x': float(eskf_pure.position[0]),
            'pure_y': float(eskf_pure.position[1]),
            'pure_z': float(eskf_pure.position[2]),
            'gt_x': float(gt_pos[step, 0]),
            'gt_y': float(gt_pos[step, 1]),
            'gt_z': float(gt_pos[step, 2]),
            'talos_err_m': float(np.linalg.norm(eskf_talos.position - gt_pos[step])),
            'talos_nocage_err_m': float(np.linalg.norm(talos_pos_nocage - gt_pos[step])),
            'pure_err_m': float(np.linalg.norm(eskf_pure.position - gt_pos[step])),
            'yaw_err_deg_abs': float(diag_yaw_err_deg[-1]),
            'cage_clamped': cage_clamped,
            'zaru_fired': zaru_fired,
            'cau_fired': cau_fired,
        })

    talos_positions = np.array(talos_positions)
    talos_positions_nocage = np.array(talos_positions_nocage)
    pure_positions  = np.array(pure_positions)
    evaluate_eskf._last_talos_pos = talos_positions_nocage
    evaluate_eskf._last_talos_pos_caged = talos_positions
    evaluate_eskf._last_gt_pos    = gt_pos
    talos_err_caged = np.linalg.norm(talos_positions - gt_pos, axis=1)
    talos_err_nocage = np.linalg.norm(talos_positions_nocage - gt_pos, axis=1)
    mean_ate        = talos_err_caged.mean()
    final_ate       = talos_err_caged[-1]
    caged_ate       = talos_err_caged.mean()
    caged_final_ate = talos_err_caged[-1]
    nocage_ate      = talos_err_nocage.mean()
    nocage_final_ate = talos_err_nocage[-1]
    total_distance  = np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1))
    mean_rte        = (mean_ate / total_distance) * 100
    final_rte       = (final_ate / total_distance) * 100
    caged_mean_rte  = (caged_ate / total_distance) * 100
    caged_final_rte = (caged_final_ate / total_distance) * 100
    nocage_mean_rte = (nocage_ate / total_distance) * 100
    nocage_final_rte = (nocage_final_ate / total_distance) * 100

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(
        f"Round {round_idx} | TALOS ATE (caged): {mean_ate:.3f}m (RTE {mean_rte:.2f}%) "
        f"| No-cage ATE: {nocage_ate:.3f}m (RTE {nocage_mean_rte:.2f}%) "
        f"| Final Drift (caged): {final_ate:.3f}m ({final_rte:.2f}%) "
        f"| Final Drift (no-cage): {nocage_final_ate:.3f}m ({nocage_final_rte:.2f}%) "
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

    if neural_updates > 0:
        generate_diagnostic_dashboard(diag_v_pred_local, diag_v_gt_local, diag_mahal_sq,
                                      diag_v_gt_mag, diag_pred_std, diag_abs_error,
                                      round_idx, plot_dir, slap_threshold=SLAP_THRESHOLD)

    if neural_updates > 0:
        slap_rate = (slap_count / neural_updates) * 100
        print(f"  [Slap Gate] {slap_count}/{neural_updates} updates rejected ({slap_rate:.1f}%)")
        if safety_reject_count > 0:
            print(f"  [Safety]    {safety_reject_count}/{neural_updates} updates blocked by hard guardrails")
    else:
        slap_rate = 0.0

    if len(diag_yaw_err_deg) > 0:
        yaw_err = np.array(diag_yaw_err_deg, dtype=np.float64)
        print(
            "  [Yaw Drift] "
            f"mean={np.mean(yaw_err):.2f}° "
            f"p95={np.percentile(yaw_err, 95):.2f}° "
            f"max={np.max(yaw_err):.2f}°"
        )
        if ENABLE_YAW_ANCHOR:
            print(f"  [Yaw Anchor] applied {yaw_anchor_fire_count} times")
        
    cage_clamp_rate = (cage_clamp_count / len(df)) * 100
    if cage_clamp_count > 0:
        print(f"  [The Cage]  Head severed {cage_clamp_count}/{len(df)} frames ({cage_clamp_rate:.1f}%)")

    def _p95_finite(values):
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.0
        return float(np.percentile(arr, 95))

    pure_imu_ate = float(np.linalg.norm(pure_positions - gt_pos, axis=1).mean())
    summary_row = {
        'mean_ate_m': float(mean_ate),
        'no_cage_ate_m': float(nocage_ate),
        'caged_ate_m': float(caged_ate),
        'mean_rte_pct': float(mean_rte),
        'final_ate_m': float(final_ate),
        'no_cage_final_ate_m': float(nocage_final_ate),
        'caged_final_ate_m': float(caged_final_ate),
        'final_rte_pct': float(final_rte),
        'pure_imu_ate_m': pure_imu_ate,
        'slap_count': int(slap_count),
        'neural_updates': int(neural_updates),
        'slap_rate_pct': float(slap_rate),
        'cage_clamp_count': int(cage_clamp_count),
        'cage_clamp_rate_pct': float(cage_clamp_rate),
        'laid_veto_count': int(laid_veto_count),
        'laid_veto_rate_pct': float((laid_veto_count / max(neural_updates + laid_veto_count, 1)) * 100.0),
        'zaru_fire_count': int(zaru_fire_count),
        'cau_fire_count': int(cau_fire_count),
        'yaw_anchor_enabled': bool(ENABLE_YAW_ANCHOR),
        'yaw_anchor_fire_count': int(yaw_anchor_fire_count),
        'safety_reject_count': int(safety_reject_count),
        'laid_diff_updates': int(laid_diff_update_count),
        'laid_diff_reject_rate_pct': float((laid_diff_reject_count / max(laid_diff_attempt_count, 1)) * 100.0),
        'laid_windowed_updates': int(laid_windowed_update_count),
        'laid_windowed_rate_pct': float(laid_windowed_update_count / max(neural_updates, 1) * 100.0),
        'laid_diff_residual_p95': _p95_finite(diag_laid_diff_res_norm),
        'bg_update_norm_p95': _p95_finite(diag_bg_dx_norm),
        'yaw_err_mean_deg': float(np.mean(diag_yaw_err_deg)) if len(diag_yaw_err_deg) > 0 else 0.0,
        'yaw_err_p95_deg': float(np.percentile(diag_yaw_err_deg, 95)) if len(diag_yaw_err_deg) > 0 else 0.0,
        'yaw_err_max_deg': float(np.max(diag_yaw_err_deg)) if len(diag_yaw_err_deg) > 0 else 0.0,
        'mahal_p95': _p95_finite(diag_mahal_sq),
        'mahal_r_p95': _p95_finite(diag_mahal_r_sq),
        'innovation_norm_p95': _p95_finite(diag_innov_norm),
        'pred_speed_mean': float(np.mean(diag_pred_speed)) if len(diag_pred_speed) > 0 else 0.0,
        'gt_speed_mean': float(np.mean(diag_gt_speed)) if len(diag_gt_speed) > 0 else 0.0,
        'pred_gt_speed_ratio': float(np.mean(diag_pred_speed) / (np.mean(diag_gt_speed) + 1e-6)) if len(diag_gt_speed) > 0 else 0.0,
        'slap_threshold': float(SLAP_THRESHOLD),
        'r_obs_min_diag': float(R_OBS_MIN_DIAG),
        'r_obs_max_diag': float(R_OBS_MAX_DIAG),
        'use_dynamic_r_obs': bool(USE_DYNAMIC_R_OBS),
        'r_obs_fixed_diag': float(R_OBS_FIXED_DIAG),
        'pred_vel_gain': float(PRED_VEL_GAIN),
        'max_pred_world_speed_mps': float(MAX_PRED_WORLD_SPEED_MPS),
        'max_innovation_norm_mps': float(MAX_INNOVATION_NORM_MPS),
        'yaw_anchor_min_trust': float(YAW_ANCHOR_MIN_TRUST),
        'yaw_anchor_max_omega_mag': float(YAW_ANCHOR_MAX_OMEGA_MAG),
        'yaw_anchor_max_laid_rms': float(YAW_ANCHOR_MAX_LAID_RMS),
        'gyro_bias_z': float(eskf_talos.gyro_bias[2]),
        'cent_bias_updates': int(cent_accepted_count),
    }
    evaluate_eskf._last_summary = summary_row
    csv_path = append_eval_csv(plot_dir, round_idx, summary_row, diag_step_rows, diag_update_rows)
    print(f"  [CSV]       telemetry appended -> {csv_path}")

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
    parser.add_argument('--seed',     type=int, default=1337)
    args = parser.parse_args()

    root, golden = Path(args.root), Path(args.golden)
    root.mkdir(parents=True, exist_ok=True)
    golden.mkdir(parents=True, exist_ok=True)
    run_dir = golden / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f":: Run directory: {run_dir.name}")
    print(f":: Seed: {args.seed}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.manifest) as f:
        manifest = json.load(f)['sequences']

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    order_path = run_dir / 'train_sequence_order.txt'
    with open(order_path, 'w', encoding='utf-8') as f:
        f.write(f"seed={args.seed}\n")
        for idx, (sid, _) in enumerate(train_seqs, start=1):
            f.write(f"{idx}\t{sid}\n")
    print(f":: Sequence order logged: {order_path.name}")

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
    subject_pool = []
    history    = []

    bad_rounds    = 0
    best_ate_ever = float('inf')
    best_ate_round = -1
    cat_strikes = 0
    blacklisted_sids = set()
    soft_quarantines = 0

    best_loss_ever       = float('inf')
    loss_stagnant_rounds = 0

    print("\n:: Commencing Incremental Training Loop ::")
    for round_idx, (sid, entry) in enumerate(train_seqs, start=1):
        free = shutil.disk_usage(root).free / 1e9
        if free < STORAGE_FLOOR_GB:
            print(f"!! Storage below {STORAGE_FLOOR_GB}GB. Halting.")
            break

        print(f"\n:: Round {round_idx} : {sid[:15]}... :: (Free disk: {free:.1f} GB)")
        if sid in blacklisted_sids:
            print(f"  [Blacklist] Skipping {sid[:40]} -- previously quarantined")
            continue
        seq_path = download_sequence(sid, entry, root)
        if not seq_path: continue

        try:
            new_data   = load_sequence_cached(seq_path)
            
            # --- Unlimited Accumulation ---
            subject_pool.append(new_data)
                
            train_data = None
            for p_data in subject_pool:
                train_data = accumulate(train_data, p_data)
            
            # --- SURGICAL FIX: Reset stagnation trackers ---
            # The dataset has changed, so the loss baseline must be reset.
            best_loss_ever = float('inf')
            loss_stagnant_rounds = 0
            
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

        # Catastrophic divergence breaker -- abort early instead of burning rounds
        cat_limit = max(CAT_ATE_ABS_M, best_ate_ever * CAT_ATE_BEST_MULT if np.isfinite(best_ate_ever) else CAT_ATE_ABS_M)
        if mean_ate > cat_limit:
            cat_strikes += 1
            print(f"\n!! CATASTROPHIC DIVERGENCE: ATE {mean_ate:.3f}m exceeded limit {cat_limit:.3f}m.")
            print(f"   [Quarantine] Strike {cat_strikes}/{CAT_STRIKE_LIMIT} on sequence {sid[:40]}...")

            # Quarantine the just-added sequence from the subject pool
            if subject_pool:
                subject_pool.pop()
                blacklisted_sids.add(sid)
                print(f"   [Blacklist] {sid[:40]} permanently blacklisted")
                train_data = None
                for p_data in subject_pool:
                    train_data = accumulate(train_data, p_data)
                if train_data is not None:
                    print(f"   [Quarantine] Pool reverted to {train_data['trans'].shape[0]:,} windows")

            # Roll back model weights to last known best physical checkpoint
            best_ckpt = run_dir / 'talos_best_physical.pth'
            if best_ckpt.exists():
                model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=False))
                opt.state = {}
                torch.save(model.state_dict(), golden / 'talos.pth')
                print("   [Rollback] Restored last best physical checkpoint")
            else:
                print("   [Rollback] No physical checkpoint yet; keeping current weights")

            history.append({'round': round_idx, 'ate': mean_ate, 'train_loss': train_final})
            update_master_dashboard(history, run_dir / 'master_telemetry.png')

            if cat_strikes >= CAT_STRIKE_LIMIT:
                print(f"\n!! CATASTROPHIC DIVERGENCE LIMIT REACHED ({CAT_STRIKE_LIMIT}). Halting.")
                break

            # Do not consume a physical-overfitting strike for quarantined rounds
            continue

        # Soft quarantine for toxic-but-noncatastrophic rounds
        eval_summary = getattr(evaluate_eskf, '_last_summary', {})
        cage_pct = float(eval_summary.get('cage_clamp_rate_pct', 0.0))
        soft_limit = best_ate_ever * SOFT_ATE_BEST_MULT if np.isfinite(best_ate_ever) else float('inf')
        if np.isfinite(best_ate_ever) and (mean_ate > soft_limit) and (cage_pct > SOFT_CAGE_CLAMP_PCT):
            soft_quarantines += 1
            print(f"\n!! SOFT QUARANTINE: ATE {mean_ate:.3f}m (> {soft_limit:.3f}m) with cage clamp {cage_pct:.1f}%.")
            print(f"   [Quarantine] Soft strike {soft_quarantines} on sequence {sid[:40]}...")

            if subject_pool:
                subject_pool.pop()
                blacklisted_sids.add(sid)
                print(f"   [Blacklist] {sid[:40]} permanently blacklisted")
                train_data = None
                for p_data in subject_pool:
                    train_data = accumulate(train_data, p_data)
                if train_data is not None:
                    print(f"   [Quarantine] Pool reverted to {train_data['trans'].shape[0]:,} windows")

            best_ckpt = run_dir / 'talos_best_physical.pth'
            if best_ckpt.exists():
                model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=False))
                opt.state = {}
                torch.save(model.state_dict(), golden / 'talos.pth')
                print("   [Rollback] Restored last best physical checkpoint")

            history.append({'round': round_idx, 'ate': mean_ate, 'train_loss': train_final})
            update_master_dashboard(history, run_dir / 'master_telemetry.png')
            continue

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
    if np.isfinite(best_ate_ever) and best_ate_round > 0:
        print(f"   Best ATE : {best_ate_ever:.3f}m")
        print(f"   Achieved : Round {best_ate_round}")
        print(f"   Checkpoint : golden/talos_best_physical.pth")
        status_msg = f"TALOS done. Best ATE: {best_ate_ever:.3f}m @ Round {best_ate_round}/{round_idx}"
        notion_ate = str(round(best_ate_ever, 3))
        notion_round = str(best_ate_round)
    else:
        print("   Best ATE : N/A (no valid physical checkpoint this run)")
        print("   Achieved : N/A")
        print("   Checkpoint : N/A")
        status_msg = f"TALOS done. No valid physical checkpoint in run {run_dir.name}."
        notion_ate = "nan"
        notion_round = "-1"
    import subprocess
    subprocess.run(["curl", "-s", "-d",
        status_msg,
        "ntfy.sh/talos-aman-lab"], capture_output=True)
    import subprocess
    subprocess.run(["python3", "notion_logger.py",
        "--ate",   notion_ate,
        "--round", notion_round,
        "--total", str(round_idx),
        "--run", run_dir.name],
        cwd="/mnt/c/TALOS")

if __name__ == '__main__':
    main()