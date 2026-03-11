"""
laid_poc.py
Proof of Concept: Dual-IMU Lever Arm Inertial Disambiguation (LAID)
Demonstrates anchoring unobservable yaw via continuous dynamic gyro bias observation.
"""

import torch, numpy as np, sys, pandas as pd
from pathlib import Path
sys.path.insert(0, '/mnt/c/TALOS')
from SMLP import SpectralMLP
from incremental_train import ESKF, WINDOW_SIZE, ESKF_DT, ZARU_WINDOW, ZARU_THRESHOLD
from scipy.spatial.transform import Rotation
from projectaria_tools.core import data_provider
from nymeria_loader import load_imu_stream, align_imu_streams, load_gt_trajectory, interpolate_gt, SID_RIGHT, SID_LEFT, TARGET_HZ

class LAID_ESKF(ESKF):
    def __init__(self, dt=0.01, gravity=None):
        super().__init__(dt, gravity)
        # TALOS Rev. 1 lever arm: ~15cm separation on the Y-axis
        self.L = np.array([0.0, -0.15, 0.0])
        self.delta_v = np.zeros(3)
        
    def update_laid(self, a1, a2, gyro):
        """
        Calculates the differential velocity between IMU1 and IMU2, 
        and uses the discrepancy from rigid body kinematics to correct the Gyro Bias.
        """
        # 1. Gravity perfectly cancels out during subtraction
        delta_a = a1 - a2
        
        # 2. Leaky integration filters out low-frequency noise while preserving dynamic spikes
        self.delta_v = self.delta_v * 0.95 + delta_a * self.dt
        
        # 3. Rigid body expectation: v_diff = omega x L
        omega = gyro - self.bg
        expected_v = np.cross(omega, self.L)
        r = self.delta_v - expected_v
        
        # 4. H Matrix: d(expected_v) / d(bg) = skew(L)
        H = np.zeros((3, 15))
        H[0:3, 9:12] = self._skew(self.L)
        
        R_obs = np.eye(3) * 1e-3
        S = H @ self.P @ H.T + R_obs
        Si = np.linalg.inv(S)
        
        # Mahalanobis safety gate
        if r @ Si @ r > 16.0: return 0.0  # Return 0 divergence if slap gate triggers
        
        K = self.P @ H.T @ Si
        
        # --- THE LAID QUARANTINE ---
        # Prevent differential math from injecting lateral noise into position/velocity
        K[0:6, :] = 0.0 
        
        dx = np.clip(K @ r, -1.0, 1.0)
        self.orientation = self.orientation @ Rotation.from_rotvec(dx[6:9]).as_matrix()
        self.bg += dx[9:12]
        self.ba += dx[12:15]
        
        self.P = (np.eye(15) - K @ H) @ self.P
        
        # Return physical divergence for the Bouncer logic
        return np.linalg.norm(r)

def main():
    print('[LAID PoC] Engaging Dual-IMU Extraction Protocol...')
    dp = data_provider.create_vrs_data_provider('/mnt/c/TALOS/nymeria/Nymeria_v0.0_20230608_s0_shelby_arroyo_act0_3ciwl8_recording_head/recording_head/data/motion.vrs')
    ts_r, imu_r = load_imu_stream(dp, SID_RIGHT)
    ts_l, imu_l = load_imu_stream(dp, SID_LEFT)
    grid_ns, imu1_reg, imu2_reg = align_imu_streams(ts_r, imu_r, ts_l, imu_l, TARGET_HZ)

    traj_path = '/mnt/c/TALOS/nymeria/Nymeria_v0.0_20230608_s0_shelby_arroyo_act0_3ciwl8_recording_head/recording_head/mps/slam/closed_loop_trajectory.csv'
    gt_ts, gt_pos, gt_quat = load_gt_trajectory(traj_path)
    pos_at_imu, quat_at_imu = interpolate_gt(gt_ts, gt_pos, gt_quat, grid_ns)

    df_traj = pd.read_csv(traj_path)
    true_gravity = df_traj[['gravity_x_world', 'gravity_y_world', 'gravity_z_world']].iloc[0].values.astype(np.float32)

    mask = (grid_ns >= gt_ts[0]) & (grid_ns <= gt_ts[-1])
    a1 = imu1_reg[mask, 0:3]; g1 = imu1_reg[mask, 3:6]
    a2 = imu2_reg[mask, 0:3]

    gt_pos_synced = pos_at_imu[mask]
    gt_pos_synced = gt_pos_synced - gt_pos_synced[0]

    device = torch.device('cuda')
    model = SpectralMLP(198).to(device)
    model.load_state_dict(torch.load('/mnt/c/TALOS/golden/talos_best_physical.pth', map_location=device, weights_only=False))
    model.eval()

    eskf = LAID_ESKF(dt=ESKF_DT, gravity=true_gravity)
    init_rot = Rotation.from_quat(quat_at_imu[mask][0]).as_matrix()
    eskf.orientation = init_rot
    eskf.velocity = init_rot @ df_traj[['device_linear_velocity_x_device', 'device_linear_velocity_y_device', 'device_linear_velocity_z_device']].iloc[0].values

    print('[LAID PoC] Architecture active. Quarantines enforced.')
    accel_buf, gyro_buf, positions = [], [], []

    for step in range(len(a1)):
        eskf.predict(a1[step], g1[step])
        
        # 1. LAID Dynamic Yaw Anchor (Continuous physical correction)
        laid_divergence = eskf.update_laid(a1[step], a2[step], g1[step])
        
        accel_buf.append(a1[step]); gyro_buf.append(g1[step])
        if len(accel_buf) > WINDOW_SIZE: accel_buf.pop(0); gyro_buf.pop(0)
        
        if len(accel_buf) == WINDOW_SIZE and step % 10 == 0:
            win = np.concatenate([accel_buf, gyro_buf], axis=-1)
            fft_flat = np.log1p(np.abs(np.fft.rfft(win[np.newaxis], axis=1))).reshape(1,-1).astype(np.float32)
            with torch.no_grad():
                pred_delta, _ = model(torch.tensor(fft_flat).to(device))
            v_world = eskf.orientation @ (pred_delta.cpu().numpy()[0] / (WINDOW_SIZE * ESKF_DT))
            
            # 2. LAID Bouncer (Physical Veto on Neural Overlord)
            trust = np.clip(1.0 - (laid_divergence * 5.0), 0.01, 1.0)
            R_obs_overlord = np.eye(3) * (0.1 / trust)
            
            # Overlord updates PV, biases are protected by K[6:15]=0 inside update_velocity
            eskf.update_velocity(v_world, R_obs=R_obs_overlord)
            
        if len(gyro_buf) >= ZARU_WINDOW and step % ZARU_WINDOW == 0:
            if np.var(np.array(gyro_buf), axis=0).sum() < ZARU_THRESHOLD:
                eskf.update_zaru(g1[step])
                eskf.update_velocity(np.zeros(3), R_obs=np.eye(3)*1e-4)
                
        positions.append(eskf.position.copy())

    ate = np.linalg.norm(np.array(positions) - gt_pos_synced, axis=1)
    print(f'\n>>> LAID PoC Final Result | Mean ATE: {ate.mean():.3f}m | Final ATE: {ate[-1]:.3f}m <<<')

if __name__ == '__main__':
    main()