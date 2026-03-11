"""
eval_rte.py — TALOS NIO Publication Metrics
Calculates Absolute Trajectory Error (ATE) and Relative Trajectory Error (RTE)
against the James Johnson validation baseline.
"""

import json
import torch
import numpy as np
from pathlib import Path

# Import your hardened architecture and physics engine
from SMLP import SpectralMLP
from incremental_train import (ESKF, load_continuous_val_stream, download_sequence, 
                               WINDOW_SIZE, ESKF_DT, ZARU_WINDOW, ZARU_THRESHOLD, INPUT_DIM)

def compute_total_distance(gt_pos: np.ndarray) -> float:
    """Calculates the cumulative path length of the ground truth trajectory."""
    diffs = np.diff(gt_pos, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return float(np.sum(distances))

def main():
    root = Path('/mnt/c/TALOS/nymeria')
    golden = Path('/mnt/c/TALOS/golden')
    weights_path = golden / 'talos_best_physical.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading hardened weights from {weights_path}...")
    model = SpectralMLP(INPUT_DIM).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    model.eval()

    # Retrieve the Shelby Arroyo validation sequence
    with open('Nymeria_download_urls.json') as f:
        manifest = json.load(f)['sequences']
    
    val_sid = next(sid for sid in manifest.keys() if 'shelby_arroyo_act0' in sid)
    val_entry = manifest[val_sid] # Fix 5: Restored assignment
    
    print(f"Loading validation sequence: {val_sid}...")
    seq_path = download_sequence(val_sid, val_entry, root)
    df, true_gravity = load_continuous_val_stream(seq_path)

    # Calculate Total Ground Truth Distance
    gt_pos = df[['px','py','pz']].values
    gt_pos = gt_pos - gt_pos[0]
    total_distance = compute_total_distance(gt_pos)

    print("\nExecuting ESKF Integration...")
    dt = ESKF_DT
    eskf = ESKF(dt=dt, gravity=-true_gravity)   
    
    accel = df[['ax','ay','az']].values.astype(np.float32)
    gyro  = df[['wx','wy','wz']].values.astype(np.float32)
    
    from scipy.spatial.transform import Rotation
    init_rot = Rotation.from_quat(df[['qx','qy','qz','qw']].iloc[0].values).as_matrix()
    eskf.velocity    = init_rot @ df[['vx','vy','vz']].iloc[0].values.copy()
    eskf.orientation = init_rot

    accel_buf, gyro_buf, positions = [], [], []
    window_time = WINDOW_SIZE * dt

    for step in range(len(df)):
        a, g = accel[step], gyro[step]
        eskf.predict(a, g)
        positions.append(eskf.position.copy())
        
        accel_buf.append(a)
        gyro_buf.append(g)

        if len(accel_buf) > WINDOW_SIZE:
            accel_buf.pop(0)
            gyro_buf.pop(0)

        # 10Hz Neural Correction
        if len(accel_buf) == WINDOW_SIZE and step % 10 == 0:
            win = np.concatenate([accel_buf, gyro_buf], axis=-1)
            fft_flat = np.log1p(np.abs(np.fft.rfft(win[np.newaxis], axis=1))).reshape(1, -1).astype(np.float32)
            
            with torch.no_grad():
                pred_delta, _ = model(torch.tensor(fft_flat).to(device))
            
            v_world = eskf.orientation @ (pred_delta.cpu().numpy()[0] / window_time)
            eskf.update_velocity(v_world)

        # ZARU
        if len(gyro_buf) >= ZARU_WINDOW and step % ZARU_WINDOW == 0:
            if np.var(np.array(gyro_buf[-ZARU_WINDOW:]), axis=0).sum() < ZARU_THRESHOLD:
                eskf.update_zaru(g)
                eskf.update_velocity(np.zeros(3), R_obs=np.eye(3) * 1e-4)

    # Compute Final Publication Metrics
    positions = np.array(positions)
    errors = np.linalg.norm(positions - gt_pos, axis=1)
    
    mean_ate = errors.mean()
    final_ate = errors[-1]
    
    mean_rte = (mean_ate / total_distance) * 100
    final_rte = (final_ate / total_distance) * 100

    print("\n" + "="*50)
    print(" TALOS NIO — FINAL PUBLICATION METRICS")
    print("="*50)
    print(f" Sequence Duration : {len(df) * dt:.1f} seconds")
    print(f" Total Distance    : {total_distance:.3f} meters")
    print("-" * 50)
    print(f" Mean ATE          : {mean_ate:.3f} meters")
    print(f" Mean RTE          : {mean_rte:.3f} %")
    print("-" * 50)
    print(f" Final ATE (Drift) : {final_ate:.3f} meters")
    print(f" Final RTE         : {final_rte:.3f} %")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()