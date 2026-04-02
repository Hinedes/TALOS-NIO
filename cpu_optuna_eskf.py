import json
import numba
import numpy as np
import optuna
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation
import glob
import time
import os

# --- Numba ESKF Math (Identical to incremental_train.py) ---
@numba.njit(cache=True, fastmath=True)
def _eskf_skew(v):
    return np.array([[ 0.0,  -v[2],  v[1]],
                     [ v[2],   0.0, -v[0]],
                     [-v[1],  v[0],  0.0 ]])

@numba.njit(cache=True, fastmath=True)
def _rotvec_to_matrix(rotvec):
    angle = np.linalg.norm(rotvec)
    if angle < 1e-9:
        return np.eye(3)
    K = _eskf_skew(rotvec / angle)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

@numba.njit(cache=True, fastmath=True)
def _eskf_predict_math(dt, position, velocity, orientation, bg, ba, P, Q, accel, gyro, gravity):
    u_a = accel - ba
    u_g = gyro - bg
    aw = orientation @ u_a + gravity
    pos_new = position + velocity * dt + 0.5 * aw * dt**2
    vel_new = velocity + aw * dt
    ori_new = orientation @ _rotvec_to_matrix(u_g * dt)
    F = np.eye(15)
    F[0:3, 3:6]   = np.eye(3) * dt
    F[3:6, 6:9]   = -orientation @ _eskf_skew(u_a) * dt
    F[3:6, 12:15] = -orientation * dt
    F[6:9, 9:12]  = -orientation * dt
    P_new = F @ P @ F.T + Q
    U, _, Vt = np.linalg.svd(ori_new)
    ori_new = U @ Vt
    return pos_new, vel_new, ori_new, P_new, Q


class CPU_ESKF:
    """Lite ESKF that only implements Predict and Neural Update."""
    def __init__(self, dt=0.01, gravity=None):
        self.dt      = dt
        self.gravity = gravity if gravity is not None else np.array([0., 0., -9.81], dtype=np.float32)
        self.position    = np.zeros(3)
        self.velocity    = np.zeros(3)
        self.orientation = np.eye(3)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.state_dim = 15
        self.P  = np.eye(self.state_dim) * 0.1
        self.Q  = np.diag([1e-6]*3 + [1e-4]*3 + [1e-5]*3 + [1e-3]*3 + [1e-2]*3)

    @staticmethod
    def _skew(v):
        return _eskf_skew(v)

    def predict(self, accel, gyro):
        pos, vel, ori, P, Q = _eskf_predict_math(
            self.dt, self.position, self.velocity, self.orientation, 
            self.bg, self.ba, self.P, self.Q, accel, gyro, self.gravity
        )
        self.position = pos
        self.velocity = vel
        self.orientation = ori
        self.P = P
        self.Q = Q

    def update_local_velocity(self, v_local_meas, R_obs, slap_threshold=5.0):
        v_local_pred = self.orientation.T @ self.velocity
        y = v_local_meas - v_local_pred
        H = np.zeros((3, self.state_dim))
        H[0:3, 3:6] = self.orientation.T
        H[0:3, 6:9] = self._skew(v_local_pred)

        R_eff = np.array(R_obs, dtype=np.float64, copy=True)
        S = H @ self.P @ H.T + R_eff
        S_inv = np.linalg.inv(S)

        mahal_sq = float(y @ S_inv @ y)
        R_inv = np.linalg.inv(R_eff)
        mahal_r_sq = float(y @ R_inv @ y)
        mahal_max = max(mahal_sq, mahal_r_sq)

        if mahal_max > slap_threshold ** 2:
            return False

        K = self.P @ H.T @ S_inv
        K[12:15, :] = 0.0   # accel bias quarantined
        dx = K @ y

        world_z_local = self.orientation.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        yaw_correction_mag = np.dot(dx[6:9], world_z_local)
        dx[6:9] = yaw_correction_mag * world_z_local * 1.0

        self.position += dx[0:3]
        self.velocity += dx[3:6]
        self.orientation = self.orientation @ Rotation.from_rotvec(dx[6:9]).as_matrix()
        self.bg += np.clip(dx[9:12], -1e-4, 1e-4)

        I = np.eye(self.state_dim)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_eff @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True


def evaluate_trajectory(params, run_dir, val_df, val_gravity, npz_path):
    """Replays the static neural predictions against the ESKF with new params."""
    try:
        # Load static predictions bypassing PyTorch
        npz = np.load(npz_path)
        inference_steps = npz['steps']
        all_pred_vels = npz['pred_vels']
        all_pred_covs = npz['pred_covs']
    except Exception:
        return float('inf')

    neural_preds = {step: (all_pred_vels[i], all_pred_covs[i]) for i, step in enumerate(inference_steps)}
    
    eskf = CPU_ESKF(dt=0.01, gravity=val_gravity)
    
    accel = val_df[['ax','ay','az']].values.astype(np.float32)
    gyro  = val_df[['wx','wy','wz']].values.astype(np.float32)
    gt_pos = val_df[['px','py','pz']].values
    gt_pos = gt_pos - gt_pos[0]
    
    init_rot = Rotation.from_quat(val_df[['qx','qy','qz','qw']].iloc[0].values).as_matrix()
    init_vel_device = val_df[['vx','vy','vz']].iloc[0].values.copy()
    init_vel_world  = init_rot @ init_vel_device

    eskf.velocity    = init_vel_world.copy()
    eskf.orientation = init_rot.copy()

    fp_pred_vel_gain = params.get('PRED_VEL_GAIN', 1.0)
    fp_use_dynamic_r_obs = params.get('USE_DYNAMIC_R_OBS', False)
    fp_r_obs_fixed_diag = params.get('R_OBS_FIXED_DIAG', 0.10)
    fp_slap_threshold  = params.get('SLAP_THRESHOLD', 4.0)
    fp_max_pred_speed  = params.get('MAX_PRED_WORLD_SPEED_MPS', 5.0)
    fp_max_innov_norm  = params.get('MAX_INNOVATION_NORM_MPS', 5.0)
    fp_cage_radius     = params.get('CAGE_RADIUS', 0.30)

    talos_positions = []
    npp_center = np.zeros(3)
    
    for step in range(len(val_df)):
        a, g = accel[step], gyro[step]
        eskf.predict(a, g)
        
        if step in neural_preds:
            pred_vel_raw, pred_cov_raw = neural_preds[step]
            pred_vel_local = pred_vel_raw * fp_pred_vel_gain

            pred_speed = np.linalg.norm(eskf.orientation @ pred_vel_local)
            innov_norm = np.linalg.norm(pred_vel_local - eskf.orientation.T @ eskf.velocity)

            if pred_speed <= fp_max_pred_speed and innov_norm <= fp_max_innov_norm:
                if fp_use_dynamic_r_obs:
                    pred_var = np.exp(pred_cov_raw)
                    r_obs_diag = np.clip(pred_var, 0.05, 2.00)
                    R_obs_used = np.diag(r_obs_diag.astype(np.float64))
                else:
                    R_obs_used = np.eye(3) * fp_r_obs_fixed_diag
                eskf.update_local_velocity(pred_vel_local, R_obs_used, slap_threshold=fp_slap_threshold)

        displacement = eskf.position - npp_center
        dist = np.linalg.norm(displacement)
        if dist > fp_cage_radius:
            eskf.position = npp_center + displacement * (fp_cage_radius / dist)
        npp_center += 0.001 * (eskf.position - npp_center)

        talos_positions.append(eskf.position.copy())

    talos_positions = np.array(talos_positions)
    mean_ate = np.linalg.norm(talos_positions - gt_pos, axis=1).mean()
    return mean_ate

def optimize_run(run_dir_str, n_trials=500):
    run_dir = Path(run_dir_str)
    
    # We need the val_df. The easiest way without PyTorch is using the cache.
    import pickle
    cache_path = glob.glob("/home/iclab/TALOS/golden/cache/*_val_stream.pkl")[0]
    val_df, val_gravity = pickle.load(open(cache_path, "rb"))
    val_df_walk = val_df.iloc[313*100:313*100+30000].reset_index(drop=True) # 300 seconds

    last_processed_npz = None

    print(f"Daemon watching for neural inferences in {run_dir}...")
    
    while True:
        npz_files = list(run_dir.glob("val_predictions_R*.npz"))
        if not npz_files:
            time.sleep(10)
            continue
            
        # Sort by round number and pick the latest
        npz_path = sorted(npz_files, key=lambda x: int(x.stem.split('_R')[1]))[-1]
        
        if npz_path == last_processed_npz:
            time.sleep(10)
            continue
            
        print(f"\n[Daemon] Found NEW target: {npz_path.name}. Starting Massively Parallel ESKF Optuna Search.")
        last_processed_npz = npz_path

        def objective(trial):
            # Let Optuna battle-test the dynamic covariance against a fixed baseline
            use_dynamic = trial.suggest_categorical("USE_DYNAMIC_R_OBS", [True, False])
            
            params = {
                'SLAP_THRESHOLD': trial.suggest_float("SLAP_THRESHOLD", 1.5, 8.0),
                'PRED_VEL_GAIN': trial.suggest_float("PRED_VEL_GAIN", 0.5, 1.5),
                'USE_DYNAMIC_R_OBS': use_dynamic,
            }
            
            # Only tune the fixed diagonal if we are actually using it
            if not use_dynamic:
                params['R_OBS_FIXED_DIAG'] = trial.suggest_float("R_OBS_FIXED_DIAG", 0.01, 1.0, log=True)
            
            params['CAGE_RADIUS']              = trial.suggest_float("CAGE_RADIUS", 0.10, 1.00)
            params['MAX_PRED_WORLD_SPEED_MPS'] = trial.suggest_float("MAX_PRED_WORLD_SPEED_MPS", 2.0, 10.0)
            params['MAX_INNOVATION_NORM_MPS']  = trial.suggest_float("MAX_INNOVATION_NORM_MPS", 2.0, 10.0)
            return evaluate_trajectory(params, run_dir, val_df_walk, val_gravity, npz_path)

        # Use SQLite for multi-process safe synchronization
        db_path = run_dir / "optuna_eskf.db"
        study_name = "talos_fusion"
        
        # To run this purely as a daemon script, we can just run a study here
        study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{db_path}",
            load_if_exists=True,
            direction="minimize"
        )
        
        # Mute optuna for massive parallelism
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        print(f"Brute forcing {n_trials} trajectories...")
        study.optimize(objective, n_trials=n_trials, n_jobs=18) # Smash the Ultra 7 265K
        
        best_params = study.best_params
        best_ate = study.best_value
        
        print(f"\n[Optuna Done] Best ATE found: {best_ate:.3f}m")
        print(f"Best Params: {best_params}")
        
        config_path = run_dir / 'darwin_config.json'
        config_path.write_text(json.dumps(best_params, indent=2))
        print(f"Saved to {config_path.name}. GPU incremental_train.py will use this automatically on next round.")
        print(f"[Daemon] Optimization cycle complete. Resuming watch...")
        
if __name__ == "__main__":
    # Point this at the latest run directory
    runs = sorted(glob.glob("/home/iclab/TALOS/golden/run_*"))
    if runs:
        optimize_run(runs[-1])
