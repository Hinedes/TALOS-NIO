"""
laid.py — Lever Arm Inertial Disambiguation
TALOS NIO — LAID Bouncer

Physics:
  Two IMUs separated by lever arm r (meters, device frame).
  For pure rotation with angular velocity ω and angular acceleration α:
    a_diff_predicted = ω × (ω × r) + α × r

  For pure translation:
    a_diff_predicted = 0  (both IMUs accelerate identically)

  Measured differential:
    a_diff_measured = a2 - a1

  Residual:
    e = a_diff_measured - a_diff_predicted

  If |e| is large → neural prediction is inconsistent with physics → veto.

Lever arm (device frame, Aria glasses, from factory extrinsics):
  r = p_imu_left - p_imu_right = [-0.00403, 0.10088, 0.08071] meters
  magnitude = 12.9 cm

Usage:
  bouncer = LAIDBouncer()
  veto = bouncer.check(imu1_window, imu2_window, dt=0.01)
  if veto:
      # suppress neural update, trust ESKF propagation only
"""

import numpy as np

# Lever arm vector: IMU2 (left) - IMU1 (right) in device frame [meters]
LEVER_ARM = np.array([-0.00402517, 0.10088029, 0.08070602], dtype=np.float64)

# Veto threshold — residual magnitude above this triggers suppression
# Tuned to ~3-sigma of expected sensor noise contribution
DEFAULT_THRESHOLD = 0.35  # m/s²


class LAIDBouncer:
    """
    LAID veto gate. Checks whether the differential accelerometer signal
    is consistent with the gyroscope-predicted lever arm kinematics.

    If inconsistent, the neural displacement prediction is physically
    implausible and should be suppressed.
    """

    def __init__(self, lever_arm=LEVER_ARM, threshold=DEFAULT_THRESHOLD, dt=0.01):
        self.r         = lever_arm.copy()
        self.threshold = threshold
        self.dt        = dt
        self._prev_gyro = None  # For fast loop angular acceleration estimate


    def check_sample(self, a1, g1, a2, threshold_scale=3.0):
        """Per-sample LAID check for the fast loop.
        
        Compares instantaneous accel differential (a2 - a1) against
        predicted differential from lever arm kinematics using current gyro.
        
        Args:
            a1 : (3,) accel IMU1 (right)
            g1 : (3,) gyro IMU1 (right)
            a2 : (3,) accel IMU2 (left)
            threshold_scale : multiplier on self.threshold for per-sample noise
        
        Returns:
            veto (bool): True = this sample is physically inconsistent, coast.
        """
        # Estimate angular acceleration from consecutive gyro samples
        if self._prev_gyro is None:
            self._prev_gyro = g1.copy()
            return False  # Can't check first sample
        
        alpha = (g1 - self._prev_gyro) / self.dt
        self._prev_gyro = g1.copy()
        
        # Predicted differential: a_diff = ω × (ω × r) + α × r
        a_diff_pred = self._predict_diff(g1.astype(np.float64), alpha.astype(np.float64))
        
        # Measured differential
        a_diff_meas = (a2 - a1).astype(np.float64)
        
        # Residual magnitude
        residual = np.linalg.norm(a_diff_meas - a_diff_pred)
        
        # Per-sample is noisier than window RMS, so use a scaled threshold
        return residual > self.threshold * threshold_scale

    def _predict_diff(self, omega, alpha):
        """
        Predict differential accel from angular motion.
        a_diff = ω × (ω × r) + α × r
        All in device frame.
        """
        centripetal  = np.cross(omega, np.cross(omega, self.r))  # ω × (ω × r)
        tangential   = np.cross(alpha, self.r)                    # α × r
        return centripetal + tangential

    def check(self, imu1_window, imu2_window, dt=None):
        """
        Args:
            imu1_window : (64, 6) float32 — accel+gyro IMU1 (right), mean-subtracted
            imu2_window : (64, 6) float32 — accel+gyro IMU2 (left),  mean-subtracted
            dt          : sample period in seconds (default: self.dt)

        Returns:
            veto (bool): True = neural prediction should be suppressed
            residual_rms (float): RMS residual for logging
        """
        if dt is None:
            dt = self.dt

        a1 = imu1_window[:, :3].astype(np.float64)  # (64, 3)
        g1 = imu1_window[:, 3:].astype(np.float64)  # (64, 3)
        a2 = imu2_window[:, :3].astype(np.float64)  # (64, 3)

        # Measured differential
        a_diff_measured = a2 - a1  # (64, 3)

        # Angular acceleration via finite difference of gyro
        alpha = np.gradient(g1, dt, axis=0)  # (64, 3)

        # Predict differential for each sample
        residuals = np.zeros_like(a_diff_measured)
        for i in range(len(g1)):
            a_diff_pred  = self._predict_diff(g1[i], alpha[i])
            residuals[i] = a_diff_measured[i] - a_diff_pred

        residual_rms = float(np.sqrt(np.mean(residuals**2)))
        veto = residual_rms > self.threshold

        return veto, residual_rms

    def check_batch(self, imu1_batch, imu2_batch, dt=None):
        """
        Batch version for training diagnostics.
        Args:
            imu1_batch : (N, 64, 6)
            imu2_batch : (N, 64, 6)
        Returns:
            vetos        : (N,) bool
            residual_rms : (N,) float
        """
        N = len(imu1_batch)
        vetos = np.zeros(N, dtype=bool)
        residuals = np.zeros(N, dtype=np.float64)
        for i in range(N):
            vetos[i], residuals[i] = self.check(imu1_batch[i], imu2_batch[i], dt)
        return vetos, residuals


    def yaw_anchor(self, imu1_window, imu2_window, dt=None):
        """
        Compute yaw pseudo-measurement from tangential velocity differential.

        During rotation, the two IMUs experience differential tangential velocity:
            v_diff = v_imu2 - v_imu1 = omega x r

        Invert to extract angular velocity observation:
            omega_obs = (r x v_diff) / |r|^2

        Returns the vertical (yaw) component as a pseudo-measurement for ESKF,
        plus a trust weight based on |omega| magnitude.

        Returns:
            omega_yaw : float  -- yaw rate observation [rad/s]
            trust     : float  -- [0, 1], 0 = unreliable, 1 = high confidence
            omega_mag : float  -- magnitude of angular rate for logging
        """
        if dt is None:
            dt = self.dt

        g1 = imu1_window[:, 3:].astype(np.float64)  # (64, 3) gyro IMU1
        g2 = imu2_window[:, 3:].astype(np.float64)  # (64, 3) gyro IMU2
        a1 = imu1_window[:, :3].astype(np.float64)
        a2 = imu2_window[:, :3].astype(np.float64)

        # Mean angular velocity from IMU1 gyro
        omega = np.mean(g1, axis=0)  # (3,)
        omega_mag = np.linalg.norm(omega)

        # Below threshold -- rotation too slow, differential signal unreliable
        if omega_mag < 0.05:
            return 0.0, 0.0, omega_mag

        # Integrate gyro to get delta-angle velocity estimate
        # v_tangential = omega x r  (predicted)
        v_pred = np.cross(omega, self.r)  # (3,)

        # Integrate accel differential over window to get velocity differential
        a_diff = a2 - a1  # (64, 3)
        v_diff = np.cumsum(a_diff, axis=0)[-1] * dt  # integrate -> velocity differential

        # Extract omega from differential: omega_obs = (r x v_diff) / |r|^2
        r_mag_sq = np.dot(self.r, self.r)
        omega_obs = np.cross(self.r, v_diff) / r_mag_sq  # (3,)

        # Yaw component (Z axis in device frame)
        omega_yaw = omega_obs[2]

        # Trust: normalized omega magnitude, capped at 1
        trust = float(np.clip(omega_mag / 1.0, 0.0, 1.0))

        return float(omega_yaw), trust, float(omega_mag)



if __name__ == '__main__':
    # Smoke test
    bouncer = LAIDBouncer()
    print(f"Lever arm: {bouncer.r}  magnitude: {np.linalg.norm(bouncer.r)*100:.1f}cm")

    # Pure noise — should NOT veto
    np.random.seed(42)
    imu1 = np.random.normal(0, 0.01, (64, 6)).astype(np.float32)
    imu2 = np.random.normal(0, 0.01, (64, 6)).astype(np.float32)
    veto, rms = bouncer.check(imu1, imu2)
    print(f"Noise test  — veto={veto}  rms={rms:.4f}  (expect False, low rms)")

    # Simulated strong rotation — differential should be large
    t = np.linspace(0, 0.64, 64)
    omega = np.column_stack([np.zeros(64), np.zeros(64), 2.0*np.ones(64)])  # 2 rad/s yaw
    imu1_rot = np.zeros((64, 6), dtype=np.float32)
    imu2_rot = np.zeros((64, 6), dtype=np.float32)
    imu1_rot[:, 3:] = omega
    imu2_rot[:, 3:] = omega
    # Add lever arm effect to imu2 accel
    for i in range(64):
        imu2_rot[i, :3] += np.cross(omega[i], np.cross(omega[i], LEVER_ARM)).astype(np.float32)
    veto, rms = bouncer.check(imu1_rot, imu2_rot)
    print(f"Rotation test — veto={veto}  rms={rms:.4f}  (expect False, physics consistent)")

    # Corrupted prediction — imu2 has garbage
    imu2_corrupt = imu2.copy()
    imu2_corrupt[:, :3] += np.random.normal(0, 5.0, (64, 3)).astype(np.float32)
    veto, rms = bouncer.check(imu1, imu2_corrupt)
    print(f"Corrupt test  — veto={veto}  rms={rms:.4f}  (expect True, high rms)")
