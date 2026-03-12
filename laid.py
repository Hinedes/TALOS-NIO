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
DEFAULT_THRESHOLD = 2.0  # m/s²


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
