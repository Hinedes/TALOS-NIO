"""
npp.py — Neck Pivot Point + NPPTracker
TALOS NIO — Dynamic Centre of Rotation

Computes the Instantaneous Centre of Rotation (ICR) from dual-IMU data,
smooths it via NPPTracker (EMA + Z-lock), and provides the floating cage
center for HALO.

Physics (rigid body identity):
  v(p) = v_ref + omega x (p - r_ref)
  Set v(NPP) = 0:
    omega x delta_r = -v_imu1
    delta_r = (omega x v_imu1) / |omega|^2   (minimum-norm solution)

NPPTracker:
  X, Y: EMA with omega-weighted updates (alpha=0.1)
        At low |omega|, ICR is unreliable — updates damped automatically.
  Z:    Locked via deadband (z_deadband=3cm).
        Z updates slowly (z_alpha=0.02) only when running mean drifts
        beyond the deadband. Neck height doesn't bounce during head rotation.

Anatomical clamp on raw ICR output:
  Lateral (X):  ±8cm
  Depth   (Y):  ±20cm
  Height  (Z):  ±10cm

Reference:
  TALOS.md Section 9 — LAID, Dynamic Neck Pivot Point + NPPTracker
"""

import numpy as np

# Anatomical clamp limits on ICR [meters]
NPP_LATERAL_LIMIT = 0.08   # ±8cm
NPP_DEPTH_LIMIT   = 0.20   # ±20cm
NPP_HEIGHT_LIMIT  = 0.10   # ±10cm

# NPPTracker parameters
NPP_ALPHA         = 0.1    # EMA rate for X, Y (omega-weighted)
NPP_Z_ALPHA       = 0.02   # slow Z update rate
NPP_Z_DEADBAND    = 0.03   # 3cm deadband on Z

# Omega threshold below which ICR is degenerate
OMEGA_THRESHOLD   = 0.05   # rad/s

# Anatomical prior (fallback when head nearly still)
NPP_PRIOR = np.array([0.0, -0.08, -0.05], dtype=np.float64)  # same as HALO offset


class NPPTracker:
    """
    Tracks the Neck Pivot Point (Instantaneous Centre of Rotation).

    X, Y: omega-weighted EMA — low angular velocity damps updates.
    Z:    deadband lock — only updates on postural shifts.

    The NPP is the cage center. HALO's floating sphere is anchored here.
    """

    def __init__(self, prior: np.ndarray = NPP_PRIOR):
        self.npp      = prior.copy()          # current best estimate
        self._z_mean  = prior[2]              # running Z mean for deadband
        self._z_buf   = []                    # Z sample buffer

    def _solve_icr(self, omega: np.ndarray, v_imu1: np.ndarray) -> np.ndarray | None:
        """
        Minimum-norm ICR solve:
          delta_r = (omega x v_imu1) / |omega|^2

        Returns None if |omega| < threshold (degenerate).
        """
        omega_mag = np.linalg.norm(omega)
        if omega_mag < OMEGA_THRESHOLD:
            return None

        delta_r = np.cross(omega, v_imu1) / (omega_mag ** 2)

        # Anatomical clamp
        delta_r[0] = np.clip(delta_r[0], -NPP_LATERAL_LIMIT, NPP_LATERAL_LIMIT)
        delta_r[1] = np.clip(delta_r[1], -NPP_DEPTH_LIMIT,   NPP_DEPTH_LIMIT)
        delta_r[2] = np.clip(delta_r[2], -NPP_HEIGHT_LIMIT,  NPP_HEIGHT_LIMIT)

        return delta_r

    def update(self, omega: np.ndarray, v_imu1: np.ndarray) -> np.ndarray:
        """
        Update NPP estimate from current angular velocity and IMU1 velocity.

        Args:
            omega  : (3,) angular velocity from gyro [rad/s], device frame
            v_imu1 : (3,) linear velocity of IMU1 [m/s], device frame

        Returns:
            npp : (3,) current NPP estimate in device frame [meters]
        """
        omega_mag = np.linalg.norm(omega)
        icr = self._solve_icr(omega, v_imu1)

        if icr is not None:
            # Omega-weighted EMA -- low rotation damps the update
            w = np.clip(omega_mag / 1.0, 0.0, 1.0)  # normalize to [0, 1]
            alpha_eff = NPP_ALPHA * w

            # X, Y: standard EMA
            self.npp[0] = (1 - alpha_eff) * self.npp[0] + alpha_eff * icr[0]
            self.npp[1] = (1 - alpha_eff) * self.npp[1] + alpha_eff * icr[1]

            # Z: deadband lock
            self._z_buf.append(icr[2])
            if len(self._z_buf) > 50:
                self._z_buf.pop(0)
            z_running = np.mean(self._z_buf)

            if abs(z_running - self._z_mean) > NPP_Z_DEADBAND:
                # Postural shift detected -- update Z slowly
                self.npp[2] = (1 - NPP_Z_ALPHA) * self.npp[2] + NPP_Z_ALPHA * z_running
                self._z_mean = z_running
        # else: head nearly still -- hold current NPP estimate

        return self.npp.copy()

    def world_position(self, R_current: np.ndarray, imu1_position: np.ndarray) -> np.ndarray:
        """
        Compute NPP world-frame position.
        NPP_world = imu1_position + R_current @ npp_device

        Args:
            R_current     : (3,3) current orientation matrix
            imu1_position : (3,) IMU1 world-frame position

        Returns:
            npp_world : (3,) NPP world-frame position
        """
        return imu1_position + R_current @ self.npp


if __name__ == '__main__':
    print('=== NPPTracker Smoke Test ===')

    tracker = NPPTracker()
    print(f'Initial NPP: {tracker.npp}')

    # Test 1 -- below omega threshold, NPP should not move
    omega_still = np.array([0.01, 0.0, 0.0])
    v_imu1_still = np.array([0.0, 0.0, 0.0])
    npp_before = tracker.npp.copy()
    tracker.update(omega_still, v_imu1_still)
    moved = np.linalg.norm(tracker.npp - npp_before)
    print(f'[1] Below threshold — NPP moved {moved:.6f}m  (expect ~0)')

    # Test 2 -- active rotation, ICR should update X, Y
    omega_active = np.array([0.0, 1.0, 0.0])  # 1 rad/s yaw
    v_imu1_active = np.array([0.1, 0.0, 0.0]) # lateral velocity
    for _ in range(20):
        npp = tracker.update(omega_active, v_imu1_active)
    print(f'[2] Active rotation — NPP: {npp}  (X, Y should have moved)')

    # Test 3 -- ICR anatomy clamp
    omega_strong = np.array([0.0, 0.1, 0.0])
    v_huge = np.array([10.0, 0.0, 0.0])  # would produce huge ICR
    tracker2 = NPPTracker()
    icr = tracker2._solve_icr(omega_strong, v_huge)
    print(f'[3] Anatomy clamp  — ICR X: {icr[0]:.4f}m  (expect ±0.08)')

    # Test 4 -- world position
    R = np.eye(3)
    imu_pos = np.array([1.0, 2.0, 1.7])
    npp_world = tracker.world_position(R, imu_pos)
    print(f'[4] World position — {npp_world}')

    print('=== Done ===')
