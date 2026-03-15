"""
halo.py — Head Anatomical Limit Observer
TALOS NIO — HALO

The HALO subsystem enforces anatomical constraints on ESKF orientation state.
It is the Observer component of the NPP+Cage pipeline.

Design Principle — Egocentric Cage:
  The cage is NOT fixed in world coordinates. R_ref tracks the body's gross
  orientation via SLERP EMA so that constraints are always head-relative-to-
  torso, NOT head-relative-to-initial-pose.

  Without a torso IMU, the body's heading is unobservable directly. Instead,
  R_ref is updated each step toward the current (unclamped) ESKF orientation.
  Large slow heading changes pass through (body is turning), while fast or
  extreme deviations are caught (drift or anatomical violation).

  "Drift cannot accumulate by walking. The cage walks with you." — TALOS.md

Anatomical limits (head relative to torso):
  Yaw   ±80°    (head turn left/right)
  Pitch +45°/-70° (look up / look down)
  Roll  ±45°    (lateral tilt)

NPP offset from IMU-right in device frame (atlanto-occipital joint):
  r_npp = [0.0, -0.08, -0.05] meters

Usage:
  halo = HALOObserver(R_ref)
  R_clamped, violated = halo.observe(R_current)
  if violated:
      eskf.orientation = R_clamped
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# NPP offset from IMU-right in device frame [meters]
NPP_OFFSET = np.array([0.0, -0.08, -0.05], dtype=np.float64)

# Anatomical limits [degrees]
YAW_LIMIT_DEG   = 80.0
PITCH_UP_DEG    = 45.0
PITCH_DOWN_DEG  = 70.0
ROLL_LIMIT_DEG  = 45.0

# Convert to radians
YAW_LIMIT   = np.radians(YAW_LIMIT_DEG)
PITCH_UP    = np.radians(PITCH_UP_DEG)
PITCH_DOWN  = np.radians(PITCH_DOWN_DEG)
ROLL_LIMIT  = np.radians(ROLL_LIMIT_DEG)

# EMA tracking rate for the egocentric reference.
# At 100Hz: time constant tau ~ 1/(100 * alpha).
# alpha=0.005 → tau ≈ 2s. Slow enough that head-snap doesn't shift
# the reference, fast enough that walking turns pass through.
REF_ALPHA = 0.01


class HALOObserver:
    """
    Observes ESKF orientation state and enforces anatomical bounds.

    Egocentric: R_ref slowly tracks the ESKF orientation via SLERP EMA,
    so the cage "walks with" the user. Constraints are always relative
    to the body's current gross heading, not the initial pose.
    """

    def __init__(self, R_ref: np.ndarray, npp_offset: np.ndarray = NPP_OFFSET,
                 alpha: float = REF_ALPHA):
        """
        Args:
            R_ref      : (3,3) reference orientation matrix at sequence start
            npp_offset : (3,) NPP position in device frame [meters]
            alpha      : EMA tracking rate for egocentric reference
        """
        self.R_ref  = R_ref.copy()
        self.r_npp  = npp_offset.copy()
        self.alpha  = alpha

    def _clamp(self, R_rel: np.ndarray):
        """
        Clamp R_rel to anatomical bounds via axis-angle decomposition.

        Uses the ZYX intrinsic convention:
          angles[0] = rotation around Z  (First axis)
          angles[1] = rotation around Y  (Middle axis — limited to ±90°)
          angles[2] = rotation around X  (Last axis)

        The key insight: the EULER CONVENTION DOES NOT NEED TO MATCH THE
        DEVICE FRAME when using a tracking reference. Because R_ref tracks
        the body's gross orientation, R_rel = R_ref.T @ R_current is always
        a SMALL rotation (head relative to torso). For small rotations, all
        Euler conventions converge — axis coupling and gimbal effects are
        negligible. The cage only clamps within ±80° / ±70° / ±45°, well
        within the safe zone for any 3-axis decomposition.

        We use ZYX because the middle axis (Y) is gimbal-limited to ±90°,
        and our tightest constraint is ±45° on roll — safely within that.
        """
        rot = Rotation.from_matrix(R_rel)
        e0, e1, e2 = rot.as_euler('ZYX')

        clamped = False

        # First axis (Z rotation) — map to yaw ±80°
        if abs(e0) > YAW_LIMIT:
            e0 = np.clip(e0, -YAW_LIMIT, YAW_LIMIT)
            clamped = True

        # Middle axis (Y rotation) — map to pitch +45°/-70°
        if e1 > PITCH_UP:
            e1 = PITCH_UP
            clamped = True
        elif e1 < -PITCH_DOWN:
            e1 = -PITCH_DOWN
            clamped = True

        # Last axis (X rotation) — map to roll ±45°
        if abs(e2) > ROLL_LIMIT:
            e2 = np.clip(e2, -ROLL_LIMIT, ROLL_LIMIT)
            clamped = True

        if clamped:
            R_clamped = Rotation.from_euler('ZYX', [e0, e1, e2]).as_matrix()
        else:
            R_clamped = R_rel

        return R_clamped, clamped

    def _track_ref(self, R_current: np.ndarray):
        """
        Update R_ref toward R_current via SLERP EMA.
        This makes the cage egocentric — it slowly follows the body.
        """
        r_ref = Rotation.from_matrix(self.R_ref)
        r_cur = Rotation.from_matrix(R_current)
        slerp = Slerp([0.0, 1.0], Rotation.concatenate([r_ref, r_cur]))
        self.R_ref = slerp(self.alpha).as_matrix()

    def observe(self, R_current: np.ndarray):
        """
        Observe current ESKF orientation and enforce anatomical bounds.

        Args:
            R_current : (3,3) current ESKF orientation matrix

        Returns:
            R_out     : (3,3) clamped orientation (= R_current if within bounds)
            violated  : bool -- True if cage was activated
        """
        # Relative rotation from egocentric reference
        R_rel = self.R_ref.T @ R_current

        # Clamp to anatomical bounds
        R_rel_clamped, violated = self._clamp(R_rel)

        # Recompose world-frame orientation
        R_out = self.R_ref @ R_rel_clamped

        # Update egocentric reference — track the UNCLAMPED orientation.
        # If we tracked the clamped output, a sustained violation would
        # drag the reference toward the cage wall, slowly opening a gap
        # for drift. Tracking the raw ESKF state means the reference
        # follows the body truthfully, and only the output gets clamped.
        self._track_ref(R_current)

        return R_out, violated

    def npp_position(self, R_current: np.ndarray, imu_position: np.ndarray):
        """
        Compute NPP world-frame position given current orientation and IMU position.
        NPP = imu_position + R_current @ r_npp

        Args:
            R_current    : (3,3) current orientation
            imu_position : (3,) IMU world-frame position

        Returns:
            npp_world : (3,) NPP position in world frame
        """
        return imu_position + R_current @ self.r_npp


if __name__ == '__main__':
    import numpy as np
    from scipy.spatial.transform import Rotation

    print('=== HALO Smoke Test (Egocentric Cage) ===')

    # Test 1 — Small rotation, no violation
    R_ref = np.eye(3)
    halo  = HALOObserver(R_ref)
    R_small = Rotation.from_euler('ZYX', [10, 20, 15], degrees=True).as_matrix()
    R_out, violated = halo.observe(R_small)
    print(f'[1] Within bounds     — violated={violated}  (expect False)')

    # Test 2 — Instant 95° yaw (no tracking time) → should violate
    halo2 = HALOObserver(np.eye(3))
    R_yaw = Rotation.from_euler('ZYX', [95, 0, 0], degrees=True).as_matrix()
    R_out, violated = halo2.observe(R_yaw)
    e0 = Rotation.from_matrix(R_out).as_euler('ZYX', degrees=True)[0]
    print(f'[2] Yaw 95° instant   — violated={violated}  yaw_out={e0:.1f}°  (expect True, ~80°)')

    # Test 3 — Gradual 180° turn (simulates walking) → ref tracks, no violation
    print('[3] Gradual 180° turn (200 steps):')
    halo3 = HALOObserver(np.eye(3))
    violations = 0
    for step in range(200):
        deg = step * 0.9  # 0.9°/step × 200 = 180°
        R_walk = Rotation.from_euler('ZYX', [deg, 0, 0], degrees=True).as_matrix()
        R_out, v = halo3.observe(R_walk)
        if v:
            violations += 1
    print(f'    Violations: {violations}/200  (expect low — ref tracks the turn)')

    # Test 4 — Pitch exceeds +45°
    halo4 = HALOObserver(np.eye(3))
    R_pitch = Rotation.from_euler('ZYX', [0, 55, 0], degrees=True).as_matrix()
    R_out, violated = halo4.observe(R_pitch)
    e1 = Rotation.from_matrix(R_out).as_euler('ZYX', degrees=True)[1]
    print(f'[4] Pitch 55°         — violated={violated}  pitch_out={e1:.1f}°  (expect True, ~45°)')

    # Test 5 — Roll exceeds ±45°
    halo5 = HALOObserver(np.eye(3))
    R_roll = Rotation.from_euler('ZYX', [0, 0, 60], degrees=True).as_matrix()
    R_out, violated = halo5.observe(R_roll)
    e2 = Rotation.from_matrix(R_out).as_euler('ZYX', degrees=True)[2]
    print(f'[5] Roll 60°          — violated={violated}  roll_out={e2:.1f}°  (expect True, ~45°)')

    # Test 6 — NPP position
    imu_pos = np.array([1.0, 2.0, 1.7])
    npp = halo5.npp_position(np.eye(3), imu_pos)
    print(f'[6] NPP position      — {npp}  (expect [1.0, 1.92, 1.65])')

    print('=== Done ===')
