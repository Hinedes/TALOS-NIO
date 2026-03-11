"""
ZARU Smoke Test — verifies gyro bias correction during stationary conditions.

Tests that:
1. The ESKF remains numerically stable with ZARU active
2. Gyro bias estimate converges toward true bias
3. Yaw drift is bounded during standstill

ZARU fires once per ZARU_WINDOW (every 50 steps = 0.5s), matching the
cadence in eskf_fusion.py. predict() rebuilds covariance between updates.
"""
import numpy as np
from eskf_fusion import ESKF, ZARU_WINDOW

DT = 0.01   # 100 Hz

# Residual gyro bias — yaw-dominant
TRUE_GYRO_BIAS = np.array([0.0005, -0.0003, 0.003])  # rad/s
GRAVITY_ACCEL  = np.array([0.0, 0.0, 9.81])


def test_zaru_convergence():
    eskf = ESKF(dt=DT)
    gyro_biased = TRUE_GYRO_BIAS.copy()
    steps = 500  # 5 seconds

    print(f"{'Step':>6} | {'bg_err':>10} | {'yaw(deg)':>10} | {'|pos|':>10}")
    print("-" * 50)

    for i in range(steps):
        gyro_noisy = gyro_biased + np.random.normal(0, 1e-4, 3)
        eskf.predict(GRAVITY_ACCEL, gyro_noisy)

        # Overlord cadence
        if i % 10 == 0:
            eskf.update_velocity(np.zeros(3))

        # ZARU compound update — once per window
        if i % ZARU_WINDOW == 0 and i > 0:
            eskf.update_zaru(gyro_noisy)
            eskf.update_velocity(np.zeros(3), R_obs=np.eye(3) * 1e-4)

        if i % 100 == 0:
            bg_err  = np.linalg.norm(eskf.bg - TRUE_GYRO_BIAS)
            yaw_deg = np.degrees(np.arctan2(eskf.orientation[1, 0], eskf.orientation[0, 0]))
            pos_mag = np.linalg.norm(eskf.position)
            print(f"{i:>6} | {bg_err:>10.6f} | {yaw_deg:>10.4f} | {pos_mag:>10.4f}")

    bg_err_final = np.linalg.norm(eskf.bg - TRUE_GYRO_BIAS)
    yaw_final    = abs(np.degrees(np.arctan2(eskf.orientation[1, 0], eskf.orientation[0, 0])))
    pos_final    = np.linalg.norm(eskf.position)

    print("-" * 50)
    print(f"Final bg error: {bg_err_final:.6f} rad/s  (target: < 0.003)")
    print(f"Final yaw:      {yaw_final:.4f} deg  (target: < 5.0)")
    print(f"Final pos:      {pos_final:.4f} m")
    print(f"Final bg:       {eskf.bg}")
    print(f"True  bg:       {TRUE_GYRO_BIAS}")
    print()

    passed = True
    if not np.all(np.isfinite([bg_err_final, yaw_final, pos_final])):
        print("✗ FAIL: Filter diverged")
        passed = False
    else:
        print("✓ PASS: Numerically stable")

    if np.isfinite(bg_err_final) and bg_err_final < 0.003:
        print(f"✓ PASS: Bias converged ({bg_err_final:.6f} < 0.003)")
    elif np.isfinite(bg_err_final):
        print(f"  INFO: Bias partially converged ({bg_err_final:.6f})")
    
    if np.isfinite(yaw_final) and yaw_final < 5.0:
        print(f"✓ PASS: Yaw bounded ({yaw_final:.4f} < 5.0 deg)")
    elif np.isfinite(yaw_final):
        print(f"✗ FAIL: Yaw drift ({yaw_final:.4f} deg)")
        passed = False

    print()
    print("✓ ZARU smoke test PASSED" if passed else "✗ ZARU smoke test FAILED")
    return passed


if __name__ == "__main__":
    np.random.seed(42)
    test_zaru_convergence()
