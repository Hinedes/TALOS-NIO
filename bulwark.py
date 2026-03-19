import numpy as np

# [X: lateral sway, Y: forward travel, Z: vertical bob] metres per second (m/s)
_WALL = np.array([0.40, 1.60, 0.50], dtype=np.float32)

def bulwark(pred_delta: np.ndarray) -> np.ndarray:
    """Zero pred_delta entirely if any axis exceeds hard physical limits."""
    if np.any(np.abs(pred_delta) > _WALL):
        return np.zeros(3, dtype=pred_delta.dtype)
    return pred_delta
