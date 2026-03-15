import numpy as np

# [X: lateral sway, Y: forward travel, Z: vertical bob] metres per 0.64s window
_WALL = np.array([0.25, 1.00, 0.20], dtype=np.float32)

def bulwark(pred_delta: np.ndarray) -> np.ndarray:
    """Zero pred_delta entirely if any axis exceeds hard physical limits."""
    if np.any(np.abs(pred_delta) > _WALL):
        return np.zeros(3, dtype=pred_delta.dtype)
    return pred_delta
