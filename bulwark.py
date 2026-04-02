import numpy as np

# [X: lateral sway, Y: forward travel, Z: vertical bob] metres per second (m/s)
_WALL = np.array([0.40, 1.60, 0.10], dtype=np.float32)

def bulwark(pred_delta: np.ndarray) -> np.ndarray:
    """Per-axis clip to hard physical limits."""
    return np.clip(pred_delta, -_WALL, _WALL)
