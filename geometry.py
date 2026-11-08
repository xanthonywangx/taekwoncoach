"""Pure math helpers. No MediaPipe or OpenCV imports, so these are easy to test."""
import numpy as np


def angle_at(a, b, c):
    """Angle ABC in degrees (the angle at point b)."""
    return angle_between(np.asarray(a) - b, np.asarray(c) - b)


def angle_between(v1, v2):
    """Angle between two vectors in degrees."""
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
