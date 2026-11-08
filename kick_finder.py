"""Python's own kick finder: the backup (and cross-check) for Nemotron pass 1.

Runs the fast lite pose model on every Nth frame and finds the moment an ankle rises highest.
"""
import numpy as np

import config
import pose
from pose import P


def ankle_lift(f):
    """How high the higher ankle is above its usual spot, in torso lengths (bigger = higher)."""
    hip_y = (f[P.LEFT_HIP][1] + f[P.RIGHT_HIP][1]) / 2
    sh_y = (f[P.LEFT_SHOULDER][1] + f[P.RIGHT_SHOULDER][1]) / 2
    torso = abs(hip_y - sh_y) + 1e-9
    top_ankle_y = min(f[P.LEFT_ANKLE][1], f[P.RIGHT_ANKLE][1])
    return (hip_y - top_ankle_y) / torso


def find_peak(landmarks, times):
    """landmarks: list of (frame_idx, array|None). Returns dict with found / peak_s / lift."""
    lifts = [(i, ankle_lift(f)) for i, f in landmarks if f is not None]
    if len(lifts) < 3:
        return {"found": False, "reason": "no body detected"}
    values = np.array([v for _, v in lifts])
    i_peak, v_peak = max(lifts, key=lambda x: x[1])
    rise = float(v_peak - np.median(values))
    if rise < config.MIN_KICK_LIFT:
        return {"found": False, "reason": "no clear kick motion", "rise": round(rise, 2)}
    return {"found": True, "peak_s": float(times[i_peak]), "rise": round(rise, 2)}


def scan(clip):
    idx = list(range(0, len(clip.frames), config.SCAN_STRIDE))
    lm = pose.detect(clip, idx, config.POSE_MODEL_LITE)
    return find_peak(lm, clip.times)
