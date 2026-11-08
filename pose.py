"""Frames -> MediaPipe pose landmarks."""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

import config


class P:
    """MediaPipe 33-point pose indices we use."""
    NOSE = 0
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_KNEE, RIGHT_KNEE = 25, 26
    LEFT_ANKLE, RIGHT_ANKLE = 27, 28


LEGS = {
    "left":  dict(hip=P.LEFT_HIP,  knee=P.LEFT_KNEE,  ankle=P.LEFT_ANKLE),
    "right": dict(hip=P.RIGHT_HIP, knee=P.RIGHT_KNEE, ankle=P.RIGHT_ANKLE),
}


def detect(clip, indices, model_path=config.POSE_MODEL_HEAVY):
    """Runs pose on clip.frames[i] for each i in indices.

    Returns a list of (frame_index, landmarks) where landmarks is a (33, 3) array of
    [x_px, y_px, visibility] or None if no body was found in that frame.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path.name}. Run: python download_models.py")
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
    out, last_ts = [], -1
    with vision.PoseLandmarker.create_from_options(opts) as landmarker:
        for i in sorted(indices):
            img = clip.frames[i]
            h, w = img.shape[:2]
            ts = max(int(clip.times[i] * 1000), last_ts + 1)  # must strictly increase
            last_ts = ts
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            res = landmarker.detect_for_video(mp_img, ts)
            if not res.pose_landmarks:
                out.append((i, None))
                continue
            out.append((i, np.array([[l.x * w, l.y * h, l.visibility or 0.0] for l in res.pose_landmarks[0]])))
    return out


# Leg points per side: hip, knee, ankle, heel, foot index
_LEFT_LEG = [23, 25, 27, 29, 31]
_RIGHT_LEG = [24, 26, 28, 30, 32]


def fix_leg_swaps(landmarks, ratio=0.6):
    """From the side, one leg hides the other and MediaPipe sometimes swaps left/right for a few frames.
    Relabel by continuity: if swapping the legs makes this frame line up much better with the previous
    one, swap them. Returns (fixed_landmarks, number_of_frames_swapped)."""
    out, prev, swaps = [], None, 0
    for i, f in landmarks:
        if f is None:
            out.append((i, f))
            continue
        if prev is not None:
            keep = (np.linalg.norm(f[_LEFT_LEG, :2] - prev[_LEFT_LEG, :2], axis=1).sum()
                    + np.linalg.norm(f[_RIGHT_LEG, :2] - prev[_RIGHT_LEG, :2], axis=1).sum())
            swap = (np.linalg.norm(f[_RIGHT_LEG, :2] - prev[_LEFT_LEG, :2], axis=1).sum()
                    + np.linalg.norm(f[_LEFT_LEG, :2] - prev[_RIGHT_LEG, :2], axis=1).sum())
            if swap < keep * ratio:
                f = f.copy()
                f[_LEFT_LEG], f[_RIGHT_LEG] = f[_RIGHT_LEG].copy(), f[_LEFT_LEG].copy()
                swaps += 1
        out.append((i, f))
        prev = f
    return out, swaps