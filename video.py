"""Loading clips, building the contact sheet for Nemotron, and drawing debug images."""
from dataclasses import dataclass, field

import cv2
import numpy as np

import config


@dataclass
class Clip:
    frames: list = field(default_factory=list)  # BGR images, downscaled
    times: list = field(default_factory=list)   # seconds, strictly increasing

    @property
    def duration(self):
        return self.times[-1] if self.times else 0.0

    def indices_between(self, start_s, end_s):
        return [i for i, t in enumerate(self.times) if start_s <= t <= end_s]


def load_clip(path, max_seconds=config.MAX_CLIP_SECONDS, width=config.PROCESS_WIDTH):
    """Reads the whole clip into memory (downscaled). Works with mp4 and browser webm."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Couldn't open the video file.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 240:  # browser webm often reports nonsense here
        fps = 30.0
    clip, i = Clip(), 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if clip.times and not t > clip.times[-1]:
            t = clip.times[-1] + 1.0 / fps     # timestamp missing or not increasing
        elif not clip.times:
            t = 0.0
        if t > max_seconds:
            break
        h, w = img.shape[:2]
        if w > width:
            img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        clip.frames.append(img)
        clip.times.append(t)
        i += 1
    cap.release()
    return clip


def contact_sheet(clip, fps=config.CONTACT_SHEET_FPS, max_tiles=config.CONTACT_SHEET_MAX_TILES,
                  tile_w=config.CONTACT_TILE_WIDTH, cols=6):
    """Grid of evenly spaced frames, each labeled with its timestamp. Returns (jpeg_bytes, tile_times)."""
    n = min(max_tiles, max(1, int(clip.duration * fps) + 1))
    targets = np.linspace(0, clip.duration, n)
    picks = [int(np.argmin([abs(t - tt) for t in clip.times])) for tt in targets]
    h0, w0 = clip.frames[0].shape[:2]
    tile_h = int(h0 * tile_w / w0)
    rows = (len(picks) + cols - 1) // cols
    sheet = np.full((rows * tile_h, cols * tile_w, 3), 30, np.uint8)
    for k, idx in enumerate(picks):
        tile = cv2.resize(clip.frames[idx], (tile_w, tile_h))
        label = f"{clip.times[idx]:.2f}s"
        cv2.rectangle(tile, (0, 0), (92, 24), (0, 0, 0), -1)
        cv2.putText(tile, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        r, c = divmod(k, cols)
        sheet[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile
    return to_jpeg(sheet, quality=80), [clip.times[i] for i in picks]


def draw_legs(img, lm, kick_leg_idx, base_leg_idx, title):
    """Overlay the kicking leg (orange) and base leg (blue) on a frame."""
    out = img.copy()
    for leg, color in ((base_leg_idx, (255, 160, 0)), (kick_leg_idx, (0, 140, 255))):
        pts = [tuple(int(v) for v in lm[leg[k]][:2]) for k in ("hip", "knee", "ankle")]
        cv2.line(out, pts[0], pts[1], color, 4)
        cv2.line(out, pts[1], pts[2], color, 4)
        for p in pts:
            cv2.circle(out, p, 6, (255, 255, 255), -1)
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def leg_sharpness(img, lm, leg_idx, margin=40):
    """Blur score for the kicking-leg area: variance of the Laplacian (higher = sharper)."""
    pts = np.array([lm[leg_idx[k]][:2] for k in ("hip", "knee", "ankle")])
    h, w = img.shape[:2]
    x0, y0 = np.maximum(pts.min(axis=0).astype(int) - margin, 0)
    x1, y1 = np.minimum(pts.max(axis=0).astype(int) + margin, [w, h])
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def to_jpeg(img, quality=90):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()