"""Webcam recorder for testing without the frontend.

SPACE = start recording, SPACE again = stop and save. ESC = quit without saving.
Recording stops on its own after MAX_CLIP_SECONDS.
In the real app the browser records and uploads to api.py instead.
"""
import time

import cv2

import config


def record_clip(path="kick.mp4", cam=0):
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise RuntimeError("Couldn't open the webcam. Close other apps using it (Zoom, Teams, browser tabs).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 120:
        fps = 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer, rec_start, saved = None, None, False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if writer is not None:
            writer.write(frame)
            elapsed = time.time() - rec_start
            cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
            cv2.putText(frame, f"REC {elapsed:4.1f}s  (SPACE to stop)", (50, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            if elapsed >= config.MAX_CLIP_SECONDS:
                saved = True
                break
        else:
            cv2.putText(frame, "SPACE to start recording, ESC to quit", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow("Taekwoncoach recorder", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 32:  # SPACE
            if writer is None:
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                rec_start = time.time()
            else:
                saved = True
                break
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if not saved:
        raise SystemExit("Recording cancelled.")
    return path
