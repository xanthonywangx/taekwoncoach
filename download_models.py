"""Downloads the MediaPipe pose models into models/. Run once: python download_models.py"""
import urllib.request

import config

BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
MODELS = {
    config.POSE_MODEL_HEAVY: f"{BASE}/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    config.POSE_MODEL_LITE: f"{BASE}/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}

if __name__ == "__main__":
    config.MODELS_DIR.mkdir(exist_ok=True)
    for path, url in MODELS.items():
        if path.exists() and path.stat().st_size > 1_000_000:
            print(f"{path.name} already downloaded")
            continue
        print(f"Downloading {path.name}...")
        urllib.request.urlretrieve(url, path)
        print(f"  done ({path.stat().st_size / 1e6:.1f} MB)")
