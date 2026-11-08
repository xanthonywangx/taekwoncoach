"""Every tunable setting in one place. Secrets come from the .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)  # .env wins over anything set in the terminal

# ---- Grading thresholds (tune on real club footage) ---------------------------------
STRAIGHT_DEG = 160         # spec says 175; webcam pose noise is ~5-10 deg, so 175 fails almost everyone
KICK_ANGLE_DEG = 90        # angle between base leg and kicking leg at extension
CHAMBER_FWD_TOL = 0.05     # ankle may sit this fraction of thigh length ahead of knee and still pass
CHAMBER_HEIGHT_TOL = 0.10  # knee may sit this fraction of thigh length below hip and still pass
MIN_VISIBILITY = 0.5       # below this, a landmark is flagged as unreliable

# ---- Frame selection ------------------------------------------------------------------
EXTENSION_MIN_KNEE_DEG = 90   # the leg must open past this knee angle or there was no extension
SHARPNESS_ANGLE_TOL_DEG = 5   # among extension frames within this many degrees of the max, pick the sharpest
NEMOTRON_HINT_TOL_S = 0.3     # Python searches this far either side of Nemotron's chamber/extension timestamps

# ---- Video handling --------------------------------------------------------------------
MAX_CLIP_SECONDS = 15      # anything longer is cut off
PROCESS_WIDTH = 640        # frames are downscaled to this width (MediaPipe works at 256px anyway)

# ---- Finding the kick --------------------------------------------------------------------
WINDOW_PAD_S = 0.5         # padding added around the kick window before detailed pose runs
SCAN_STRIDE = 3            # backup scan looks at every Nth frame with the lite model
SCAN_HALF_WINDOW_S = 0.7   # backup scan's window = ankle peak +/- this
MIN_KICK_LIFT = 0.25       # ankle must rise at least this much (in torso lengths) to count as a kick

# ---- Pose models (downloaded by download_models.py) -------------------------------------
MODELS_DIR = ROOT / "models"
POSE_MODEL_HEAVY = MODELS_DIR / "pose_landmarker_heavy.task"
POSE_MODEL_LITE = MODELS_DIR / "pose_landmarker_lite.task"

# ---- Nemotron ----------------------------------------------------------------------------
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
NEMOTRON_BASE_URL = os.getenv("NEMOTRON_BASE_URL", "https://integrate.api.nvidia.com/v1")
NEMOTRON_MODEL = os.getenv("NEMOTRON_MODEL", "nvidia/nemotron-3.5-super-vl")
NEMOTRON_TIMEOUT_S = float(os.getenv("NEMOTRON_TIMEOUT_S", "60"))
CONTACT_SHEET_FPS = 5      # frames per second shown to Nemotron in pass 1
CONTACT_SHEET_MAX_TILES = 48
CONTACT_TILE_WIDTH = 240

# ---- Storage -----------------------------------------------------------------------------
RESULTS_DIR = ROOT / "results"   # images, clips and full result JSON stay here

# ---- SpacetimeDB ---------------------------------------------------------------------------
SPACETIME_URL = os.getenv("SPACETIME_URL", "http://localhost:3000")
SPACETIME_DB = os.getenv("SPACETIME_DB", "").strip()   # the name used with spacetime publish
SPACETIME_TOKEN = os.getenv("SPACETIME_TOKEN", "").strip()  # optional; a token is fetched automatically
UPLOAD_MEDIA = os.getenv("UPLOAD_MEDIA", "1") != "0"          # store clip + frame images in SpacetimeDB too
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "8"))        # skip files bigger than this (module limit is 8)


def spacetime_enabled():
    return bool(SPACETIME_DB) and os.getenv("USE_SPACETIME", "1") != "0"


def nemotron_available():
    return bool(NVIDIA_API_KEY) and os.getenv("USE_NEMOTRON", "1") != "0"
