"""Saves each graded kick to results/<id>/ (result.json, debug images, the clip).

Images, the clip and the full result.json always live here. The pipeline also records a summary row
in SpacetimeDB (db.py) when a username is given and SPACETIME_DB is set.
"""
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import config


def new_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def save(result, images, video_path=None):
    rid = result.get("id") or new_id()
    result["id"] = rid
    d = config.RESULTS_DIR / rid
    d.mkdir(parents=True, exist_ok=True)
    for name, data in images.items():
        (d / f"{name}.jpg").write_bytes(data)
    if video_path:
        clip_name = "clip" + (Path(video_path).suffix or ".mp4")
        shutil.copy(video_path, d / clip_name)
        result["clip_url"] = f"/results/{rid}/clip"
    # URLs the frontend can put straight into <img src> (prefix with the API base URL)
    result["images"] = {name: f"/results/{rid}/{name}.jpg" for name in images}
    result["saved_at"] = datetime.now().isoformat(timespec="seconds")
    (d / "result.json").write_text(json.dumps(result, indent=2))
    return rid


def load(rid):
    p = config.RESULTS_DIR / rid / "result.json"
    return json.loads(p.read_text()) if p.exists() else None


def image_path(rid, name):
    p = config.RESULTS_DIR / rid / f"{name}.jpg"
    return p if p.exists() else None


def result_dir(rid):
    return config.RESULTS_DIR / rid


def clip_path(rid):
    d = config.RESULTS_DIR / rid
    matches = sorted(d.glob("clip.*")) if d.exists() else []
    return matches[0] if matches else None


def list_results(limit=20):
    """Newest first. Short summaries for a history list or leaderboard."""
    if not config.RESULTS_DIR.exists():
        return []
    out = []
    for d in sorted(config.RESULTS_DIR.iterdir(), reverse=True):
        r = load(d.name) if d.is_dir() else None
        if r is None:
            continue
        out.append({k: r.get(k) for k in ("id", "saved_at", "username", "status", "passed", "tier", "feedback_source")}
                   | {"thumbnail": r.get("images", {}).get("extension")})
        if len(out) >= limit:
            break
    return out


def update(result):
    """Rewrites result.json for an already-saved result (keeps its images and clip)."""
    (config.RESULTS_DIR / result["id"] / "result.json").write_text(json.dumps(result, indent=2))


def delete(rid):
    """Removes a result folder (images, clip, result.json). Returns True if it existed."""
    d = config.RESULTS_DIR / rid
    if not d.exists() or not d.is_dir():
        return False
    shutil.rmtree(d)
    return True
