"""End-to-end test against a REAL running SpacetimeDB (skipped unless you ask for it).

  spacetime start                                    # in another terminal
  SPACETIME_LIVE_DB=<your database name> python -m pytest tests/test_spacetime_live.py -v

Creates throwaway users (live_xxxxxx) in that database. Pose and Nemotron are faked so no camera,
model files, or API key are needed.
"""
import os
import random
import string

import pytest

LIVE_DB = os.getenv("SPACETIME_LIVE_DB")
pytestmark = pytest.mark.skipif(not LIVE_DB, reason="set SPACETIME_LIVE_DB to run against a real SpacetimeDB")

from test_backend import fake_detect, video_file  # noqa: E402,F401  (reuse the fake athlete + test clip)

import config  # noqa: E402
import db  # noqa: E402
import pose  # noqa: E402


@pytest.fixture
def live(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SPACETIME_DB", LIVE_DB)
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(pose, "detect", fake_detect)
    from fastapi.testclient import TestClient
    import api
    return TestClient(api.app)


def new_name():
    return "live_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=6))


def test_full_user_flow(live, video_file):
    c, name = live, new_name()

    # sign up (mixed case + spaces are normalized), duplicate and invalid names rejected
    assert c.post("/users", json={"username": f"  {name.upper()} "}).json() == {"username": name}
    assert c.post("/users", json={"username": name}).status_code == 409
    assert c.post("/users", json={"username": "no!"}).status_code == 400

    # log in
    assert c.post("/login", json={"username": name}).json()["username"] == name
    assert c.post("/login", json={"username": new_name()}).status_code == 404

    # grade a kick under that user
    with open(video_file, "rb") as f:
        r = c.post("/grade", files={"video": ("kick.mp4", f, "video/mp4")}, data={"username": name}).json()
    assert r["status"] == "graded" and r["db_saved"] is True, r.get("notes")

    # unknown user is refused before grading
    with open(video_file, "rb") as f:
        assert c.post("/grade", files={"video": ("kick.mp4", f, "video/mp4")},
                      data={"username": new_name()}).status_code == 404

    # their kick list + statline
    kicks = c.get(f"/users/{name}/kicks").json()
    assert len(kicks) == 1
    k = kicks[0]
    assert k["result_id"] == r["id"] and k["status"] == "completed"
    assert k["passed"] == r["passed"] and k["tier"] == r["tier"]
    assert k["checks"]["kick_straightness"]["knee_angle_deg"] == pytest.approx(
        r["checks"]["kick_straightness"]["knee_angle_deg"])
    assert k["images"]["chamber"] == r["images"]["chamber"] and k["clip_url"] == r["clip_url"]
    assert k["recorded_time"].startswith("20")   # ISO timestamp

    # profile, and someone else can find them by search
    p = c.get(f"/users/{name}").json()
    assert p["total_kicks"] == 1 and p["completed_kicks"] == 1 and p["best_passed"] == r["passed"]
    assert name in [u["username"] for u in c.get("/users", params={"search": name[5:]}).json()]

    # media bytes are in SpacetimeDB: delete the local copies and the API still serves them
    assert r["media_saved"] == ["clip", "chamber", "extension"], r.get("notes")
    import shutil
    local_clip = (config.RESULTS_DIR / r["id"] / "clip.mp4").read_bytes()
    shutil.rmtree(config.RESULTS_DIR / r["id"])
    clip = c.get(f"/results/{r['id']}/clip")
    assert clip.status_code == 200 and clip.content == local_clip
    assert clip.headers["content-type"] == "video/mp4"
    img = c.get(f"/results/{r['id']}/chamber.jpg")
    assert img.status_code == 200 and img.content[:2] == b"\xff\xd8"   # JPEG

    # delete removes it from their list
    assert c.delete(f"/results/{r['id']}").json()["db_deleted"] is True
    assert c.get(f"/users/{name}/kicks").json() == []
    assert c.get(f"/results/{r['id']}/clip").status_code == 404   # bytes deleted too


def test_retake_is_recorded_as_failed(live, video_file, monkeypatch):
    c, name = live, new_name()
    c.post("/users", json={"username": name})
    import pipeline
    monkeypatch.setattr(pipeline.kick_finder, "scan", lambda clip: {"found": False})   # no kick found
    with open(video_file, "rb") as f:
        r = c.post("/grade", files={"video": ("kick.mp4", f, "video/mp4")}, data={"username": name}).json()
    assert r["status"] == "retake" and r["db_saved"] is True
    k = c.get(f"/users/{name}/kicks").json()[0]
    assert k["status"] == "failed" and k["fail_reason"] == r["retake_reason"] and k["passed"] is None


def test_one_megabyte_clip_round_trip(live):
    """A ~1 MB clip (typical browser recording) stores and comes back byte-identical."""
    import os
    c, name = live, new_name()
    c.post("/users", json={"username": name})
    rid = "mb-" + os.urandom(4).hex()
    db.start_kick(name, rid)
    db.complete_kick(rid, {"id": rid, "status": "retake", "retake_reason": "test"})
    clip = config.RESULTS_DIR / "upload.webm"
    clip.parent.mkdir(parents=True, exist_ok=True)
    data = os.urandom(1_000_000)
    clip.write_bytes(data)
    db.upload_media(rid, "clip", clip)
    got, mime = db.get_media(rid, "clip")
    assert got == data and mime == "video/webm"
    db.delete_kick(rid)
