"""Backend tests. No camera, pose model files, or API key needed: pose and Nemotron are faked.

  python -m pytest -q
"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config        # noqa: E402
import feedback      # noqa: E402
import kick_finder   # noqa: E402
import nemotron      # noqa: E402
import pipeline      # noqa: E402
import pose          # noqa: E402
import scoring       # noqa: E402
from pose import P   # noqa: E402


# ---- Keep tests offline -------------------------------------------------------------------
# Tests never use your real .env key or the network. Tests that need Nemotron set a fake key
# and replace nemotron.locate_kick / review_kick themselves.

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "")

    def no_network():
        raise RuntimeError("tests must not call the real Nemotron API")
    monkeypatch.setattr(nemotron, "client", no_network)


# ---- Fake athlete -------------------------------------------------------------------------

def frame(knee, ankle, flip=False):
    """Side-view pose, right leg kicking, facing right (flip=True mirrors to face left)."""
    f = np.zeros((33, 3)); f[:, 2] = 1
    f[P.NOSE] = [560, 200, 1]
    f[P.LEFT_SHOULDER] = f[P.RIGHT_SHOULDER] = [520, 260, 1]
    f[P.RIGHT_HIP] = f[P.LEFT_HIP] = [500, 450, 1]
    f[P.LEFT_KNEE] = [500, 575, 1]; f[P.LEFT_ANKLE] = [500, 700, 1]
    f[P.RIGHT_KNEE] = [*knee, 1]; f[P.RIGHT_ANKLE] = [*ankle, 1]
    if flip:
        f[:, 0] = 1000 - f[:, 0]
    return f


STAND = ((500, 575), (500, 700))
GOOD = (((610, 440), (560, 560)), ((610, 440), (735, 430)))  # tucked + high chamber, straight hip-height kick
BAD = (((600, 520), (630, 610)), ((600, 520), (700, 560)))   # low chamber (~146 deg, foot ahead of knee), low bent kick (~164 deg)


def clip_landmarks(chamber, extension, flip=False):
    seq = [STAND] * 5 + [chamber] * 3 + [extension] * 3 + [STAND] * 3
    return [(i, frame(*s, flip)) for i, s in enumerate(seq)]


# ---- Scoring ----------------------------------------------------------------------------

@pytest.mark.parametrize("flip", [False, True])
def test_good_kick_passes_all(flip):
    r = scoring.grade(clip_landmarks(*GOOD, flip))
    assert r["passed"] == 4 and r["tier"] == "strong" and r["kicking_leg"] == "right"


@pytest.mark.parametrize("flip", [False, True])
def test_bad_kick(flip):
    r = scoring.grade(clip_landmarks(*BAD, flip))
    assert r["checks"]["chamber_tightness"]["pass"] == 0
    assert r["checks"]["chamber_height"]["pass"] == 0
    assert r["checks"]["kick_angle"]["pass"] == 0
    assert r["tier"] == "building"


@pytest.mark.parametrize("passed,tier", [(4, "strong"), (3, "strong"), (2, "even"), (1, "building"), (0, "building")])
def test_tiers(passed, tier):
    assert scoring.tier_for(passed) == tier


def test_no_body_raises():
    with pytest.raises(ValueError):
        scoring.grade([(i, None) for i in range(10)])


def test_building_feedback_is_soft():
    text = feedback.fallback_feedback(scoring.grade(clip_landmarks(*BAD)))
    assert "Consider these tips" in text and "bad" not in text.lower()


# ---- Kick finder + window ----------------------------------------------------------------

def test_kick_finder_finds_peak():
    lm = clip_landmarks(*GOOD)
    times = [i / 30 for i in range(len(lm))]
    r = kick_finder.find_peak(lm, times)
    assert r["found"] and 8 / 30 <= r["peak_s"] <= 10 / 30


def test_kick_finder_no_kick():
    lm = [(i, frame(*STAND)) for i in range(20)]
    assert not kick_finder.find_peak(lm, [i / 30 for i in range(20)])["found"]


def test_window_agree_uses_both():
    w = pipeline.choose_window({"start_s": 2.0, "end_s": 3.0}, {"found": True, "peak_s": 2.6}, 10)
    assert w["source"] == "both" and w["start_s"] <= 1.9 and w["end_s"] >= 3.3


def test_window_disagree_covers_both():
    w = pipeline.choose_window({"start_s": 6.0, "end_s": 7.0}, {"found": True, "peak_s": 2.0}, 10)
    assert w["source"] == "both (disagreed)" and w["start_s"] < 2.0 and w["end_s"] > 7.0


def test_window_nemotron_only_and_clamped():
    w = pipeline.choose_window({"start_s": 0.1, "end_s": 9.9}, {"found": False}, 10)
    assert w == {"start_s": 0.0, "end_s": 10.0, "source": "nemotron"}


def test_window_none():
    assert pipeline.choose_window(None, {"found": False}, 10) is None


# ---- Nemotron reply parsing ----------------------------------------------------------------

def test_parse_json_with_fences_and_chatter():
    text = 'Sure!\n```json\n{"usable": true, "start_s": 1.5, "end_s": 2.5}\n```\nHope that helps.'
    assert nemotron.parse_json(text)["start_s"] == 1.5


def test_parse_json_garbage_raises():
    with pytest.raises(ValueError):
        nemotron.parse_json("no json here")


# ---- Full pipeline on a real video file (pose + Nemotron faked) ---------------------------

KICK_FRAMES = range(30, 45)   # frames where the fake athlete kicks (1.0s - 1.5s at 30fps)


def fake_detect(clip, indices, model_path=None):
    out = []
    for i in indices:
        if i in range(30, 35):
            out.append((i, frame(*GOOD[0])))
        elif i in range(35, 40):
            out.append((i, frame(*GOOD[1])))
        else:
            out.append((i, frame(*STAND)))
    return out


@pytest.fixture
def video_file(tmp_path):
    path = tmp_path / "kick.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
    for i in range(90):  # 3 seconds
        img = np.full((360, 640, 3), 40, np.uint8)
        cv2.putText(img, str(i), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        w.write(img)
    w.release()
    return str(path)


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(pose, "detect", fake_detect)
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "SPACETIME_DB", "")   # tests never touch a real database


def test_pipeline_offline(video_file, isolated):
    r = pipeline.grade_video(video_file, use_nemotron=False)
    assert r["status"] == "graded" and r["passed"] == 4
    assert r["feedback_source"] == "fallback"
    assert r["window"]["source"] == "python"
    assert (config.RESULTS_DIR / r["id"] / "chamber.jpg").exists()


def test_pipeline_with_nemotron(video_file, isolated, monkeypatch):
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "fake")
    monkeypatch.setattr(nemotron, "locate_kick",
                        lambda sheet: {"usable": True, "reason": None, "kick_type": "front snap kick",
                                       "start_s": 0.9, "chamber_s": 1.05, "extension_s": 1.25, "end_s": 1.6})
    monkeypatch.setattr(nemotron, "review_kick",
                        lambda g, a, b: {"frames_ok": True, "frame_issue": None,
                                         "uncertain_checks": ["kick_angle", "made_up_check"],
                                         "extra_observations": ["Guard hands dropped"],
                                         "feedback": "Great kick!"})
    r = pipeline.grade_video(video_file)
    assert r["status"] == "graded" and r["window"]["source"] == "both"
    assert r["checks"]["kick_angle"]["uncertain"] is True
    assert r["checks"]["kick_angle"]["pass"] == 1          # never flipped
    assert r["feedback_source"] == "nemotron"
    assert r["frames_chosen_by"] == "nemotron"
    assert r["images"]["chamber"] == f"/results/{r['id']}/chamber.jpg"


def test_pipeline_nemotron_rejects_clip(video_file, isolated, monkeypatch):
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "fake")
    monkeypatch.setattr(nemotron, "locate_kick",
                        lambda sheet: {"usable": False, "reason": "Your feet left the frame.",
                                       "kick_type": "none", "start_s": None, "end_s": None})
    r = pipeline.grade_video(video_file)
    assert r["status"] == "retake" and "feet" in r["retake_reason"]


def test_pipeline_survives_nemotron_crash(video_file, isolated, monkeypatch):
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "fake")

    def boom(*a, **k):
        raise TimeoutError("network down")
    monkeypatch.setattr(nemotron, "locate_kick", boom)
    monkeypatch.setattr(nemotron, "review_kick", boom)
    r = pipeline.grade_video(video_file)
    assert r["status"] == "graded" and r["feedback_source"] == "fallback" and len(r["notes"]) == 2


def test_api_grade(video_file, isolated):
    from fastapi.testclient import TestClient
    import api
    c = TestClient(api.app)
    with open(video_file, "rb") as f:
        r = c.post("/grade", files={"video": ("kick.mp4", f, "video/mp4")})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("graded", "retake")
    assert c.get(f"/results/{body['id']}").status_code == 200
    assert c.get(f"/results/{body['id']}/chamber.jpg").status_code == 200


# ---- Frame selection rules -----------------------------------------------------------------

RECHAMBER = ((610, 440), (540, 520))     # tighter fold-back AFTER the kick (~44 deg vs ~62 deg chamber)
NO_EXTENSION = ((610, 440), (620, 565))  # knee up but lower leg only opens to ~89 deg
EXT_SLIGHTLY_BENT = ((610, 440), (735, 437))  # ~176 deg, within 5 deg of the ~179 deg extension


def seq_landmarks(seq):
    return [(i, frame(*s)) for i, s in enumerate(seq)]


def test_rechamber_after_kick_is_not_picked_as_chamber():
    lm = seq_landmarks([STAND] * 3 + [GOOD[0]] * 2 + [GOOD[1]] * 2 + [RECHAMBER] * 2 + [STAND] * 3)
    r = scoring.grade(lm)
    assert r["chamber_frame"] in (3, 4)
    assert r["extension_frame"] in (5, 6)


def test_chamber_comes_before_extension():
    r = scoring.grade(clip_landmarks(*GOOD))
    assert r["chamber_frame"] < r["extension_frame"]
    assert r["chamber_knee_deg"] < 90 < r["extension_knee_deg"]


def test_no_extension_past_90_asks_for_retake():
    lm = seq_landmarks([STAND] * 3 + [GOOD[0]] * 2 + [NO_EXTENSION] * 2 + [STAND] * 3)
    with pytest.raises(ValueError, match="didn't extend"):
        scoring.grade(lm)


def test_sharpest_frame_wins_among_near_max_extensions():
    lm = seq_landmarks([STAND] * 3 + [GOOD[0]] * 2 + [GOOD[1], EXT_SLIGHTLY_BENT] + [STAND] * 3)
    blurry_first = lambda i, f: 100.0 if i == 6 else 10.0   # frame 6 (176 deg) is the sharp one
    assert scoring.grade(lm, sharpness=blurry_first)["extension_frame"] == 6
    assert scoring.grade(lm)["extension_frame"] == 5          # without blur info: highest angle wins


# ---- Nemotron-guided frame choice ------------------------------------------------------------

def test_guided_picks_frames_near_nemotron_moments():
    # two chamber-like moments; Nemotron points at the second one (frames 7-8)
    lm = seq_landmarks([STAND] * 2 + [GOOD[0]] * 2 + [STAND] * 2 + [GOOD[0]] * 2 + [GOOD[1]] * 2 + [STAND] * 2)
    r = scoring.grade(lm, chamber_near={6, 7}, extension_near={8, 9})
    assert r["frames_chosen_by"] == "nemotron"
    assert r["chamber_frame"] in (6, 7) and r["extension_frame"] in (8, 9)


def test_guided_falls_back_when_nemotron_moments_dont_fit():
    lm = clip_landmarks(*GOOD)
    r = scoring.grade(lm, chamber_near={0, 1}, extension_near={2})   # standing frames: nothing lifted there
    assert r["frames_chosen_by"] == "python" and r["passed"] == 4


def test_leg_swap_fix():
    lm = clip_landmarks(*GOOD)
    swapped = []
    for i, f in lm:
        if i == 6:  # MediaPipe mixes up the legs for one frame
            f = f.copy()
            f[[23, 25, 27, 29, 31]], f[[24, 26, 28, 30, 32]] = f[[24, 26, 28, 30, 32]].copy(), f[[23, 25, 27, 29, 31]].copy()
        swapped.append((i, f))
    fixed, n = pose.fix_leg_swaps(swapped)
    assert n == 1
    assert np.allclose(fixed[6][1], lm[6][1])


def test_api_history_and_clip(video_file, isolated):
    from fastapi.testclient import TestClient
    import api
    c = TestClient(api.app)
    with open(video_file, "rb") as f:
        rid = c.post("/grade", files={"video": ("kick.mp4", f, "video/mp4")}).json()["id"]
    hist = c.get("/results").json()
    assert hist[0]["id"] == rid and hist[0]["thumbnail"]
    assert c.get(f"/results/{rid}/clip").status_code == 200


# ---- SpacetimeDB --------------------------------------------------------------------------------
import json  # noqa: E402

import db  # noqa: E402


def test_kick_summary_has_what_complete_kick_reads():
    r = {"id": "abc", "status": "graded", "passed": 3, "tier": "strong", "checks": {},
         "feedback": "Great kick!", "feedback_source": "nemotron", "clip_url": "/results/abc/clip",
         "images": {"extension": "/results/abc/extension.jpg"}, "timings_s": {"total": 3}, "notes": []}
    assert set(db.kick_summary(r)) == {"id", "status", "passed", "tier", "checks", "feedback",
                                       "feedback_source", "images", "clip_url"}


def test_complete_kick_sends_id_and_json(monkeypatch):
    sent = []
    monkeypatch.setattr(db, "call", lambda reducer, *args: sent.append((reducer, args)))
    db.complete_kick("abc", {"id": "abc", "status": "graded", "passed": 4})
    reducer, (rid, payload) = sent[0]
    assert reducer == "complete_kick" and rid == "abc"
    assert json.loads(payload) == {"id": "abc", "status": "graded", "passed": 4}


def test_call_falls_back_to_camel_case(monkeypatch):
    urls = []

    class Resp:
        def __init__(self, code):
            self.status_code, self.text = code, ""
    monkeypatch.setattr(db, "_headers", lambda: {})
    monkeypatch.setattr(db.httpx, "post", lambda url, **k: (urls.append(url), Resp(404 if "start_kick" in url else 200))[1])
    db.call("start_kick", "Haris", "abc")
    assert urls[0].endswith("/call/start_kick") and urls[1].endswith("/call/startKick")


def test_sql_response_parsing(monkeypatch):
    class Resp:
        status_code = 200
        def json(self):
            return [{"schema": {"elements": [{"name": {"some": "user_id"}}, {"name": {"some": "username"}}]},
                     "rows": [[1, "Haris"], [2, "Anthony"]]}]
    monkeypatch.setattr(db, "_headers", lambda: {})
    monkeypatch.setattr(db.httpx, "post", lambda *a, **k: Resp())
    assert db.sql("SELECT * FROM user") == [{"user_id": 1, "username": "Haris"}, {"user_id": 2, "username": "Anthony"}]


class FakeDB:
    """Records the lifecycle calls the pipeline makes."""
    def __init__(self, monkeypatch, fail_on=None):
        self.calls = []
        for name in ("start_kick", "set_status", "complete_kick", "delete_kick", "create_user", "upload_media"):
            monkeypatch.setattr(db, name, self._make(name, fail_on))

    def _make(self, name, fail_on):
        def f(*args):
            if name == fail_on:
                raise ConnectionError("refused")
            self.calls.append((name, args))
        return f

    def names(self):
        return [c[0] for c in self.calls]


def storage_json(r):
    return json.loads((config.RESULTS_DIR / r["id"] / "result.json").read_text())


def test_pipeline_kick_lifecycle(video_file, isolated, monkeypatch):
    monkeypatch.setattr(config, "SPACETIME_DB", "taekwon-db")
    fake = FakeDB(monkeypatch)
    r = pipeline.grade_video(video_file, use_nemotron=False, username="Haris")
    assert fake.names() == ["start_kick", "set_status", "complete_kick", "upload_media", "upload_media", "upload_media"]
    assert [c[1][1] for c in fake.calls[3:]] == ["clip", "chamber", "extension"]
    assert r["media_saved"] == ["clip", "chamber", "extension"]
    assert fake.calls[0][1] == ("Haris", r["id"])
    assert fake.calls[1][1] == (r["id"], db.STATUS_PROCESSING)
    assert fake.calls[2][1][1]["status"] == "graded"
    assert r["db_saved"] is True and storage_json(r)["db_saved"] is True


def test_pipeline_survives_spacetime_down(video_file, isolated, monkeypatch):
    monkeypatch.setattr(config, "SPACETIME_DB", "taekwon-db")
    FakeDB(monkeypatch, fail_on="start_kick")
    r = pipeline.grade_video(video_file, use_nemotron=False, username="Haris")
    assert r["status"] == "graded" and r["db_saved"] is False
    assert any("SpacetimeDB" in n for n in r["notes"])


def test_pipeline_crash_marks_kick_failed(video_file, isolated, monkeypatch):
    monkeypatch.setattr(config, "SPACETIME_DB", "taekwon-db")
    fake = FakeDB(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("pose crashed")
    monkeypatch.setattr(pipeline.kick_finder, "scan", boom)
    with pytest.raises(RuntimeError):
        pipeline.grade_video(video_file, use_nemotron=False, username="Haris")
    assert fake.names()[-1] == "complete_kick" and fake.calls[-1][1][1]["status"] == "retake"


def test_no_username_skips_db(video_file, isolated, monkeypatch):
    monkeypatch.setattr(config, "SPACETIME_DB", "taekwon-db")
    fake = FakeDB(monkeypatch)
    assert pipeline.grade_video(video_file, use_nemotron=False)["db_saved"] is False
    assert fake.calls == []


def test_api_passes_username(video_file, isolated, monkeypatch):
    from fastapi.testclient import TestClient
    import api
    monkeypatch.setattr(config, "SPACETIME_DB", "taekwon-db")
    fake = FakeDB(monkeypatch)
    monkeypatch.setattr(db, "get_user", lambda u: {"user_id": 2, "username": "anthony", "created_at": "x"})
    with open(video_file, "rb") as f:
        body = TestClient(api.app).post("/grade", files={"video": ("kick.mp4", f, "video/mp4")},
                                        data={"username": "Anthony"}).json()
    assert fake.calls[0][1][0] == "Anthony" and body["db_saved"] is True


def test_api_grade_unknown_user_is_404_before_grading(video_file, isolated, monkeypatch):
    from fastapi.testclient import TestClient
    import api
    monkeypatch.setattr(config, "SPACETIME_DB", "taekwon-db")
    fake = FakeDB(monkeypatch)
    monkeypatch.setattr(db, "get_user", lambda u: None)
    with open(video_file, "rb") as f:
        r = TestClient(api.app).post("/grade", files={"video": ("kick.mp4", f, "video/mp4")},
                                     data={"username": "ghost"})
    assert r.status_code == 404 and "Sign up" in r.json()["detail"]
    assert fake.calls == []   # nothing was graded or saved


def test_api_users_endpoints(monkeypatch):
    from fastapi.testclient import TestClient
    import api
    c = TestClient(api.app)
    monkeypatch.setattr(config, "SPACETIME_DB", "")
    assert c.post("/users", json={"username": "Haris"}).status_code == 503
    monkeypatch.setattr(config, "SPACETIME_DB", "taekwon-db")
    fake = FakeDB(monkeypatch)
    monkeypatch.setattr(db, "sql", lambda q: [{"user_id": 1, "username": "haris"}])
    assert c.post("/users", json={"username": "Haris"}).status_code == 201
    assert fake.calls == [("create_user", ("Haris",))]
    assert c.get("/users").json()[0]["username"] == "haris"


def test_api_signup_errors_map_to_status_codes(monkeypatch):
    from fastapi.testclient import TestClient
    import api
    c = TestClient(api.app)
    monkeypatch.setattr(config, "SPACETIME_DB", "taekwon-db")

    def taken(u):
        raise db.UserError('Username "haris" is taken')
    monkeypatch.setattr(db, "create_user", taken)
    assert c.post("/users", json={"username": "haris"}).status_code == 409
    monkeypatch.setattr(db, "create_user", lambda u: db.clean_username(u))
    assert c.post("/users", json={"username": "a!"}).status_code == 400


def test_api_login_and_search(monkeypatch):
    from fastapi.testclient import TestClient
    import api
    c = TestClient(api.app)
    monkeypatch.setattr(config, "SPACETIME_DB", "taekwon-db")
    users = [{"user_id": 1, "username": "haris", "created_at": "2026-11-08T12:00:00+00:00"},
             {"user_id": 2, "username": "anthony", "created_at": "2026-11-08T12:00:00+00:00"}]
    monkeypatch.setattr(db, "sql", lambda q: [u for u in users if "WHERE" not in q or f"'{u['username']}'" in q])
    assert c.post("/login", json={"username": " Haris "}).json()["username"] == "haris"
    assert c.post("/login", json={"username": "nobody"}).status_code == 404
    assert [u["username"] for u in c.get("/users", params={"search": "ant"}).json()] == ["anthony"]
    assert [u["username"] for u in c.get("/users").json()] == ["anthony", "haris"]


def test_kicks_for_builds_statlines(monkeypatch):
    tables = {
        "user": [{"user_id": 1, "username": "haris", "created_at": "t"}],
        "kick": [{"kick_id": 10, "user_id": 1, "result_id": "r1", "recorded_time": "2026-11-08T12:00:00+00:00",
                  "status_id": 1, "fail_reason": ""},
                 {"kick_id": 11, "user_id": 1, "result_id": "r2", "recorded_time": "2026-11-08T12:05:00+00:00",
                  "status_id": 4, "fail_reason": "Step back"}],
        "evaluation": [{"kick_id": 10, "chamber_height_result": True, "knee_below_hip_ratio": -0.5,
                        "chamber_tightness_result": False, "ankle_forward_knee_ratio": 0.2,
                        "kick_angle_result": True, "leg_angle": 95.0, "kick_straightness_result": True,
                        "knee_angle": 170.0, "kick_tier": "strong", "feedback": "Great", "total_score": 3,
                        "feedback_source": "nemotron"}],
        "media_asset": [{"asset_id": 1, "kick_id": 10, "file_type_id": 2, "file_name": "/results/r1/chamber.jpg"},
                        {"asset_id": 2, "kick_id": 10, "file_type_id": 1, "file_name": "/results/r1/clip"}],
    }
    monkeypatch.setattr(db, "sql", lambda q: tables[q.split("FROM ")[1].split()[0]])
    kicks = db.kicks_for("haris")
    assert [k["result_id"] for k in kicks] == ["r2", "r1"]          # newest first
    assert kicks[0]["status"] == "failed" and kicks[0]["fail_reason"] == "Step back"
    assert kicks[1]["passed"] == 3 and kicks[1]["checks"]["chamber_tightness"]["pass"] == 0
    assert kicks[1]["images"]["chamber"] == "/results/r1/chamber.jpg" and kicks[1]["clip_url"] == "/results/r1/clip"
    p = db.user_profile("haris")
    assert p["total_kicks"] == 2 and p["completed_kicks"] == 1 and p["average_passed"] == 3
    assert p["check_pass_rates"]["chamber_tightness"] == 0


def test_api_delete_kick(video_file, isolated, monkeypatch):
    from fastapi.testclient import TestClient
    import api
    c = TestClient(api.app)
    with open(video_file, "rb") as f:
        rid = c.post("/grade", files={"video": ("kick.mp4", f, "video/mp4")}).json()["id"]
    monkeypatch.setattr(config, "SPACETIME_DB", "taekwon-db")
    fake = FakeDB(monkeypatch)
    assert c.delete(f"/results/{rid}").json() == {"ok": True, "files_deleted": True, "db_deleted": True}
    assert fake.calls == [("delete_kick", (rid,))]
    assert c.get(f"/results/{rid}").status_code == 404


def test_db_sync_skips_existing(video_file, isolated, monkeypatch):
    r1 = pipeline.grade_video(video_file, use_nemotron=False)
    r2 = pipeline.grade_video(video_file, use_nemotron=False)
    fake = FakeDB(monkeypatch)
    monkeypatch.setattr(db, "sql", lambda q: [{"result_id": r1["id"]}])
    assert db.sync_results("Haris") == (1, 1)
    assert fake.names() == ["start_kick", "complete_kick"] and fake.calls[0][1] == ("Haris", r2["id"])


def test_media_upload_skips_big_files_and_can_be_turned_off(video_file, isolated, monkeypatch):
    monkeypatch.setattr(config, "SPACETIME_DB", "taekwon-db")
    fake = FakeDB(monkeypatch)
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0.0001)          # everything is "too big"
    r = pipeline.grade_video(video_file, use_nemotron=False, username="Haris")
    assert r["media_saved"] == [] and "upload_media" not in fake.names()
    assert any("kept on disk only" in n for n in r["notes"])
    monkeypatch.setattr(config, "UPLOAD_MEDIA", False)
    r = pipeline.grade_video(video_file, use_nemotron=False, username="Haris")
    assert "media_saved" not in r


def test_api_serves_media_from_db_when_not_on_disk(monkeypatch):
    from fastapi.testclient import TestClient
    import api
    monkeypatch.setattr(config, "SPACETIME_DB", "taekwon-db")
    monkeypatch.setattr(db, "get_media", lambda rid, kind: (b"VIDEOBYTES", "video/webm") if kind == "clip" else None)
    c = TestClient(api.app)
    r = c.get("/results/not-on-this-laptop/clip")
    assert r.status_code == 200 and r.content == b"VIDEOBYTES" and r.headers["content-type"] == "video/webm"
    assert c.get("/results/not-on-this-laptop/chamber.jpg").status_code == 404
