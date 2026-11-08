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