"""The full flow. Both the CLI (main.py) and the API (api.py) call grade_video().

1. Load clip
2. Find the kick: Nemotron pass 1 (contact sheet) and Python's lite-model scan, in parallel
3. Pad the window
4. MediaPipe heavy model on the window only
5. Python grading engine: exact frames (guided by Nemotron's chamber/extension moments) + 4 checks
6. Nemotron pass 2: check frames, flag uncertain checks, coach
7. Save: full result + images to results/<id>/, kick + evaluation + media rows to SpacetimeDB (if a username is given)
"""
import time
from concurrent.futures import ThreadPoolExecutor

import config
import db
import feedback
import kick_finder
import nemotron
import pose
import scoring
import storage
import video
from pose import LEGS


def grade_video(video_path, use_nemotron=None, save=True, username=None):
    """Returns the result dict. result["status"] is "graded" or "retake".
    username: username to record the kick under in SpacetimeDB (must already have an account).
              None = results/ folder only.

    SpacetimeDB lifecycle (live for the frontend): Uploaded -> Processing -> Completed / Failed."""
    username = (username or "").strip() or None
    rid = storage.new_id()
    pre_notes = []
    use_db = bool(save and username and config.spacetime_enabled())
    if use_db:
        try:
            db.start_kick(username, rid)
            db.set_status(rid, db.STATUS_PROCESSING)
        except db.UserError as e:  # e.g. no account with that username
            pre_notes.append(f"Not saved to SpacetimeDB: {e}")
            use_db = False
        except Exception as e:  # the grade still runs even if the database is down
            pre_notes.append(f"SpacetimeDB unavailable ({type(e).__name__}: {e}); saved to results/ only.")
            use_db = False
    try:
        result = _grade(video_path, use_nemotron, save, rid, username, pre_notes)
    except Exception:
        if use_db:  # mark the kick Failed so the frontend doesn't show "Processing" forever
            try:
                db.complete_kick(rid, {"id": rid, "status": "retake",
                                       "retake_reason": "Something went wrong while grading. Please try again."})
            except Exception:
                pass
        raise
    result["db_saved"] = False
    if use_db:
        try:
            db.complete_kick(rid, result)
            result["db_saved"] = True
        except Exception as e:
            result["notes"].append(f"SpacetimeDB save failed ({type(e).__name__}: {e}).")
        if result["db_saved"] and save and config.UPLOAD_MEDIA:
            result["media_saved"] = _upload_media(rid, result["notes"])
    if save:
        storage.update(result)
    return result


def _upload_media(rid, notes):
    """Sends the clip and frame images to SpacetimeDB. Returns the list of kinds that were stored."""
    files = {"clip": storage.clip_path(rid),
             "chamber": storage.image_path(rid, "chamber"),
             "extension": storage.image_path(rid, "extension")}
    saved = []
    for kind, path in files.items():
        if path is None:
            continue   # e.g. a retake has no chamber/extension images
        mb = path.stat().st_size / 1_048_576
        if mb > config.MAX_UPLOAD_MB:
            notes.append(f"{kind} is {mb:.1f} MB, over MAX_UPLOAD_MB={config.MAX_UPLOAD_MB}; kept on disk only.")
            continue
        try:
            db.upload_media(rid, kind, path)
            saved.append(kind)
        except Exception as e:
            notes.append(f"Couldn't store {kind} in SpacetimeDB ({type(e).__name__}: {e}); kept on disk.")
    return saved


def _grade(video_path, use_nemotron, save, rid, username, notes):
    use_nemotron = config.nemotron_available() if use_nemotron is None else (use_nemotron and config.nemotron_available())
    timings, images = {}, {}
    result = {"id": rid, "status": None, "username": username, "nemotron_used": use_nemotron,
              "notes": notes, "timings_s": timings}
    t0 = time.perf_counter()

    def lap(name, since):
        timings[name] = round(time.perf_counter() - since, 2)
        return time.perf_counter()

    def finish(status, message=None):
        result["status"] = status
        if message:
            result["retake_reason"] = message
        timings["total"] = round(time.perf_counter() - t0, 2)
        if save:
            storage.save(result, images, video_path)
        return result

    # 1. Load
    t = time.perf_counter()
    clip = video.load_clip(video_path)
    if len(clip.frames) < 10:
        return finish("retake", "The video is too short. Record a little before and after the kick.")
    result["clip_seconds"] = round(clip.duration, 2)
    t = lap("load", t)

    # 2. Find the kick: Nemotron pass 1 + Python scan in parallel
    sheet, _ = video.contact_sheet(clip)
    images["contact_sheet"] = sheet
    with ThreadPoolExecutor(max_workers=2) as ex:
        scan_future = ex.submit(kick_finder.scan, clip)
        loc_future = ex.submit(nemotron.locate_kick, sheet) if use_nemotron else None
        scan = scan_future.result()
        loc = None
        if loc_future:
            try:
                loc = loc_future.result()
            except Exception as e:
                notes.append(f"Nemotron pass 1 unavailable ({type(e).__name__}: {e}); used Python scan.")
    result["kick_search"] = {"nemotron": loc, "python": scan}
    t = lap("find_kick", t)

    if loc and not loc["usable"]:
        return finish("retake", loc["reason"] or "Couldn't grade this clip. Try again with your whole body in frame.")

    # 3. Decide the window
    window = choose_window(loc, scan, clip.duration)
    if window is None:
        return finish("retake", "Couldn't find a kick in the video. Stand still for a second, kick once, then hold still again.")
    result["window"] = window

    # 4. Detailed pose on the window only
    idx = clip.indices_between(window["start_s"], window["end_s"])
    landmarks = pose.detect(clip, idx, config.POSE_MODEL_HEAVY)
    landmarks, swaps = pose.fix_leg_swaps(landmarks)
    if swaps:
        notes.append(f"Fixed MediaPipe left/right leg mix-ups in {swaps} frame(s).")
    t = lap("pose", t)

    # Nemotron's chamber/extension moments -> nearby frame indices for Python to search
    tol = config.NEMOTRON_HINT_TOL_S
    chamber_near = extension_near = None
    if loc and loc.get("chamber_s") is not None:
        chamber_near = set(clip.indices_between(loc["chamber_s"] - tol, loc["chamber_s"] + tol))
    if loc and loc.get("extension_s") is not None:
        extension_near = set(clip.indices_between(loc["extension_s"] - tol, loc["extension_s"] + tol))

    # 5. Grade
    def sharpness(i, lm):  # blur score around whichever leg is higher in this frame (the kicking leg)
        leg = min(LEGS.values(), key=lambda L: lm[L["ankle"]][1])
        return video.leg_sharpness(clip.frames[i], lm, leg)

    try:
        grade = scoring.grade(landmarks, sharpness=sharpness,
                              chamber_near=chamber_near, extension_near=extension_near)
    except ValueError as e:
        return finish("retake", str(e))
    result.update(grade)
    lm = dict(landmarks)
    K, B = LEGS[grade["kicking_leg"]], LEGS["left" if grade["kicking_leg"] == "right" else "right"]
    for name in ("chamber", "extension"):
        i = grade[f"{name}_frame"]
        result[f"{name}_time_s"] = round(clip.times[i], 2)
        images[name] = video.to_jpeg(video.draw_legs(clip.frames[i], lm[i], K, B, name.upper()))
    t = lap("grade", t)

    # 6. Nemotron pass 2
    review = None
    if use_nemotron:
        try:
            review = nemotron.review_kick(grade, images["chamber"], images["extension"])
        except Exception as e:
            notes.append(f"Nemotron pass 2 unavailable ({type(e).__name__}: {e}); used built-in feedback.")
    result["review"] = review
    if review:
        review["uncertain_checks"] = [c for c in review["uncertain_checks"] if c in result["checks"]]
        for c in review["uncertain_checks"]:
            result["checks"][c]["uncertain"] = True
        result["extra_observations"] = review["extra_observations"]
        result["feedback"] = review["feedback"] or feedback.fallback_feedback(grade)
        result["feedback_source"] = "nemotron" if review["feedback"] else "fallback"
    else:
        result["extra_observations"] = []
        result["feedback"] = feedback.fallback_feedback(grade)
        result["feedback_source"] = "fallback"
    lap("review", t)

    if review and not review["frames_ok"]:
        return finish("retake", review["frame_issue"] or "The camera didn't catch the kick clearly. Please try again.")
    return finish("graded")


def choose_window(loc, scan, duration):
    """Combines Nemotron's rough window with Python's ankle-peak scan. Returns dict or None."""
    pad = config.WINDOW_PAD_S
    nem = None
    if loc and loc.get("start_s") is not None and loc.get("end_s") is not None:
        nem = (loc["start_s"], loc["end_s"])
    py = None
    if scan and scan.get("found"):
        py = (scan["peak_s"] - config.SCAN_HALF_WINDOW_S, scan["peak_s"] + config.SCAN_HALF_WINDOW_S)

    if nem and py:
        agree = nem[0] - pad <= scan["peak_s"] <= nem[1] + pad
        start, end = min(nem[0], py[0]), max(nem[1], py[1])
        # agree: one window covering both. disagree: still cover both so Nemotron's moments stay
        # usable, and the frame rules decide (they fall back to Python if Nemotron's moments don't fit).
        source = "both" if agree else "both (disagreed)"
    elif nem:
        start, end, source = nem[0], nem[1], "nemotron"
    elif py:
        start, end, source = py[0], py[1], "python"
    else:
        return None
    return {"start_s": round(max(0.0, start - pad), 2),
            "end_s": round(min(duration, end + pad), 2),
            "source": source}
