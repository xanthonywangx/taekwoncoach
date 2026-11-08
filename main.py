"""Command line for testing the backend without a frontend.

  python main.py --check                    # checks your setup
  python main.py --record                   # record with the webcam, then grade
  python main.py --video kick.mp4           # grade an existing clip
  add --no-nemotron to either to skip Nemotron (offline mode)
"""
import argparse
import json
import sys

import config


def print_result(r):
    print("\n" + "=" * 60)
    if r["status"] == "retake":
        print(f"RETAKE: {r.get('retake_reason')}")
    else:
        print(f"Passed {r['passed']}/4   tier: {r['tier']}   kicking leg: {r['kicking_leg']}")
        for name, c in r["checks"].items():
            mark = "PASS" if c["pass"] else "----"
            flag = "  (uncertain)" if c.get("uncertain") else ""
            vals = {k: v for k, v in c.items() if k not in ("pass", "uncertain")}
            print(f"  [{mark}] {name:<18} {vals}{flag}")
        print(f"\nCoach ({r['feedback_source']}): {r['feedback']}")
    if r.get("frames_chosen_by"):
        print(f"Frames chosen by: {r['frames_chosen_by']}  "
              f"(chamber {r.get('chamber_knee_deg')} deg at {r.get('chamber_time_s')}s, "
              f"extension {r.get('extension_knee_deg')} deg at {r.get('extension_time_s')}s)")
    w = r.get("window")
    if w:
        print(f"\nKick window: {w['start_s']}s - {w['end_s']}s (found by {w['source']})")
    if r.get("db_saved"):
        print("Saved to SpacetimeDB: yes")
    if r.get("username"):
        print(f"SpacetimeDB: {'saved' if r.get('db_saved') else 'NOT saved'} under user '{r['username']}'")
    for n in r.get("notes", []):
        print(f"Note: {n}")
    print(f"Timings: {r.get('timings_s')}")
    if r.get("id"):
        from storage import result_dir
        print(f"Saved to: {result_dir(r['id'])}  (open chamber.jpg / extension.jpg to check the frames)")
    print("=" * 60)


def check_setup():
    ok = True
    print(f"Python {sys.version.split()[0]}", "OK" if sys.version_info[:2] in [(3, 10), (3, 11), (3, 12)]
          else "(3.12 recommended; may still work)")
    try:
        import mediapipe, cv2, fastapi, openai  # noqa: F401
        print(f"Packages OK (mediapipe {mediapipe.__version__}, opencv {cv2.__version__})")
    except ImportError as e:
        ok = False
        print(f"Missing package: {e}. Run: pip install -r requirements.txt")
    for p in (config.POSE_MODEL_HEAVY, config.POSE_MODEL_LITE):
        exists = p.exists() and p.stat().st_size > 1_000_000
        ok &= exists
        print(f"{p.name}: {'OK' if exists else 'MISSING, run: python download_models.py'}")
    import cv2
    cap = cv2.VideoCapture(0)
    cam_ok = cap.isOpened() and cap.read()[0]
    cap.release()
    print(f"Webcam: {'OK' if cam_ok else 'NOT AVAILABLE (only needed for --record)'}")
    if not config.NVIDIA_API_KEY:
        print("Nemotron: no NVIDIA_API_KEY in .env, so it will run in offline mode (fine for testing)")
    else:
        try:
            import nemotron
            reply = nemotron.ping()
            print(f"Nemotron: OK, model {config.NEMOTRON_MODEL} replied: {reply[:40]!r}")
        except Exception as e:
            ok = False
            print(f"Nemotron: FAILED ({type(e).__name__}: {e})")
            print("  Check the key in .env and that NEMOTRON_MODEL matches the model ID on build.nvidia.com")
    if not config.spacetime_enabled():
        print("SpacetimeDB: SPACETIME_DB not set in .env (kicks only saved to results/)")
    else:
        try:
            import db
            n = len(db.sql("SELECT * FROM user"))
            print(f"SpacetimeDB: OK, database '{config.SPACETIME_DB}' at {config.SPACETIME_URL}, {n} user(s)")
        except Exception as e:
            ok = False
            print(f"SpacetimeDB: FAILED ({type(e).__name__}: {e})")
            print("  Is `spacetime start` running, did you `spacetime publish taekwoncoach --server local`, and is SPACETIME_DB=taekwoncoach in .env?")
    print("\nAll good." if ok else "\nFix the items above, then run --check again.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--record", action="store_true")
    src.add_argument("--check", action="store_true")
    ap.add_argument("--no-nemotron", action="store_true")
    ap.add_argument("--user", dest="user", help="username to save the kick under in SpacetimeDB (created if new)")
    ap.add_argument("--json", action="store_true", help="print the raw JSON the API would return")
    a = ap.parse_args()

    if a.check:
        check_setup()
        sys.exit()

    from pipeline import grade_video
    if a.user and config.spacetime_enabled():   # command line convenience: sign up if needed
        import db
        try:
            if db.get_user(a.user) is None:
                print(f"Creating account '{db.create_user(a.user)}'")
        except db.UserError as e:
            sys.exit(str(e))
    path = a.video
    if a.record:
        from capture import record_clip
        path = record_clip()
    print("Grading... (Nemotron calls can take 10-30s)" if not a.no_nemotron and config.nemotron_available()
          else "Grading (offline mode)...")
    result = grade_video(path, use_nemotron=not a.no_nemotron, username=a.user)
    if a.json:
        print(json.dumps(result, indent=2))
    print_result(result)
