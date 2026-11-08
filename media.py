"""
Fix clips saved before this existed (re-encodes them and re-uploads to SpacetimeDB):
  python media.py fix-all
"""
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

# Encoders to try, best first. libx264 is in imageio-ffmpeg's build; h264_videotoolbox is the
# Mac's built-in hardware encoder, used if libx264 is missing.
ENCODERS = [
    ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26"],
    ["-c:v", "h264_videotoolbox", "-b:v", "1500k"],
]


@lru_cache(maxsize=1)
def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def normalize_clip(src, dst, timeout=90):
    """Converts src to a browser-safe H.264 mp4 at dst. Returns (True, None) or (False, reason)."""
    exe = ffmpeg_exe()
    if not exe:
        return False, "ffmpeg not found (pip install imageio-ffmpeg)"
    src, dst = Path(src), Path(dst)
    tmp = dst.with_name(dst.stem + ".converting.mp4")
    last = "unknown error"
    for enc in ENCODERS:
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-an",
               "-vf", "scale='min(1280,iw)':-2", *enc, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
        try:
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            last = "conversion timed out"
            continue
        if run.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(dst)
            return True, None
        last = (run.stderr or "").strip().splitlines()[-1:] or ["failed"]
        last = last[0][:200]
    tmp.unlink(missing_ok=True)
    return False, last


def describe(path):
    """Short codec summary, e.g. 'h264 yuv420p 1280x720' (for checks and tests)."""
    exe = ffmpeg_exe()
    run = subprocess.run([exe, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    for line in run.stderr.splitlines():
        if "Video:" in line:
            return line.split("Video:", 1)[1].strip()
    return ""


def is_browser_safe(path):
    d = describe(path)
    return d.startswith("h264") and "yuv420p" in d and Path(path).suffix == ".mp4"


def fix_all():
    """Re-encode every saved clip that isn't browser-safe yet, and re-upload it to SpacetimeDB."""
    import config
    import db
    fixed = skipped = failed = 0
    for d in sorted(p for p in config.RESULTS_DIR.iterdir() if p.is_dir()) if config.RESULTS_DIR.exists() else []:
        clips = sorted(d.glob("clip.*"))
        if not clips:
            continue
        clip = clips[0]
        if is_browser_safe(clip):
            skipped += 1
            continue
        target = d / "clip.mp4"
        ok, why = normalize_clip(clip, target)
        if not ok:
            failed += 1
            print(f"  {d.name}: couldn't convert ({why})")
            continue
        if clip != target:
            clip.unlink()
        fixed += 1
        if config.spacetime_enabled():
            try:
                db.upload_media(d.name, "clip", target)
            except Exception as e:  # kick may not be in the database (no username), that's fine
                print(f"  {d.name}: converted, not re-uploaded ({type(e).__name__}: {str(e)[:80]})")
    print(f"Converted {fixed}, already fine {skipped}, failed {failed}.")


if __name__ == "__main__":
    if sys.argv[1:] == ["fix-all"]:
        fix_all()
    else:
        print(__doc__)
