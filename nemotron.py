"""The two Nemotron 3.5 Super VL passes.

Pass 1 locate_kick: looks at a contact sheet of the clip, finds the kick, and rejects unusable ones
Pass 2 review_kick: looks at the chamber + extension frames and the measured scores, confirms the
frames, flags doubtful checks as uncertain n then coaches feedback
"""
import base64
import json
import re

import config
from scoring import CHECK_NAMES

_client = None


def client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=config.NEMOTRON_BASE_URL, api_key=config.NVIDIA_API_KEY,
                         timeout=config.NEMOTRON_TIMEOUT_S, max_retries=1)
    return _client


def _image(jpeg_bytes):
    b64 = base64.b64encode(jpeg_bytes).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def parse_json(text):
    """Pulls the JSON object out of a model reply (handles ```json fences and stray text)."""
    if not text:
        raise ValueError("empty reply")
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON in reply: {text[:200]}")
    return json.loads(text[start:end + 1])


def _ask(system, content, max_tokens=4096):
    r = client().chat.completions.create(
        model=config.NEMOTRON_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    msg = r.choices[0].message
    text = msg.content or getattr(msg, "reasoning_content", None) or ""
    return parse_json(text)


def ping():
    """Tiny text-only call to check the key and model name."""
    r = client().chat.completions.create(
        model=config.NEMOTRON_MODEL,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        max_tokens=512,
    )
    return (r.choices[0].message.content or "").strip()


# ---- Pass 1 ----------------------------------------------------------------------------------

LOCATE_SYSTEM = """You review webcam clips for a taekwondo training app. You will get a contact sheet:
a grid of frames from ONE short clip, read left to right, top to bottom. Each tile shows its timestamp in seconds.
The person should be filmed from the side doing ONE front snap kick.

Find these moments (use the tile timestamps):
- start_s: the knee starts to lift.
- chamber_s: the CHAMBER = the moment the kicking knee is lifted and bent the MOST (tightest knee angle).
  It always happens BEFORE the extension. Ignore the fold-back after the kick.
- extension_s: the EXTENSION = after the chamber, the moment the kicking leg is opened the MOST
  (largest knee angle). If two tiles look equally extended, pick the less blurry one.
  A real extension opens the knee past 90 degrees.
- end_s: the kicking foot is back on the floor.

Set usable=false ONLY if: there is no kick, the person's head or feet leave the frame during the kick,
more than one person is kicking, or it is too dark/blurry to see the legs.
Bad technique is NOT a reason for usable=false; technique is graded later.

Reply with ONLY this JSON, no other text:
{"usable": true, "reason": null, "kick_type": "front snap kick", "start_s": 1.2, "chamber_s": 1.6, "extension_s": 1.8, "end_s": 2.4}
reason: one short sentence the athlete can act on, only when usable=false.
kick_type: "front snap kick", "other kick", or "none"."""


def locate_kick(sheet_jpeg):
    data = _ask(LOCATE_SYSTEM, [
        {"type": "text", "text": "Here is the contact sheet. Find the kick."},
        _image(sheet_jpeg),
    ], max_tokens=4096)
    out = {
        "usable": bool(data.get("usable", True)),
        "reason": data.get("reason"),
        "kick_type": data.get("kick_type"),
        "start_s": _num(data.get("start_s")),
        "chamber_s": _num(data.get("chamber_s")),
        "extension_s": _num(data.get("extension_s")),
        "end_s": _num(data.get("end_s")),
    }
    if out["chamber_s"] is not None and out["extension_s"] is not None and out["extension_s"] <= out["chamber_s"]:
        out["chamber_s"] = out["extension_s"] = None   # chamber must come first; drop hints that break that
    if out["start_s"] is not None and out["end_s"] is not None and out["end_s"] < out["start_s"]:
        out["start_s"], out["end_s"] = out["end_s"], out["start_s"]
    return out


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---- Pass 2 ----------------------------------------------------------------------------------

REVIEW_SYSTEM = """You are a friendly taekwondo coach for BEGINNERS reviewing one front snap kick filmed from the side.
A measurement system already graded 4 checks (pass 1 = met, 0 = not met):
- chamber_tightness: in the chamber, the kicking foot should stay behind the knee (heel tucked).
- chamber_height: in the chamber, the knee should be at or above hip height.
- kick_straightness: at extension, the kicking leg should be nearly straight.
- kick_angle: at extension, the angle between standing leg and kicking leg should be more than 90 degrees.

You get two images with the legs drawn on (orange = kicking leg, blue = standing leg):
image 1 = the frame chosen as the CHAMBER (the tightest knee bend, before the extension),
image 2 = the frame chosen as the EXTENSION (after the chamber, the leg opened the most; knee past 90 degrees).

Your jobs:
1. frames_ok: false only if image 1 does not show a lifted, folded knee, or image 2 does not show the leg
   kicking out at or near its furthest point, or the drawn lines clearly don't sit on the person's legs.
2. uncertain_checks: list check names where what you SEE clearly contradicts the measured pass value.
   Do not change any pass value yourself. Use an empty list when the measurements look right.
3. extra_observations: up to 2 short things you can see that the checks don't cover
   (guard hands dropped, leaning far back, standing heel lifted). Empty list if nothing stands out.
4. feedback: coaching text for the athlete, max 90 words, plain text:
   - tier "strong": congratulate first, then a short tip for each failed check.
   - tier "even": brief encouragement, then a tip for each failed check.
   - tier "building": warm tone; phrase every fix as "try this" or "consider this". Never say bad, wrong, or failed.
   - One concrete physical cue per failed check. You may add one extra_observation as a bonus tip.

Reply with ONLY this JSON, no other text:
{"frames_ok": true, "frame_issue": null, "uncertain_checks": [], "extra_observations": [], "feedback": "..."}"""


def review_kick(grade, chamber_jpeg, extension_jpeg):
    summary = {k: grade[k] for k in ("checks", "passed", "tier", "low_visibility")}
    data = _ask(REVIEW_SYSTEM, [
        {"type": "text", "text": "Measured grade:\n" + json.dumps(summary)},
        {"type": "text", "text": "Image 1 (chamber):"},
        _image(chamber_jpeg),
        {"type": "text", "text": "Image 2 (extension):"},
        _image(extension_jpeg),
    ], max_tokens=4096)
    uncertain = [c for c in data.get("uncertain_checks") or [] if c in CHECK_NAMES]
    extras = [str(x) for x in (data.get("extra_observations") or [])][:2]
    feedback = str(data.get("feedback") or "").strip()
    return {
        "frames_ok": bool(data.get("frames_ok", True)),
        "frame_issue": data.get("frame_issue"),
        "uncertain_checks": uncertain,
        "extra_observations": extras,
        "feedback": feedback,
    }