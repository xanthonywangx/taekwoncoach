"""the 4 binary checks (0s n 1s ideology) + tier. deterministic aka same input, same grade."""
import numpy as np

import config
from geometry import angle_at, angle_between
from pose import P, LEGS

CHECK_NAMES = ["chamber_tightness", "chamber_height", "kick_straightness", "kick_angle"]


def _xy(frame, idx):
    return frame[idx][:2]


def find_kicking_leg(valid):
    """Kicking leg = the one whose ankle gets highest (smallest y)."""
    def peak(leg):
        return min(f[LEGS[leg]["ankle"]][1] for _, f in valid)
    kick = min(LEGS, key=peak)
    return kick, ("left" if kick == "right" else "right")


def find_facing(valid, hip_idx):
    """+1 if the athlete faces right in the image, -1 if left (nose sits ahead of the hips)."""
    dx = np.median([f[P.NOSE][0] - f[hip_idx][0] for _, f in valid])
    return 1 if dx > 0 else -1


def knee_angle(f, K):
    return angle_at(_xy(f, K["hip"]), _xy(f, K["knee"]), _xy(f, K["ankle"]))


def lifted_frames(valid, K, B):
    """Frames where the kicking knee is raised above the standing knee (chamber, extension, rechamber).
    Standing frames are excluded, otherwise a straight standing leg would look like a perfect extension."""
    return [(i, f) for i, f in valid if f[K["knee"]][1] < f[B["knee"]][1]]


def _best_extension(after, angles, sharpness):
    """Highest knee angle after the chamber; among frames within SHARPNESS_ANGLE_TOL_DEG of it, the sharpest.
    Raises ValueError if the knee never opens past EXTENSION_MIN_KNEE_DEG."""
    if not after:
        raise ValueError("Couldn't find the extension. Keep the camera recording until your foot is back down.")
    best = max(angles[i] for i, _ in after)
    if best <= config.EXTENSION_MIN_KNEE_DEG:
        raise ValueError("Your leg didn't extend. Snap the lower leg out from the chamber, then pull it back.")
    near_best = [(i, f) for i, f in after if angles[i] >= best - config.SHARPNESS_ANGLE_TOL_DEG]
    if sharpness is not None and len(near_best) > 1:
        return max(near_best, key=lambda x: (sharpness(x[0], x[1]), angles[x[0]]))
    return max(near_best, key=lambda x: angles[x[0]])


def _guided(lifted, angles, sharpness, chamber_near, extension_near):
    """Use Nemotron's chamber/extension moments: search only the frames near each one, same rules."""
    ch_pool = [(i, f) for i, f in lifted if i in chamber_near]
    if not ch_pool:
        raise ValueError("no lifted frames near Nemotron's chamber time")
    ch_i, ch = min(ch_pool, key=lambda x: angles[x[0]])
    after = [(i, f) for i, f in lifted if i > ch_i and (not extension_near or i in extension_near)]
    return (ch_i, ch), _best_extension(after, angles, sharpness)


def _unguided(lifted, angles, sharpness):
    """Python-only rules. Anchor on the most-open lifted frame so a tight rechamber AFTER the kick
    can't be picked as the chamber."""
    peak_i = max(lifted, key=lambda x: angles[x[0]])[0]
    before_peak = [(i, f) for i, f in lifted if i < peak_i]
    if not before_peak:
        raise ValueError("Couldn't find the chamber. Start recording before you lift your knee.")
    ch_i, ch = min(before_peak, key=lambda x: angles[x[0]])
    after = [(i, f) for i, f in lifted if i > ch_i]
    return (ch_i, ch), _best_extension(after, angles, sharpness)


def find_chamber_and_extension(valid, K, B, sharpness=None, chamber_near=None, extension_near=None):
    """Chamber = tightest knee angle, before the extension.
    Extension = after the chamber, highest knee angle, least blurry among near-ties, knee angle > 90.

    chamber_near / extension_near: optional sets of frame indices near Nemotron's timestamps.
    When given, frames are chosen near them first; if that fails, the Python-only rules run.
    Returns ((ch_i, ch), (ext_i, ext), source). Raises ValueError with a user-facing message.
    """
    lifted = lifted_frames(valid, K, B)
    if len(lifted) < 2:
        raise ValueError("Couldn't see the knee come up. Make sure your kicking leg is the one closest to the camera.")
    angles = {i: knee_angle(f, K) for i, f in lifted}
    if chamber_near:
        try:
            ch, ext = _guided(lifted, angles, sharpness, chamber_near, extension_near)
            return ch, ext, "nemotron"
        except ValueError:
            pass  # Nemotron's moments didn't line up with the pose data; fall back to Python rules
    ch, ext = _unguided(lifted, angles, sharpness)
    return ch, ext, "python"


# ---- The 4 checks. Each returns (pass 0/1, measured value) -------------------------------

def check_chamber_tightness(ch, K, facing, thigh):
    """Ankle must NOT be in front of the knee."""
    ankle_fwd = float((ch[K["ankle"]][0] - ch[K["knee"]][0]) * facing / thigh)
    return int(ankle_fwd <= config.CHAMBER_FWD_TOL), round(ankle_fwd, 2)


def check_chamber_height(ch, K, thigh):
    """Knee must NOT be lower than the hip (image y grows downward)."""
    knee_below = float((ch[K["knee"]][1] - ch[K["hip"]][1]) / thigh)
    return int(knee_below <= config.CHAMBER_HEIGHT_TOL), round(knee_below, 2)


def check_straightness(ext, K):
    """Hip-knee-ankle angle at extension above threshold."""
    ang = angle_at(_xy(ext, K["hip"]), _xy(ext, K["knee"]), _xy(ext, K["ankle"]))
    return int(ang > config.STRAIGHT_DEG), round(ang, 1)


def check_kick_angle(ext, K, B):
    """Angle between base leg and kicking leg, measured from the hip."""
    hip = _xy(ext, K["hip"])
    ang = angle_between(_xy(ext, B["ankle"]) - hip, _xy(ext, K["ankle"]) - hip)
    return int(ang > config.KICK_ANGLE_DEG), round(ang, 1)


def tier_for(passed):
    """Majority 1s -> congratulate; tie -> feedback; majority 0s -> soft 'consider these tips'."""
    return "strong" if passed >= 3 else "even" if passed == 2 else "building"


def grade(landmarks, sharpness=None, chamber_near=None, extension_near=None):
    """landmarks: list of (frame_idx, array|None) from pose.detect. Returns a JSON-safe dict.
    sharpness: optional callable(frame_idx, landmarks) -> float used to pick the least blurry extension.
    chamber_near / extension_near: optional sets of frame indices near Nemotron's chosen moments.

    Raises ValueError with a user-facing message if the clip can't be graded.
    """
    valid = [(i, f) for i, f in landmarks if f is not None]
    if len(valid) < 5:
        raise ValueError("Couldn't see your body clearly during the kick. Step back so head to feet are in frame.")

    kick, base = find_kicking_leg(valid)
    K, B = LEGS[kick], LEGS[base]
    facing = find_facing(valid, K["hip"])

    (ch_i, ch), (ext_i, ext), frame_source = find_chamber_and_extension(
        valid, K, B, sharpness, chamber_near, extension_near)

    thigh = float(np.linalg.norm(_xy(ch, K["hip"]) - _xy(ch, K["knee"]))) + 1e-9

    t_pass, t_val = check_chamber_tightness(ch, K, facing, thigh)
    h_pass, h_val = check_chamber_height(ch, K, thigh)
    s_pass, s_val = check_straightness(ext, K)
    a_pass, a_val = check_kick_angle(ext, K, B)

    checks = {
        "chamber_tightness": {"pass": t_pass, "ankle_forward_of_knee_ratio": t_val},
        "chamber_height":    {"pass": h_pass, "knee_below_hip_ratio": h_val},
        "kick_straightness": {"pass": s_pass, "knee_angle_deg": s_val, "target_deg": config.STRAIGHT_DEG},
        "kick_angle":        {"pass": a_pass, "leg_angle_deg": a_val, "target_deg": config.KICK_ANGLE_DEG},
    }
    for c in checks.values():
        c["uncertain"] = False
    passed = sum(c["pass"] for c in checks.values())
    low_vis = [name for name, idx in [("kicking ankle", K["ankle"]), ("kicking knee", K["knee"]),
                                      ("standing ankle", B["ankle"])]
               if ext[idx][2] < config.MIN_VISIBILITY]

    return {
        "kicking_leg": kick,
        "facing": "right" if facing == 1 else "left",
        "chamber_frame": ch_i,
        "extension_frame": ext_i,
        "frames_chosen_by": frame_source,
        "chamber_knee_deg": round(knee_angle(ch, K), 1),
        "extension_knee_deg": round(knee_angle(ext, K), 1),
        "checks": checks,
        "passed": passed,
        "tier": tier_for(passed),
        "low_visibility": low_vis,
    }