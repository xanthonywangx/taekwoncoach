"""Built-in coaching text. Used when Nemotron is off or unreachable, so the demo never breaks."""

TIPS = {
    "chamber_tightness": "Fold the kicking leg tighter in the chamber: bring your heel back toward your butt so your foot stays behind your knee.",
    "chamber_height": "Lift the knee higher before you kick. Aim for knee at hip height or above.",
    "kick_straightness": "Snap the lower leg all the way out so your leg finishes nearly straight, then pull it back.",
    "kick_angle": "Kick higher: try to get your kicking foot up to at least hip level.",
}

OPENERS = {
    "strong": "Great kick!",
    "even": "Good start, a couple of things will make this cleaner.",
    "building": "Nice effort. Here are a few things to try on the next one.",
}


def fallback_feedback(grade):
    misses = [TIPS[k] for k, c in grade["checks"].items() if not c["pass"]]
    lead = "Consider these tips: " if grade["tier"] == "building" else ""
    text = " ".join([OPENERS[grade["tier"]]] + ([lead + " ".join(misses)] if misses else []))
    if grade.get("low_visibility"):
        text += " Tip for next time: better lighting or stepping back a bit will help the camera see your legs."
    return text
