"""SpacetimeDB over its HTTP API. Off unless SPACETIME_DB is set in .env.

Tables follow the whiteboard design: user, kick, evaluation, media_asset, record_status, file_type.
Python writes the kick lifecycle (start_kick -> set_kick_status -> complete_kick). React reads the
tables live and can add users / delete kicks. Database-side names are snake_case (user_id, kick_tier).

Command line, for checking the database without a frontend:
  python db.py add-user Haris
  python db.py users
  python db.py kicks
  python db.py evaluations
  python db.py assets
  python db.py sync Haris          # copy every saved result in results/ in under Haris
"""
import json
import re
import sys
from datetime import datetime, timezone

import httpx

import config

_token = None

USERNAME_RE = re.compile(r"^[a-z0-9_]{3,20}$")
CHECKS = ["chamber_tightness", "chamber_height", "kick_straightness", "kick_angle"]


class UserError(ValueError):
    """Something the user can fix (name taken, bad name, unknown user). The message is user-facing."""


def clean_username(raw):
    """Same rule as the module: trimmed, lowercase, 3-20 letters/numbers/_."""
    name = (raw or "").strip().lower()
    if not USERNAME_RE.match(name):
        raise UserError("Username must be 3-20 characters: letters, numbers, or _")
    return name


def _url(path=""):
    return f"{config.SPACETIME_URL.rstrip('/')}/v1/database/{config.SPACETIME_DB}{path}"


def _headers():
    """Uses SPACETIME_TOKEN if set; otherwise asks the server for an identity once and reuses it."""
    global _token
    token = config.SPACETIME_TOKEN or _token
    if not token:
        try:
            r = httpx.post(f"{config.SPACETIME_URL.rstrip('/')}/v1/identity", timeout=5)
            r.raise_for_status()
            _token = token = r.json()["token"]
        except Exception:
            return {}  # fall back to an anonymous call
    return {"Authorization": f"Bearer {token}"}


def call(reducer, *args):
    """Runs a reducer by its snake_case name (falls back to camelCase if the server doesn't know it)."""
    last = None
    for name in dict.fromkeys([reducer, _camel(reducer)]):
        r = httpx.post(_url(f"/call/{name}"), json=list(args), headers=_headers(), timeout=10)
        if r.status_code == 404:
            last = r
            continue
        if r.status_code == 530:   # the module threw SenderError: a message meant for the user
            raise UserError(r.text.strip() or f"{name} was rejected")
        if r.status_code >= 400:
            raise RuntimeError(f"{name} failed ({r.status_code}): {r.text[:300]}")
        return
    raise RuntimeError(f"Reducer {reducer} not found in '{config.SPACETIME_DB}'. Is the module published? "
                       f"({last.status_code}: {last.text[:200]})")


def sql(query):
    """Runs a read query. Returns a list of dicts (one per row)."""
    r = httpx.post(_url("/sql"), content=query, headers=_headers(), timeout=10)
    if r.status_code >= 400:
        raise RuntimeError(f"SQL failed ({r.status_code}): {r.text[:300]}")
    out = []
    for stmt in r.json():
        names = []
        for el in stmt["schema"]["elements"]:
            n = el.get("name")
            names.append(n.get("some") if isinstance(n, dict) else n)
        out += [{n: _decode(v) for n, v in zip(names, row)} for row in stmt["rows"]]
    return out


def _decode(v):
    """SpacetimeDB timestamps arrive as [microseconds]; return an ISO string the frontend can show."""
    if isinstance(v, list) and len(v) == 1 and isinstance(v[0], int) and v[0] > 10**15:
        return datetime.fromtimestamp(v[0] / 1e6, tz=timezone.utc).isoformat(timespec="seconds")
    return v


# ---- Users and kicks ---------------------------------------------------------------------------

STATUS_COMPLETED, STATUS_PROCESSING, STATUS_UPLOADED, STATUS_FAILED = 1, 2, 3, 4


def create_user(username):
    """Sign up. Raises UserError if the name is invalid or taken. Returns the stored (lowercase) name."""
    name = clean_username(username)
    call("create_user", name)
    return name


def get_user(username):
    """The user row, or None. Used for log in."""
    name = clean_username(username)
    rows = sql(f"SELECT * FROM user WHERE username = '{name}'")   # safe: name passed the regex
    return rows[0] if rows else None


def search_users(query="", limit=20):
    """Usernames containing the query (empty query = everyone), alphabetical."""
    q = (query or "").strip().lower()
    names = sorted(r["username"] for r in sql("SELECT * FROM user") if q in r["username"])
    return [{"username": n} for n in names[:limit]]


STATUS_NAMES = {1: "completed", 2: "processing", 3: "uploaded", 4: "failed"}
FILE_KEYS = {1: "clip", 2: "chamber", 3: "extension", 4: "result"}


def kicks_for(username, limit=50):
    """One user's kicks, newest first, each with its statline. Returns None if the user doesn't exist."""
    user = get_user(username)
    if user is None:
        return None
    kicks = sql(f"SELECT * FROM kick WHERE user_id = {int(user['user_id'])}")
    return _statlines(kicks, {user["user_id"]: user["username"]}, limit)


def recent_kicks(limit=50):
    """Everyone's kicks, newest first (the home feed). Each statline includes its username."""
    users = {u["user_id"]: u["username"] for u in sql("SELECT * FROM user")}
    return _statlines(sql("SELECT * FROM kick"), users, limit)


def kick_owner(result_id):
    """Username that owns a kick, or None if the kick isn't in the database."""
    if not RESULT_ID_RE.match(result_id or ""):
        return None
    kicks = sql(f"SELECT * FROM kick WHERE result_id = '{result_id}'")
    if not kicks:
        return None
    users = sql(f"SELECT * FROM user WHERE user_id = {int(kicks[0]['user_id'])}")
    return users[0]["username"] if users else None


def _statlines(kicks, usernames, limit):
    """Newest `limit` kicks -> statlines (joins evaluation + media_asset; never touches media_data)."""
    kicks = sorted(kicks, key=lambda k: k["recorded_time"], reverse=True)[:limit]
    ids = {k["kick_id"] for k in kicks}
    if not ids:
        return []
    evals = {e["kick_id"]: e for e in sql("SELECT * FROM evaluation") if e["kick_id"] in ids}
    files = {}
    for a in sql("SELECT * FROM media_asset"):
        if a["kick_id"] in ids:
            files.setdefault(a["kick_id"], {})[FILE_KEYS.get(a["file_type_id"], "other")] = a["file_name"]
    out = []
    for k in kicks:
        line = _statline(k, evals.get(k["kick_id"]), files.get(k["kick_id"], {}))
        line["username"] = usernames.get(k["user_id"], "")
        out.append(line)
    return out


def _statline(kick, ev, files):
    line = {
        "result_id": kick["result_id"],
        "recorded_time": kick["recorded_time"],
        "status": STATUS_NAMES.get(kick["status_id"], "unknown"),
        "fail_reason": kick["fail_reason"],
        "passed": None, "tier": None, "feedback": None, "feedback_source": None, "checks": None,
        "images": {k: v for k, v in files.items() if k in ("chamber", "extension")},
        "clip_url": files.get("clip"),
    }
    if ev:
        line.update({
            "passed": ev["total_score"],
            "tier": ev["kick_tier"],
            "feedback": ev["feedback"],
            "feedback_source": ev["feedback_source"],
            "checks": {
                "chamber_tightness": {"pass": int(ev["chamber_tightness_result"]),
                                      "ankle_forward_of_knee_ratio": ev["ankle_forward_knee_ratio"]},
                "chamber_height": {"pass": int(ev["chamber_height_result"]),
                                   "knee_below_hip_ratio": ev["knee_below_hip_ratio"]},
                "kick_straightness": {"pass": int(ev["kick_straightness_result"]), "knee_angle_deg": ev["knee_angle"]},
                "kick_angle": {"pass": int(ev["kick_angle_result"]), "leg_angle_deg": ev["leg_angle"]},
            },
        })
    return line


def user_profile(username):
    """Profile header + overall statline (averages across completed kicks). None if no such user."""
    user = get_user(username)
    if user is None:
        return None
    kicks = kicks_for(username, limit=10_000)
    done = [k for k in kicks if k["status"] == "completed"]
    rate = lambda c: round(sum(k["checks"][c]["pass"] for k in done) / len(done), 2) if done else None
    return {
        "username": user["username"],
        "created_at": user["created_at"],
        "total_kicks": len(kicks),
        "completed_kicks": len(done),
        "best_passed": max((k["passed"] for k in done), default=None),
        "average_passed": round(sum(k["passed"] for k in done) / len(done), 2) if done else None,
        "check_pass_rates": {c: rate(c) for c in CHECKS},
        "latest_kick": kicks[0] if kicks else None,
    }


def start_kick(username, result_id):
    """Creates the kick row (status Uploaded). Raises UserError if the username has no account."""
    call("start_kick", username, result_id)


def set_status(result_id, status_id):
    call("set_kick_status", result_id, status_id)


def kick_summary(result):
    """The fields complete_kick reads. Keep in sync with spacetimedb/src/index.ts."""
    keys = ("id", "status", "retake_reason", "passed", "tier", "checks", "feedback",
            "feedback_source", "images", "clip_url")
    return {k: result[k] for k in keys if k in result}


def complete_kick(result_id, result):
    """Writes the evaluation + media assets and sets the status to Completed (or Failed for a retake)."""
    call("complete_kick", result_id, json.dumps(kick_summary(result)))


MIME_BY_SUFFIX = {".webm": "video/webm", ".mp4": "video/mp4", ".mov": "video/quicktime", ".jpg": "image/jpeg"}
FILE_TYPE_IDS = {"clip": 1, "chamber": 2, "extension": 3}
RESULT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def upload_media(result_id, kind, path):
    """Stores one file's bytes in SpacetimeDB (media_data). kind: "clip", "chamber" or "extension".
    Call after complete_kick. Raises UserError if the module rejects it (e.g. over the size limit)."""
    import base64
    from pathlib import Path
    path = Path(path)
    mime = MIME_BY_SUFFIX.get(path.suffix.lower(), "application/octet-stream")
    data = base64.b64encode(path.read_bytes()).decode()
    call("upload_media", result_id, FILE_TYPE_IDS[kind], mime, data)


def get_media(result_id, kind):
    """Returns (bytes, mime_type) for a kick's clip/chamber/extension from SpacetimeDB, or None."""
    import base64
    if not RESULT_ID_RE.match(result_id or "") or kind not in FILE_TYPE_IDS:
        return None
    kicks = sql(f"SELECT * FROM kick WHERE result_id = '{result_id}'")
    if not kicks:
        return None
    kick_id = int(kicks[0]["kick_id"])
    assets = [a for a in sql(f"SELECT * FROM media_asset WHERE kick_id = {kick_id}")
              if a["file_type_id"] == FILE_TYPE_IDS[kind]]
    if not assets:
        return None
    rows = sql(f"SELECT * FROM media_data WHERE asset_id = {int(assets[0]['asset_id'])}")
    if not rows:
        return None
    return base64.b64decode(rows[0]["file_data"]), rows[0]["mime_type"]


def delete_kick(result_id):
    call("delete_kick", result_id)


def sync_results(username):
    """Pushes every result in results/ that isn't in the kick table yet. Returns (added, skipped)."""
    import storage
    if get_user(username) is None:
        create_user(username)
    existing = {row.get("result_id") for row in sql("SELECT * FROM kick")}
    added = skipped = 0
    for summary in reversed(storage.list_results(limit=10_000)):   # oldest first
        if summary["id"] in existing:
            skipped += 1
            continue
        result = storage.load(summary["id"])
        start_kick(username, result["id"])
        complete_kick(result["id"], result)
        result["username"], result["db_saved"] = username, True
        storage.update(result)
        added += 1
    return added, skipped


def _camel(s):
    head, *rest = s.split("_")
    return head + "".join(p.title() for p in rest)


# ---- Command line -------------------------------------------------------------------------------

def _print_rows(rows):
    if not rows:
        print("(no rows)")
    for row in rows:
        print(row)


if __name__ == "__main__":
    if not config.SPACETIME_DB:
        sys.exit("Set SPACETIME_DB in .env (the name you published with, e.g. taekwoncoach).")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    tables = {"users": "user", "kicks": "kick", "evaluations": "evaluation", "assets": "media_asset",
              "statuses": "record_status", "filetypes": "file_type"}
    if cmd == "add-user" and len(sys.argv) >= 3:
        create_user(sys.argv[2])
        print("Added. Current users:")
        _print_rows(sql("SELECT * FROM user"))
    elif cmd in tables:
        _print_rows(sql(f"SELECT * FROM {tables[cmd]}"))
    elif cmd == "sync" and len(sys.argv) >= 3:
        added, skipped = sync_results(sys.argv[2])
        print(f"Added {added} kick(s), skipped {skipped} already in the database.")
    else:
        print(__doc__)
