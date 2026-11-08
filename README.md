# Taekwon-Coach backend

Grades a beginner front snap kick from a side-view webcam clip.

## Flow
1. Video in (one kick, recording started and stopped by hand)
2. Find the kick, two ways at once:
   - Nemotron 3.5 Super VL, pass 1: looks at a grid of labeled frames, finds the kick window AND the
     chamber and extension moments, rejects unusable clips
   - Python backup: fast MediaPipe "lite" scan for the moment the ankle is highest
3. Pad the window by 0.5 s
4. MediaPipe "heavy" pose on the window only, then fix any left/right leg mix-ups
5. Python grading engine: evaluates frames within the selected timeframe and runs the 4 checks (1 if passes threshold, 0 if not)
   - Chamber = tightest knee angle, before the extension
   - Extension = after the chamber, largest knee angle, least blurry among near-ties, must pass 90 degrees
   - If Nemotron's moments don't fit the pose data, the same rules run on Python alone
6. Nemotron pass 2: checks the chosen frames, marks doubtful checks "uncertain" (never flips them), writes the feedback
7. Saved: full result + images to `results/<id>/`; kick, evaluation and media rows to SpacetimeDB when a username is given

Without an API key, steps 2 and 6 fall back to Python and the built-in tips, so everything runs offline.

## Files
| File | Job |
|---|---|
| `config.py` | Rule Configuration |
| `video.py` | Load clips, contact sheet, debug images |
| `pose.py` | MediaPipe landmarks |
| `kick_finder.py` | Python's kick finder |
| `scoring.py` | Going through 4 checks and tiers |
| `nemotron.py` | Nemotron pass 1 and pass 2 |
| `feedback.py` | Defining each criterias and return messages for each tiers |
| `pipeline.py` | Runs the whole flow (`grade_video`) |
| `storage.py` | Saves full results, images and clips to `results/` |
| `db.py` | SpacetimeDB: kick lifecycle, users, command-line checks |
| `spacetimedb/src/index.ts` | SpacetimeDB tables and reducers (publish from this folder) |
| `spacetime.json` | tells the `spacetime` CLI where the module is |
| `api.py` | HTTP API for the frontend |
| `main.py` | Command line for testing |
| `capture.py` | Webcam recorder for testing |

## Setup (once per laptop)
Needs Python 3.12 (any 3.12.x). Run everything in the VS Code terminal from this folder.

Windows (PowerShell):
```
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python download_models.py
copy .env.example .env
```
Mac:
```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_models.py
cp .env.example .env
```
Then: `python main.py --check`

## Testing
```
python -m pytest -q                       # logic tests, no camera needed
python main.py --record --no-nemotron     # SPACE start, kick, SPACE stop
python main.py --video kick.mp4           # re-grade a saved clip
python main.py --record                   # full flow with Nemotron (key in .env)
```
Check `results/<id>/chamber.jpg` and `extension.jpg`: they show which frames were graded,
with the kicking leg in orange and the standing leg in blue.

## API (for the frontend)
```
uvicorn api:app --reload --port 8000
```
Open http://localhost:8000/docs to try it.

| Endpoint | What it does |
|---|---|
| `POST /users` | **sign up**. JSON `{"username": "..."}`. 201, or 409 taken, 400 invalid |
| `POST /login` | **log in**. JSON `{"username": "..."}`. Returns the user, or 404 |
| `GET /users?search=ha` | **search** users by name (empty = everyone) |
| `GET /users/{username}` | **profile**: total/completed kicks, best and average score, pass rate per check, latest kick |
| `GET /users/{username}/kicks` | **that user's videos**, newest first, each with its statline |
| `POST /grade` | multipart form: `video` (webm or mp4) + `username`. Returns the result JSON. 404 before grading if the username has no account |
| `DELETE /results/{id}` | deletes a kick's files and its SpacetimeDB rows |
| `GET /results?limit=20` | newest results first, for a history list |
| `GET /results/{id}` | one saved result |
| `GET /results/{id}/chamber.jpg` (also `extension`, `contact_sheet`) | debug images |
| `GET /results/{id}/clip` | the original recording |
| `GET /health` | server status, model files present, Nemotron on/off |

Main fields in the result:
- `status`: `"graded"` or `"retake"` (if retake, show `retake_reason`)
- `checks.<name>.pass`: 1 or 0, plus `uncertain` true/false
- `passed`: 0 to 4; `tier`: `strong` / `even` / `building`
- `feedback` and `feedback_source` (`nemotron` or `fallback`)
- `extra_observations`: up to 2 strings from Nemotron
- `frames_chosen_by`: `nemotron` or `python`
- `images.chamber`, `images.extension`: paths to prefix with the API URL; `clip_url`
- `id`

Grading takes 1-5 s offline and roughly 10-30 s with Nemotron, so show a loading state.

### Accounts
Username only, no password (hackathon demo). Usernames are stored lowercase, 3-20 letters, numbers or `_`.
Signing up `Haris` stores `haris`; logging in as `HARIS` works. Grading never creates accounts, so a typo
gets a 404 instead of a new user.

Frontend flow: sign up / log in -> dashboard (`GET /users/{me}` + `GET /users/{me}/kicks`) -> new kick
(`POST /grade` with `username`) -> back to dashboard -> search (`GET /users?search=`) -> open someone
else's dashboard (same two GETs with their name).

Kick statline (`GET /users/{username}/kicks`):
```json
{"result_id": "...", "recorded_time": "2026-11-08T12:00:00+00:00", "status": "completed",
 "fail_reason": "", "passed": 3, "tier": "strong", "feedback": "...", "feedback_source": "nemotron",
 "checks": {"chamber_tightness": {"pass": 1, "ankle_forward_of_knee_ratio": -0.2}, "...": {}},
 "images": {"chamber": "/results/<id>/chamber.jpg", "extension": "/results/<id>/extension.jpg"},
 "clip_url": "/results/<id>/clip"}
```
`status` is `completed`, `processing`, `uploaded` or `failed` (a retake: show `fail_reason`; `passed` is null).

### SpacetimeDB
Tables (`spacetimedb/src/index.ts`), following the whiteboard design. SQL names are snake_case.

| Table | Columns |
|---|---|
| `user` | user_id, username (unique, lowercase, no password), created_at |
| `kick` | kick_id, user_id, result_id, recorded_time, status_id, fail_reason |
| `evaluation` | kick_id, chamber_height_result, knee_below_hip_ratio, chamber_tightness_result, ankle_forward_knee_ratio, kick_angle_result, leg_angle, kick_straightness_result, knee_angle, kick_tier, feedback, total_score, feedback_source |
| `media_asset` | asset_id, kick_id, file_type_id, file_name (URL path on this API) |
| `media_data` | asset_id, kick_id, mime_type, size_bytes, file_data (the file, base64). One row per clip/image |
| `record_status` | 1 Completed, 2 Processing, 3 Uploaded, 4 Failed |
| `file_type` | 1 Video, 2 Chamber, 3 Extension, 4 Result |

Lifecycle when `/grade` gets a `username`: `start_kick` (Uploaded; the user must exist) ->
`set_kick_status` (Processing) -> `complete_kick` (evaluation + media rows, Completed; or Failed with
`fail_reason` for a retake). React subscribed to `kick` sees each status change live.
After `complete_kick`, Python calls `upload_media` once per file (clip, chamber, extension), so the bytes
live in SpacetimeDB as well as in `results/`. `GET /results/{id}/clip` (and the `.jpg` images) serve the
local copy if this machine has it, otherwise they fetch it from SpacetimeDB, so a kick graded on one
laptop plays on any other. Files over 8 MB are refused (`MAX_UPLOAD_MB`); browser clips are ~1 MB.
Turn uploads off with `UPLOAD_MEDIA=0`.

Frontend: use the API URLs (`clip_url`, `images.*`) rather than reading `media_data` directly. If you do
read it through the SpacetimeDB SDK, subscribe to one row (`SELECT * FROM media_data WHERE asset_id = X`),
never the whole table, or every client downloads every video.

Clips recorded with `python main.py --record` use OpenCV's `mp4v` codec, which browsers can't play;
clips recorded in the browser (webm) play fine.

Setup (once): install the CLI (`curl -sSf https://install.spacetimedb.com | sh` on Mac) and Node.js, then
```
cd spacetimedb && npm install && cd ..
```
Every session:
```
spacetime start                                         # terminal 1: keep running
spacetime publish taekwoncoach --server local --yes     # after any change to spacetimedb/src/index.ts
uvicorn api:app --reload --port 8000                    # terminal 2
```
`.env`: `SPACETIME_DB=taekwoncoach` (empty = SpacetimeDB off; grading still works and saves to `results/`).

Checks without a frontend:
```
python db.py add-user Haris
python main.py --record --user Haris      # grade a kick and save it (creates the account if missing)
python db.py kicks                        # also: users, evaluations, assets, statuses, filetypes
python db.py sync Haris                   # push older results/ kicks in under Haris
SPACETIME_LIVE_DB=taekwoncoach python -m pytest tests/test_spacetime_live.py -v   # full flow, real DB
```
If you change a table's columns and publish fails, wipe the local data and republish:
`spacetime publish taekwoncoach --server local --delete-data=always --yes`

### React example
```jsx
const API = "http://localhost:8000";

async function gradeKick(blob, username) {             // blob from MediaRecorder
  const form = new FormData();
  form.append("video", blob, "kick.webm");
  form.append("username", username);              // saves the kick to SpacetimeDB
  const res = await fetch(`${API}/grade`, { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail);
  return res.json();
}

// <img src={API + result.images.extension} />
```
