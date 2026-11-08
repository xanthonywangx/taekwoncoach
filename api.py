"""HTTP API the frontend calls.

  uvicorn api:app --reload --port 8000
  then open http://localhost:8000/docs to try it in the browser

Accounts (username only, stored lowercase, 3-20 letters/numbers/_):
POST /users                    sign up. JSON {"username": "..."} -> 201, 409 if taken, 400 if invalid
POST /login                    log in.  JSON {"username": "..."} -> the user, 404 if no such account
GET  /users?search=ha          search users by name (empty = everyone)
GET  /users/{username}         profile + overall statline (averages, best, pass rate per check)
GET  /users/{username}/kicks   that user's kicks, newest first, each with its statline

Grading:
POST /grade                    multipart form: "video" = the clip (webm or mp4), "username" = who kicked.
                               404 before grading if the username has no account. No username = not saved
                               to SpacetimeDB (results/ folder only).
DELETE /results/{id}           deletes the kick's files and its SpacetimeDB rows
GET  /results                  newest results first (history list); ?limit=20
GET  /results/{id}             the saved result JSON
GET  /results/{id}/{image}.jpg chamber, extension, or contact_sheet
GET  /results/{id}/clip        the original recording
GET  /health                   is the server up, are models present, is Nemotron on
"""
import os
import shutil
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

import config
import db
import storage
from pipeline import grade_video

app = FastAPI(title="Taekwoncoach")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {
        "ok": True,
        "pose_models": config.POSE_MODEL_HEAVY.exists() and config.POSE_MODEL_LITE.exists(),
        "nemotron": config.nemotron_available(),
        "nemotron_model": config.NEMOTRON_MODEL,
        "spacetime": config.spacetime_enabled(),
        "spacetime_db": config.SPACETIME_DB,
    }


@app.post("/grade")
def grade(video: UploadFile = File(...), username: str = Form("")):
    # plain def (not async) so FastAPI runs the slow grading in a worker thread
    if username.strip() and config.spacetime_enabled():
        _require_user(username)   # fail fast, before 10-30 s of grading
    suffix = os.path.splitext(video.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(video.file, tmp)
        path = tmp.name
    try:
        return grade_video(path, username=username or None)
    except FileNotFoundError as e:   # pose model missing
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:          # unreadable video
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.remove(path)


class UsernameBody(BaseModel):
    username: str


def _need_db():
    if not config.spacetime_enabled():
        raise HTTPException(status_code=503, detail="SpacetimeDB is off (set SPACETIME_DB in .env)")


def _db_call(fn, *args):
    """Runs a db function and turns its errors into HTTP errors the frontend can show."""
    _need_db()
    try:
        return fn(*args)
    except db.UserError as e:
        status = 409 if "taken" in str(e) else 404 if "No account" in str(e) else 400
        raise HTTPException(status_code=status, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"SpacetimeDB error: {e}")


def _require_user(username):
    user = _db_call(db.get_user, username)
    if user is None:
        raise HTTPException(status_code=404, detail=f'No account named "{username.strip().lower()}". Sign up first.')
    return user


@app.post("/users", status_code=201)
def sign_up(body: UsernameBody):
    name = _db_call(db.create_user, body.username)
    return {"username": name}


@app.post("/login")
def log_in(body: UsernameBody):
    user = _require_user(body.username)
    return {"username": user["username"], "created_at": user["created_at"]}


@app.get("/users")
def search_users(search: str = "", limit: int = 20):
    return _db_call(db.search_users, search, limit)


@app.get("/users/{username}")
def user_profile(username: str):
    profile = _db_call(db.user_profile, username)
    if profile is None:
        raise HTTPException(status_code=404, detail=f'No account named "{username.lower()}"')
    return profile


@app.get("/users/{username}/kicks")
def user_kicks(username: str, limit: int = 50):
    kicks = _db_call(db.kicks_for, username, limit)
    if kicks is None:
        raise HTTPException(status_code=404, detail=f'No account named "{username.lower()}"')
    return kicks


@app.delete("/results/{rid}")
def delete_result(rid: str):
    deleted_files = storage.delete(rid)
    db_deleted = False
    if config.spacetime_enabled():
        try:
            db.delete_kick(rid)
            db_deleted = True
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Files deleted={deleted_files}, database failed: {e}")
    if not deleted_files and not db_deleted:
        raise HTTPException(status_code=404, detail="Result not found")
    return {"ok": True, "files_deleted": deleted_files, "db_deleted": db_deleted}


@app.get("/results")
def list_results(limit: int = 20):
    return storage.list_results(limit)


@app.get("/results/{rid}/clip")
def get_clip(rid: str):
    p = storage.clip_path(rid)
    if p is not None:
        return FileResponse(p, media_type=db.MIME_BY_SUFFIX.get(p.suffix.lower(), "video/mp4"))
    return _from_db(rid, "clip", "Clip not found")


def _from_db(rid, kind, missing):
    """Files not on this machine's disk (graded on a teammate's laptop) come from SpacetimeDB."""
    if config.spacetime_enabled():
        try:
            found = db.get_media(rid, kind)
        except Exception:
            found = None
        if found:
            data, mime = found
            return Response(content=data, media_type=mime)
    raise HTTPException(status_code=404, detail=missing)


@app.get("/results/{rid}")
def get_result(rid: str):
    r = storage.load(rid)
    if r is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return r


@app.get("/results/{rid}/{name}.jpg")
def get_image(rid: str, name: str):
    p = storage.image_path(rid, name)
    if p is not None:
        return FileResponse(p, media_type="image/jpeg")
    if name in ("chamber", "extension"):
        return _from_db(rid, name, "Image not found")
    raise HTTPException(status_code=404, detail="Image not found")
