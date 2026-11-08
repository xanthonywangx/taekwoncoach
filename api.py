"""HTTP API the frontend calls.

  uvicorn api:app --reload --port 8000

POST /grade                    multipart form, field "video" = the recorded clip (webm or mp4)
GET  /results                  newest results first (history list); ?limit=20
GET  /results/{id}             the saved result JSON
GET  /results/{id}/{image}.jpg chamber, extension, or contact_sheet
GET  /results/{id}/clip        the original recording
GET  /health                   is the server up, are models present, is Nemotron on
"""
import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import config
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
    }


@app.post("/grade")
def grade(video: UploadFile = File(...)):
    # plain def (not async) so FastAPI runs the slow grading in a worker thread
    suffix = os.path.splitext(video.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(video.file, tmp)
        path = tmp.name
    try:
        return grade_video(path)
    except FileNotFoundError as e:   # pose model missing
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:          # unreadable video
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.remove(path)


@app.get("/results")
def list_results(limit: int = 20):
    return storage.list_results(limit)


@app.get("/results/{rid}/clip")
def get_clip(rid: str):
    p = storage.clip_path(rid)
    if p is None:
        raise HTTPException(status_code=404, detail="Clip not found")
    media = "video/webm" if p.suffix == ".webm" else "video/mp4"
    return FileResponse(p, media_type=media)


@app.get("/results/{rid}")
def get_result(rid: str):
    r = storage.load(rid)
    if r is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return r


@app.get("/results/{rid}/{name}.jpg")
def get_image(rid: str, name: str):
    p = storage.image_path(rid, name)
    if p is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(p, media_type="image/jpeg")