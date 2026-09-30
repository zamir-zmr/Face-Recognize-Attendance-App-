"""Run:  pip install -r requirements.txt && python download_models.py && uvicorn app:app --host 0.0.0.0 --port 8000
Env (optional): FACE_API_KEY, CORS_ORIGINS="https://your-app.vercel.app,http://localhost:5500", MATCH_THRESHOLD, POSE_TIMEOUT_S"""
import base64
import os
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import download_models
import sessions
from engine import engine, prepare
from store import store

API_KEY = os.getenv("FACE_API_KEY", "")
ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
MAX_IMG_BYTES = 1_500_000
ENROLL_MIN_SCORE = 0.7
DUPLICATE_SIM = 0.6


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        engine.load()
    except RuntimeError:          # models missing -> fetch once, then retry
        download_models.main()
        engine.load()
    yield


app = FastAPI(title="Face Attendance Backend", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def private_network(request: Request, call_next):   # Chrome Private Network Access preflight
    resp = await call_next(request)
    if request.method == "OPTIONS":
        resp.headers["Access-Control-Allow-Private-Network"] = "true"
    return resp


def auth(x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")


def decode(data: str) -> np.ndarray:
    if "," in data[:64]:
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=False)
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "Bad image encoding")
    if len(raw) > MAX_IMG_BYTES:
        raise HTTPException(413, "Image too large")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Unreadable image")
    return img


class EnrollBody(BaseModel):
    employee_id: str
    images: list[str]
    force: bool = False
    relaxed: bool = False     # legacy photo sync: accept small / low-res photos


class FrameBody(BaseModel):
    session_id: str
    image: str


class PruneBody(BaseModel):
    keep: list[str]


INDEX = Path(__file__).parent / "index.html"


@app.get("/", include_in_schema=False)
def root():
    if INDEX.exists():
        return FileResponse(INDEX, media_type="text/html")
    return {"service": "face-backend", "health": "/api/health", "docs": "/docs"}


@app.get("/api/health")
def health():
    return {"ok": engine.ready, "detector": engine.det_name, "recognizer": engine.rec_name,
            "enrolled": len(store), "sessions": sessions.BACKEND, "store": store.backend,
            "hands": getattr(engine, "hands_ready", False), "hands_api": getattr(engine, "_hands_api", "old-engine")}


@app.get("/api/enrolled", dependencies=[Depends(auth)])
def enrolled():
    return {"ids": store.ids()}


@app.post("/api/enroll", dependencies=[Depends(auth)])
def enroll(body: EnrollBody):
    emp_id = body.employee_id.strip()
    if not emp_id or not body.images:
        raise HTTPException(400, "employee_id and images required")
    vecs, reasons = [], []
    min_score, min_frac = (0.5, 0.06) if body.relaxed else (ENROLL_MIN_SCORE, 0.15)
    for data in body.images[:12]:
        raw = decode(data)
        found = None
        for scale in ((1.0, 2.0) if body.relaxed else (1.0,)):      # legacy 160px photos: retry upscaled
            img = raw if scale == 1.0 else cv2.resize(raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            frame, _, low = prepare(img)
            faces = engine.detect(frame, low)
            if faces:
                found = (frame, faces)
                break
        if found is None:
            reasons.append("no face detected")
            continue
        frame, faces = found
        f = faces[0]
        if f[14] < min_score:
            reasons.append("face confidence %.2f too low" % f[14])
            continue
        if f[2] < min_frac * frame.shape[1]:
            reasons.append("face too small in photo")
            continue
        if len(faces) > 1 and faces[1][2] * faces[1][3] > 0.5 * f[2] * f[3]:
            reasons.append("more than one face")
            continue
        vecs.append(engine.embed(frame, f))
    if not vecs:
        raise HTTPException(422, "No clear single face found (%s)" % "; ".join(sorted(set(reasons))))
    if len(vecs) > 1 and not body.relaxed:      # relaxed (distorted legacy variants) skips the cross-check
        sims = np.stack(vecs) @ np.stack(vecs).T
        if float(sims.min()) < 0.35:
            raise HTTPException(422, "Images look like different people")
    mean = np.mean(vecs, axis=0)
    mean = (mean / (np.linalg.norm(mean) + 1e-9)).astype(np.float32)
    other, sim, _ = store.match(mean, exclude=emp_id)
    if other and sim >= DUPLICATE_SIM and not body.force:
        raise HTTPException(409, "Face already enrolled as %s" % other)
    store.put(emp_id, mean)
    return {"ok": True, "employee_id": emp_id, "samples": len(vecs)}


@app.delete("/api/enroll/{employee_id}", dependencies=[Depends(auth)])
def unenroll(employee_id: str):
    return {"ok": store.delete(employee_id)}


@app.post("/api/prune", dependencies=[Depends(auth)])
def prune(body: PruneBody):
    if not body.keep:
        return {"removed": 0}
    return {"removed": store.keep_only(body.keep)}


@app.post("/api/verify/start", dependencies=[Depends(auth)])
def verify_start():
    return {"session_id": sessions.start().sid}


@app.post("/api/verify/frame", dependencies=[Depends(auth)])
def verify_frame(body: FrameBody):
    res = sessions.advance(body.session_id, decode(body.image))
    if res is None:
        raise HTTPException(404, "Unknown or expired session")
    return res
