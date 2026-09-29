"""Run:  pip install -r requirements.txt && python download_models.py && uvicorn app:app --host 0.0.0.0 --port 8000
Env (optional): FACE_API_KEY, CORS_ORIGINS="https://your-app.vercel.app,http://localhost:5500", MATCH_THRESHOLD, POSE_TIMEOUT_S"""
import base64
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
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


class FrameBody(BaseModel):
    session_id: str
    image: str


class PruneBody(BaseModel):
    keep: list[str]


@app.get("/api/health")
def health():
    return {"ok": engine.ready, "detector": engine.det_name, "recognizer": engine.rec_name, "enrolled": len(store)}


@app.get("/api/enrolled", dependencies=[Depends(auth)])
def enrolled():
    return {"ids": store.ids()}


@app.post("/api/enroll", dependencies=[Depends(auth)])
def enroll(body: EnrollBody):
    emp_id = body.employee_id.strip()
    if not emp_id or not body.images:
        raise HTTPException(400, "employee_id and images required")
    vecs = []
    for data in body.images[:12]:
        frame, _, low = prepare(decode(data))
        faces = engine.detect(frame, low)
        if not faces:
            continue
        f = faces[0]
        if f[14] < ENROLL_MIN_SCORE or f[2] < 0.15 * frame.shape[1]:
            continue
        if len(faces) > 1 and faces[1][2] * faces[1][3] > 0.5 * f[2] * f[3]:
            continue
        vecs.append(engine.embed(frame, f))
    if not vecs:
        raise HTTPException(422, "No clear single face found - improve lighting / move closer")
    if len(vecs) > 1:
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
    s = sessions.get(body.session_id)
    if s is None:
        raise HTTPException(404, "Unknown or expired session")
    return sessions.step(s, decode(body.image))
