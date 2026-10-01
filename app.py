"""Run:  pip install -r requirements.txt && python download_models.py && uvicorn app:app --host 0.0.0.0 --port 8000
Env (optional): FACE_API_KEY, CORS_ORIGINS="https://your-app.vercel.app,http://localhost:5500", MATCH_THRESHOLD, POSE_TIMEOUT_S"""
import base64
import os
import platform
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


def valid_id(raw: str) -> str:
    emp = (raw or "").strip()
    if not emp or len(emp) > 128:
        raise HTTPException(400, "valid employee id required")
    return emp


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
            "hands": getattr(engine, "hands_ready", False), "hands_api": getattr(engine, "_hands_api", "old-engine"),
            "hands_error": getattr(engine, "hands_error", ""),
            "python": platform.python_version(), "firebase": store.backend == "firebase"}


@app.get("/api/enrolled", dependencies=[Depends(auth)])
def enrolled():
    return {"ids": store.ids()}


def _enroll(emp_id: str, images: list, force: bool, relaxed: bool) -> dict:
    emp_id = emp_id.strip()
    if not emp_id or not images:
        raise HTTPException(400, "employee_id and images required")
    vecs, reasons = [], []
    min_score, min_frac = (0.5, 0.06) if relaxed else (ENROLL_MIN_SCORE, 0.15)
    for data in images[:12]:
        raw = decode(data)
        found = None
        for scale in ((1.0, 2.0) if relaxed else (1.0,)):      # legacy 160px photos: retry upscaled
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
    if len(vecs) > 1 and not relaxed:      # relaxed (distorted legacy variants) skips the cross-check
        sims = np.stack(vecs) @ np.stack(vecs).T
        if float(sims.min()) < 0.35:
            raise HTTPException(422, "Images look like different people")
    mean = np.mean(vecs, axis=0)
    mean = (mean / (np.linalg.norm(mean) + 1e-9)).astype(np.float32)
    other, sim, _ = store.match(mean, exclude=emp_id)
    if other and sim >= DUPLICATE_SIM and not force:
        raise HTTPException(409, "Face already enrolled as %s" % other)
    store.put(emp_id, mean)
    return {"ok": True, "employee_id": emp_id, "samples": len(vecs)}


@app.post("/api/enroll", dependencies=[Depends(auth)])
def enroll(body: EnrollBody):
    return _enroll(body.employee_id, body.images, body.force, body.relaxed)


@app.delete("/api/enroll/{employee_id}", dependencies=[Depends(auth)])
def unenroll(employee_id: str):                     # face data of exactly this id (employee record is NOT touched)
    return {"ok": store.delete(valid_id(employee_id))}


@app.post("/api/prune", dependencies=[Depends(auth)])
def prune(body: PruneBody):
    if not body.keep:
        return {"removed": 0}
    try:
        return {"removed": store.keep_only(body.keep)}
    except ValueError as e:                         # safety guard: pruning most profiles at once is refused
        raise HTTPException(409, str(e))


@app.post("/api/verify/start", dependencies=[Depends(auth)])
def verify_start():
    return {"session_id": sessions.start().sid}


@app.post("/api/verify/frame", dependencies=[Depends(auth)])
def verify_frame(body: FrameBody):
    res = sessions.advance(body.session_id, decode(body.image))
    if res is None:
        raise HTTPException(404, "Unknown or expired session")
    return res


# ---------- Firebase business data (employees, attendance logs, dashboard) ----------
@app.get("/api/dashboard", dependencies=[Depends(auth)])
def dashboard():
    return store.dashboard()


@app.get("/api/employees", dependencies=[Depends(auth)])
def employees_list():
    return {"employees": store.get_employees()}


@app.put("/api/employees/{employee_id}", dependencies=[Depends(auth)])
def employee_put(employee_id: str, body: dict):
    employee_id = valid_id(employee_id)
    if str(body.get("id", employee_id)).strip() != employee_id:
        raise HTTPException(400, "id in body does not match url")
    return {"ok": store.upsert_employee(employee_id, body)}


@app.delete("/api/employees/{employee_id}", dependencies=[Depends(auth)])
def employee_delete(employee_id: str):
    return {"ok": store.delete(valid_id(employee_id), full=True)}        # employee record + face data + image + local cache, everywhere


@app.get("/api/logs", dependencies=[Depends(auth)])
def logs_range(start: str, end: str = ""):
    return {"logs": store.get_logs(start, end or start)}


@app.post("/api/attendance", dependencies=[Depends(auth)])
def attendance_add(body: dict):
    emp = str(body.get("empId") or body.get("employee_id") or "").strip()
    if not emp:
        raise HTTPException(400, "empId required")
    return {"ok": store.log_attendance(emp, body)}


class ImportBody(BaseModel):
    employees: list[dict]
    enroll: bool = True


def _norm_employee(raw):
    if not isinstance(raw, dict):
        return None
    emp = str(raw.get("id") or raw.get("emp_id") or raw.get("employee_id") or "").strip()
    if not emp or len(emp) > 128:
        return None
    rec = {k: v for k, v in raw.items() if v is not None and v != "" and k not in ("photoUrl", "emp_id", "employee_id")}
    rec["id"] = emp
    rec["name"] = str(rec.get("name") or emp)
    d = rec.get("descriptor")
    if d is not None and not (isinstance(d, list) and all(isinstance(x, (int, float)) for x in d)):
        rec.pop("descriptor")
    return rec


@app.post("/api/employees/import", dependencies=[Depends(auth)])
def employees_import(body: ImportBody):
    """Upserts employee records (profile, photo, descriptor, metadata) into Firebase and enrolls each photo as a face embedding."""
    recs, seen, skipped = [], set(), 0
    for raw in body.employees[:500]:
        r = _norm_employee(raw)
        if r is None or r["id"] in seen:
            skipped += 1
            continue
        seen.add(r["id"])
        recs.append(r)
    saved = enrolled = 0
    failed = []
    for r in recs:
        if store.upsert_employee(r["id"], r):
            saved += 1
        photo = r.get("photo")
        if body.enroll and isinstance(photo, str) and photo.startswith("data:"):
            try:
                _enroll(r["id"], [photo], True, True)
                enrolled += 1
            except HTTPException as e:
                failed.append({"id": r["id"], "reason": e.detail})
            except Exception as e:  # noqa: BLE001
                failed.append({"id": r["id"], "reason": str(e)})
    return {"ok": True, "received": len(body.employees), "saved": saved, "enrolled": enrolled, "skipped": skipped, "failed": failed}
