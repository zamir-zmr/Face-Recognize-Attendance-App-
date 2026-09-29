"""Verification flow: search -> identify -> baseline -> random pose challenge -> passed/failed. Challenge is chosen server-side."""
import os
import pickle
import secrets
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from engine import engine, pose_metrics, prepare
from store import store

MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", 0.40))    # SFace cosine (0.363 = zoo default)
POSE_MATCH_THRESHOLD = float(os.getenv("POSE_MATCH_THRESHOLD", 0.32))  # relaxed: turned head lowers similarity
MATCH_MARGIN = 0.03
IDENT_FRAMES = 2
BASE_FRAMES = 4
MIN_FACE_FRAC = 0.16            # face width / frame width
YAW_DELTA = float(os.getenv("POSE_YAW_DELTA", 0.17))
PITCH_UP = float(os.getenv("POSE_PITCH_UP", 0.075))      # separate thresholds: up/down need less movement than left/right
PITCH_DOWN = float(os.getenv("POSE_PITCH_DOWN", 0.085))
HOLD_FRAMES = 2
HOLD_RELEASE = 0.75             # hysteresis: once in the target zone, only 75% of the threshold is needed to stay
WRONG_STRENGTH = 1.35           # a wrong direction must clearly exceed its threshold...
WRONG_FRAMES = 3                # ...for this many frames before failing
CROSS_TALK = 1.6                # off-axis motion may be at most 1.6x the target-axis motion
SMOOTH = 0.7                    # EMA weight of the newest sample (after 3-sample median)
BASE_SPREAD = 0.08              # baseline frames must be this steady (yaw & pitch range)
POSE_TIMEOUT = float(os.getenv("POSE_TIMEOUT_S", 8))
SEARCH_TIMEOUT = 3.0
NOFACE_FAIL_S = 2.5
HARD_TIMEOUT = 30.0
MOTION_MIN = 0.45               # mean abs pixel diff (0-255) of fixed face ROI across baseline
UNKNOWN_LIMIT = 5
POSES = ("UP", "DOWN", "LEFT", "RIGHT")

_lock = threading.Lock()
SESSION_TTL = 90


@dataclass
class Session:
    sid: str = field(default_factory=lambda: secrets.token_urlsafe(16))
    started: float = field(default_factory=time.time)
    stage: str = "search"
    emp: str | None = None
    ident_hits: int = 0
    unknown: int = 0
    base: list = field(default_factory=list)
    roi: tuple | None = None
    prev_crop: np.ndarray | None = None
    diffs: list = field(default_factory=list)
    baseline: tuple | None = None
    target: str | None = None
    deadline: float = 0.0
    hold: int = 0
    wrong: int = 0
    hist: list = field(default_factory=list)
    smooth: tuple | None = None
    last_face_t: float = field(default_factory=time.time)
    done: bool = False
    result: dict | None = None
    score: float = 0.0


# ---------- session storage: Redis (multi-worker safe) when REDIS_URL is set, else in-memory ----------
_redis = None
if os.getenv("REDIS_URL"):
    try:
        import redis
        _redis = redis.Redis.from_url(os.environ["REDIS_URL"], socket_timeout=2, socket_connect_timeout=3)
        _redis.ping()
    except Exception as e:  # noqa: BLE001
        print("[sessions] Redis unavailable, using in-memory:", e)
        _redis = None
BACKEND = "redis" if _redis else "memory"

_mem: dict[str, Session] = {}
_mem_pose: dict[str, str] = {}


def _save(s: Session):
    if _redis:
        _redis.setex("face:sess:" + s.sid, SESSION_TTL, pickle.dumps(s, protocol=4))
    else:
        _mem[s.sid] = s


def get(sid: str) -> Session | None:
    if _redis:
        raw = _redis.get("face:sess:" + sid)
        return pickle.loads(raw) if raw else None
    return _mem.get(sid)


def start() -> Session:
    if not _redis:
        now = time.time()
        with _lock:
            for k in [k for k, v in _mem.items() if now - v.started > SESSION_TTL]:
                del _mem[k]
            if len(_mem) > 300:
                _mem.pop(next(iter(_mem)))
    s = Session()
    _save(s)
    return s


def _last_pose(emp: str) -> str | None:
    if _redis:
        v = _redis.hget("face:lastpose", emp)
        return v.decode() if v else None
    return _mem_pose.get(emp)


def _set_last_pose(emp: str, pose: str):
    if _redis:
        _redis.hset("face:lastpose", emp, pose)
    else:
        _mem_pose[emp] = pose


def advance(sid: str, bgr: np.ndarray) -> dict | None:
    """Load session -> run one step -> persist. None if session unknown/expired."""
    s = get(sid)
    if s is None:
        return None
    res = step(s, bgr)
    _save(s)
    return res


def _axes(m, base):
    """Signed movement in threshold units: |v| >= 1 means the pose threshold is reached."""
    dy, dp = m[0] - base[0], m[1] - base[1]
    return dy / YAW_DELTA, dp / (PITCH_DOWN if dp > 0 else PITCH_UP)


def _toward(pose, yv, pv):
    # raw (un-mirrored) frame: nose to image-right = user's LEFT; pitch grows when looking DOWN
    return {"LEFT": yv, "RIGHT": -yv, "DOWN": pv, "UP": -pv}[pose]


def _crop(gray, roi):
    x, y, w, h = roi
    return cv2.resize(gray[y:y + h, x:x + w], (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32)


def _resp(s: Session, state: str, message: str, face=None, frame_shape=None, **extra):
    out = {"state": state, "stage": s.stage, "message": message, "pose": s.target if s.stage == "challenge" else None}
    if face is not None and frame_shape is not None:
        H, W = frame_shape[:2]
        out["box"] = [float(face[0]) / W, float(face[1]) / H, float(face[2]) / W, float(face[3]) / H]
    out.update(extra)
    return out


def _finish(s: Session, state: str, message: str, face=None, shape=None, **extra):
    s.done = True
    s.result = _resp(s, state, message, face, shape, **extra)
    if state == "passed":
        s.result["employee_id"] = s.emp
        s.result["score"] = round(s.score, 3)
    return s.result


def step(s: Session, bgr: np.ndarray) -> dict:
    if s.done and s.result:
        return s.result
    now = time.time()
    if now - s.started > HARD_TIMEOUT:
        return _finish(s, "failed", "Session timed out", reason="timeout")

    frame, luma, low = prepare(bgr)
    shape = frame.shape
    faces = engine.detect(frame, low)
    info = {"luma": round(luma, 1), "low_light": low}

    if not faces:
        if s.stage in ("search", "identify"):
            s.ident_hits = 0
            if now - s.started > SEARCH_TIMEOUT:
                return _finish(s, "idle", "No face", **info)
            return _resp(s, "running", "Scanning face...", **info)
        s.hold = 0
        if now - s.last_face_t > NOFACE_FAIL_S:
            msg = "Lighting too low - improve lighting" if low and luma < 40 else "Face not clearly visible"
            return _finish(s, "failed", msg, reason="noface", **info)
        return _resp(s, "running", "Face lost - look at the camera", **info)

    s.last_face_t = now
    face = faces[0]
    H, W = shape[:2]

    if len(faces) > 1 and faces[1][2] * faces[1][3] > 0.5 * face[2] * face[3]:
        return _resp(s, "running", "Multiple faces - one person only", face, shape, **info)
    if face[2] < MIN_FACE_FRAC * W:
        return _resp(s, "running", "Move closer to the camera", face, shape, **info)

    # ---- identify ----
    if s.stage in ("search", "identify"):
        s.stage = "identify"
        vec = engine.embed(frame, face)
        emp, sim, margin = store.match(vec)
        if emp is None:
            return _finish(s, "unknown", "No face records enrolled", face, shape, **info)
        if sim >= MATCH_THRESHOLD and margin >= MATCH_MARGIN:
            s.ident_hits = s.ident_hits + 1 if s.emp in (None, emp) else 1
            s.emp, s.score = emp, sim
            if s.ident_hits >= IDENT_FRAMES:
                s.stage = "baseline"
                return _resp(s, "running", "Look straight at the camera...", face, shape, **info)
            return _resp(s, "running", "Recognizing...", face, shape, **info)
        s.ident_hits, s.unknown = 0, s.unknown + 1
        if s.unknown >= UNKNOWN_LIMIT:
            return _finish(s, "unknown", "Face not recognized", face, shape, **info)
        return _resp(s, "running", "Recognizing...", face, shape, **info)

    yaw, pitch = pose_metrics(face)

    # ---- baseline (neutral pose + micro-motion liveness) ----
    if s.stage == "baseline":
        if abs(yaw) > 0.22:
            return _resp(s, "running", "Look straight at the camera...", face, shape, **info)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if s.roi is None:
            x, y, w, h = [int(v) for v in face[:4]]
            x, y = max(0, x), max(0, y)
            s.roi = (x, y, max(8, min(w, W - x)), max(8, min(h, H - y)))
        crop = _crop(gray, s.roi)
        if s.prev_crop is not None:
            s.diffs.append(float(np.abs(crop - s.prev_crop).mean()))
        s.prev_crop = crop
        s.base.append((yaw, pitch))
        s.base = s.base[-BASE_FRAMES:]
        if len(s.base) < BASE_FRAMES:
            return _resp(s, "running", "Look straight at the camera...", face, shape, **info)
        ys, ps = [b[0] for b in s.base], [b[1] for b in s.base]
        if max(ys) - min(ys) > BASE_SPREAD or max(ps) - min(ps) > BASE_SPREAD:   # still moving -> keep collecting
            return _resp(s, "running", "Hold still, look straight...", face, shape, **info)
        if float(np.mean(s.diffs[-BASE_FRAMES:])) < MOTION_MIN:
            return _finish(s, "failed", "Spoof suspected (static image/frozen video)", face, shape, reason="spoof", **info)
        s.baseline = (float(np.median([b[0] for b in s.base])), float(np.median([b[1] for b in s.base])))
        pool = [p for p in POSES if p != _last_pose(s.emp)]
        s.target = secrets.choice(pool)
        _set_last_pose(s.emp, s.target)
        s.stage, s.deadline, s.hold, s.wrong, s.smooth, s.hist = "challenge", now + POSE_TIMEOUT, 0, 0, None, []
        return _resp(s, "running", "Pose Challenge: " + s.target, face, shape, **info)

    # ---- challenge ----
    if now > s.deadline:
        return _finish(s, "failed", "Timed out - %s pose not detected" % s.target, face, shape, reason="timeout", **info)
    s.hist = (s.hist + [(yaw, pitch)])[-3:]                       # median of last 3 kills landmark spikes
    m = (float(np.median([h[0] for h in s.hist])), float(np.median([h[1] for h in s.hist])))
    s.smooth = m if s.smooth is None else (SMOOTH * m[0] + (1 - SMOOTH) * s.smooth[0],
                                           SMOOTH * m[1] + (1 - SMOOTH) * s.smooth[1])
    yv, pv = _axes(s.smooth, s.baseline)
    tv = _toward(s.target, yv, pv)
    off = abs(pv) if s.target in ("LEFT", "RIGHT") else abs(yv)
    need = HOLD_RELEASE if s.hold else 1.0
    if tv >= need and off <= max(1.0, CROSS_TALK * tv):
        s.hold += 1
        s.wrong = 0
        if s.hold >= HOLD_FRAMES:
            vec = engine.embed(frame, face)   # same person must finish the challenge
            sim = store.similarity(s.emp, vec)
            if sim < POSE_MATCH_THRESHOLD:
                return _finish(s, "failed", "Identity changed during challenge", face, shape, reason="identity", **info)
            s.score = max(s.score, sim)
            s.stage = "passed"
            return _finish(s, "passed", "Verified", face, shape, **info)
    else:
        s.hold = 0
        wrong_dir = any(_toward(p, yv, pv) >= WRONG_STRENGTH and _toward(p, yv, pv) > 1.2 * max(tv, 0)
                        for p in POSES if p != s.target)
        s.wrong = s.wrong + 1 if wrong_dir else max(0, s.wrong - 1)
        if s.wrong >= WRONG_FRAMES:
            return _finish(s, "failed", "Wrong pose! Required: %s" % s.target, face, shape, reason="wrong", **info)
    return _resp(s, "running", "Pose Challenge: " + s.target, face, shape, **info)
