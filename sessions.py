"""Verification flow: search -> identify -> baseline -> random pose challenge -> passed/failed. Challenge is chosen server-side."""
import os
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
BASE_FRAMES = 5
MIN_FACE_FRAC = 0.16            # face width / frame width
YAW_DELTA = float(os.getenv("POSE_YAW_DELTA", 0.20))
PITCH_DELTA = float(os.getenv("POSE_PITCH_DELTA", 0.13))
HOLD_FRAMES = 3
SMOOTH = 0.6
POSE_TIMEOUT = float(os.getenv("POSE_TIMEOUT_S", 8))
SEARCH_TIMEOUT = 3.0
NOFACE_FAIL_S = 2.5
HARD_TIMEOUT = 30.0
MOTION_MIN = 0.45               # mean abs pixel diff (0-255) of fixed face ROI across baseline
UNKNOWN_LIMIT = 5
POSES = ("UP", "DOWN", "LEFT", "RIGHT")

_last_pose: dict[str, str] = {}
_lock = threading.Lock()


@dataclass
class Session:
    sid: str = field(default_factory=lambda: secrets.token_urlsafe(16))
    started: float = field(default_factory=time.monotonic)
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
    smooth: tuple | None = None
    last_face_t: float = field(default_factory=time.monotonic)
    done: bool = False
    result: dict | None = None
    score: float = 0.0


_sessions: dict[str, Session] = {}


def start() -> Session:
    now = time.monotonic()
    with _lock:
        for k in [k for k, s in _sessions.items() if now - s.started > 90]:
            del _sessions[k]
        if len(_sessions) > 300:
            _sessions.pop(next(iter(_sessions)))
        s = Session()
        _sessions[s.sid] = s
    return s


def get(sid: str) -> Session | None:
    return _sessions.get(sid)


def _classify(m, base):
    dy, dp = m[0] - base[0], m[1] - base[1]
    yaw_hit, pit_hit = abs(dy) >= YAW_DELTA, abs(dp) >= PITCH_DELTA
    if yaw_hit and (not pit_hit or abs(dy) / YAW_DELTA >= abs(dp) / PITCH_DELTA):
        return "LEFT" if dy > 0 else "RIGHT"   # raw (un-mirrored) frame: nose to image-right = user's LEFT
    if pit_hit:
        return "DOWN" if dp > 0 else "UP"
    return None


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
    now = time.monotonic()
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
        if len(s.base) < BASE_FRAMES:
            return _resp(s, "running", "Look straight at the camera...", face, shape, **info)
        if float(np.mean(s.diffs)) < MOTION_MIN:
            return _finish(s, "failed", "Spoof suspected (static image/frozen video)", face, shape, reason="spoof", **info)
        s.baseline = (float(np.median([b[0] for b in s.base])), float(np.median([b[1] for b in s.base])))
        with _lock:
            pool = [p for p in POSES if p != _last_pose.get(s.emp)]
            s.target = secrets.choice(pool)
            _last_pose[s.emp] = s.target
        s.stage, s.deadline, s.hold, s.smooth = "challenge", now + POSE_TIMEOUT, 0, None
        return _resp(s, "running", "Pose Challenge: " + s.target, face, shape, **info)

    # ---- challenge ----
    if now > s.deadline:
        return _finish(s, "failed", "Timed out - %s pose not detected" % s.target, face, shape, reason="timeout", **info)
    m = (yaw, pitch)
    s.smooth = m if s.smooth is None else (SMOOTH * m[0] + (1 - SMOOTH) * s.smooth[0],
                                           SMOOTH * m[1] + (1 - SMOOTH) * s.smooth[1])
    seen = _classify(s.smooth, s.baseline)
    if seen == s.target:
        s.hold += 1
        if s.hold >= HOLD_FRAMES:
            vec = engine.embed(frame, face)   # same person must finish the challenge
            sim = store.similarity(s.emp, vec)
            if sim < POSE_MATCH_THRESHOLD:
                return _finish(s, "failed", "Identity changed during challenge", face, shape, reason="identity", **info)
            s.score = max(s.score, sim)
            s.stage = "passed"
            return _finish(s, "passed", "Verified", face, shape, **info)
    elif seen:
        return _finish(s, "failed", "Wrong pose! Required: %s" % s.target, face, shape, reason="wrong", **info)
    else:
        s.hold = 0
    return _resp(s, "running", "Pose Challenge: " + s.target, face, shape, **info)
