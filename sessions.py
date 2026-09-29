"""Verification flow: search -> identify -> baseline -> random challenge sequence (POSE left/right + FINGER count 1-5, random order) -> passed/failed.
Everything (order, pose, finger count) is chosen server-side; fingers are counted server-side too."""
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
BASE_FRAMES = 3                 # neutral-yaw samples (collected while identifying, so the challenge starts instantly)
MIN_FACE_FRAC = 0.16            # face width / frame width
YAW_DELTA = float(os.getenv("POSE_YAW_DELTA", 0.17))   # head-turn needed, in eye-distance units
HOLD_FRAMES = 2
HOLD_RELEASE = 0.75             # hysteresis: once in the target zone, only 75% of the threshold is needed to stay
WRONG_STRENGTH = 1.35           # opposite direction must clearly exceed the threshold...
WRONG_FRAMES = 3                # ...for this many frames before failing
SMOOTH = 0.7                    # EMA weight of the newest sample (after 3-sample median)
BASE_SPREAD = 0.08              # baseline yaw samples must be this steady
POSE_TIMEOUT = float(os.getenv("POSE_TIMEOUT_S", 8))
SEARCH_TIMEOUT = 3.0
NOFACE_FAIL_S = 2.5
HARD_TIMEOUT = 30.0
MOTION_MIN = 0.45               # mean abs pixel diff (0-255) of fixed face ROI across baseline
UNKNOWN_LIMIT = 5
POSES = ("LEFT", "RIGHT")
FINGER_CHOICES = (1, 2, 3, 4, 5)
FINGER_TIMEOUT = float(os.getenv("FINGER_TIMEOUT_S", 10))
FINGER_HOLD = int(os.getenv("FINGER_HOLD_FRAMES", 3))   # consecutive frames with the exact count

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
    baseline: float | None = None
    target: str | None = None
    deadline: float = 0.0
    hold: int = 0
    wrong: int = 0
    hist: list = field(default_factory=list)
    smooth: float | None = None
    last_face_t: float = field(default_factory=time.time)
    done: bool = False
    result: dict | None = None
    score: float = 0.0
    steps: list = field(default_factory=list)        # e.g. [("fingers", 3), ("pose", "LEFT")] - random order
    step_i: int = 0
    kind: str | None = None                          # current challenge: "pose" | "fingers"
    fingers: int | None = None
    fhold: int = 0
    ticks: int = 0
    completed: list = field(default_factory=list)    # [{"kind": "fingers", "value": 3}, ...]


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


def advance(sid: str, bgr: np.ndarray) -> dict | None:
    """Load session -> run one step -> persist. None if session unknown/expired."""
    s = get(sid)
    if s is None:
        return None
    res = step(s, bgr)
    _save(s)
    return res


def _crop(gray, roi):
    x, y, w, h = roi
    return cv2.resize(gray[y:y + h, x:x + w], (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32)


def _collect_baseline(s: Session, frame, face, yaw: float):
    """Neutral-yaw sample + fixed-ROI micro-motion (liveness). Called from identify AND baseline stages."""
    if abs(yaw) > 0.22:
        return
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if s.roi is None:
        x, y, w, h = [int(v) for v in face[:4]]
        x, y = max(0, x), max(0, y)
        s.roi = (x, y, max(8, min(w, W - x)), max(8, min(h, H - y)))
    crop = _crop(gray, s.roi)
    if s.prev_crop is not None:
        s.diffs.append(float(np.abs(crop - s.prev_crop).mean()))
    s.prev_crop = crop
    s.base = (s.base + [yaw])[-BASE_FRAMES:]


def _resp(s: Session, state: str, message: str, face=None, frame_shape=None, **extra):
    chal = s.stage == "challenge"
    out = {"state": state, "stage": s.stage, "message": message,
           "pose": s.target if chal and s.kind == "pose" else None,
           "kind": s.kind if chal else None,
           "fingers": s.fingers if chal and s.kind == "fingers" else None,
           "ticks": s.ticks, "completed": s.completed,
           "step": min(s.step_i + 1, len(s.steps)), "steps": len(s.steps)}
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


def _make_steps() -> list:
    steps = [("pose", secrets.choice(POSES))]
    if engine.hands_ready:
        steps.append(("fingers", secrets.choice(FINGER_CHOICES)))
        if secrets.randbelow(2):
            steps.reverse()
    return steps


def _begin_step(s: Session, now: float, face, shape, info):
    kind, val = s.steps[s.step_i]
    s.kind, s.stage = kind, "challenge"
    s.hold = s.wrong = s.fhold = s.ticks = 0
    s.smooth, s.hist = None, []
    s.target = val if kind == "pose" else None
    s.fingers = val if kind == "fingers" else None
    s.deadline = now + (POSE_TIMEOUT if kind == "pose" else FINGER_TIMEOUT)
    msg = "Pose Challenge: %s" % val if kind == "pose" else "Finger Challenge: show %d finger%s" % (val, "" if val == 1 else "s")
    return _resp(s, "running", msg, face, shape, **info)


def _complete_step(s: Session, now: float, frame, face, shape, info):
    sim = store.similarity(s.emp, engine.embed(frame, face))   # same person must finish every challenge
    if sim < POSE_MATCH_THRESHOLD:
        return _finish(s, "failed", "Identity changed during challenge", face, shape, reason="identity", **info)
    s.score = max(s.score, sim)
    kind, val = s.steps[s.step_i]
    if kind == "fingers":
        s.ticks = val
    s.completed = s.completed + [{"kind": kind, "value": val}]
    s.step_i += 1
    if s.step_i >= len(s.steps):
        s.stage = "passed"
        return _finish(s, "passed", "Verified", face, shape, **info)
    if s.steps[s.step_i][0] == "pose":       # head pose needs a fresh neutral baseline after the hand step
        s.stage, s.kind, s.target, s.fingers, s.base = "baseline", None, None, None, []
        return _resp(s, "running", "Great! Look straight at the camera...", face, shape, **info)
    return _begin_step(s, now, face, shape, info)


def _fingers_step(s: Session, now: float, frame, face, shape, info):
    count, hbox = engine.count_fingers(frame)
    extra = dict(info, hand=hbox)
    if count is None:
        s.fhold = s.ticks = 0
        return _resp(s, "running", "Show your hand to the camera", face, shape, **extra)
    if count == s.fingers:
        s.fhold += 1
        s.ticks = count
        if s.fhold >= FINGER_HOLD:
            return _complete_step(s, now, frame, face, shape, extra)
        return _resp(s, "running", "Hold steady...", face, shape, **extra)
    s.fhold = 0
    s.ticks = count if count < s.fingers else 0
    msg = "Too many fingers - show exactly %d" % s.fingers if count > s.fingers else "Show %d finger%s (seeing %d)" % (s.fingers, "" if s.fingers == 1 else "s", count)
    return _resp(s, "running", msg, face, shape, **extra)


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

    yaw, _ = pose_metrics(face)
    collected = False

    # ---- identify (baseline samples gathered in the same frames) ----
    if s.stage in ("search", "identify"):
        s.stage = "identify"
        _collect_baseline(s, frame, face, yaw)
        collected = True
        vec = engine.embed(frame, face)
        emp, sim, margin = store.match(vec)
        if emp is None:
            return _finish(s, "unknown", "No face records enrolled", face, shape, **info)
        if sim >= MATCH_THRESHOLD and margin >= MATCH_MARGIN:
            s.ident_hits = s.ident_hits + 1 if s.emp in (None, emp) else 1
            s.emp, s.score = emp, sim
            if s.ident_hits >= IDENT_FRAMES:
                s.stage = "baseline"           # fall through: challenge can start on this very frame
            else:
                return _resp(s, "running", "Recognizing...", face, shape, **info)
        else:
            s.ident_hits, s.unknown = 0, s.unknown + 1
            if s.unknown >= UNKNOWN_LIMIT:
                return _finish(s, "unknown", "Face not recognized", face, shape, **info)
            return _resp(s, "running", "Recognizing...", face, shape, **info)

    # ---- baseline -> pick random LEFT/RIGHT ----
    if s.stage == "baseline":
        if not collected:
            _collect_baseline(s, frame, face, yaw)
        if len(s.base) < BASE_FRAMES or len(s.diffs) < BASE_FRAMES - 1:
            return _resp(s, "running", "Look straight at the camera...", face, shape, **info)
        if max(s.base) - min(s.base) > BASE_SPREAD:
            return _resp(s, "running", "Hold still, look straight...", face, shape, **info)
        if float(np.mean(s.diffs[-BASE_FRAMES:])) < MOTION_MIN:
            return _finish(s, "failed", "Spoof suspected (static image/frozen video)", face, shape, reason="spoof", **info)
        s.baseline = float(np.median(s.base))
        if not s.steps:
            s.steps, s.step_i = _make_steps(), 0     # random order + random values, decided server-side
        return _begin_step(s, now, face, shape, info)

    # ---- challenge ----
    if now > s.deadline:
        what = "%s pose" % s.target if s.kind == "pose" else "%s finger challenge" % s.fingers
        return _finish(s, "failed", "Timed out - %s not detected" % what, face, shape, reason="timeout", **info)
    if s.kind == "fingers":
        return _fingers_step(s, now, frame, face, shape, info)
    s.hist = (s.hist + [yaw])[-3:]                                  # median of last 3 kills landmark spikes
    m = float(np.median(s.hist))
    s.smooth = m if s.smooth is None else SMOOTH * m + (1 - SMOOTH) * s.smooth
    yv = (s.smooth - s.baseline) / YAW_DELTA     # raw (un-mirrored) frame: nose to image-right = user's LEFT
    tv = yv if s.target == "LEFT" else -yv
    need = HOLD_RELEASE if s.hold else 1.0
    if tv >= need:
        s.hold += 1
        s.wrong = 0
        if s.hold >= HOLD_FRAMES:
            return _complete_step(s, now, frame, face, shape, info)
    else:
        s.hold = 0
        s.wrong = s.wrong + 1 if -tv >= WRONG_STRENGTH else max(0, s.wrong - 1)
        if s.wrong >= WRONG_FRAMES:
            return _finish(s, "failed", "Wrong direction! Required: %s" % s.target, face, shape, reason="wrong", **info)
    return _resp(s, "running", "Pose Challenge: " + s.target, face, shape, **info)
