"""Two-step verification. Step 1/2: straight-face recognition (no head-pose challenge) with a passive micro-motion
liveness check. Step 2/2: random finger-count challenge (hand tracked, overlay points returned). Attendance passes only
after both, and the face seen while showing fingers must still match the employee recognised in step 1.
Challenges are chosen server-side. When hand tracking is unavailable and REQUIRE_FINGER=0, step 1 alone is enough."""
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
MATCH_MARGIN = 0.03
IDENT_FRAMES = 2
LIVE_FRAMES = 3                 # straight-face frames used for the passive liveness (micro-motion) check
STRAIGHT_YAW = 0.22             # face must look roughly at the camera (eye-distance units) - no head turning needed
STRAIGHT_TIMEOUT = 6.0          # seconds allowed to look straight once the face has been matched
MIN_FACE_FRAC = 0.16            # face width / frame width
SEARCH_TIMEOUT = 3.0
HARD_TIMEOUT = 60.0
MOTION_MIN = 0.45               # mean abs pixel diff (0-255) of fixed face ROI across the liveness frames
UNKNOWN_LIMIT = 5
FINGER_TARGETS = tuple(int(x) for x in os.getenv("FINGER_TARGETS", "1,2,3,4,5").split(",") if x.strip())
FINGER_HOLD = 3                     # consecutive frames with the exact requested count
FINGER_TIMEOUT = float(os.getenv("FINGER_TIMEOUT_S", 12))
FINGER_FACE_GRACE = 3.5             # face may be hidden briefly by the hand
STEP_LINK_THRESHOLD = float(os.getenv("STEP_LINK_THRESHOLD", 0.25))   # same person in step 1 and step 2 (relaxed: hand may cover face)
SWAP_FRAMES = 3                     # consecutive non-matching face frames in step 2 before "person changed"
REQUIRE_FINGER = os.getenv("REQUIRE_FINGER", "1") != "0"   # 0 = face-only fallback when hand tracking is unavailable

_lock = threading.Lock()
SESSION_TTL = 90


@dataclass
class Session:
    sid: str = field(default_factory=lambda: secrets.token_urlsafe(16))
    started: float = field(default_factory=time.time)
    stage: str = "search"           # search -> identify -> finger -> passed
    emp: str | None = None
    ident_hits: int = 0
    ident_t: float = 0.0
    unknown: int = 0
    roi: tuple | None = None
    prev_crop: np.ndarray | None = None
    diffs: list = field(default_factory=list)
    last_face_t: float = field(default_factory=time.time)
    done: bool = False
    result: dict | None = None
    score: float = 0.0
    face_done: bool = False         # Step 1/2 complete: face matched an employee + liveness ok
    use_fingers: bool = True
    f_target: int | None = None
    f_deadline: float = 0.0
    f_hold: int = 0
    link_t: float = 0.0             # last time the face seen in step 2 matched the step-1 employee
    swap: int = 0


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
    s = Session(use_fingers=bool(getattr(engine, "hands_ready", False)))
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


def _collect_liveness(s: Session, frame, face) -> bool:
    """Fixed-ROI micro-motion sample (passive liveness). Only straight-facing frames count. Returns True if counted."""
    yaw, _ = pose_metrics(face)
    if abs(yaw) > STRAIGHT_YAW:
        return False
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
    return True


def _resp(s: Session, state: str, message: str, face=None, frame_shape=None, **extra):
    out = {"state": state, "stage": s.stage, "message": message,
           "step": (2 if s.stage in ("finger", "passed") else 1) if s.use_fingers else 1,
           "steps": 2 if s.use_fingers else 1,
           "face_verified": s.face_done}
    if s.face_done:
        out["employee_id"] = s.emp
    if s.stage == "finger":
        out["target_fingers"] = s.f_target
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


def _fmsg(n: int) -> str:
    return "Show %d finger%s" % (n, "" if n == 1 else "s")


def _start_finger(s: Session, now: float):
    s.stage, s.f_target = "finger", secrets.choice(FINGER_TARGETS)
    s.f_deadline, s.f_hold, s.swap, s.link_t = now + FINGER_TIMEOUT, 0, 0, now


def _finger_step(s: Session, frame, faces, shape, now: float, info: dict) -> dict:
    """Step 2/2. The employee is already known from step 1: the face in view must keep matching that employee
    (a hand may cover it briefly), so nobody can swap in between the two steps."""
    face = faces[0] if faces else None
    if face is not None:
        s.last_face_t = now
    if now > s.f_deadline:
        return _finish(s, "failed", "Timed out - %s" % _fmsg(s.f_target).lower(), face, shape, reason="timeout",
                       target_fingers=s.f_target, **info)
    if now - s.last_face_t > FINGER_FACE_GRACE:
        return _finish(s, "failed", "Keep your face in view", reason="noface", target_fingers=s.f_target, **info)
    if len(faces) > 1 and faces[1][2] * faces[1][3] > 0.5 * face[2] * face[3]:
        s.f_hold = 0
        return _resp(s, "running", "Multiple faces - one person only", face, shape, target_fingers=s.f_target, **info)
    if face is not None:                                        # identity must stay the same person as step 1
        sim = store.similarity(s.emp, engine.embed(frame, face))
        if sim >= STEP_LINK_THRESHOLD:
            s.swap, s.link_t = 0, now
        else:
            s.swap += 1
            if s.swap >= SWAP_FRAMES:
                return _finish(s, "failed", "Person changed between steps", face, shape, reason="person_changed",
                               target_fingers=s.f_target, **info)
    hand = engine.analyze_hand(frame)
    count = hand["count"] if hand else None
    extra = dict(count=count, fingers=hand["fingers"] if hand else [], hand_box=hand["box"] if hand else None,
                 target_fingers=s.f_target, **info)
    if count == s.f_target and now - s.link_t <= FINGER_FACE_GRACE:
        s.f_hold += 1
        if s.f_hold >= FINGER_HOLD:
            s.stage = "passed"                                  # Step 2/2 done -> attendance can be logged
            return _finish(s, "passed", "Verified", face, shape, **extra)
    else:
        s.f_hold = 0
    return _resp(s, "running", "Step 2/2: " + _fmsg(s.f_target), face, shape, **extra)


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

    if s.stage == "finger":
        return _finger_step(s, frame, faces, shape, now, info)

    if not faces:
        if s.stage == "search":
            if now - s.started > SEARCH_TIMEOUT:
                return _finish(s, "idle", "No face", **info)
            return _resp(s, "running", "Scanning face...", **info)
        s.ident_hits = 0                                        # identify stage
        if now - s.last_face_t > SEARCH_TIMEOUT:
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

    if s.stage == "search":
        if not s.use_fingers and REQUIRE_FINGER:
            return _finish(s, "failed", "Finger challenge unavailable on server - see notice", face, shape,
                           reason="finger_unavailable", **info)
        s.stage, s.ident_t = "identify", now

    # ---- Step 1/2: straight-face recognition (+ passive liveness sampled in the same frames) ----
    straight = _collect_liveness(s, frame, face)
    vec = engine.embed(frame, face)
    emp, sim, margin = store.match(vec)
    if emp is None:
        return _finish(s, "unknown", "No face records enrolled", face, shape, **info)
    if not (sim >= MATCH_THRESHOLD and margin >= MATCH_MARGIN):
        s.ident_hits, s.unknown = 0, s.unknown + 1
        if s.unknown >= UNKNOWN_LIMIT:
            return _finish(s, "unknown", "Face not recognized", face, shape, **info)
        return _resp(s, "running", "Recognizing...", face, shape, **info)

    s.ident_hits = s.ident_hits + 1 if s.emp in (None, emp) else 1
    s.emp, s.score = emp, sim
    if s.ident_hits < IDENT_FRAMES:
        return _resp(s, "running", "Recognizing...", face, shape, **info)

    if not straight or len(s.diffs) < LIVE_FRAMES - 1:          # need a few straight-facing frames for liveness
        if now - s.ident_t > STRAIGHT_TIMEOUT:
            return _finish(s, "failed", "Face not clearly visible", face, shape, reason="noface", **info)
        return _resp(s, "running", "Look straight at the camera...", face, shape, **info)
    if float(np.mean(s.diffs[-LIVE_FRAMES:])) < MOTION_MIN:
        return _finish(s, "failed", "Spoof suspected (static image/frozen video)", face, shape, reason="spoof", **info)

    s.face_done = True                                          # Step 1/2 complete
    if s.use_fingers:
        _start_finger(s, now)
        return _resp(s, "running", "Face Verification Complete", face, shape, **info)
    s.stage = "passed"                                          # hand tracking off + REQUIRE_FINGER=0 -> face-only
    return _finish(s, "passed", "Verified", face, shape, **info)
