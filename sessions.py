"""Two-step verification. Step 1/2: straight-face recognition (no head-pose challenge) with a passive micro-motion
liveness check. Step 2/2: random finger-count challenge (hand tracked, overlay points returned). Attendance passes only
after both, and the face seen while showing fingers must still match the employee recognised in step 1.
Challenges are chosen server-side. If hand tracking is unavailable the session fails (attendance is never granted on the face step alone)."""
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

MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", 0.37))    # SFace cosine (0.363 = zoo default)
MATCH_MARGIN = 0.03
IDENT_FRAMES = 2
LIVE_FRAMES = 3                 # straight-face frames used for the passive liveness (micro-motion) check
STRAIGHT_YAW = 0.22             # face must look roughly at the camera (eye-distance units) - no head turning needed
STRAIGHT_TIMEOUT = 6.0          # seconds allowed to look straight once the face has been matched
MIN_FACE_FRAC = 0.16            # face width / frame width
SEARCH_TIMEOUT = 3.0
HARD_TIMEOUT = 60.0
MOTION_MIN = 0.45               # mean abs pixel diff (0-255) of fixed face ROI across the liveness frames
UNKNOWN_LIMIT = 8
FINGER_TARGETS = tuple(int(x) for x in os.getenv("FINGER_TARGETS", "1,2,3,4").split(",") if x.strip())
FINGER_HOLD = 3                     # consecutive frames with the exact requested count
FINGER_TIMEOUT = float(os.getenv("FINGER_TIMEOUT_S", 12))
FINGER_FACE_GRACE = 0.0             # STRICT: no grace - face must be present in EVERY finger-step frame
FINGER_MIN_SCORE = 0.6              # detector confidence below this = face obstructed / not a clear real face
FINGER_MAX_YAW = 0.35               # face turned away beyond this = lost
FINGER_CENTER_X = 0.30              # face centre must stay within +-30% of frame centre (x)
FINGER_CENTER_Y = 0.32              # ... and +-32% (y)
STEP_LINK_THRESHOLD = float(os.getenv("STEP_LINK_THRESHOLD", 0.25))   # same person in step 1 and step 2 (relaxed: hand may cover face)
SWAP_FRAMES = 1                     # consecutive non-matching identity checks in step 2 before "person changed"
FACE_CHECK_S = 0.0                  # STRICT: face detect + identity runs on EVERY frame of step 2
REQUIRE_FINGER = True               # BOTH steps are mandatory: no face-only fallback, ever

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
    finger_done: bool = False       # Step 2/2 complete: requested finger count held
    use_fingers: bool = True
    f_target: int | None = None
    f_deadline: float = 0.0
    f_hold: int = 0
    link_t: float = 0.0             # last time the face seen in step 2 matched the step-1 employee
    swap: int = 0
    face_chk_t: float = 0.0         # last time the (slow) identity check ran in step 2
    face_box: list | None = None    # last known face box, reused between identity checks


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


def save(s: Session):
    _save(s)


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
           "face_verified": s.face_done, "finger_verified": s.finger_done}
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
    if state == "passed" and not (s.face_done and s.finger_done and s.emp):   # hard gate: both challenges required
        state, message, extra = "failed", "Both face and finger challenges are required", dict(extra, reason="incomplete")
    s.done = True
    if state != "passed":                                       # any failure -> back to Step 1, nothing stays verified
        s.face_done = s.finger_done = False
        s.stage = "search"
    s.result = _resp(s, state, message, face, shape, **extra)
    if state == "passed":
        s.stage = "passed"
        s.result["stage"] = "passed"
        s.result["employee_id"] = s.emp
        s.result["score"] = round(s.score, 3)
        s.result["status"], s.result["action"] = "VERIFIED / SUCCESS", "ACCEPT"
    elif extra.get("reason") == "face_lost":
        s.result["status"], s.result["action"] = "FACE_LOST", "REJECT/RESET"
        s.result["reset_to_step"] = 1
    else:
        s.result["action"] = "REJECT/RESET"
        s.result["reset_to_step"] = 1
    return s.result


def _fmsg(n: int) -> str:
    return "Show %d finger%s" % (n, "" if n == 1 else "s")


def _start_finger(s: Session, now: float):
    s.stage, s.f_target = "finger", secrets.choice(FINGER_TARGETS)
    s.f_deadline, s.f_hold, s.swap, s.link_t = now + FINGER_TIMEOUT, 0, 0, now
    s.face_chk_t, s.last_face_t = now, now


def _face_lost(s: Session, why: str, **info) -> dict:
    """Face missing / off-centre / obstructed / turned away during the finger challenge -> instant FACE_LOST + reset to Step 1.
    Finger detection is NEVER run in this case."""
    return _finish(s, "failed", "FACE_LOST - " + why + " - restarting from Step 1", reason="face_lost",
                   target_fingers=s.f_target, face_present=False, **info)


def _finger_step(s: Session, bgr: np.ndarray, now: float) -> dict:
    """Step 2/2 (STRICT). Every single frame: the real face must be detected, clear, centred, facing the camera, alone, and
    still the same employee as in step 1. Only then is the hand analysed - on that SAME frame. One failed frame = FACE_LOST."""
    frame, luma, low = prepare(bgr)
    shape = frame.shape
    H, W = shape[:2]
    info = {"luma": round(luma, 1), "low_light": low}
    if now > s.f_deadline:
        return _finish(s, "failed", "Timed out - %s" % _fmsg(s.f_target).lower(), reason="timeout",
                       target_fingers=s.f_target, box=s.face_box, **info)

    faces = engine.detect(frame, low)
    if not faces:
        return _face_lost(s, "face not visible", **info)
    face = faces[0]
    s.face_box = [float(face[0]) / W, float(face[1]) / H, float(face[2]) / W, float(face[3]) / H]
    if len(faces) > 1 and faces[1][2] * faces[1][3] > 0.5 * face[2] * face[3]:
        return _face_lost(s, "multiple faces", box=s.face_box, **info)
    if face[14] < FINGER_MIN_SCORE or face[2] < MIN_FACE_FRAC * W:
        return _face_lost(s, "face obstructed or too small", box=s.face_box, **info)
    cx, cy = (face[0] + face[2] / 2) / W, (face[1] + face[3] / 2) / H
    if abs(cx - 0.5) > FINGER_CENTER_X or abs(cy - 0.5) > FINGER_CENTER_Y:
        return _face_lost(s, "face out of frame / not centered", box=s.face_box, **info)
    yaw, _ = pose_metrics(face)
    if abs(yaw) > FINGER_MAX_YAW:
        return _face_lost(s, "face turned away", box=s.face_box, **info)
    sim = store.similarity(s.emp, engine.embed(frame, face))
    if sim < STEP_LINK_THRESHOLD:
        return _finish(s, "failed", "Person changed between steps - restarting from Step 1", reason="person_changed",
                       target_fingers=s.f_target, box=s.face_box, **info)
    s.last_face_t = s.link_t = now

    hand = engine.analyze_hand(frame)                           # same frame as the face above
    count = hand["count"] if hand else None
    extra = dict(count=count, fingers=hand["fingers"] if hand else [], hand_box=hand["box"] if hand else None,
                 target_fingers=s.f_target, box=s.face_box, face_present=True, **info)
    if count == s.f_target:
        s.f_hold += 1
        if s.f_hold >= FINGER_HOLD:
            s.finger_done = True
            s.stage = "passed"
            return _finish(s, "passed", "VERIFIED / SUCCESS", **extra)
    else:
        s.f_hold = 0
    return _resp(s, "running", "Step 2/2: " + _fmsg(s.f_target), **extra)


def step(s: Session, bgr: np.ndarray) -> dict:
    if s.done and s.result:
        return s.result
    now = time.time()
    if now - s.started > HARD_TIMEOUT:
        return _finish(s, "failed", "Session timed out", reason="timeout")

    if s.stage == "finger":                                     # fast path: no per-frame face detection
        return _finger_step(s, bgr, now)

    frame, luma, low = prepare(bgr)
    shape = frame.shape
    faces = engine.detect(frame, low)
    info = {"luma": round(luma, 1), "low_light": low}

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
        if not s.use_fingers:
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

    s.face_done = True                                          # Step 1/2 complete (attendance NOT granted yet)
    if not s.use_fingers:
        return _finish(s, "failed", "Finger challenge unavailable on server - see notice", face, shape,
                       reason="finger_unavailable", **info)
    _start_finger(s, now)
    s.face_box = [float(face[0]) / W, float(face[1]) / H, float(face[2]) / W, float(face[3]) / H]
    return _resp(s, "running", "Face Verification Complete", face, shape, **info)
