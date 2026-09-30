"""OpenCV engine: low-light enhancement + YuNet detection + SFace embeddings (int8 ONNX, loaded once)."""
import math
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

_MP_ERR = ""
try:                                   # finger counting (optional: challenge falls back to pose-only if missing)
    import mediapipe as mp
except Exception as _e:  # noqa: BLE001
    mp = None
    _MP_ERR = "mediapipe import failed: %s" % _e

MODEL_DIR = Path(os.getenv("FACE_MODEL_DIR", Path(__file__).parent / "models"))
DET_FILES = ["face_detection_yunet_2023mar_int8.onnx", "face_detection_yunet_2023mar.onnx"]
REC_FILES = ["face_recognition_sface_2021dec_int8.onnx", "face_recognition_sface_2021dec.onnx"]

HAND_MODEL = MODEL_DIR / "hand_landmarker.task"   # MediaPipe >= 0.10.30 (no mp.solutions) needs this file

MAX_W = 640
LOW_LIGHT_LUMA = float(os.getenv("LOW_LIGHT_LUMA", 105))   # enhance below this mean brightness (0-255)
TARGET_LUMA = 125.0

_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


# ---------- low-light enhancement ----------
def mean_luma(bgr: np.ndarray) -> float:
    small = cv2.resize(bgr, (64, 48), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())


def enhance_low_light(bgr: np.ndarray, luma: float) -> np.ndarray:
    """Adaptive gamma -> CLAHE on L channel -> light bilateral denoise (only when very dark)."""
    luma = max(luma, 4.0)
    gamma = min(max(math.log(luma / 255.0) / math.log(TARGET_LUMA / 255.0), 1.0), 3.0)  # out = in^(1/gamma)
    lut = (np.power(np.arange(256) / 255.0, 1.0 / gamma) * 255.0).clip(0, 255).astype(np.uint8)
    out = cv2.LUT(bgr, lut)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if luma < 60:
        out = cv2.bilateralFilter(out, 5, 30, 30)
    return out


def prepare(bgr: np.ndarray):
    """Returns (frame_for_detection, luma, is_low_light)."""
    h, w = bgr.shape[:2]
    if w > MAX_W:
        bgr = cv2.resize(bgr, (MAX_W, int(h * MAX_W / w)), interpolation=cv2.INTER_AREA)
    luma = mean_luma(bgr)
    if luma < LOW_LIGHT_LUMA:
        return enhance_low_light(bgr, luma), luma, True
    return bgr, luma, False


# ---------- pose (from YuNet's 5 landmarks) ----------
def pose_metrics(face: np.ndarray):
    """Roll-compensated pose. Landmarks are rotated so the eye line is horizontal (head tilt / phone tilt no longer
    leaks into yaw/pitch). yaw: nose offset from eye-midpoint / eye distance. pitch: nose depth between eye-line and mouth-line."""
    rx, ry, lx, ly, nx, ny, rmx, rmy, lmx, lmy = [float(v) for v in face[4:14]]
    ang = math.atan2(ly - ry, lx - rx)
    c, s = math.cos(-ang), math.sin(-ang)
    ex, ey = (rx + lx) / 2, (ry + ly) / 2

    def rot(x, y):
        dx, dy = x - ex, y - ey
        return dx * c - dy * s, dx * s + dy * c

    n_x, n_y = rot(nx, ny)
    _, m_y = rot((rmx + lmx) / 2, (rmy + lmy) / 2)
    eye_d = max(1.0, math.hypot(lx - rx, ly - ry))
    return n_x / eye_d, n_y / max(1.0, m_y)


# ---------- fingers (MediaPipe Hands landmarks) ----------
FINGER_PAIRS = ((8, 6), (12, 10), (16, 14), (20, 18))   # (tip, pip) for index..pinky
FINGER_MARGIN = 1.05     # tip must be this much farther from wrist than pip
THUMB_MARGIN = 1.2       # thumb tip vs thumb base, measured from pinky knuckle
MIN_HAND_FRAC = 0.12     # hand bbox (max side) / frame


# ---------- engine ----------
class Engine:
    def __init__(self):
        self.lock = threading.Lock()   # cv2.dnn nets are not thread-safe
        self.detector = None
        self.recognizer = None
        self.det_name = self.rec_name = ""
        self.hands = None
        self._hands_api = ""
        self.hands_error = ""          # why the finger challenge is off (shown in the app popup via /api/health)
        self.hlock = threading.Lock()
        self._hts = 0                  # monotonic ms timestamp required by the tasks VIDEO mode
        cv2.setNumThreads(int(os.getenv("CV_THREADS", "2")))

    def load(self):
        for name in DET_FILES:
            p = MODEL_DIR / name
            if p.exists():
                try:
                    self.detector = cv2.FaceDetectorYN.create(str(p), "", (320, 320), 0.6, 0.3, 200)
                    self.det_name = name
                    break
                except cv2.error:
                    continue
        for name in REC_FILES:
            p = MODEL_DIR / name
            if p.exists():
                try:
                    self.recognizer = cv2.FaceRecognizerSF.create(str(p), "")
                    self.rec_name = name
                    break
                except cv2.error:
                    continue
        if self.detector is None or self.recognizer is None:
            raise RuntimeError("Models missing/unreadable in %s - run: python download_models.py" % MODEL_DIR)
        if mp is None:
            self.hands_error = _MP_ERR or "mediapipe not installed"
            print("[engine] finger challenge disabled:", self.hands_error)
        elif self.hands is None:
            self.hands, self._hands_api = self._init_hands()
        warm = np.zeros((240, 320, 3), np.uint8)   # warm-up so first real request is fast
        self.detect(warm, False)
        return self

    def _init_hands(self):
        """Legacy mp.solutions first (mediapipe <= 0.10.21: hand models + GL libs are bundled in the wheel).
        Tasks API only as fallback (needs hand_landmarker.task AND system libGLESv2/libEGL)."""
        try:
            if hasattr(mp, "solutions"):
                # static_image_mode=False: palm detector only runs when the hand is lost, otherwise the cheap
                # landmark tracker follows it from frame to frame -> much faster finger tracking.
                h = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0,
                                             min_detection_confidence=0.6, min_tracking_confidence=0.5)
                print("[engine] MediaPipe Hands (legacy solutions) ready")
                return h, "legacy"
            if HAND_MODEL.exists():
                from mediapipe.tasks.python import BaseOptions, vision
                opts = vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
                    running_mode=vision.RunningMode.VIDEO, num_hands=1,   # VIDEO = tracking between frames
                    min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
                h = vision.HandLandmarker.create_from_options(opts)     # may raise (e.g. missing libGLESv2)
                print("[engine] MediaPipe HandLandmarker (tasks) ready")
                return h, "tasks"
            self.hands_error = "%s missing - run python download_models.py" % HAND_MODEL.name
            print("[engine] MediaPipe Hands unavailable, finger challenge disabled: %s" % self.hands_error)
        except Exception as e:  # noqa: BLE001
            self.hands_error = "hand landmarker init failed: %s" % e
            print("[engine] MediaPipe Hands unavailable, finger challenge disabled:", e)
        return None, ""

    @property
    def hands_ready(self) -> bool:
        return self.hands is not None

    def analyze_hand(self, bgr: np.ndarray):
        """One hand -> {"count": 0-5, "box": [x,y,w,h], "fingers": [{"n": name, "tip": [x,y], "base": [x,y]}, ...]}.
        `fingers` lists only extended fingers (normalized 0-1 image coords) for the UI tracking overlay.
        Returns None when no usable hand is visible or hand tracking is unavailable."""
        if self.hands is None:
            return None
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with self.hlock:
            if self._hands_api == "tasks":
                self._hts = max(self._hts + 1, int(time.monotonic() * 1000))
                res = self.hands.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self._hts)
                lms = res.hand_landmarks[0] if res.hand_landmarks else None
            else:
                res = self.hands.process(rgb)
                lms = res.multi_hand_landmarks[0].landmark if res.multi_hand_landmarks else None
        if lms is None:
            return None
        pts = np.array([[p.x, p.y] for p in lms], np.float32)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        if max(x1 - x0, y1 - y0) < MIN_HAND_FRAC:
            return None
        px = pts * np.array([w, h], np.float32)          # pixel space: no aspect distortion

        def d(a, b):
            return float(np.linalg.norm(px[a] - px[b]))

        ext = []                                          # (name, tip landmark, base landmark)
        if d(4, 17) > d(2, 17) * THUMB_MARGIN:
            ext.append(("thumb", 4, 2))
        for name, (tip, pip), mcp in zip(("index", "middle", "ring", "pinky"), FINGER_PAIRS, (5, 9, 13, 17)):
            if d(tip, 0) > d(pip, 0) * FINGER_MARGIN:
                ext.append((name, tip, mcp))
        fingers = [{"n": n, "tip": [float(pts[t][0]), float(pts[t][1])], "base": [float(pts[m][0]), float(pts[m][1])]}
                   for n, t, m in ext]
        box = [float(max(0, x0)), float(max(0, y0)), float(min(1, x1) - max(0, x0)), float(min(1, y1) - max(0, y0))]
        return {"count": len(ext), "box": box, "fingers": fingers}

    def count_fingers(self, bgr: np.ndarray):
        """Returns (count 0-5, hand_box normalized [x,y,w,h]) or (None, None) when no usable hand is visible."""
        r = self.analyze_hand(bgr)
        return (r["count"], r["box"]) if r else (None, None)

    @property
    def ready(self) -> bool:
        return self.detector is not None and self.recognizer is not None

    def detect(self, bgr: np.ndarray, low_light: bool):
        """All faces as rows [x,y,w,h, 5 landmarks(10), score], largest first."""
        h, w = bgr.shape[:2]
        with self.lock:
            self.detector.setInputSize((w, h))
            self.detector.setScoreThreshold(0.45 if low_light else 0.6)
            _, faces = self.detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return []
        return sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    def embed(self, bgr: np.ndarray, face: np.ndarray) -> np.ndarray:
        with self.lock:
            aligned = self.recognizer.alignCrop(bgr, face)
            feat = self.recognizer.feature(aligned)
        v = feat.reshape(-1).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


engine = Engine()
