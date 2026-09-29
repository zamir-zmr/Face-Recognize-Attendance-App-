"""OpenCV engine: low-light enhancement + YuNet detection + SFace embeddings (int8 ONNX, loaded once)."""
import math
import os
import threading
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(os.getenv("FACE_MODEL_DIR", Path(__file__).parent / "models"))
DET_FILES = ["face_detection_yunet_2023mar_int8.onnx", "face_detection_yunet_2023mar.onnx"]
REC_FILES = ["face_recognition_sface_2021dec_int8.onnx", "face_recognition_sface_2021dec.onnx"]

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
    """yaw: nose offset from eye-midpoint / eye distance. pitch: nose position between eye-line and mouth-line."""
    rx, ry, lx, ly, nx, ny, rmx, rmy, lmx, lmy = [float(v) for v in face[4:14]]
    eye_mx, eye_my = (rx + lx) / 2, (ry + ly) / 2
    mouth_my = (rmy + lmy) / 2
    eye_d = max(1.0, abs(lx - rx))
    yaw = (nx - eye_mx) / eye_d
    pitch = (ny - eye_my) / max(1.0, mouth_my - eye_my)
    return yaw, pitch


# ---------- engine ----------
class Engine:
    def __init__(self):
        self.lock = threading.Lock()   # cv2.dnn nets are not thread-safe
        self.detector = None
        self.recognizer = None
        self.det_name = self.rec_name = ""
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
        warm = np.zeros((240, 320, 3), np.uint8)   # warm-up so first real request is fast
        self.detect(warm, False)
        return self

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
