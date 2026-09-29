"""Finger counter. MediaPipe >= 0.10.30 removed `mp.solutions`; this uses the Tasks API (HandLandmarker) and
falls back to the legacy API only if it exists. Never raises: if unavailable, `hands.ready` is False."""
import math
import os
import threading
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(os.getenv("FACE_MODEL_DIR", Path(__file__).parent / "models"))
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"


def count_fingers(lm) -> int:
    """lm: 21 landmarks with .x/.y (normalized). Rotation-invariant: compares distances, not axes."""
    def d(a, b):
        return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)

    n = 0
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):      # index, middle, ring, pinky
        if d(tip, 0) > d(pip, 0) * 1.08:
            n += 1
    if d(4, 17) > d(3, 17) * 1.10 and d(4, 5) > d(3, 5) * 0.9:    # thumb: tip away from pinky base & index base
        n += 1
    return n


class Hands:
    def __init__(self):
        self.lock = threading.Lock()
        self._lm = None
        self._legacy = None
        self._mp = None
        self.error = ""

    @property
    def ready(self) -> bool:
        return self._lm is not None or self._legacy is not None

    def load(self):
        try:
            import mediapipe as mp
            self._mp = mp
        except Exception as e:  # noqa: BLE001
            self.error = "mediapipe import failed: %s" % e
            print("[hands]", self.error)
            return self
        try:
            if HAND_MODEL.exists():
                from mediapipe.tasks.python import BaseOptions, vision
                opts = vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
                    running_mode=vision.RunningMode.IMAGE, num_hands=1,
                    min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
                self._lm = vision.HandLandmarker.create_from_options(opts)
            elif hasattr(mp, "solutions"):                       # old mediapipe (<= 0.10.21)
                self._legacy = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
            else:
                self.error = "hand_landmarker.task missing - run: python download_models.py"
        except Exception as e:  # noqa: BLE001
            self.error = "hand landmarker init failed: %s" % e
        print("[hands]", "ready (%s)" % ("tasks" if self._lm else "legacy") if self.ready else "disabled: " + self.error)
        return self

    def count(self, bgr: np.ndarray):
        """Returns finger count (0-5) or None if no hand / unavailable."""
        if not self.ready:
            return None
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with self.lock:
            if self._lm is not None:
                img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
                res = self._lm.detect(img)
                hands = res.hand_landmarks
            else:
                res = self._legacy.process(rgb)
                hands = [h.landmark for h in (res.multi_hand_landmarks or [])]
        return count_fingers(hands[0]) if hands else None


hands = Hands()