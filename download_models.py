"""Downloads compressed (int8-quantized) OpenCV Zoo models: YuNet (detector, ~0.1 MB) + SFace (recognizer, ~9.7 MB)."""
import sys
import urllib.request
from pathlib import Path

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
FILES = {
    "face_detection_yunet_2023mar_int8.onnx": f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar_int8.onnx",
    "face_recognition_sface_2021dec_int8.onnx": f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec_int8.onnx",
}
FALLBACK = {
    "face_detection_yunet_2023mar.onnx": f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MIN_BYTES = 50_000  # guards against Git-LFS pointer files / HTML error pages


def fetch(name: str, url: str, out: Path) -> bool:
    dest = out / name
    if dest.exists() and dest.stat().st_size > MIN_BYTES:
        print(f"ok   {name}")
        return True
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:  # noqa: BLE001
        print(f"fail {name}: {e}")
        dest.unlink(missing_ok=True)
        return False
    if dest.stat().st_size < MIN_BYTES:
        print(f"fail {name}: file too small")
        dest.unlink(missing_ok=True)
        return False
    print(f"ok   {name} ({dest.stat().st_size / 1024:.0f} KB)")
    return True


def main() -> int:
    out = Path(__file__).parent / "models"
    out.mkdir(exist_ok=True)
    ok = True
    for (n, u), (fn, fu) in zip(FILES.items(), FALLBACK.items()):
        if not fetch(n, u, out):
            ok = fetch(fn, fu, out) and ok
    if not fetch("hand_landmarker.task", HAND_URL, out):     # finger challenge only: warn, don't fail the build
        print("WARNING: hand model missing -> finger challenge disabled")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
