"""Downloads compressed (int8-quantized) OpenCV Zoo models: YuNet (detector, ~0.1 MB) + SFace (recognizer, ~9.7 MB)."""
import subprocess
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


MEDIAPIPE_VERSION = "0.10.21"   # last release with mp.solutions + bundled hand models + bundled libGLESv2/libEGL (0.10.30+ needs system GL libs)
MEDIAPIPE = "mediapipe==" + MEDIAPIPE_VERSION
MP_DEPS = ["absl-py", "attrs>=19.1.0", "flatbuffers>=2.0", "protobuf>=4.25.3,<5", "matplotlib", "numpy<2"]


def ensure_mediapipe() -> None:
    """Build-time self-heal: install the pinned mediapipe with --no-deps (its opencv-contrib-python pin needs libGL)."""
    probe = "import importlib.metadata as m, mediapipe; print(m.version('mediapipe'))"

    def installed() -> str:
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""

    if installed() == MEDIAPIPE_VERSION:
        print("ok   %s already installed" % MEDIAPIPE)
        return
    print("mediapipe %s -> installing %s (--no-deps)" % (installed() or "missing", MEDIAPIPE))
    subprocess.call([sys.executable, "-m", "pip", "install", "-q", *MP_DEPS])
    subprocess.call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", MEDIAPIPE])
    print("ok   %s installed" % MEDIAPIPE if installed() == MEDIAPIPE_VERSION else "WARNING: mediapipe not importable -> finger challenge disabled")


def main() -> int:
    ensure_mediapipe()
    out = Path(__file__).parent / "models"
    out.mkdir(exist_ok=True)
    ok = True
    for (n, u), (fn, fu) in zip(FILES.items(), FALLBACK.items()):
        if not fetch(n, u, out):
            ok = fetch(fn, fu, out) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
