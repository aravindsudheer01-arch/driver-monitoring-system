"""Download Google's pretrained MediaPipe Face Landmarker model."""
from pathlib import Path
from urllib.request import urlretrieve

URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_FILE = Path(__file__).resolve().parent / "models" / "face_landmarker.task"

if __name__ == "__main__":
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_FILE.is_file() and MODEL_FILE.stat().st_size > 1_000_000:
        print(f"Model already exists: {MODEL_FILE}")
    else:
        print("Downloading the official MediaPipe model...")
        urlretrieve(URL, MODEL_FILE)
        print(f"Saved: {MODEL_FILE} ({MODEL_FILE.stat().st_size:,} bytes)")
