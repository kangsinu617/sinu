"""OpenCV YuNet 기반 얼굴 탐지. 첫 실행 시 ONNX 모델 자동 다운로드."""
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import cv2

MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"


@dataclass
class Face:
    bbox: tuple[float, float, float, float]
    confidence: float


class FaceDetector:
    def __init__(self, score_threshold: float = 0.6, model_dir: Path | None = None) -> None:
        model_dir = model_dir or (Path(__file__).resolve().parent.parent / "models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / MODEL_FILENAME
        if not model_path.exists():
            print(f"[face] downloading {MODEL_FILENAME} ...")
            urlretrieve(MODEL_URL, model_path)
        self._detector = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
        )

    def detect(self, frame) -> list[Face]:
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame)
        if faces is None:
            return []
        out: list[Face] = []
        for f in faces:
            x, y, fw, fh = f[:4]
            score = float(f[-1])
            out.append(Face(bbox=(float(x), float(y), float(x + fw), float(y + fh)), confidence=score))
        return out
