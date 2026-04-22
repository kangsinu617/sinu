"""YOLOv8n 기반 사람(=아기) 탐지. COCO class 0(person)만 필터."""
from dataclasses import dataclass

from ultralytics import YOLO


@dataclass
class Person:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float


class PersonDetector:
    def __init__(self, weights: str = "yolov8n.pt") -> None:
        self.model = YOLO(weights)

    def detect(self, frame) -> list[Person]:
        res = self.model(frame, verbose=False, classes=[0])[0]
        if len(res.boxes) == 0:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        return [Person(bbox=tuple(b.tolist()), confidence=float(c)) for b, c in zip(boxes, confs)]
