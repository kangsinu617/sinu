"""YOLOv8n-pose 래퍼와 person ↔ pose 매칭 유틸리티."""
from dataclasses import dataclass
from typing import Optional

from ultralytics import YOLO

from .person import Person

KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


@dataclass
class Pose:
    bbox: tuple[float, float, float, float]
    keypoints: dict[str, tuple[float, float, float]]  # name → (x, y, conf)


class PoseDetector:
    def __init__(self, weights: str = "yolov8n-pose.pt") -> None:
        self.model = YOLO(weights)

    def detect(self, frame) -> list[Pose]:
        res = self.model(frame, verbose=False, classes=[0])[0]
        if len(res.boxes) == 0 or res.keypoints is None:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        kps_all = res.keypoints.data.cpu().numpy()  # (N, 17, 3)
        out: list[Pose] = []
        for b, kps in zip(boxes, kps_all):
            kp_dict = {
                KP_NAMES[i]: (float(kps[i][0]), float(kps[i][1]), float(kps[i][2]))
                for i in range(17)
            }
            out.append(Pose(bbox=tuple(b.tolist()), keypoints=kp_dict))
        return out


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_pose_to_person(person: Person, poses: list[Pose]) -> Optional[Pose]:
    if not poses:
        return None
    best_iou, best_pose = 0.0, None
    for p in poses:
        score = iou(person.bbox, p.bbox)
        if score > best_iou:
            best_iou, best_pose = score, p
    return best_pose
