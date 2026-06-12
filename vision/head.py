"""CrowdHuman YOLOv5m 기반 머리 탐지. cls==1(head)만 반환.

질식 cause 판별용 — 엎드리면 뒤통수가 잡혀 head 검출(prone), 천에 덮이면
검출 안 됨(face_covered). 측정상 prone head_conf 0.40 vs 천 0.00으로 갈린다.
torch.hub로 yolov5 커스텀 가중치를 로드한다(ultralytics YOLO()는 못 읽음).
"""
from dataclasses import dataclass

import cv2
import torch


@dataclass
class Head:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float


class HeadDetector:
    RAW_CONF_FLOOR = 0.05  # 모델 내부 컷 — 임계 미달 검출도 conf 계측에 노출

    def __init__(self, weights_path: str, conf_threshold: float = 0.25) -> None:
        self.conf_threshold = conf_threshold  # "present" 판정은 호출부에서 적용
        self.model = torch.hub.load("ultralytics/yolov5", "custom",
                                    path=weights_path, trust_repo=True, verbose=False)
        self.model.conf = self.RAW_CONF_FLOOR

    def detect(self, frame_bgr) -> list[Head]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        det = self.model(rgb).xyxy[0].cpu().numpy()  # [x1,y1,x2,y2,conf,cls]
        return [Head(bbox=(float(d[0]), float(d[1]), float(d[2]), float(d[3])),
                     confidence=float(d[4]))
                for d in det if int(d[5]) == 1]
