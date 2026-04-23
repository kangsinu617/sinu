"""영상 기반 위험 평가 — v1.

규칙:
  1. suffocation_risk: person 안에 face 없음 지속 → cause(flipped|blanket|unknown) 분기
  2. climbing_risk: pose ankle이 난간 ROI 안 + 서있음 자세 지속
  3. roi_exit_risk: person 중심이 안전 ROI 밖

각 evaluate_* 는 순수 함수로, 스무딩된 좌표 입력을 받아 (판정, 진단) 반환.
"""
from dataclasses import dataclass, field
from typing import Optional

from .face import Face
from .person import Person
from .pose import Pose


@dataclass
class RiskSignal:
    type: str
    confidence: float
    metadata: dict = field(default_factory=dict)


def main_person(persons: list[Person]) -> Optional[Person]:
    if not persons:
        return None
    return max(persons, key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]))


def evaluate_roi_exit(
    center: Optional[tuple[float, float]],
    roi: tuple[int, int, int, int],
) -> tuple[bool, dict]:
    rx1, ry1, rx2, ry2 = roi
    diag: dict = {"roi": (rx1, ry1, rx2, ry2)}
    if center is None:
        diag["block"] = "no_center"
        return False, diag
    cx, cy = center
    diag["center"] = (round(cx), round(cy))
    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
        diag["block"] = "inside_roi"
        return False, diag
    return True, diag
