"""영상 기반 위험 평가 (단순화 버전).

규칙:
  1. 얼굴 미탐 위험: person이 감지되지만 그 bbox 안에 face가 없음이 N초 지속 → 뒤집힘/얼굴덮힘
  2. ROI 이탈 위험: person 중심이 지정된 안전 ROI를 벗어남 → 낙상/등반

각 evaluate_* 함수는 (판정, 진단 dict)를 반환해 HUD·로그에서 원인 확인.
"""
from dataclasses import dataclass, field
from typing import Optional

from .face import Face
from .person import Person


@dataclass
class RiskSignal:
    type: str
    confidence: float
    metadata: dict = field(default_factory=dict)


def main_person(persons: list[Person]) -> Optional[Person]:
    if not persons:
        return None
    return max(persons, key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]))


def _face_inside_person(face: Face, person: Person) -> bool:
    fx = (face.bbox[0] + face.bbox[2]) / 2
    fy = (face.bbox[1] + face.bbox[3]) / 2
    px1, py1, px2, py2 = person.bbox
    return px1 <= fx <= px2 and py1 <= fy <= py2


def evaluate_face_missing(
    person: Optional[Person], faces: list[Face]
) -> tuple[bool, dict]:
    diag: dict = {"person_n": 1 if person else 0, "face_n": len(faces)}
    if person is None:
        diag["block"] = "no_person"
        return False, diag
    matching = [f for f in faces if _face_inside_person(f, person)]
    diag["face_in_p"] = len(matching)
    if not matching:
        return True, diag
    diag["block"] = "face_detected"
    return False, diag


def evaluate_roi_exit(
    person: Optional[Person], roi: tuple[int, int, int, int]
) -> tuple[bool, dict]:
    rx1, ry1, rx2, ry2 = roi
    diag: dict = {"roi": (rx1, ry1, rx2, ry2)}
    if person is None:
        diag["block"] = "no_person"
        return False, diag
    cx = (person.bbox[0] + person.bbox[2]) / 2
    cy = (person.bbox[1] + person.bbox[3]) / 2
    diag["center"] = (round(cx), round(cy))
    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
        diag["block"] = "inside_roi"
        return False, diag
    return True, diag
