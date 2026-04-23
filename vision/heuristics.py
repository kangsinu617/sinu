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


def evaluate_climbing(
    smoothed_ankle: Optional[tuple[float, float]],
    pose: Optional[Pose],
    climb_roi: tuple[int, int, int, int],
    keypoint_conf_threshold: float,
    standing_y_margin: float,
) -> tuple[bool, dict]:
    diag: dict = {"climb_roi": climb_roi}
    if pose is None:
        diag["block"] = "no_pose"
        return False, diag
    if smoothed_ankle is None:
        diag["block"] = "no_ankle"
        return False, diag
    ax, ay = smoothed_ankle
    diag["ankle"] = (round(ax), round(ay))
    cx1, cy1, cx2, cy2 = climb_roi
    if not (cx1 <= ax <= cx2 and cy1 <= ay <= cy2):
        diag["block"] = "ankle_outside_roi"
        return False, diag

    shoulders = [pose.keypoints[k] for k in ("left_shoulder", "right_shoulder")
                 if pose.keypoints[k][2] >= keypoint_conf_threshold]
    hips = [pose.keypoints[k] for k in ("left_hip", "right_hip")
            if pose.keypoints[k][2] >= keypoint_conf_threshold]
    if not shoulders or not hips:
        diag["block"] = "shoulder_or_hip_invisible"
        return False, diag

    sy = sum(k[1] for k in shoulders) / len(shoulders)
    hy = sum(k[1] for k in hips) / len(hips)
    margin = hy - sy
    diag["standing_margin"] = round(margin, 1)
    if margin < standing_y_margin:
        diag["block"] = "not_standing"
        return False, diag
    return True, diag
