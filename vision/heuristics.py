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
    climb_rois: list[tuple[int, int, int, int]],
    keypoint_conf_threshold: float,
    standing_y_margin: float,
) -> tuple[bool, dict]:
    diag: dict = {"climb_rois_n": len(climb_rois)}
    if not climb_rois:
        diag["block"] = "no_climb_roi"
        return False, diag
    if pose is None:
        diag["block"] = "no_pose"
        return False, diag
    if smoothed_ankle is None:
        diag["block"] = "no_ankle"
        return False, diag
    ax, ay = smoothed_ankle
    diag["ankle"] = (round(ax), round(ay))
    matched = next(
        (i for i, (cx1, cy1, cx2, cy2) in enumerate(climb_rois)
         if cx1 <= ax <= cx2 and cy1 <= ay <= cy2),
        None,
    )
    if matched is None:
        diag["block"] = "ankle_outside_roi"
        return False, diag
    diag["matched_roi"] = matched

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


def _face_inside_person(face: Face, person: Person) -> bool:
    fx = (face.bbox[0] + face.bbox[2]) / 2
    fy = (face.bbox[1] + face.bbox[3]) / 2
    px1, py1, px2, py2 = person.bbox
    return px1 <= fx <= px2 and py1 <= fy <= py2


def evaluate_suffocation(
    person: Optional[Person],
    faces: list[Face],
    pose: Optional[Pose],
    keypoint_conf_threshold: float,
    flipped_min_visible: int,
    blanket_max_visible: int,
) -> tuple[bool, Optional[str], dict]:
    diag: dict = {"person_n": 1 if person else 0, "face_n": len(faces)}
    if person is None:
        diag["block"] = "no_person"
        return False, None, diag
    matching = [f for f in faces if _face_inside_person(f, person)]
    diag["face_in_p"] = len(matching)
    if matching:
        diag["block"] = "face_detected"
        return False, None, diag

    visible = 0
    if pose is not None:
        for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            if pose.keypoints[name][2] >= keypoint_conf_threshold:
                visible += 1
    diag["visible_keypoints"] = visible

    assert flipped_min_visible > blanket_max_visible, (
        f"flipped_min_visible ({flipped_min_visible}) must exceed "
        f"blanket_max_visible ({blanket_max_visible})"
    )
    if visible >= flipped_min_visible:
        cause = "flipped"
    elif visible <= blanket_max_visible:
        cause = "blanket"
    else:
        cause = "unknown"
    return True, cause, diag
