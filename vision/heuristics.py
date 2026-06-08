"""영상 기반 위험 평가 — v1.

규칙:
  1. suffocation_risk: face가 최근 보였는데 person 안에 face 없음 지속
  2. climbing_risk: pose wrist가 난간 ROI 안 + 서있음 자세 지속
  3. roi_exit_risk: person 중심이 안전 ROI 밖
  4. fall_risk: person 중심 y가 짧은 윈도우 동안 크게 하강 (낙상)

각 evaluate_* 는 순수 함수로 (판정, 진단) 반환.
"""
from dataclasses import dataclass, field
from typing import Optional

from .face import Face
from .person import Person
from .pose import Pose
from .roi_geometry import EDGE_LABELS, nearest_edge, point_in_polygon

__all__ = [
    "RiskSignal", "main_person",
    "evaluate_roi_exit", "evaluate_fall", "evaluate_climbing", "evaluate_suffocation",
    "face_inside_person", "pose_face_visible", "pose_torso_visible",
]

_FACE_KP = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
_TORSO_KP = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


def pose_face_visible(pose: Optional[Pose], conf_threshold: float, min_visible: int) -> bool:
    """pose의 얼굴 키포인트(코·눈·귀)가 충분히 보이면 True.

    YuNet face 검출이 카메라 각도 때문에 실패해도, 천장을 보고 누운(supine)
    상태면 pose는 얼굴 키포인트를 높은 conf로 잡는다(측정 5/5, conf 0.93~0.98).
    엎드리면(prone) 얼굴이 매트를 향해 죽는다(측정 1/5, nose 0.08). 이 차이로
    'face 미검출이지만 정상 누움'을 질식 오탐에서 제외한다.

    face_visible의 보조(OR) 신호로만 쓰고 필수 전제로 쓰지 않는다 — 이불덮힘
    (face_covered)은 얼굴 키포인트도 안 보여 OR 양쪽이 False라 그대로 판정된다.
    """
    if pose is None:
        return False
    visible = sum(1 for n in _FACE_KP if pose.keypoints[n][2] >= conf_threshold)
    return visible >= min_visible


def pose_torso_visible(pose: Optional[Pose], conf_threshold: float, min_visible: int) -> bool:
    """pose의 몸통 키포인트(어깨·엉덩이)가 충분히 보이면 True.

    face가 안 보이는 상태에서 몸통 키포인트가 잡힌다 = 등을 카메라로 향한
    엎드림(prone). 측정상 엎드림은 4/4가 0.95~0.99로 잡히고, 천에 덮이면
    키포인트가 0.0으로 죽는다(0/4) — edge_density(prone 0.046~0.159 vs
    covered 0.012~0.042)보다 구간이 압도적으로 멀어 견고하게 갈린다.

    천에 완전히 덮여 키포인트가 죽으면 False가 되어 face_covered로 분류되므로
    (이불덮힘의 정답), 키포인트를 위험의 필수 전제가 아니라 flipped vs
    face_covered 갈림길로만 쓴다. 측면 누움은 한쪽만 보일 수 있어 min_visible로
    조절한다.
    """
    if pose is None:
        return False
    visible = sum(1 for n in _TORSO_KP if pose.keypoints[n][2] >= conf_threshold)
    return visible >= min_visible


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
    safe_polygon: list[tuple[float, float]],
) -> tuple[bool, dict]:
    diag: dict = {"polygon_n": len(safe_polygon)}
    if center is None:
        diag["block"] = "no_center"
        return False, diag
    cx, cy = center
    diag["center"] = (round(cx), round(cy))
    if point_in_polygon((cx, cy), safe_polygon):
        diag["block"] = "inside_polygon"
        return False, diag
    return True, diag


def evaluate_fall(
    center: Optional[tuple[float, float]],
    past_center: Optional[tuple[float, float]],
    min_drop_px: float,
) -> tuple[bool, dict]:
    """짧은 윈도우(호출자가 관리) 동안의 순 하강 거리로 낙상 판정.

    past_center는 약 window_s 전의 center. 단일 프레임 속도가 아니라
    윈도우 누적 하강을 보므로 카메라 흔들림·검출 튐 같은 왕복성 노이즈가
    상쇄되어 걸러진다. 진짜 낙상은 윈도우 동안 한 방향으로 크게 하강한다.
    """
    diag: dict = {}
    if center is None or past_center is None:
        diag["block"] = "no_center"
        return False, diag
    drop = center[1] - past_center[1]  # 양수 = 아래로
    diag["fall_drop"] = round(drop, 1)
    if drop < min_drop_px:
        diag["block"] = "drop_too_small"
        return False, diag
    return True, diag


def evaluate_climbing(
    smoothed_wrist: Optional[tuple[float, float]],
    pose: Optional[Pose],
    safe_polygon: list[tuple[float, float]],
    rail_band_px: float,
    keypoint_conf_threshold: float,
    standing_y_margin: float,
) -> tuple[bool, dict]:
    diag: dict = {"polygon_n": len(safe_polygon)}
    if not safe_polygon:
        diag["block"] = "no_polygon"
        return False, diag
    if pose is None:
        diag["block"] = "no_pose"
        return False, diag
    if smoothed_wrist is None:
        diag["block"] = "no_wrist"
        return False, diag
    wx, wy = smoothed_wrist
    diag["wrist"] = (round(wx), round(wy))
    if not point_in_polygon((wx, wy), safe_polygon):
        diag["block"] = "wrist_outside_polygon"
        return False, diag
    edge_idx, dist = nearest_edge((wx, wy), safe_polygon)
    diag["rail_dist"] = round(dist, 1)
    if dist > rail_band_px:
        diag["block"] = "not_near_rail"
        return False, diag
    diag["rail_edge"] = EDGE_LABELS[edge_idx]

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


def face_inside_person(face: Face, person: Person) -> bool:
    fx = (face.bbox[0] + face.bbox[2]) / 2
    fy = (face.bbox[1] + face.bbox[3]) / 2
    px1, py1, px2, py2 = person.bbox
    return px1 <= fx <= px2 and py1 <= fy <= py2


def evaluate_suffocation(
    subject_present: bool,
    torso_visible: bool,
    face_visible_now: bool,
    face_recently_seen: bool,
    person_was_in_roi: bool,
    roi_containment: float = 1.0,
    out_of_view_roi_threshold: float = 0.0,
    motion_level: float = 0.0,
    motion_threshold: float = float("inf"),
    head_present: Optional[bool] = None,
) -> tuple[bool, Optional[str], dict]:
    """질식 위험을 감지하고 원인을 몸통 키포인트 가시성(torso_visible)으로 구분.

    face가 안 보이는 상태에서 원인은 pose의 몸통 키포인트(어깨·엉덩이) 가시성으로
    가른다. 엎드린(prone) 인형은 등을 카메라로 향해 몸통 키포인트가 잘 잡히고
    (측정 4/4, 0.95~0.99), 천에 덮이면 키포인트가 0.0으로 죽는다(0/4). 둘이
    압도적으로 멀어 견고하게 갈린다(이전엔 edge_density로 갈랐으나 prone
    0.046~0.159 vs covered 0.012~0.042로 구간이 붙어 있어 교체).
      - flipped:      subject 있음 + torso_visible (몸통 노출 = 등 향한 엎드림).
      - face_covered: torso_visible False(천에 덮여 키포인트 소실), 또는 subject가
                      아예 없음(몸·머리까지 완전히 파묻혀 검출 붕괴).

    단, subject bbox가 안전 ROI 밖으로 많이 벗어나 있으면
    (roi_containment < out_of_view_roi_threshold) 인형이 카메라 각도 안에
    제대로 안 잡힌 상태(발만 보임 등)다. 엎드림 정탐은 ROI 포함율이 높고
    (측정 88%) 이 엣지케이스는 낮아(68%) 갈리므로, 이때는 위험이 아니라
    "안 보임"(out_of_view)으로 처리한다. 이 가드는 flipped/face_covered 양쪽에
    공통 적용한다 — torso로 원인을 가르기 전에 먼저 시야 밖을 거른다. 단
    subject 자체가 아예 없는(완전 파묻힘) 경우는 이 가드보다 앞서 face_covered로
    반환해, ROI 포함율 가드가 진짜 질식을 out_of_view로 놓치지 않게 한다.

    subject_present 는 pose 또는 person 검출 여부, torso_visible 은 pose 몸통
    키포인트가 충분히 보이는지를 호출자가 pose_torso_visible로 계산해 넘긴다.
    face_recently_seen / person_was_in_roi 는 프레임 간 추적값. ROI 안에서 본 적
    없으면(빈 방) 오탐 방지를 위해 판정하지 않는다.

    또 flipped 후보라도 subject 영역의 프레임 간 활동량(motion_level)이
    motion_threshold 이상이면 살아 움직이는 중(배 시간·능동적 버둥거림)이라
    무반응 질식이 아니다 → active_motion(위험 아님)으로 처리한다. 인형·천 덮인
    무반응은 활동량이 낮게 유지된다. 이 가드도 flipped 분기에만 적용한다
    (face_covered는 천에 덮인 채 버둥거려도 위험할 수 있어 면제하지 않는다).
    움직임은 단일 프레임 신호라 산발적이어도, 5초 지속 트리거가 연속 정지만
    위험으로 채택하므로 가끔의 움직임으로도 카운트가 리셋된다.

    head 검출(head_present)을 넘기면 cause를 head 우선으로 가른다 — head 보이면
    flipped, 안 보이면 face_covered. 얼굴만 천 덮인 경우(torso 4/4지만 head 없음,
    측정 prone 0.40 vs 천 0.00)를 face_covered로 정확히 잡는다. head_present=None
    (검출기 미사용/실패)이면 기존 torso_visible 폴백으로 동작한다.
    """
    diag: dict = {"subject": int(subject_present), "torso": int(torso_visible),
                  "roiin": round(roi_containment, 2), "motion": round(motion_level, 3),
                  "head": ("fallback" if head_present is None else int(head_present))}
    if face_visible_now:
        diag["block"] = "face_detected"
        return False, None, diag
    if not person_was_in_roi:
        diag["block"] = "not_in_roi"
        return False, None, diag
    if not face_recently_seen:
        diag["block"] = "face_never_seen"
        return False, None, diag
    if not subject_present:
        return True, "face_covered", diag
    if roi_containment < out_of_view_roi_threshold:
        diag["block"] = "out_of_view"
        return False, "out_of_view", diag
    is_prone = torso_visible if head_present is None else head_present
    if is_prone:
        if motion_level >= motion_threshold:
            diag["block"] = "active_motion"
            return False, "active_motion", diag
        return True, "flipped", diag
    return True, "face_covered", diag
