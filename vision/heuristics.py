"""영상 기반 위험 평가 — v1.

규칙:
  1. suffocation_risk: 2층 분리 — detect_suffocation(face 미가시 AND 진입 게이트)
     + label_suffocation_cause(원인 flipped/face_covered, flags는 비차단 메타데이터)
  2. climbing_risk: pose wrist가 난간 ROI 안 + 서있음 자세 지속
  3. fall_risk: person 중심 y가 짧은 윈도우 동안 크게 하강 (낙상)

각 evaluate_* / detect_* / label_* 는 순수 함수로 (판정, 진단) 반환.
"""
import math
from dataclasses import dataclass, field
from typing import Optional

from .face import Face
from .person import Person
from .pose import Pose
from .roi_geometry import EDGE_LABELS, nearest_edge, point_in_polygon

__all__ = [
    "RiskSignal", "main_person",
    "evaluate_fall", "evaluate_climbing", "detect_suffocation", "label_suffocation_cause",
    "face_inside_person", "pose_face_visible", "pose_face_kp_count", "pose_torso_visible",
    "side_lying_features", "clearly_side_lying",
    "memory_fresh", "suffocation_latched", "presence_sustained",
]

_FACE_KP = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
_TORSO_KP = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
_SIDE_KP_PAIRS = (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"))


def pose_face_visible(pose: Optional[Pose], conf_threshold: float, min_visible: int) -> bool:
    """pose의 얼굴 키포인트(코·눈·귀)가 충분히 보이면 True.

    YuNet face 검출이 카메라 각도 때문에 실패해도, 천장을 보고 누운(supine)
    상태면 pose는 얼굴 키포인트를 높은 conf로 잡는다(측정 5/5, conf 0.93~0.98).
    엎드리면(prone) 얼굴이 매트를 향해 죽는다(측정 1/5, nose 0.08). 이 차이로
    'face 미검출이지만 정상 누움'을 질식 오탐에서 제외한다.

    face_visible의 보조(OR) 신호로만 쓰고 필수 전제로 쓰지 않는다 — 이불덮힘
    (face_covered)은 얼굴 키포인트도 안 보여 OR 양쪽이 False라 그대로 판정된다.
    """
    return pose_face_kp_count(pose, conf_threshold) >= min_visible


def pose_face_kp_count(pose: Optional[Pose], conf_threshold: float) -> int:
    """conf 임계 이상인 얼굴 키포인트(코·눈·귀) 개수 — HUD 계측용."""
    if pose is None:
        return 0
    return sum(1 for n in _FACE_KP if pose.keypoints[n][2] >= conf_threshold)


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


def side_lying_features(pose: Pose) -> tuple[float, float, float]:
    """어깨·엉덩이 4점으로 (shoulder_spread, hip_spread, lr_symmetry) 계산.

    spread는 좌우 키포인트 거리를 torso_length(어깨중심~엉덩이중심)로 정규화한
    비율 — 카메라 거리 불변. prone은 좌우가 벌어져 크고(측정 최저 1.11), 옆누움은
    한쪽이 포개져 작다(측정 최고 0.63). lr_symmetry(좌/우 conf의 min/max)는 측정
    결과 전 자세 0.93~1.00로 판별 불능이라 — pose 모델이 가려진 쪽도 높은 conf로
    추정해버림 — 판정엔 쓰지 않고 진단(diag/HUD)용으로만 반환한다.

    torso_length가 0이면(자세 붕괴/중심 겹침) (0,0,0) 센티넬 — 호출자는 가드를
    발동하지 않는다(안전측: 애매하면 다운그레이드 금지).
    """
    kp = pose.keypoints
    ls, rs = kp["left_shoulder"], kp["right_shoulder"]
    lh, rh = kp["left_hip"], kp["right_hip"]
    scx, scy = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
    hcx, hcy = (lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2
    torso_len = math.hypot(scx - hcx, scy - hcy)
    if torso_len <= 1e-6:
        return (0.0, 0.0, 0.0)
    shoulder_spread = math.hypot(ls[0] - rs[0], ls[1] - rs[1]) / torso_len
    hip_spread = math.hypot(lh[0] - rh[0], lh[1] - rh[1]) / torso_len
    left_conf = ls[2] + lh[2]
    right_conf = rs[2] + rh[2]
    hi = max(left_conf, right_conf)
    lr_symmetry = min(left_conf, right_conf) / hi if hi > 1e-6 else 0.0
    return (shoulder_spread, hip_spread, lr_symmetry)


def clearly_side_lying(pose: Optional[Pose], spread_max: float, min_conf: float) -> bool:
    """명백한 옆누움일 때만 True — DANGER를 안전으로 내리는 가드라 보수적으로.

    조건(AND):
      1. 어깨쌍 또는 엉덩이쌍 중 하나 이상이 좌우 모두 min_conf 이상 — spread를
         신뢰할 수 있는 상태. 천에 덮이면 키포인트가 0.0으로 죽어 여기서 False →
         face_covered(위험) 유지, 천 덮임 보호는 깨지지 않는다.
      2. max(shoulder_spread, hip_spread) <= spread_max — 좁게 포개진 기하.
    pose 없음·센티넬 등 애매하면 False → flipped(위험) 유지.
    """
    if pose is None:
        return False
    kp = pose.keypoints
    judgeable = any(kp[l][2] >= min_conf and kp[r][2] >= min_conf
                    for l, r in _SIDE_KP_PAIRS)
    if not judgeable:
        return False
    shoulder_spread, hip_spread, _ = side_lying_features(pose)
    if shoulder_spread <= 0.0 and hip_spread <= 0.0:
        return False
    return max(shoulder_spread, hip_spread) <= spread_max


@dataclass
class RiskSignal:
    type: str
    confidence: float
    metadata: dict = field(default_factory=dict)


def main_person(persons: list[Person]) -> Optional[Person]:
    if not persons:
        return None
    return max(persons, key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]))


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


def memory_fresh(last_seen: float, now: float, window_s: float) -> bool:
    """타임스탬프 메모리가 아직 유효한가 (last_seen 0 = 한 번도 못 봄)."""
    return last_seen > 0 and (now - last_seen) < window_s


def suffocation_latched(latch_since: float, now: float,
                        face_visible: bool, latch_max_s: float) -> bool:
    """진행 중 질식 래치가 유효한가.

    face_memory/roi_memory는 판정 *시작* 게이트(빈 방·인형 오탐 방지)인데, 만료가
    진행 중인 위험까지 차단해 END(해제)가 오발행되는 결함이 있었다. main 루프는
    위험 활성 시점에 latch_since를 기록하고, 래치가 유효한 동안 두 게이트를
    통과시킨다. 해제 조건: 얼굴 재출현(아기든 들어올린 보호자든 = 상황 종료),
    또는 latch_max_s 초과(얼굴이 끝내 안 잡힌 채 아기를 데려간 극단 케이스에서
    알람이 영원히 안 꺼지는 것 방지 — README 한계로 문서화).
    """
    if latch_since <= 0 or face_visible:
        return False
    return (now - latch_since) < latch_max_s


def presence_sustained(presence_since: float, last_present: float,
                       now: float, entry_s: float, gap_s: float) -> bool:
    """subject가 ROI 안에 연속 entry_s 이상 존재했는가 (gap_s 이내 깜빡임은 연속).

    질식 진입 게이트의 신규 경로 — 입장부터 face를 한 번도 안 보여준 엎드림은
    face_memory 기반 게이트(face_never_seen)가 판정 자체를 막아 미탐이 됐다
    (2026-06-10 카메라 검증). "사람이 ROI 안에 오래 있는데 얼굴이 안 보인다"는
    그 자체로 판정을 시작할 근거가 된다. presence_since 0 = 미존재(센티넬).
    """
    if presence_since <= 0:
        return False
    if (now - last_present) > gap_s:
        return False
    return (now - presence_since) >= entry_s


def detect_suffocation(face_visible_now: bool, entry_ok: bool) -> tuple[bool, dict]:
    """1층 감지: face 미가시 + 진입 게이트 통과 = 위험. 다른 억제 없음.

    분류용 신호(out_of_view·active_motion·side_lying·head/torso)는 감지에
    관여하지 않는다 — 전부 2층(label_suffocation_cause)의 라벨/flag.
    함수가 자명하지만, 감지 조건이 이 둘뿐임을 코드로 고정하는 것이 목적 —
    억제 조건을 추가하려면 이 시그니처를 바꿔야 한다(의도된 마찰). 게이트
    사슬에서 분류 게이트의 오류가 그대로 질식 미탐이 되던 결함(2026-06-10
    카메라 검증, side 가드·climbing 베토)이 재설계 동기.
    """
    diag: dict = {"face_visible": int(face_visible_now), "entry_ok": int(entry_ok)}
    return (not face_visible_now) and entry_ok, diag


def label_suffocation_cause(
    subject_present: bool,
    torso_visible: bool,
    head_present: Optional[bool],
    side_lying: bool,
    motion_level: float,
    motion_threshold: float,
    roi_containment: Optional[float],
    out_of_view_roi_threshold: float,
) -> tuple[str, dict]:
    """2층 라벨: cause는 항상 'flipped' 또는 'face_covered' — unknown 없음(전수).

    감지(detect_suffocation)를 차단하지 않는다. 서버 이벤트가
    PRONE_SUFFOCATION/BLANKET_SUFFOCATION 둘뿐이라 애매해도 둘 중 하나로
    보내고, 신뢰 맥락은 flags(메타데이터)로 싣는다 — "판별 불가면 이벤트
    보류"는 분류 실패가 감지를 죽이는 기존 결함의 재현이라 채택하지 않음.

    cause 결정:
      1. subject 없음 → face_covered (몸·머리까지 완전히 파묻혀 검출 붕괴)
      2. head_present True → flipped (뒤통수 노출=엎드림) / False → face_covered
         (얼굴만 천 덮임 포함 — torso가 살아 있어도 head가 우선)
      3. head_present None(검출기 미사용·예외) → torso 폴백:
         torso_visible(등 노출, 실측 prone 4/4 conf 0.95~0.99) → flipped /
         소실(천 덮임 0/4) → face_covered

    head를 양성 증거로만 쓰는 변형(미검출 시 torso 폴백)은 시도 후 철회 —
    "얼굴만 천"(torso 생존)이 전부 flipped로 빠져 BLANKET 케이스를 사실상
    포기하게 된다(2026-06-13). 잔존 비용: head 검출기 간헐 맹점 동안 prone이
    face_covered로 라벨될 수 있음 — head_memory_s(충전 윈도우 전체)로 완화.

    flags는 표시·진단용 메타데이터일 뿐이다 — 기존 구조에선 이들이 위험을
    차단하는 게이트였고 그 오류가 그대로 질식 미탐이 됐다(2026-06-10).
    side_lying은 호출자가 clearly_side_lying으로 계산해 주입(가드 비활성 시
    상시 False). roi_containment None(계산 불가)은 out_of_view False.
    """
    diag: dict = {
        "subject": int(subject_present),
        "torso": int(torso_visible),
        "head": ("fallback" if head_present is None else int(head_present)),
        "out_of_view": (roi_containment is not None
                        and roi_containment < out_of_view_roi_threshold),
        "active_motion": motion_level >= motion_threshold,
        "side_lying": side_lying,
        "motion": round(motion_level, 3),
        "roi_in": (round(roi_containment, 2) if roi_containment is not None else None),
    }
    if not subject_present:
        return "face_covered", diag
    is_prone = torso_visible if head_present is None else head_present
    return ("flipped" if is_prone else "face_covered"), diag
