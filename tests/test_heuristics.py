from vision.heuristics import evaluate_fall, evaluate_roi_exit

SAFE_POLY = [(0, 0), (100, 0), (100, 100), (0, 100)]  # TL,TR,BR,BL


def test_roi_exit_none_center():
    active, diag = evaluate_roi_exit(None, SAFE_POLY)
    assert active is False
    assert diag["block"] == "no_center"


def test_roi_exit_inside():
    active, diag = evaluate_roi_exit((50.0, 50.0), SAFE_POLY)
    assert active is False
    assert diag["block"] == "inside_polygon"


def test_roi_exit_outside():
    active, diag = evaluate_roi_exit((150.0, 50.0), SAFE_POLY)
    assert active is True


from vision.heuristics import evaluate_climbing
from vision.pose import Pose


def _pose(**overrides):
    base = {
        "nose": (0.0, 0.0, 0.0), "left_eye": (0.0, 0.0, 0.0), "right_eye": (0.0, 0.0, 0.0),
        "left_ear": (0.0, 0.0, 0.0), "right_ear": (0.0, 0.0, 0.0),
        "left_shoulder": (10.0, 100.0, 0.9), "right_shoulder": (20.0, 100.0, 0.9),
        "left_elbow": (0.0, 0.0, 0.0), "right_elbow": (0.0, 0.0, 0.0),
        "left_wrist": (0.0, 0.0, 0.0), "right_wrist": (0.0, 0.0, 0.0),
        "left_hip": (10.0, 200.0, 0.9), "right_hip": (20.0, 200.0, 0.9),
        "left_knee": (0.0, 0.0, 0.0), "right_knee": (0.0, 0.0, 0.0),
        "left_ankle": (10.0, 300.0, 0.9), "right_ankle": (20.0, 300.0, 0.9),
    }
    base.update(overrides)
    return Pose(bbox=(0, 0, 100, 400), keypoints=base)


# 사람 bbox/포즈 좌표계에 맞춘 안전 폴리곤. 하단 변(y=300 근처)을 난간으로 사용
SAFE_POLY_CLIMB = [(0, 0), (100, 0), (100, 300), (0, 300)]
RAIL_BAND_PX = 40.0


def test_climbing_empty_polygon():
    active, diag = evaluate_climbing((50.0, 290.0), _pose(), [], RAIL_BAND_PX, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "no_polygon"


def test_climbing_pose_none():
    active, diag = evaluate_climbing(None, None, SAFE_POLY_CLIMB, RAIL_BAND_PX, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "no_pose"


def test_climbing_no_wrist():
    active, diag = evaluate_climbing(None, _pose(), SAFE_POLY_CLIMB, RAIL_BAND_PX, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "no_wrist"


def test_climbing_wrist_outside_polygon():
    active, diag = evaluate_climbing((500.0, 500.0), _pose(), SAFE_POLY_CLIMB, RAIL_BAND_PX, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "wrist_outside_polygon"


def test_climbing_wrist_far_from_rail():
    # 폴리곤 내부지만 어느 변과도 band보다 멀리 (중앙 부근)
    active, diag = evaluate_climbing((50.0, 150.0), _pose(), SAFE_POLY_CLIMB, RAIL_BAND_PX, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "not_near_rail"


def test_climbing_shoulders_invisible():
    pose = _pose(left_shoulder=(10.0, 100.0, 0.1), right_shoulder=(20.0, 100.0, 0.1))
    active, diag = evaluate_climbing((50.0, 290.0), pose, SAFE_POLY_CLIMB, RAIL_BAND_PX, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "shoulder_or_hip_invisible"


def test_climbing_hips_invisible():
    pose = _pose(left_hip=(10.0, 200.0, 0.1), right_hip=(20.0, 200.0, 0.1))
    active, diag = evaluate_climbing((50.0, 290.0), pose, SAFE_POLY_CLIMB, RAIL_BAND_PX, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "shoulder_or_hip_invisible"


def test_climbing_margin_too_small():
    # 어깨와 엉덩이 y가 가까움 → 서있지 않음
    pose = _pose(left_hip=(10.0, 105.0, 0.9), right_hip=(20.0, 105.0, 0.9))
    active, diag = evaluate_climbing((50.0, 290.0), pose, SAFE_POLY_CLIMB, RAIL_BAND_PX, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "not_standing"


def test_climbing_all_conditions_met():
    # 폴리곤 내부 + 하단 변(y=300)까지 거리 10 ≤ band + 서있음
    active, diag = evaluate_climbing((50.0, 290.0), _pose(), SAFE_POLY_CLIMB, RAIL_BAND_PX, 0.5, 20.0)
    assert active is True
    assert diag["rail_edge"] == "bottom"


from vision.heuristics import evaluate_suffocation

EDGE_T = 0.044


def test_suffocation_never_in_roi():
    # ROI 안에서 본 적 없으면(빈 방) 판정 안 함
    active, cause, diag = evaluate_suffocation(
        True, 0.12, face_visible_now=False, face_recently_seen=True,
        person_was_in_roi=False, flipped_edge_threshold=EDGE_T)
    assert active is False
    assert cause is None
    assert diag["block"] == "not_in_roi"


def test_suffocation_face_visible_now():
    # 지금 얼굴이 보이면 위험 아님
    active, cause, diag = evaluate_suffocation(
        True, 0.03, face_visible_now=True, face_recently_seen=True,
        person_was_in_roi=True, flipped_edge_threshold=EDGE_T)
    assert active is False
    assert diag["block"] == "face_detected"


def test_suffocation_face_never_seen():
    # face를 최근에 본 적 없으면 오탐 방지
    active, cause, diag = evaluate_suffocation(
        True, 0.12, face_visible_now=False, face_recently_seen=False,
        person_was_in_roi=True, flipped_edge_threshold=EDGE_T)
    assert active is False
    assert diag["block"] == "face_never_seen"


def test_suffocation_prone_high_edge():
    # subject 있음 + 구조 노출(edge ≥ 임계) → 엎드림(flipped)
    active, cause, diag = evaluate_suffocation(
        True, 0.12, face_visible_now=False, face_recently_seen=True,
        person_was_in_roi=True, flipped_edge_threshold=EDGE_T)
    assert active is True
    assert cause == "flipped"


def test_suffocation_covered_low_edge():
    # subject 있음 + 매끈한 천(edge < 임계) → 천에 덮임
    active, cause, diag = evaluate_suffocation(
        True, 0.03, face_visible_now=False, face_recently_seen=True,
        person_was_in_roi=True, flipped_edge_threshold=EDGE_T)
    assert active is True
    assert cause == "face_covered"


def test_suffocation_buried_no_subject():
    # 몸·머리까지 완전히 파묻혀 검출 붕괴 → 얼굴까지 덮임
    active, cause, diag = evaluate_suffocation(
        False, 0.0, face_visible_now=False, face_recently_seen=True,
        person_was_in_roi=True, flipped_edge_threshold=EDGE_T)
    assert active is True
    assert cause == "face_covered"


def test_suffocation_out_of_view_blocks_flipped():
    # flipped 후보(edge ≥ 임계)지만 ROI 포함율 낮음(발만 보임) → 위험 아님
    active, cause, diag = evaluate_suffocation(
        True, 0.12, face_visible_now=False, face_recently_seen=True,
        person_was_in_roi=True, flipped_edge_threshold=EDGE_T,
        roi_containment=0.68, out_of_view_roi_threshold=0.72)
    assert active is False
    assert cause == "out_of_view"
    assert diag["block"] == "out_of_view"


def test_suffocation_prone_in_view_still_fires():
    # flipped 후보 + ROI 포함율 충분(엎드림 정탐) → flipped 유지
    active, cause, diag = evaluate_suffocation(
        True, 0.12, face_visible_now=False, face_recently_seen=True,
        person_was_in_roi=True, flipped_edge_threshold=EDGE_T,
        roi_containment=0.88, out_of_view_roi_threshold=0.72)
    assert active is True
    assert cause == "flipped"


from vision.heuristics import pose_face_visible


def test_pose_face_visible_supine():
    # 천장 보고 누움(supine): 얼굴 키포인트 5/5 높은 conf → True
    pose = _pose(
        nose=(50.0, 10.0, 0.98), left_eye=(45.0, 8.0, 0.95), right_eye=(55.0, 8.0, 0.95),
        left_ear=(40.0, 12.0, 0.93), right_ear=(60.0, 12.0, 0.93))
    assert pose_face_visible(pose, 0.5, 4) is True


def test_pose_face_visible_prone():
    # 엎드림(prone): 얼굴이 매트 향해 1/5, nose 0.08 → False (위험 유지)
    pose = _pose(
        nose=(50.0, 10.0, 0.08), left_eye=(45.0, 8.0, 0.10), right_eye=(55.0, 8.0, 0.05),
        left_ear=(40.0, 12.0, 0.44), right_ear=(60.0, 12.0, 0.20))
    assert pose_face_visible(pose, 0.5, 4) is False


def test_pose_face_visible_none():
    assert pose_face_visible(None, 0.5, 4) is False


def test_suffocation_out_of_view_does_not_affect_face_covered():
    # 가드는 flipped 분기 전용 — edge 낮은 face_covered는 ROI 포함율 낮아도 유지
    active, cause, diag = evaluate_suffocation(
        True, 0.03, face_visible_now=False, face_recently_seen=True,
        person_was_in_roi=True, flipped_edge_threshold=EDGE_T,
        roi_containment=0.50, out_of_view_roi_threshold=0.72)
    assert active is True
    assert cause == "face_covered"


def test_fall_none_center():
    active, diag = evaluate_fall(None, (50.0, 100.0), 200.0)
    assert active is False
    assert diag["block"] == "no_center"


def test_fall_none_prev():
    active, diag = evaluate_fall((50.0, 200.0), None, 200.0)
    assert active is False
    assert diag["block"] == "no_center"


def test_fall_small_drop():
    # 5px 하강 < 200px 임계
    active, diag = evaluate_fall((50.0, 105.0), (50.0, 100.0), 200.0)
    assert active is False
    assert diag["block"] == "drop_too_small"
    assert diag["fall_drop"] == 5.0


def test_fall_large_drop():
    # 250px 하강 >= 200px 임계
    active, diag = evaluate_fall((50.0, 350.0), (50.0, 100.0), 200.0)
    assert active is True
    assert diag["fall_drop"] == 250.0


def test_fall_ascending_ignored():
    # 위로 올라가는 것은 낙상 아님
    active, diag = evaluate_fall((50.0, 70.0), (50.0, 100.0), 200.0)
    assert active is False
    assert diag["block"] == "drop_too_small"
