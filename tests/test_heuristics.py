import itertools

from vision.heuristics import evaluate_fall, memory_fresh, suffocation_latched, presence_sustained, side_lying_features, clearly_side_lying, detect_suffocation, label_suffocation_cause

SAFE_POLY = [(0, 0), (100, 0), (100, 100), (0, 100)]  # TL,TR,BR,BL


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


from vision.heuristics import pose_torso_visible


def test_pose_torso_visible_prone():
    # 엎드림: 어깨·엉덩이 4/4 높은 conf(측정 0.95~0.99) → True
    pose = _pose(
        left_shoulder=(10.0, 100.0, 0.99), right_shoulder=(20.0, 100.0, 0.98),
        left_hip=(10.0, 200.0, 0.97), right_hip=(20.0, 200.0, 0.95))
    assert pose_torso_visible(pose, 0.5, 2) is True


def test_pose_torso_visible_covered():
    # 천 덮임: 몸통 키포인트 0/4(측정 0.0) → False (face_covered로 분류)
    pose = _pose(
        left_shoulder=(0.0, 0.0, 0.0), right_shoulder=(0.0, 0.0, 0.0),
        left_hip=(0.0, 0.0, 0.0), right_hip=(0.0, 0.0, 0.0))
    assert pose_torso_visible(pose, 0.5, 2) is False


def test_pose_torso_visible_partial_side():
    # 측면 누움: 한쪽 어깨·엉덩이만 보임(2/4) → min_visible=2면 True
    pose = _pose(
        left_shoulder=(10.0, 100.0, 0.9), right_shoulder=(20.0, 100.0, 0.1),
        left_hip=(10.0, 200.0, 0.9), right_hip=(20.0, 200.0, 0.1))
    assert pose_torso_visible(pose, 0.5, 2) is True


def test_pose_torso_visible_none():
    assert pose_torso_visible(None, 0.5, 2) is False



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



# ---------- memory_fresh / suffocation_latched (시간 헬퍼) ----------

def test_memory_fresh_within_window():
    assert memory_fresh(100.0, 105.0, 30.0) is True


def test_memory_fresh_expired():
    assert memory_fresh(100.0, 130.0, 30.0) is False


def test_memory_fresh_never_seen():
    assert memory_fresh(0.0, 50.0, 30.0) is False


def test_latch_keeps_alarm_past_memory_expiry():
    # 회귀 고정: 덮임 t=100에 래치 ON → t=135(face_memory 30s 만료 후)에도
    # 래치가 살아 있어야 진행 중 질식의 END 오발행이 막힌다.
    assert suffocation_latched(100.0, 135.0, False, 300.0) is True


def test_latch_releases_on_face_visible():
    assert suffocation_latched(100.0, 110.0, True, 300.0) is False


def test_latch_releases_after_cap():
    assert suffocation_latched(100.0, 400.0, False, 300.0) is False


def test_latch_off_when_never_set():
    assert suffocation_latched(0.0, 50.0, False, 300.0) is False


# ---------- presence_sustained (진입 게이트: ROI 연속 존재) ----------

def test_presence_never_present():
    # presence_since 0 = 한 번도 ROI 안에서 본 적 없음
    assert presence_sustained(0.0, 0.0, 100.0, 10.0, 2.0) is False


def test_presence_entry_elapsed():
    # 100s부터 연속 존재, 지금 110.5s = 10s 경과 → 통과
    assert presence_sustained(100.0, 110.5, 110.5, 10.0, 2.0) is True


def test_presence_not_yet_entry():
    # 5s밖에 안 됨 → 아직
    assert presence_sustained(100.0, 105.0, 105.0, 10.0, 2.0) is False


def test_presence_gap_within_tolerance():
    # 마지막 목격이 1.5s 전 — gap 2s 이내라 연속 유지, 누적 11.5s → 통과
    assert presence_sustained(100.0, 110.0, 111.5, 10.0, 2.0) is True


def test_presence_gap_exceeded():
    # 마지막 목격이 3s 전 — gap 2s 초과라 연속 끊김 → False
    assert presence_sustained(100.0, 110.0, 113.0, 10.0, 2.0) is False


def test_presence_exact_boundary():
    # 경과 정확히 entry_s = 포함(>=)
    assert presence_sustained(100.0, 110.0, 110.0, 10.0, 2.0) is True


def test_presence_gap_exact_boundary():
    # 마지막 목격이 정확히 gap_s(2s) 전 — strict 초과(>)만 끊김이라 연속 유지
    assert presence_sustained(100.0, 110.0, 112.0, 10.0, 2.0) is True


# ---------- detect_suffocation (1층 감지 — 2입력 진리표 전수) ----------

def test_detect_truth_table():
    # 감지 조건은 (NOT face_visible) AND entry_ok 단 둘 — 진리표 전수
    assert detect_suffocation(face_visible_now=False, entry_ok=True)[0] is True
    assert detect_suffocation(face_visible_now=True, entry_ok=True)[0] is False
    assert detect_suffocation(face_visible_now=False, entry_ok=False)[0] is False
    assert detect_suffocation(face_visible_now=True, entry_ok=False)[0] is False


def test_detect_diag_records_inputs():
    _, diag = detect_suffocation(face_visible_now=False, entry_ok=True)
    assert diag == {"face_visible": 0, "entry_ok": 1}


# ---------- label_suffocation_cause (2층 라벨 — 전수성·폴백·flags) ----------

def _label(subject=True, torso=True, head=None, side=False,
           motion=0.0, motion_thr=0.02, roi_in=1.0, oov_thr=0.72):
    return label_suffocation_cause(subject, torso, head, side,
                                   motion, motion_thr, roi_in, oov_thr)


def test_label_totality_always_binary():
    # 어떤 입력 조합에도 cause는 flipped/face_covered 둘 중 하나 — unknown 없음
    for subject, torso in itertools.product([True, False], repeat=2):
        for head in (True, False, None):
            for side in (True, False):
                for roi_in in (None, 0.3, 0.9):
                    cause, _ = _label(subject=subject, torso=torso, head=head,
                                      side=side, motion=0.5, roi_in=roi_in)
                    assert cause in ("flipped", "face_covered")


def test_label_no_subject_is_face_covered():
    # 완전 파묻힘(검출 붕괴) — head 값과 무관하게 face_covered
    assert _label(subject=False, head=True)[0] == "face_covered"


def test_label_head_present_is_flipped():
    assert _label(head=True, torso=False)[0] == "flipped"


def test_label_head_absent_is_face_covered():
    # 얼굴만 천 — torso 살아 있어도 head 미검출이면 face_covered.
    # (미검출 시 torso 폴백 변형은 BLANKET 케이스를 포기하게 돼 철회 — 2026-06-13)
    assert _label(head=False, torso=True)[0] == "face_covered"
    assert _label(head=False, torso=False)[0] == "face_covered"


def test_label_head_none_falls_back_to_torso():
    assert _label(head=None, torso=True)[0] == "flipped"
    assert _label(head=None, torso=False)[0] == "face_covered"


def test_label_flags_do_not_change_cause():
    # 재설계 핵심: out_of_view/active_motion/side_lying이 전부 발동해도
    # cause는 그대로 — flag는 메타데이터일 뿐
    cause, diag = _label(head=True, side=True, motion=0.5, roi_in=0.3)
    assert cause == "flipped"
    assert diag["out_of_view"] is True
    assert diag["active_motion"] is True
    assert diag["side_lying"] is True


def test_label_flags_off_when_below_thresholds():
    _, diag = _label(side=False, motion=0.001, roi_in=0.9)
    assert diag["out_of_view"] is False
    assert diag["active_motion"] is False
    assert diag["side_lying"] is False


def test_label_out_of_view_none_roi_is_false():
    # roi_containment 계산 불가(None) → flag False (애매하면 표시하지 않음)
    _, diag = _label(roi_in=None)
    assert diag["out_of_view"] is False


# ---------- 재설계 회귀 시나리오 (구 단층 판정 시나리오 보존) ----------

def test_regression_blanket_full_cover():
    # 이불 전체 덮임: subject는 잡히되 torso 소실, head 미검출 → face_covered
    active, _ = detect_suffocation(face_visible_now=False, entry_ok=True)
    cause, _ = _label(subject=True, torso=False, head=False)
    assert active is True and cause == "face_covered"


def test_regression_buried_subject_lost():
    # 완전 파묻힘(검출 붕괴): subject 없음 → 감지 유지 + face_covered
    active, _ = detect_suffocation(face_visible_now=False, entry_ok=True)
    cause, _ = _label(subject=False)
    assert active is True and cause == "face_covered"


def test_regression_prone_with_head():
    # 엎드림 + head(뒤통수) 검출 → flipped
    active, _ = detect_suffocation(face_visible_now=False, entry_ok=True)
    cause, _ = _label(subject=True, torso=True, head=True)
    assert active is True and cause == "flipped"


def test_regression_face_reappears_releases():
    # 얼굴 재출현 → 감지 즉시 해제 (유일한 억제 조건)
    active, _ = detect_suffocation(face_visible_now=True, entry_ok=True)
    assert active is False


def test_regression_prone_survives_all_old_gates():
    # 재설계 핵심: 구 게이트 신호(out_of_view·side·motion)가 전부 "차단" 방향
    # 값이어도 감지는 활성 유지 — flag로만 기록된다
    active, _ = detect_suffocation(face_visible_now=False, entry_ok=True)
    cause, diag = _label(subject=True, torso=True, head=True,
                         side=True, motion=0.5, roi_in=0.3)
    assert active is True
    assert cause == "flipped"
    assert (diag["out_of_view"], diag["active_motion"], diag["side_lying"]) \
        == (True, True, True)


def test_regression_empty_room_entry_gate():
    # 빈 방·인형 오탐 방지: 진입 게이트 미통과면 face가 안 보여도 감지 없음
    active, _ = detect_suffocation(face_visible_now=False, entry_ok=False)
    assert active is False


# ---------- side_lying_features / clearly_side_lying (옆누움 가드) ----------

def test_side_lying_features_wide_symmetric():
    # torso_len = |어깨중심(50,100) - 엉덩이중심(50,200)| = 100
    pose = _pose(
        left_shoulder=(10.0, 100.0, 0.9), right_shoulder=(90.0, 100.0, 0.9),
        left_hip=(20.0, 200.0, 0.9), right_hip=(80.0, 200.0, 0.9),
    )
    ss, hs, sym = side_lying_features(pose)
    assert abs(ss - 0.8) < 1e-6   # 80px / 100px
    assert abs(hs - 0.6) < 1e-6   # 60px / 100px
    assert abs(sym - 1.0) < 1e-6


def test_side_lying_features_degenerate_returns_sentinel():
    # 어깨중심 == 엉덩이중심 → torso_len 0 → (0,0,0) 센티넬
    pose = _pose(
        left_shoulder=(10.0, 100.0, 0.9), right_shoulder=(20.0, 100.0, 0.9),
        left_hip=(10.0, 100.0, 0.9), right_hip=(20.0, 100.0, 0.9),
    )
    assert side_lying_features(pose) == (0.0, 0.0, 0.0)


def test_clearly_side_lying_narrow_is_side():
    # 측정 옆누움(최고 0.63) 영역: ss=0.5, hs=0.4 ≤ 0.8 → 안전 다운그레이드
    pose = _pose(
        left_shoulder=(45.0, 100.0, 0.9), right_shoulder=(95.0, 100.0, 0.9),
        left_hip=(50.0, 200.0, 0.9), right_hip=(90.0, 200.0, 0.9),
    )
    assert clearly_side_lying(pose, spread_max=0.8, min_conf=0.5) is True


def test_clearly_side_lying_wide_prone_is_not_side():
    # 측정 prone(최저 1.11) 영역: ss=1.2 > 0.8 → flipped 유지
    pose = _pose(
        left_shoulder=(0.0, 100.0, 0.9), right_shoulder=(120.0, 100.0, 0.9),
        left_hip=(10.0, 200.0, 0.9), right_hip=(110.0, 200.0, 0.9),
    )
    assert clearly_side_lying(pose, spread_max=0.8, min_conf=0.5) is False


def test_clearly_side_lying_blanket_dead_kp_is_not_side():
    # 안전상 핵심: 천 덮임은 몸통 conf가 죽어 판단 불가 → 가드 미발동 →
    # face_covered(위험) 유지. 천 덮임 보호가 깨지지 않음을 고정.
    pose = _pose(
        left_shoulder=(10.0, 100.0, 0.0), right_shoulder=(20.0, 100.0, 0.0),
        left_hip=(10.0, 200.0, 0.0), right_hip=(20.0, 200.0, 0.0),
    )
    assert clearly_side_lying(pose, spread_max=0.8, min_conf=0.5) is False


def test_clearly_side_lying_none_pose_is_not_side():
    assert clearly_side_lying(None, spread_max=0.8, min_conf=0.5) is False


def test_clearly_side_lying_degenerate_is_not_side():
    # 센티넬 기하(애매) → 다운그레이드 금지
    pose = _pose(
        left_shoulder=(10.0, 100.0, 0.9), right_shoulder=(20.0, 100.0, 0.9),
        left_hip=(10.0, 100.0, 0.9), right_hip=(20.0, 100.0, 0.9),
    )
    assert clearly_side_lying(pose, spread_max=0.8, min_conf=0.5) is False


def test_clearly_side_lying_one_pair_alive_narrow_is_side():
    # 엉덩이쌍만 신뢰 가능해도 spread가 좁으면 판단 가능 (ss≈0.2, hs≈0.3)
    pose = _pose(
        left_shoulder=(40.0, 100.0, 0.2), right_shoulder=(60.0, 100.0, 0.2),
        left_hip=(40.0, 200.0, 0.9), right_hip=(70.0, 200.0, 0.9),
    )
    assert clearly_side_lying(pose, spread_max=0.8, min_conf=0.5) is True


