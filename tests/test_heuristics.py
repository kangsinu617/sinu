from vision.heuristics import evaluate_roi_exit


def test_roi_exit_none_center():
    active, diag = evaluate_roi_exit(None, (0, 0, 100, 100))
    assert active is False
    assert diag["block"] == "no_center"


def test_roi_exit_inside():
    active, diag = evaluate_roi_exit((50.0, 50.0), (0, 0, 100, 100))
    assert active is False
    assert diag["block"] == "inside_roi"


def test_roi_exit_outside():
    active, diag = evaluate_roi_exit((150.0, 50.0), (0, 0, 100, 100))
    assert active is True
    assert diag["center"] == (150, 50)


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


CLIMB_ROI = (0, 250, 100, 320)


def test_climbing_pose_none():
    active, diag = evaluate_climbing(None, None, CLIMB_ROI, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "no_pose"


def test_climbing_ankle_outside_roi():
    active, diag = evaluate_climbing((500.0, 500.0), _pose(), CLIMB_ROI, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "ankle_outside_roi"


def test_climbing_shoulders_invisible():
    pose = _pose(
        left_shoulder=(10.0, 100.0, 0.1),
        right_shoulder=(20.0, 100.0, 0.1),
    )
    active, diag = evaluate_climbing((15.0, 280.0), pose, CLIMB_ROI, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "shoulder_or_hip_invisible"


def test_climbing_margin_too_small():
    # shoulder y = 180, hip y = 190 → margin 10 < 20
    pose = _pose(
        left_shoulder=(10.0, 180.0, 0.9), right_shoulder=(20.0, 180.0, 0.9),
        left_hip=(10.0, 190.0, 0.9), right_hip=(20.0, 190.0, 0.9),
    )
    active, diag = evaluate_climbing((15.0, 280.0), pose, CLIMB_ROI, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "not_standing"


def test_climbing_all_conditions_met():
    # shoulder y = 100, hip y = 200 → margin 100 ≥ 20, ankle 안쪽
    active, diag = evaluate_climbing((15.0, 280.0), _pose(), CLIMB_ROI, 0.5, 20.0)
    assert active is True
    assert diag["standing_margin"] == 100.0


def test_climbing_no_ankle():
    active, diag = evaluate_climbing(None, _pose(), CLIMB_ROI, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "no_ankle"


def test_climbing_hips_invisible():
    pose = _pose(
        left_hip=(10.0, 200.0, 0.1),
        right_hip=(20.0, 200.0, 0.1),
    )
    active, diag = evaluate_climbing((15.0, 280.0), pose, CLIMB_ROI, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "shoulder_or_hip_invisible"


from vision.face import Face
from vision.heuristics import evaluate_suffocation
from vision.person import Person


def _person_at(x1, y1, x2, y2):
    return Person(bbox=(x1, y1, x2, y2), confidence=0.9)


def _face_at(x1, y1, x2, y2):
    return Face(bbox=(x1, y1, x2, y2), confidence=0.9)


def test_suffocation_no_person():
    active, cause, diag = evaluate_suffocation(None, [], None, 0.5, 3, 0)
    assert active is False
    assert cause is None


def test_suffocation_face_detected_inside_person():
    person = _person_at(0, 0, 100, 200)
    face = _face_at(20, 20, 60, 60)
    active, cause, diag = evaluate_suffocation(person, [face], None, 0.5, 3, 0)
    assert active is False
    assert cause is None


def test_suffocation_no_face_all_keypoints_visible_is_flipped():
    person = _person_at(0, 0, 100, 200)
    active, cause, diag = evaluate_suffocation(person, [], _pose(), 0.5, 3, 0)
    assert active is True
    assert cause == "flipped"
    assert diag["visible_keypoints"] == 4


def test_suffocation_no_face_no_pose_is_blanket():
    person = _person_at(0, 0, 100, 200)
    active, cause, diag = evaluate_suffocation(person, [], None, 0.5, 3, 0)
    assert active is True
    assert cause == "blanket"
    assert diag["visible_keypoints"] == 0


def test_suffocation_no_face_partial_keypoints_is_unknown():
    person = _person_at(0, 0, 100, 200)
    # 4개 중 2개만 보임 (left_hip, right_hip 가림)
    pose = _pose(
        left_hip=(10.0, 200.0, 0.1),
        right_hip=(20.0, 200.0, 0.1),
    )
    active, cause, diag = evaluate_suffocation(person, [], pose, 0.5, 3, 0)
    assert active is True
    assert cause == "unknown"
    assert diag["visible_keypoints"] == 2
