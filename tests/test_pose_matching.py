from vision.person import Person
from vision.pose import Pose, iou, match_pose_to_person


def _pose(bbox):
    kps = {
        name: (0.0, 0.0, 0.0) for name in [
            "nose","left_eye","right_eye","left_ear","right_ear",
            "left_shoulder","right_shoulder","left_elbow","right_elbow",
            "left_wrist","right_wrist","left_hip","right_hip",
            "left_knee","right_knee","left_ankle","right_ankle",
        ]
    }
    return Pose(bbox=bbox, keypoints=kps)


def test_iou_identical_boxes():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_boxes():
    assert iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0


def test_iou_half_overlap():
    # (0,0,10,10) ∩ (5,0,15,10) = 5*10 = 50, union = 100+100-50 = 150
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 50/150) < 1e-6


def test_match_returns_highest_iou_pose():
    person = Person(bbox=(0, 0, 10, 10), confidence=0.9)
    poses = [_pose((100, 100, 110, 110)), _pose((1, 0, 11, 10))]
    matched = match_pose_to_person(person, poses)
    assert matched is poses[1]


def test_match_returns_none_when_no_overlap():
    person = Person(bbox=(0, 0, 10, 10), confidence=0.9)
    poses = [_pose((100, 100, 110, 110))]
    assert match_pose_to_person(person, poses) is None


def test_match_returns_none_when_no_poses():
    person = Person(bbox=(0, 0, 10, 10), confidence=0.9)
    assert match_pose_to_person(person, []) is None
