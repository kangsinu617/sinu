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
