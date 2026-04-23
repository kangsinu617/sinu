from vision.tracker import DurationTracker


def test_false_before_required_elapsed():
    t = DurationTracker(required_duration_s=1.0, grace_s=0.5)
    assert t.update(True, 100.0) is False
    assert t.update(True, 100.5) is False


def test_true_after_required_elapsed():
    t = DurationTracker(required_duration_s=1.0, grace_s=0.5)
    t.update(True, 100.0)
    assert t.update(True, 101.0) is True


def test_sustained_true_stays_true():
    t = DurationTracker(required_duration_s=1.0, grace_s=0.5)
    t.update(True, 100.0)
    t.update(True, 101.0)
    assert t.update(True, 101.5) is True


def test_grace_tolerates_brief_false():
    t = DurationTracker(required_duration_s=1.0, grace_s=0.5)
    t.update(True, 100.0)
    t.update(True, 101.0)
    # 0.3s 동안 False, grace 이내 → 상태 유지
    assert t.update(False, 101.3) is True


def test_grace_exceeded_resets():
    t = DurationTracker(required_duration_s=1.0, grace_s=0.5)
    t.update(True, 100.0)
    t.update(True, 101.0)
    # 0.6s 동안 False, grace 초과 → 리셋
    assert t.update(False, 101.6) is False
    # 재시작: 첫 True에서 required 미충족이라 False
    assert t.update(True, 101.7) is False


def test_elapsed_reports_ongoing_time():
    t = DurationTracker(required_duration_s=1.0, grace_s=0.5)
    t.update(True, 100.0)
    assert abs(t.elapsed(100.7) - 0.7) < 1e-9
