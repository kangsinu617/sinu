from audio.yamnet_classifier import AudioClassifier

CFG = {
    "sample_rate": 16000,
    "chunk_duration_s": 0.96,
    "window_chunks": 3,
    "score_threshold": 0.3,
    "min_duration_s": 1.0,
    "whimper_score_threshold": 0.25,
    "whimper_min_duration_s": 2.0,
}


def test_window_mean_single_chunk():
    clf = AudioClassifier(CFG)
    buf: list[float] = []
    mean = clf._window_mean(buf, 0.5)
    assert mean == 0.5


def test_window_mean_averages_three_chunks():
    clf = AudioClassifier(CFG)
    buf: list[float] = []
    clf._window_mean(buf, 0.2)
    clf._window_mean(buf, 0.4)
    mean = clf._window_mean(buf, 0.6)
    assert abs(mean - 0.4) < 1e-6


def test_window_capped_at_window_chunks():
    # 5청크 추가, 마지막 3개만 유효: [0.0, 0.0, 0.9] → mean = 0.3
    clf = AudioClassifier(CFG)
    buf: list[float] = []
    for _ in range(5):
        clf._window_mean(buf, 0.0)
    mean = clf._window_mean(buf, 0.9)
    assert abs(mean - 0.3) < 1e-6


def test_window_below_threshold_cry_inactive():
    clf = AudioClassifier(CFG)
    buf: list[float] = []
    clf._window_mean(buf, 0.1)
    clf._window_mean(buf, 0.1)
    clf._update_state(clf._window_mean(buf, 0.1), 0.0)
    active, score, _, _ = clf.get_state()
    assert active is False
    assert score < 0.3


def test_window_above_threshold_cry_active():
    clf = AudioClassifier(CFG)
    buf: list[float] = []
    clf._window_mean(buf, 0.5)
    clf._window_mean(buf, 0.5)
    clf._update_state(clf._window_mean(buf, 0.5), 0.0)
    active, score, _, _ = clf.get_state()
    assert active is True
    assert score >= 0.3


def test_get_state_initial():
    clf = AudioClassifier(CFG)
    active, score, whimper_active, whimper_score = clf.get_state()
    assert active is False
    assert score == 0.0
    assert whimper_active is False
    assert whimper_score == 0.0


def test_whimper_above_threshold():
    clf = AudioClassifier(CFG)
    buf: list[float] = []
    clf._window_mean(buf, 0.3)
    clf._window_mean(buf, 0.3)
    clf._update_state(0.0, clf._window_mean(buf, 0.3))
    _, _, whimper_active, whimper_score = clf.get_state()
    assert whimper_active is True
    assert whimper_score >= 0.25


def test_whimper_below_threshold():
    clf = AudioClassifier(CFG)
    buf: list[float] = []
    clf._window_mean(buf, 0.1)
    clf._window_mean(buf, 0.1)
    clf._update_state(0.0, clf._window_mean(buf, 0.1))
    _, _, whimper_active, whimper_score = clf.get_state()
    assert whimper_active is False
    assert whimper_score < 0.25
