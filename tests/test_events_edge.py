from events.edge import transition


def test_idle_returns_none():
    states: dict = {}
    sig = transition("FALL", False, states, now=10.0, confidence=0.0, metadata={})
    assert sig is None
    assert states == {}


def test_first_active_returns_start():
    states: dict = {}
    sig = transition("FALL", True, states, now=10.0, confidence=0.85, metadata={"k": "v"})
    assert sig is not None
    assert sig.metadata["phase"] == "START"
    assert sig.metadata["started_at"] == 10.0
    assert sig.metadata["duration_s"] == 0.0
    assert sig.metadata["k"] == "v"
    assert sig.confidence == 0.85
    assert "FALL" in states


def test_continued_active_returns_none():
    states: dict = {}
    transition("FALL", True, states, 10.0, 0.5, {})
    sig = transition("FALL", True, states, 11.0, 0.7, {})
    assert sig is None


def test_continued_active_keeps_max_confidence():
    states: dict = {}
    transition("FALL", True, states, 10.0, 0.5, {})
    transition("FALL", True, states, 11.0, 0.9, {})
    transition("FALL", True, states, 12.0, 0.7, {})
    sig = transition("FALL", False, states, 13.0, 0.0, {})
    assert sig.confidence == 0.9


def test_active_to_inactive_returns_end():
    states: dict = {}
    transition("FALL", True, states, 10.0, 0.85, {"cause": "x"})
    sig = transition("FALL", False, states, now=15.0, confidence=0.0, metadata={})
    assert sig is not None
    assert sig.metadata["phase"] == "END"
    assert sig.metadata["started_at"] == 10.0
    assert sig.metadata["ended_at"] == 15.0
    assert sig.metadata["duration_s"] == 5.0
    assert sig.metadata["cause"] == "x"
    assert "FALL" not in states


def test_multiple_event_types_independent():
    states: dict = {}
    s1 = transition("FALL", True, states, 10.0, 0.8, {})
    s2 = transition("CRYING", True, states, 11.0, 0.7, {})
    assert s1 is not None and s1.metadata["phase"] == "START"
    assert s2 is not None and s2.metadata["phase"] == "START"
    assert "FALL" in states and "CRYING" in states


def test_restart_after_end():
    states: dict = {}
    transition("FALL", True, states, 10.0, 0.8, {})
    transition("FALL", False, states, 12.0, 0.0, {})
    sig = transition("FALL", True, states, 20.0, 0.6, {})
    assert sig.metadata["phase"] == "START"
    assert sig.metadata["started_at"] == 20.0
