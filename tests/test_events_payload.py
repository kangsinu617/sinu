from events.payload import build_payload
from vision.heuristics import RiskSignal


def _signal(type_: str, confidence: float = 0.85, **metadata) -> RiskSignal:
    return RiskSignal(type_, confidence, metadata)


# epoch float — 2026-05-03T10:00:00Z UTC
START_TS = 1777802400.0
END_TS = 1777802405.0


def test_suffocation_face_covered_maps_to_blanket():
    sig = _signal("suffocation_risk", cause="face_covered")
    payload = build_payload(sig, "DEV-001")
    assert payload["eventType"] == "BLANKET_SUFFOCATION"
    assert payload["severity"] == "DANGER"


def test_suffocation_flipped_maps_to_prone():
    sig = _signal("suffocation_risk", cause="flipped")
    payload = build_payload(sig, "DEV-001")
    assert payload["eventType"] == "PRONE_SUFFOCATION"
    assert payload["severity"] == "DANGER"


def test_climbing_maps_to_caution():
    payload = build_payload(_signal("climbing_risk"), "DEV-001")
    assert payload["eventType"] == "CLIMBING"
    assert payload["severity"] == "CAUTION"


def test_fall_maps_to_danger():
    payload = build_payload(_signal("fall_risk"), "DEV-001")
    assert payload["eventType"] == "FALL"
    assert payload["severity"] == "DANGER"


def test_roi_exit_maps_to_caution():
    payload = build_payload(_signal("roi_exit_risk"), "DEV-001")
    assert payload["eventType"] == "ROI_EXIT"
    assert payload["severity"] == "CAUTION"


def test_cry_maps_to_caution():
    payload = build_payload(_signal("cry_detected"), "DEV-001")
    assert payload["eventType"] == "CRYING"
    assert payload["severity"] == "CAUTION"


def test_unknown_event_returns_none():
    payload = build_payload(_signal("scream_detected"), "DEV-001")
    assert payload is None


def test_duration_rounded_to_int_seconds():
    sig = _signal("fall_risk", duration_s=4.7)
    payload = build_payload(sig, "DEV-001")
    assert payload["duration"] == 5
    assert isinstance(payload["duration"], int)


def test_payload_has_required_fields():
    sig = _signal("cry_detected", confidence=0.72)
    payload = build_payload(sig, "DEV-001")
    assert payload["deviceSerial"] == "DEV-001"
    assert payload["confidence"] == 0.72
    assert payload["snapshotUrl"] == ""
    assert payload["videoUrl"] == ""


def test_start_phase_includes_started_at_only():
    sig = _signal("fall_risk", phase="START", started_at=START_TS, duration_s=0.0)
    payload = build_payload(sig, "DEV-001")
    assert payload["phase"] == "START"
    assert payload["startedAt"] == "2026-05-03T10:00:00Z"
    assert "endedAt" not in payload
    assert payload["duration"] == 0


def test_end_phase_includes_started_and_ended():
    sig = _signal(
        "fall_risk", phase="END",
        started_at=START_TS, ended_at=END_TS, duration_s=5.0,
    )
    payload = build_payload(sig, "DEV-001")
    assert payload["phase"] == "END"
    assert payload["startedAt"] == "2026-05-03T10:00:00Z"
    assert payload["endedAt"] == "2026-05-03T10:00:05Z"
    assert payload["duration"] == 5


def test_no_phase_means_no_phase_field():
    sig = _signal("fall_risk")
    payload = build_payload(sig, "DEV-001")
    assert "phase" not in payload
    assert "startedAt" not in payload
    assert "endedAt" not in payload
