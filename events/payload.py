"""RiskSignal → 서버 MQTT 페이로드 변환.

옵션 A: 시작/끝 두 번 publish.
- START : phase, startedAt, duration=0
- END   : phase, startedAt, endedAt, duration
"""
from datetime import datetime, timezone
from typing import Optional

from vision.heuristics import RiskSignal

EVENT_TYPE_MAP: dict = {
    "climbing_risk": "CLIMBING",
    "fall_risk": "FALL",
    "roi_exit_risk": "ROI_EXIT",
    "cry_detected": "CRYING",
    "babble_detected": "WHINING",
}

SUFFOCATION_CAUSE_MAP: dict = {
    "face_covered": "BLANKET_SUFFOCATION",
    "flipped": "PRONE_SUFFOCATION",
}

SEVERITY_MAP: dict = {
    "BLANKET_SUFFOCATION": "DANGER",
    "PRONE_SUFFOCATION": "DANGER",
    "FALL": "DANGER",
    "CLIMBING": "CAUTION",
    "CRYING": "CAUTION",
    "ROI_EXIT": "CAUTION",
    "WHINING": "INFO",
}


def _to_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_event_type(signal: RiskSignal) -> Optional[str]:
    if signal.type == "suffocation_risk":
        return SUFFOCATION_CAUSE_MAP.get(signal.metadata.get("cause"))
    return EVENT_TYPE_MAP.get(signal.type)


def build_payload(signal: RiskSignal, device_serial: str) -> Optional[dict]:
    event_type = _resolve_event_type(signal)
    if event_type is None:
        return None
    severity = SEVERITY_MAP.get(event_type)
    if severity is None:
        return None
    md = signal.metadata
    payload: dict = {
        "deviceSerial": device_serial,
        "eventType": event_type,
        "severity": severity,
        "confidence": round(float(signal.confidence), 2),
        "duration": int(round(md.get("duration_s", 0))),
        "snapshotUrl": "",
        "videoUrl": "",
    }
    phase = md.get("phase")
    if phase:
        payload["phase"] = phase
    started_at = md.get("started_at")
    if started_at is not None:
        payload["startedAt"] = _to_iso(started_at)
    ended_at = md.get("ended_at")
    if ended_at is not None:
        payload["endedAt"] = _to_iso(ended_at)
    return payload
