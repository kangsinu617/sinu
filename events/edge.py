"""이벤트 START/END 전이 감지.

DurationTracker 결과(active 여부)와 이전 상태를 비교해 페이로드 한 번씩만 발행.
- False → True : START (started_at 기록)
- True  → True : 진행 중 (confidence 최댓값 갱신)
- True  → False: END (started_at, ended_at, duration 산출)
- False → False: IDLE (None)
"""
from typing import Optional

from vision.heuristics import RiskSignal


def transition(
    event_type: str,
    active: bool,
    states: dict,
    now: float,
    confidence: float,
    metadata: dict,
) -> Optional[RiskSignal]:
    info = states.get(event_type)
    if active and info is None:
        states[event_type] = {
            "started_at": now,
            "confidence": confidence,
            "metadata": dict(metadata),
        }
        return RiskSignal(event_type, confidence, {
            "phase": "START",
            "started_at": now,
            "duration_s": 0.0,
            **metadata,
        })
    if active:
        info["confidence"] = max(info["confidence"], confidence)
        return None
    if info is not None:
        states.pop(event_type)
        return RiskSignal(event_type, info["confidence"], {
            "phase": "END",
            "started_at": info["started_at"],
            "ended_at": now,
            "duration_s": now - info["started_at"],
            **info["metadata"],
        })
    return None
