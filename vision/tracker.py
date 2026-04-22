"""지속 시간 상태 머신."""
from typing import Optional


class DurationTracker:
    """조건이 required_duration_s 이상 지속되는 동안 True 반환."""

    def __init__(self, required_duration_s: float) -> None:
        self.required = required_duration_s
        self.start_ts: Optional[float] = None

    def update(self, condition: bool, now: float) -> bool:
        if not condition:
            self.start_ts = None
            return False
        if self.start_ts is None:
            self.start_ts = now
            return False
        return now - self.start_ts >= self.required

    def elapsed(self, now: float) -> float:
        return 0.0 if self.start_ts is None else now - self.start_ts
