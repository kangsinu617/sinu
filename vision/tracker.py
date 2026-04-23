"""지속 시간 상태 머신. 조건 False 튐을 grace_s 이내에서 무시."""
from typing import Optional


class DurationTracker:
    """조건이 required_duration_s 이상 지속되는 동안 True 반환.

    grace_s: 조건이 잠깐 False가 되어도 이 시간 이내면 누적 카운트 유지.
    """

    def __init__(self, required_duration_s: float, grace_s: float = 0.0) -> None:
        self.required = required_duration_s
        self.grace = grace_s
        self.start_ts: Optional[float] = None
        self.last_true_ts: Optional[float] = None

    def update(self, condition: bool, now: float) -> bool:
        if condition:
            if self.start_ts is None:
                self.start_ts = now
            self.last_true_ts = now
            return now - self.start_ts >= self.required
        if self.last_true_ts is None:
            return False
        if now - self.last_true_ts > self.grace:
            self.start_ts = None
            self.last_true_ts = None
            return False
        return now - self.start_ts >= self.required

    def elapsed(self, now: float) -> float:
        return 0.0 if self.start_ts is None else now - self.start_ts
