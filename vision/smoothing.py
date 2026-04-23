"""지수이동평균 좌표 스무더."""
from typing import Optional


class EMA:
    """(x, y) 좌표에 지수이동평균을 적용. alpha가 클수록 최신 입력 반영 ↑."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.value: Optional[tuple[float, float]] = None

    def update(self, x: float, y: float) -> tuple[float, float]:
        if self.value is None:
            self.value = (x, y)
        else:
            px, py = self.value
            self.value = (
                self.alpha * x + (1 - self.alpha) * px,
                self.alpha * y + (1 - self.alpha) * py,
            )
        return self.value

    def reset(self) -> None:
        self.value = None
