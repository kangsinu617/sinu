"""폴리곤 ROI 기하 연산 — 순수 함수, 모델 의존 없음.

폴리곤 규약: [TL, TR, BR, BL] 순서의 (x, y) 4점. 변 인덱스:
  0: TL->TR (top), 1: TR->BR (right), 2: BR->BL (bottom), 3: BL->TL (left)
"""
import math
from typing import List, Optional, Tuple

Point = Tuple[float, float]
Quad = List[Point]
EDGE_LABELS = ("top", "right", "bottom", "left")


def point_in_polygon(pt: Point, poly: Quad) -> bool:
    """ray-casting. 표준 반열림 구간 관례: 위/왼 변 위 점은 내부로,
    아래/오른 변 위 점은 외부로 판정된다."""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def point_to_segment_distance(pt: Point, a: Point, b: Point) -> float:
    px, py = pt
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def nearest_edge(pt: Point, poly: Quad) -> Tuple[int, float]:
    """가장 가까운 변의 (인덱스, 거리)."""
    n = len(poly)
    best_idx, best_dist = 0, float("inf")
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        d = point_to_segment_distance(pt, a, b)
        if d < best_dist:
            best_idx, best_dist = i, d
    return best_idx, best_dist


def average_quads(quads: List[Quad]) -> Optional[Quad]:
    """N개 폴리곤의 꼭짓점별 평균. 빈 입력은 None.

    각 quad는 [TL, TR, BR, BL] 순서의 4점이라고 가정한다.
    """
    if not quads:
        return None
    n = len(quads)
    result: Quad = []
    for vi in range(4):
        sx = sum(q[vi][0] for q in quads) / n
        sy = sum(q[vi][1] for q in quads) / n
        result.append((sx, sy))
    return result
