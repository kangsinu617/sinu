"""사각 영역 검출 — frame을 4점 폴리곤으로.

검출기 인터페이스: detect_quad(frame) -> [TL,TR,BR,BL] | None
ContourQuadDetector(부착물 없음)와 ArucoQuadDetector(마커 4개) 둘 다 동일 인터페이스.
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .roi_geometry import Point, Quad


def order_quad(pts: List[Tuple[float, float]]) -> Quad:
    """4점을 TL,TR,BR,BL 순으로 정렬.

    y로 위/아래 두 점씩 나눈 뒤 각 쌍을 x로 좌/우 구분한다.
    x+y/x-y 휴리스틱과 달리 비스듬히 기울어진(회전된) 사변형에서도
    꼭짓점이 중복 배정되지 않는다.
    """
    pts = sorted([(float(x), float(y)) for x, y in pts], key=lambda p: p[1])
    tl, tr = sorted(pts[:2], key=lambda p: p[0])
    bl, br = sorted(pts[2:], key=lambda p: p[0])
    return [(int(round(p[0])), int(round(p[1]))) for p in (tl, tr, br, bl)]


class ContourQuadDetector:
    def __init__(self, canny_low: int, canny_high: int,
                 min_area_ratio: float, approx_eps_ratio: float) -> None:
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_area_ratio = min_area_ratio
        self.approx_eps_ratio = approx_eps_ratio

    def detect_quad(self, frame) -> Optional[Quad]:
        h, w = frame.shape[:2]
        min_area = self.min_area_ratio * w * h
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        best: Optional[Quad] = None
        best_area = 0.0
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, self.approx_eps_ratio * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            area = cv2.contourArea(approx)
            if area < min_area or area <= best_area:
                continue
            best_area = area
            best = order_quad([(int(p[0][0]), int(p[0][1])) for p in approx])
        return best


class ArucoQuadDetector:
    """ArUco 마커 4개의 중심을 [TL,TR,BR,BL] 폴리곤으로.

    ID와 무관하게 위치(order_quad)로 정렬하므로, 가상 면의 네 꼭짓점에
    마커를 한 장씩 붙이기만 하면 된다. 정확히 4개가 보일 때만 검출 성공.
    """

    def __init__(self, dict_name: str = "DICT_4X4_50") -> None:
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        self.detector = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())

    def detect_quad(self, frame) -> Optional[Quad]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(corners) != 4:
            return None
        centers = [(float(c[0][:, 0].mean()), float(c[0][:, 1].mean())) for c in corners]
        return order_quad(centers)
