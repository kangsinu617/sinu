import cv2
import numpy as np

from vision.quad_detector import ArucoQuadDetector, order_quad


def _frame_with_markers(centers, dict_name="DICT_4X4_50", marker_px=80):
    """흰 배경에 ID 0..N-1 마커를 주어진 중심 좌표로 합성한 프레임."""
    frame = np.full((480, 640, 3), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    half = marker_px // 2
    for i, (cx, cy) in enumerate(centers):
        img = cv2.aruco.generateImageMarker(d, i, marker_px)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        frame[cy - half:cy + half, cx - half:cx + half] = img
    return frame


def test_order_quad_already_ordered():
    pts = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert order_quad(pts) == [(0, 0), (100, 0), (100, 100), (0, 100)]


def test_order_quad_shuffled():
    # 순서 뒤섞인 입력도 TL,TR,BR,BL로 정렬
    pts = [(100, 100), (0, 0), (0, 100), (100, 0)]
    assert order_quad(pts) == [(0, 0), (100, 0), (100, 100), (0, 100)]


def test_order_quad_skewed():
    # 비스듬 사변형
    pts = [(90, 100), (10, 0), (0, 80), (100, 20)]
    ordered = order_quad(pts)
    assert ordered[0] == (10, 0)    # TL: 위쪽 왼쪽
    assert ordered[2] == (90, 100)  # BR: 아래쪽 오른쪽


def test_order_quad_rotated_no_duplicate():
    # 실제 박스에서 나온, 시계방향으로 기울어진 사변형.
    # 옛 x+y/x-y 휴리스틱은 한 점을 BR/TR에 중복 배정해 두 점을 잃었다.
    pts = [(19, 92), (242, 48), (597, 216), (461, 305)]
    ordered = order_quad(pts)
    assert len(set(ordered)) == 4          # 중복 없음 (4점 모두 보존)
    assert set(ordered) == set(pts)        # 원본 4점과 동일 집합
    assert ordered == [(19, 92), (242, 48), (597, 216), (461, 305)]  # TL,TR,BR,BL


def test_aruco_detects_four_markers_ordered():
    # 네 꼭짓점에 마커 4개 → [TL,TR,BR,BL] 중심 반환 (오차 허용)
    centers = [(120, 110), (520, 110), (520, 370), (120, 370)]  # TL,TR,BR,BL
    frame = _frame_with_markers(centers)
    det = ArucoQuadDetector("DICT_4X4_50")
    quad = det.detect_quad(frame)
    assert quad is not None
    assert len(quad) == 4
    for (gx, gy), (qx, qy) in zip(centers, quad):
        assert abs(gx - qx) <= 3 and abs(gy - qy) <= 3


def test_aruco_returns_none_when_too_few():
    # 마커 3개만 보이면 None
    centers = [(120, 110), (520, 110), (520, 370)]
    frame = _frame_with_markers(centers)
    det = ArucoQuadDetector("DICT_4X4_50")
    assert det.detect_quad(frame) is None


def test_aruco_returns_none_when_blank():
    frame = np.full((480, 640, 3), 255, np.uint8)
    det = ArucoQuadDetector("DICT_4X4_50")
    assert det.detect_quad(frame) is None
