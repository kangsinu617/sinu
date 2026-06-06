"""수동 ROI 선택 — 가상 면의 네 꼭짓점을 마우스로 클릭해 폴리곤 정의.

카메라가 고정이므로 한 번 정의해 JSON에 저장하고 재실행 시 재사용한다.
검출기(ArUco/contour)와 달리 조명·각도에 무관하게 100% 안정적이고
시연 화면에 아무 부착물도 남지 않는다.
"""
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .quad_detector import order_quad
from .roi_geometry import Quad


def load_polygon(path: Path) -> Optional[Quad]:
    """저장된 폴리곤을 읽는다. 없거나 형식이 깨졌으면 None."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, list) or len(data) != 4:
        return None
    try:
        return [(int(x), int(y)) for x, y in data]
    except (TypeError, ValueError):
        return None


def save_polygon(path: Path, polygon: Quad) -> None:
    path.write_text(json.dumps([[int(x), int(y)] for x, y in polygon]))


def select_polygon(cam, window: str) -> Optional[Quad]:
    """카메라 화면에서 네 꼭짓점을 클릭받아 [TL,TR,BR,BL] 폴리곤 반환.

    좌클릭=점 추가(최대 4), r=초기화, Enter/c=4점 확정, q/ESC=취소(None).
    """
    points: List[Tuple[int, int]] = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    print("[ROI] 가상 면의 네 꼭짓점을 클릭하세요. r=초기화, Enter/c=확정, q=취소")
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                continue
            for i, pt in enumerate(points):
                cv2.circle(frame, pt, 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (pt[0] + 6, pt[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if len(points) >= 2:
                cv2.polylines(frame, [np.array(points, np.int32)],
                              len(points) == 4, (0, 255, 0), 2)
            msg = (f"Click 4 corners ({len(points)}/4)  r=reset  q=cancel"
                   if len(points) < 4 else
                   "Enter=confirm  r=reset  q=cancel")
            cv2.putText(frame, msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                points.clear()
            elif key in (ord("q"), 27):
                return None
            elif key in (13, ord("c")) and len(points) == 4:
                return order_quad(points)
    finally:
        cv2.setMouseCallback(window, lambda *a: None)
