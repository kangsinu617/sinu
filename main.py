"""Week 2 (재구성): 얼굴 미탐 + ROI 이탈 기반 위험 감지 프로토타입.

실행: python main.py
키:
  q  종료
  r  ROI 재선택 (드래그로 영역 지정, ENTER 확정, c 취소)
"""
from pathlib import Path
from time import time

import cv2
import yaml

from vision.face import FaceDetector
from vision.heuristics import (
    RiskSignal,
    evaluate_face_missing,
    evaluate_roi_exit,
    main_person,
)
from vision.person import PersonDetector
from vision.tracker import DurationTracker

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def draw_overlay(frame, persons, faces, roi, emitted, debug) -> None:
    cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (100, 100, 255), 1)
    cv2.putText(frame, "ROI", (roi[0] + 4, roi[1] + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
    for p in persons:
        x1, y1, x2, y2 = map(int, p.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"person {p.confidence:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"face {f.confidence:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    y = 18
    for k, v in debug.items():
        cv2.putText(frame, f"{k}: {v}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 16
    for i, sig in enumerate(emitted):
        cv2.putText(frame, f"[{sig.type}] {sig.metadata}", (10, y + 10 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def select_roi_interactive(window: str, frame) -> tuple[int, int, int, int] | None:
    sel = cv2.selectROI(window, frame, showCrosshair=True, fromCenter=False)
    if sel[2] <= 0 or sel[3] <= 0:
        return None
    x, y, w, h = sel
    return (int(x), int(y), int(x + w), int(y + h))


def main() -> None:
    cfg = load_config()
    cam = cv2.VideoCapture(cfg["camera"]["index"])
    if not cam.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다")

    person_det = PersonDetector(cfg["models"]["person"])
    face_det = FaceDetector(cfg["models"]["face"]["score_threshold"])

    face_cfg = cfg["heuristics"]["face_missing"]
    roi_cfg = cfg["heuristics"]["roi_exit"]
    roi = (roi_cfg["x1"], roi_cfg["y1"], roi_cfg["x2"], roi_cfg["y2"])

    face_tracker = DurationTracker(face_cfg["min_duration_s"])
    cooldown_s = cfg["dispatcher"]["cooldown_s"]
    last_event_ts: dict[str, float] = {}

    window = "infant-safety"
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            now = time()

            persons = person_det.detect(frame)
            faces = face_det.detect(frame)
            p = main_person(persons)

            face_missing, face_diag = evaluate_face_missing(p, faces)
            roi_exit, roi_diag = evaluate_roi_exit(p, roi)

            active_risks: list[RiskSignal] = []
            if face_tracker.update(face_missing, now):
                active_risks.append(RiskSignal(
                    "face_missing_risk",
                    p.confidence if p else 0.0,
                    {"heuristic": "face_not_in_person", **face_diag},
                ))
            if roi_exit:
                active_risks.append(RiskSignal(
                    "roi_exit_risk",
                    p.confidence if p else 0.0,
                    {"heuristic": "person_center_outside_roi", **roi_diag},
                ))

            for s in active_risks:
                if now - last_event_ts.get(s.type, 0) >= cooldown_s:
                    print(f"[EVENT] {s.type} conf={s.confidence:.2f} {s.metadata}")
                    last_event_ts[s.type] = now

            debug = {
                "persons": len(persons),
                "faces": len(faces),
                "face_in_p": face_diag.get("face_in_p", "-"),
                "face_missing": face_missing,
                "face_elapsed": f"{face_tracker.elapsed(now):.1f}s",
                "roi_exit": roi_exit,
                "center": roi_diag.get("center", "-"),
            }
            draw_overlay(frame, persons, faces, roi, active_risks, debug)
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                new_roi = select_roi_interactive(window, frame)
                if new_roi is not None:
                    roi = new_roi
                    print(f"[ROI] 업데이트: {roi}")
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
