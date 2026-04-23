"""v1: pose + 두 개 ROI + cause 분기 질식 + 안정화 레이어.

실행: python main.py
키:
  q  종료
  r  safe_roi 재선택
  c  climb_rail 재선택
"""
from pathlib import Path
from time import time

import cv2
import yaml

from vision.face import FaceDetector
from vision.heuristics import (
    RiskSignal,
    evaluate_climbing,
    evaluate_roi_exit,
    evaluate_suffocation,
    main_person,
)
from vision.person import PersonDetector
from vision.pose import KP_NAMES, PoseDetector, match_pose_to_person
from vision.smoothing import EMA
from vision.tracker import DurationTracker

CONFIG_PATH = Path(__file__).parent / "config.yaml"

KP_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def select_roi_interactive(window: str, frame) -> tuple[int, int, int, int] | None:
    sel = cv2.selectROI(window, frame, showCrosshair=True, fromCenter=False)
    if sel[2] <= 0 or sel[3] <= 0:
        return None
    x, y, w, h = sel
    return (int(x), int(y), int(x + w), int(y + h))


def clamp_roi(roi, w, h):
    x1, y1, x2, y2 = roi
    return (max(0, min(x1, w - 1)), max(0, min(y1, h - 1)),
            max(0, min(x2, w - 1)), max(0, min(y2, h - 1)))


def compute_ankle(pose, bbox_bottom_xy, conf_threshold):
    if pose is None:
        return bbox_bottom_xy
    ankles = [pose.keypoints[k] for k in ("left_ankle", "right_ankle")
              if pose.keypoints[k][2] >= conf_threshold]
    if not ankles:
        return bbox_bottom_xy
    cx = sum(a[0] for a in ankles) / len(ankles)
    cy = sum(a[1] for a in ankles) / len(ankles)
    return (cx, cy)


def draw_pose(frame, pose, conf_threshold):
    if pose is None:
        return
    kp = pose.keypoints
    for name in KP_NAMES:
        x, y, c = kp[name]
        if c >= conf_threshold:
            cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)
    for a, b in KP_EDGES:
        xa, ya, ca = kp[a]
        xb, yb, cb = kp[b]
        if ca >= conf_threshold and cb >= conf_threshold:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (255, 0, 0), 1)


def draw_overlay(frame, persons, faces, main_pose, safe_roi, climb_roi, active_risks, debug, kp_conf):
    cv2.rectangle(frame, (safe_roi[0], safe_roi[1]), (safe_roi[2], safe_roi[3]), (100, 100, 255), 1)
    cv2.putText(frame, "safe", (safe_roi[0] + 4, safe_roi[1] + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
    cv2.rectangle(frame, (climb_roi[0], climb_roi[1]), (climb_roi[2], climb_roi[3]), (0, 150, 255), 1)
    cv2.putText(frame, "climb_rail", (climb_roi[0] + 4, climb_roi[1] + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1)
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
    draw_pose(frame, main_pose, kp_conf)
    y = 18
    for k, v in debug.items():
        cv2.putText(frame, f"{k}: {v}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 16
    for i, sig in enumerate(active_risks):
        cause = sig.metadata.get("cause")
        zone = sig.metadata.get("zone")
        tag = cause or zone or ""
        label = f"[{sig.type}/{tag}]" if tag else f"[{sig.type}]"
        cv2.putText(frame, f"{label}", (10, y + 10 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def main() -> None:
    cfg = load_config()
    cam = cv2.VideoCapture(cfg["camera"]["index"])
    if not cam.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다")

    person_det = PersonDetector(cfg["models"]["person"])
    face_det = FaceDetector(cfg["models"]["face"]["score_threshold"])
    pose_det = PoseDetector(cfg["models"]["pose"])

    suf_cfg = cfg["heuristics"]["suffocation"]
    clm_cfg = cfg["heuristics"]["climbing"]
    exit_cfg = cfg["heuristics"]["roi_exit"]
    stab_cfg = cfg["stability"]

    safe_roi = (cfg["rois"]["safe"]["x1"], cfg["rois"]["safe"]["y1"],
                cfg["rois"]["safe"]["x2"], cfg["rois"]["safe"]["y2"])
    climb_roi = (cfg["rois"]["climb_rail"]["x1"], cfg["rois"]["climb_rail"]["y1"],
                 cfg["rois"]["climb_rail"]["x2"], cfg["rois"]["climb_rail"]["y2"])

    center_ema = EMA(stab_cfg["ema_alpha"])
    ankle_ema = EMA(stab_cfg["ema_alpha"])

    suf_tracker = DurationTracker(suf_cfg["min_duration_s"], stab_cfg["grace_s"])
    clm_tracker = DurationTracker(clm_cfg["min_duration_s"], stab_cfg["grace_s"])
    exit_tracker = DurationTracker(exit_cfg["min_duration_s"], stab_cfg["grace_s"])

    cooldown_s = cfg["dispatcher"]["cooldown_s"]
    last_event_ts: dict[str, float] = {}

    window = "infant-safety-v1"
    clamped_once = False
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if not clamped_once:
                safe_roi = clamp_roi(safe_roi, w, h)
                climb_roi = clamp_roi(climb_roi, w, h)
                clamped_once = True
            now = time()

            persons = person_det.detect(frame)
            faces = face_det.detect(frame)
            poses = pose_det.detect(frame)
            p = main_person(persons)
            main_pose = match_pose_to_person(p, poses) if p else None

            center_xy = None
            ankle_xy = None
            if p is not None:
                cx = (p.bbox[0] + p.bbox[2]) / 2
                cy = (p.bbox[1] + p.bbox[3]) / 2
                center_xy = center_ema.update(cx, cy)
                bbox_bottom = ((p.bbox[0] + p.bbox[2]) / 2, p.bbox[3])
                ax, ay = compute_ankle(main_pose, bbox_bottom, clm_cfg["ankle_conf_threshold"])
                ankle_xy = ankle_ema.update(ax, ay)
            else:
                center_ema.reset()
                ankle_ema.reset()

            suf_active, cause, suf_diag = evaluate_suffocation(
                p, faces, main_pose,
                suf_cfg["keypoint_conf_threshold"],
                suf_cfg["flipped_min_visible"],
                suf_cfg["blanket_max_visible"],
            )
            clm_active, clm_diag = evaluate_climbing(
                ankle_xy, main_pose, climb_roi,
                clm_cfg["ankle_conf_threshold"], clm_cfg["standing_y_margin"],
            )
            exit_active, exit_diag = evaluate_roi_exit(center_xy, safe_roi)

            active_risks: list[RiskSignal] = []
            if suf_tracker.update(suf_active, now):
                active_risks.append(RiskSignal(
                    "suffocation_risk",
                    p.confidence if p else 0.0,
                    {"cause": cause, "heuristic": "face_not_in_person", **suf_diag},
                ))
            if clm_tracker.update(clm_active, now):
                active_risks.append(RiskSignal(
                    "climbing_risk",
                    p.confidence if p else 0.0,
                    {"zone": "crib_rail", "heuristic": "ankle_in_rail_and_standing", **clm_diag},
                ))
            if exit_tracker.update(exit_active, now):
                active_risks.append(RiskSignal(
                    "roi_exit_risk",
                    p.confidence if p else 0.0,
                    {"heuristic": "person_center_outside_roi", **exit_diag},
                ))

            for s in active_risks:
                if now - last_event_ts.get(s.type, 0) >= cooldown_s:
                    print(f"[EVENT] {s.type} conf={s.confidence:.2f} {s.metadata}")
                    last_event_ts[s.type] = now

            debug = {
                "persons": len(persons),
                "faces": len(faces),
                "face_in_p": suf_diag.get("face_in_p", "-"),
                "visible_kp": suf_diag.get("visible_keypoints", "-"),
                "cause": cause or "-",
                "suf_elapsed": f"{suf_tracker.elapsed(now):.1f}s",
                "clm_elapsed": f"{clm_tracker.elapsed(now):.1f}s",
                "ankle_ema": tuple(round(x) for x in ankle_xy) if ankle_xy else "-",
                "center_ema": tuple(round(x) for x in center_xy) if center_xy else "-",
                "roi_exit": exit_active,
            }
            draw_overlay(frame, persons, faces, main_pose, safe_roi, climb_roi,
                         active_risks, debug, suf_cfg["keypoint_conf_threshold"])
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                new_roi = select_roi_interactive(window, frame)
                if new_roi is not None:
                    safe_roi = clamp_roi(new_roi, w, h)
                    print(f"[ROI] safe 업데이트: {safe_roi}")
            if key == ord("c"):
                new_roi = select_roi_interactive(window, frame)
                if new_roi is not None:
                    climb_roi = clamp_roi(new_roi, w, h)
                    print(f"[ROI] climb_rail 업데이트: {climb_roi}")
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
