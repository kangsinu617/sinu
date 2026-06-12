"""v1: pose + 두 개 ROI + cause 분기 질식 + 안정화 레이어.

실행: python main.py
키:
  q  종료
  r  ROI(가상 면) 재정의 (네 꼭짓점 다시 클릭)
"""
from collections import deque
from pathlib import Path
from time import time

import cv2
import numpy as np
import yaml

from vision.face import FaceDetector
from vision.heuristics import (
    RiskSignal,
    clearly_side_lying,
    detect_suffocation,
    evaluate_climbing,
    evaluate_fall,
    face_inside_person,
    label_suffocation_cause,
    main_person,
    memory_fresh,
    pose_face_kp_count,
    pose_face_visible,
    pose_torso_visible,
    presence_sustained,
    side_lying_features,
    suffocation_latched,
)
from vision.person import PersonDetector
from vision.head import HeadDetector
from vision.pose import KP_NAMES, PoseDetector, match_pose_to_person
from vision.smoothing import EMA
from vision.tracker import DurationTracker
from audio.yamnet_classifier import AudioClassifier
from events.edge import transition
from events.mqtt_client import MqttPublisher
from events.payload import build_payload
from vision.quad_detector import ArucoQuadDetector, ContourQuadDetector
from vision.manual_roi import load_polygon, save_polygon, select_polygon
from vision.roi_geometry import average_quads, point_in_polygon

CONFIG_PATH = Path(__file__).parent / "config.yaml"
ROI_PATH = Path(__file__).parent / "saved_roi.json"

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


def compute_wrist(pose, conf_threshold):
    if pose is None:
        return None
    wrists = [pose.keypoints[k] for k in ("left_wrist", "right_wrist")
              if pose.keypoints[k][2] >= conf_threshold]
    if not wrists:
        return None
    cx = sum(w[0] for w in wrists) / len(wrists)
    cy = sum(w[1] for w in wrists) / len(wrists)
    return (cx, cy)


def roi_containment(bbox, polygon):
    """subject bbox의 5x5 격자점 중 안전 ROI 폴리곤 안에 든 비율 (0~1).

    bbox가 카메라 안전영역 안에 제대로 잡혔는지의 지표. 발만 걸친 채 ROI
    밖으로 벗어나면 값이 낮아진다(엎드림 정탐 0.88 vs 발만 보임 0.68).
    """
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    gx = [x1 + (x2 - x1) * i / 4 for i in range(5)]
    gy = [y1 + (y2 - y1) * j / 4 for j in range(5)]
    inside = sum(1 for px in gx for py in gy if point_in_polygon((px, py), polygon))
    return inside / 25.0


def motion_level(prev_gray, gray, bbox):
    """이전↔현재 프레임의 subject bbox 영역 회색조 절대차 평균 (0~1, 색 무관).

    엎드림 정탐(정지=무반응)과 안전한 엎드림(배 시간·능동적 버둥거림)을 가르는
    활력 신호. 움직이면 값이 오르고, 인형·천 덮인 무반응은 낮게 유지된다.
    """
    if prev_gray is None or bbox is None:
        return 0.0
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(cv2.absdiff(gray[y1:y2, x1:x2], prev_gray[y1:y2, x1:x2]).mean()) / 255.0


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


POSE_DRAW_CONF = 0.5  # 이 신뢰도 미만으로 잡힌 사람은 pose 스켈레톤을 안 그림 (덮임 시 떨림 방지)


def draw_overlay(frame, persons, faces, main_pose, safe_polygon, active_risks, debug, kp_conf,
                 person_conf=1.0):
    pts = np.array(safe_polygon, np.int32)
    cv2.polylines(frame, [pts], True, (100, 100, 255), 2)
    cv2.putText(frame, "safe", (safe_polygon[0][0] + 4, safe_polygon[0][1] + 14),
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
    if person_conf >= POSE_DRAW_CONF:
        draw_pose(frame, main_pose, kp_conf)
    y = 18
    for k, v in debug.items():
        cv2.putText(frame, f"{k}: {v}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 16
    for i, sig in enumerate(active_risks):
        # suffocation은 cause(flipped/face_covered)를 구분해 표시
        tag = sig.metadata.get("cause") or sig.metadata.get("zone") or ""
        label = f"[{sig.type}/{tag}]" if tag else f"[{sig.type}]"
        cv2.putText(frame, f"{label}", (10, y + 10 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def detect_safe_polygon(cam, cfg: dict, window: str):
    """시작 시 N프레임 검출 → 꼭짓점 평균으로 안정 폴리곤 반환.

    검출 실패 또는 비활성화 시 None. 호출자가 fallback_polygon으로 폴백.
    """
    auto_cfg = cfg.get("auto_roi", {})
    if not auto_cfg.get("enabled", False):
        return None
    det_name = auto_cfg.get("detector", "contour")
    if det_name == "aruco":
        detector = ArucoQuadDetector(auto_cfg.get("aruco", {}).get("dict", "DICT_4X4_50"))
        print("[AUTO-ROI] safe zone 검출 중... 마커 4개가 화면에 보이도록 맞춰주세요 (q=폴백)")
    elif det_name == "contour":
        c = auto_cfg["contour"]
        detector = ContourQuadDetector(
            c["canny_low"], c["canny_high"], c["min_area_ratio"], c["approx_eps_ratio"],
        )
        print("[AUTO-ROI] safe zone 검출 중... (배경 깔끔하게, 박스 전체가 보이도록)")
    else:
        print(f"[AUTO-ROI] detector '{det_name}' 미구현. 폴백.")
        return None
    quads = []
    target = int(auto_cfg["init_frames"])
    max_tries = target * 10
    tries = 0
    while len(quads) < target and tries < max_tries:
        tries += 1
        ok, frame = cam.read()
        if not ok:
            continue
        quad = detector.detect_quad(frame)
        if quad is not None:
            quads.append(quad)
            cv2.polylines(frame, [np.array(quad, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, f"Detecting safe zone... ({len(quads)}/{target})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    poly = average_quads(quads)
    if poly is None:
        print(f"[AUTO-ROI] 검출 실패 ({len(quads)}/{target}). fallback_polygon 사용.")
        return None
    poly = [(int(round(x)), int(round(y))) for x, y in poly]
    print(f"[AUTO-ROI] safe polygon = {poly} ({len(quads)}/{target} hits)")
    return poly


def setup_roi(cam, cfg: dict, window: str):
    """detector 설정에 따라 safe polygon 결정. 실패 시 fallback_polygon.

    manual: 저장본 있으면 재사용, 없으면 클릭 선택 후 저장.
    aruco/contour: detect_safe_polygon으로 N프레임 자동 검출.
    """
    auto_cfg = cfg["auto_roi"]
    det_name = auto_cfg.get("detector", "manual")
    poly = None
    if det_name == "manual":
        poly = load_polygon(ROI_PATH)
        if poly is not None:
            print(f"[ROI] 저장된 ROI 사용: {poly}  (재정의: r 키)")
        else:
            poly = select_polygon(cam, window)
            if poly is not None:
                save_polygon(ROI_PATH, poly)
                print(f"[ROI] 저장: {ROI_PATH}")
    else:
        poly = detect_safe_polygon(cam, cfg, window)
    if poly is None:
        print("[ROI] fallback_polygon 사용.")
        poly = [tuple(pt) for pt in auto_cfg["fallback_polygon"]]
    return poly


def main() -> None:
    cfg = load_config()
    cam = cv2.VideoCapture(cfg["camera"]["index"])
    if not cam.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다")

    person_det = PersonDetector(cfg["models"]["person"])
    face_det = FaceDetector(cfg["models"]["face"]["score_threshold"])
    pose_det = PoseDetector(cfg["models"]["pose"])
    try:
        head_weights = str(Path(__file__).parent / cfg["models"]["head"])
        head_det = HeadDetector(head_weights,
                                cfg["heuristics"]["suffocation"].get("head_conf_threshold", 0.25))
    except Exception as e:
        print(f"[head] 모델 로드 실패 → torso 폴백: {e}")
        head_det = None

    suf_cfg = cfg["heuristics"]["suffocation"]
    clm_cfg = cfg["heuristics"]["climbing"]
    fall_cfg = cfg["heuristics"]["fall"]
    stab_cfg = cfg["stability"]

    aud_cfg = cfg["audio"]
    audio = AudioClassifier(aud_cfg)
    audio_on = audio.start()

    mqtt_cfg = cfg["mqtt"]
    publisher = MqttPublisher(mqtt_cfg["host"], mqtt_cfg["port"], mqtt_cfg["topic"])
    publisher.start()
    device_serial = mqtt_cfg["device_serial"]

    cry_tracker = DurationTracker(aud_cfg["min_duration_s"], stab_cfg["grace_s"])
    whimper_tracker = DurationTracker(aud_cfg["whimper_min_duration_s"], stab_cfg["grace_s"])

    window = "infant-safety-v1"
    safe_polygon = setup_roi(cam, cfg, window)
    rail_band_px = cfg["auto_roi"]["rail_band_px"]

    center_ema = EMA(stab_cfg["ema_alpha"])
    wrist_ema = EMA(stab_cfg["ema_alpha"])

    suf_tracker = DurationTracker(suf_cfg["min_duration_s"], stab_cfg["grace_s"])
    clm_tracker = DurationTracker(clm_cfg["min_duration_s"], stab_cfg["grace_s"])
    fall_tracker = DurationTracker(0.0, 0.0)  # 윈도우가 시간 통합을 하므로 즉시 판정

    face_memory_s: float = suf_cfg.get("face_memory_s", 30.0)
    roi_memory_s: float = suf_cfg.get("roi_memory_s", face_memory_s)
    out_of_view_roi_threshold: float = suf_cfg.get("out_of_view_roi_threshold", 0.72)
    face_kp_conf_threshold: float = suf_cfg.get("face_kp_conf_threshold", 0.5)
    face_kp_min_visible: int = suf_cfg.get("face_kp_min_visible", 2)
    torso_kp_conf_threshold: float = suf_cfg.get("torso_kp_conf_threshold", 0.5)
    torso_kp_min_visible: int = suf_cfg.get("torso_kp_min_visible", 2)
    motion_threshold: float = suf_cfg.get("motion_threshold", 0.02)
    head_memory_s: float = suf_cfg.get("head_memory_s", 2.0)
    latch_max_s: float = suf_cfg.get("latch_max_s", 300.0)
    presence_entry_s: float = suf_cfg.get("presence_entry_s", 10.0)
    presence_gap_s: float = suf_cfg.get("presence_gap_s", 2.0)
    side_cfg: dict = suf_cfg.get("side_lying_guard", {})
    side_guard_enabled: bool = side_cfg.get("enabled", True)
    side_spread_max: float = side_cfg.get("spread_max", 0.8)
    side_min_conf: float = side_cfg.get("min_conf", 0.5)

    event_states: dict = {}

    last_face_seen_time: float = 0.0
    last_in_roi_time: float = 0.0
    last_head_seen_time: float = 0.0   # subject bbox 안에서 head를 마지막으로 본 시각
    suf_latch_since: float = 0.0       # 질식 위험 활성 래치 시작 시각 (0=꺼짐)
    roi_presence_since: float = 0.0    # subject가 ROI 안 연속 존재 시작 시각 (0=미존재)
    last_roi_present_time: float = 0.0  # subject 중심을 ROI 안에서 마지막으로 본 시각
    fall_window_s: float = fall_cfg["window_s"]
    fall_center_hist: deque = deque()  # (t, (cx, cy)) — 낙상 윈도우
    last_climbing_time: float = 0.0    # climbing이 마지막으로 활성이던 시각
    climb_ref_y = None                 # climbing 중이던 기준 높이 (고정)
    prev_gray = None                   # 직전 프레임 회색조 (활동량 계산용)

    clamped_once = False
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if not clamped_once:
                safe_polygon = [(max(0, min(x, w - 1)), max(0, min(y, h - 1)))
                                for x, y in safe_polygon]
                clamped_once = True
            now = time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            persons = person_det.detect(frame)
            faces = face_det.detect(frame)
            poses = pose_det.detect(frame)
            p = main_person(persons)
            main_pose = match_pose_to_person(p, poses) if p else None

            raw_center_xy = None
            center_xy = None
            wrist_xy = None
            if p is not None:
                cx = (p.bbox[0] + p.bbox[2]) / 2
                cy = (p.bbox[1] + p.bbox[3]) / 2
                raw_center_xy = (cx, cy)
                center_xy = center_ema.update(cx, cy)
                wrist_pos = compute_wrist(main_pose, clm_cfg["wrist_conf_threshold"])
                if wrist_pos is not None:
                    wrist_xy = wrist_ema.update(*wrist_pos)
            else:
                center_ema.reset()
                wrist_ema.reset()

            # 얼굴이 보이면 안전 — 몸만 덮여 person 검출이 실패해도 ROI 안에 face가
            # 잡히면 인정(얼굴 노출=안전). YuNet face OR pose 얼굴 키포인트.
            face_in_roi = any(
                point_in_polygon(((f.bbox[0] + f.bbox[2]) / 2, (f.bbox[1] + f.bbox[3]) / 2),
                                 safe_polygon)
                for f in faces
            )
            face_in_person = (p is not None
                              and any(face_inside_person(f, p) for f in faces))
            face_kp_visible = pose_face_visible(main_pose, face_kp_conf_threshold,
                                                face_kp_min_visible)
            face_visible_now = face_in_person or face_in_roi or face_kp_visible
            # face_visible 발동 항 계측(OR 순서 첫 항) — prone 미탐 시 keypoint
            # 환각이 범인인지 HUD에서 특정하기 위함 (face_kp_min_visible 조정은 측정 후)
            face_src = ("yunet" if face_in_person else
                        "roi" if face_in_roi else
                        "kp" if face_kp_visible else "-")
            if face_visible_now:
                last_face_seen_time = now
            face_recently_seen = memory_fresh(last_face_seen_time, now, face_memory_s)

            # ROI 안에서 subject를 본 마지막 시각을 시간 메모리로 유지(face와 동일 패턴).
            # boolean 래치(중심이 ROI 안인지 여부)는 덮는 도중 bbox 중심이 ROI 밖으로 밀리면
            # False로 얼어붙어 face_covered를 영영 막으므로 시간 메모리로 대체.
            if center_xy is not None and point_in_polygon(center_xy, safe_polygon):
                last_in_roi_time = now
                if roi_presence_since <= 0 or (now - last_roi_present_time) > presence_gap_s:
                    roi_presence_since = now  # 미존재였거나 gap 초과 → 연속 구간 재시작
                last_roi_present_time = now
            person_was_in_roi = memory_fresh(last_in_roi_time, now, roi_memory_s)

            # 진입 게이트(빈 방·인형 오탐 방지): (face 이력 AND roi 이력) OR
            # ROI 연속 존재(입장부터 face를 안 보여준 엎드림 커버) OR 진행 중 래치
            # (메모리 만료가 진행 중 위험을 끊어 END 오발행하는 결함 방지).
            latched = suffocation_latched(suf_latch_since, now, face_visible_now, latch_max_s)
            presence_ok = presence_sustained(roi_presence_since, last_roi_present_time,
                                             now, presence_entry_s, presence_gap_s)
            entry_ok = (face_recently_seen and person_was_in_roi) or presence_ok or latched
            entry_src = ("face" if (face_recently_seen and person_was_in_roi) else
                         "presence" if presence_ok else
                         "latch" if latched else "-")

            # subject = pose(엎드림 인형도 잘 잡음) 우선, 없으면 person bbox
            subject_bbox = (main_pose.bbox if main_pose is not None
                            else p.bbox if p is not None else None)
            suf_torso = pose_torso_visible(main_pose, torso_kp_conf_threshold, torso_kp_min_visible)
            suf_roiin = roi_containment(subject_bbox, safe_polygon)
            suf_motion = motion_level(prev_gray, gray, subject_bbox)
            prev_gray = gray
            # head는 face 없을 때만(cause 판별 구간) 돌린다 — YOLOv5m CPU 비용 절약.
            # 제3자(보호자 등) 머리로 cause가 오염되지 않게 subject bbox 안의 head만
            # 인정하고, 프레임별 검출 깜빡임은 head_memory_s 시간 스무딩으로 흡수한다.
            head_present = None
            head_best_conf = None  # subject 안 최고 conf(임계 미달 포함) — HUD 계측용
            head_any_conf = None   # 프레임 전체 최고 conf — bbox 게이트로 버려졌는지 판별용
            if head_det is not None and not face_visible_now and subject_bbox is not None:
                try:
                    sx1, sy1, sx2, sy2 = subject_bbox
                    # 입력은 전체 프레임 그대로 — subject 크롭 업스케일은 천 덮인
                    # 머리까지 conf 0.1~0.3으로 끌어올려 prone(0.40)과의 분리를
                    # 깨므로 철회(2026-06-13). 간헐 미검출은 head_memory_s로 완화.
                    for h in head_det.detect(frame):
                        if head_any_conf is None or h.confidence > head_any_conf:
                            head_any_conf = h.confidence
                        hcx = (h.bbox[0] + h.bbox[2]) / 2
                        hcy = (h.bbox[1] + h.bbox[3]) / 2
                        if sx1 <= hcx <= sx2 and sy1 <= hcy <= sy2:
                            if head_best_conf is None or h.confidence > head_best_conf:
                                head_best_conf = h.confidence
                            if h.confidence >= head_det.conf_threshold:
                                last_head_seen_time = now
                    head_present = memory_fresh(last_head_seen_time, now, head_memory_s)
                except Exception as e:
                    print(f"[head] 검출 예외 → torso 폴백: {e}")
                    head_present = None
            suf_side = (side_guard_enabled
                        and clearly_side_lying(main_pose, side_spread_max, side_min_conf))
            suf_active, det_diag = detect_suffocation(face_visible_now, entry_ok)
            cause, label_diag = label_suffocation_cause(
                subject_bbox is not None, suf_torso, head_present, suf_side,
                suf_motion, motion_threshold,
                suf_roiin, out_of_view_roi_threshold,
            )
            suf_diag = {**det_diag, **label_diag}
            # 래치 갱신: 얼굴 재출현·상한 초과 → OFF, 위험 활성 → ON.
            # side_lying/active_motion으로 일시 안전이어도 래치는 유지 — 얼굴이
            # 안 보이는 타임라인이 이어지는 한, 다시 prone으로 구르면 재판정돼야 한다.
            if face_visible_now:
                suf_latch_since = 0.0
            elif suf_latch_since > 0 and (now - suf_latch_since) >= latch_max_s:
                suf_latch_since = 0.0
            elif suf_active and suf_latch_since <= 0:
                suf_latch_since = now
            clm_active, clm_diag = evaluate_climbing(
                wrist_xy, main_pose, safe_polygon, rail_band_px,
                clm_cfg["wrist_conf_threshold"], clm_cfg["standing_y_margin"],
            )
            if clm_active:
                last_climbing_time = now
                if raw_center_xy is not None:
                    climb_ref_y = raw_center_xy[1]

            if raw_center_xy is not None:
                fall_center_hist.append((now, raw_center_xy))
                while len(fall_center_hist) > 1 and now - fall_center_hist[0][0] > fall_window_s:
                    fall_center_hist.popleft()
            else:
                fall_center_hist.clear()
            past_center_xy = fall_center_hist[0][1] if fall_center_hist else None
            fall_active, fall_diag = evaluate_fall(
                raw_center_xy, past_center_xy, fall_cfg["min_drop_px"],
            )
            # climbing 직후 경로: 고정 기준 높이 대비 하강 (past_center 희석 회피)
            in_climb_ctx = (last_climbing_time > 0
                            and now - last_climbing_time < fall_cfg["climb_window_s"])
            climb_drop = None
            if (not fall_active and in_climb_ctx
                    and raw_center_xy is not None and climb_ref_y is not None):
                climb_drop = raw_center_xy[1] - climb_ref_y
                if climb_drop >= fall_cfg["climb_drop_px"]:
                    fall_active = True
                    fall_diag.pop("block", None)
                    fall_diag["climb_drop"] = round(climb_drop, 1)
            # climbing 직후 경로 2: bbox 자세붕괴 (세로→가로=누움)
            aspect = None
            if p is not None:
                _pw = p.bbox[2] - p.bbox[0]
                _ph = p.bbox[3] - p.bbox[1]
                if _ph > 0:
                    aspect = _pw / _ph
            if (not fall_active and in_climb_ctx and aspect is not None
                    and aspect >= fall_cfg["climb_fall_aspect"]):
                fall_active = True
                fall_diag.pop("block", None)
                fall_diag["climb_aspect"] = round(aspect, 2)

            cry_raw, cry_score, whimper_raw, whimper_score = (
                audio.get_state() if audio_on else (False, 0.0, False, 0.0)
            )
            cry_condition = cry_raw and p is not None
            whimper_condition = whimper_raw and p is not None

            suf_triggered = suf_tracker.update(suf_active, now)
            clm_triggered = clm_tracker.update(clm_active, now)
            # 같은 대상이 질식(DANGER)과 climbing(CAUTION)일 수는 없다 — 질식 우선.
            # DANGER 활성 중 CAUTION만 거두는 것이라 안전 손실 없음. 역방향
            # (climbing으로 질식 억제)은 FN 사고로 금지(2026-06-10).
            clm_triggered = clm_triggered and not suf_triggered
            fall_triggered = fall_tracker.update(fall_active, now)
            cry_triggered = cry_tracker.update(cry_condition, now)
            whimper_triggered = whimper_tracker.update(whimper_condition, now)

            p_conf = p.confidence if p else 0.0
            # face_covered(몸 전체 덮임)는 person이 없어 검출 conf가 없음 → 고정값
            suf_conf = p_conf if p is not None else 0.9
            suf_heuristic = ("prone_torso_keypoints_visible" if cause == "flipped"
                             else "face_covered_by_blanket")
            event_inputs = [
                ("suffocation_risk", suf_triggered, suf_conf,
                 {"cause": cause, "heuristic": suf_heuristic, **suf_diag}),
                ("climbing_risk", clm_triggered, p_conf,
                 {"zone": "crib_rail", "heuristic": "wrist_in_rail_and_standing", **clm_diag}),
                ("fall_risk", fall_triggered, p_conf,
                 {"heuristic": "rapid_y_descent", **fall_diag}),
                ("cry_detected", cry_triggered, cry_score,
                 {"heuristic": "yamnet_cry_and_person_present"}),
                ("babble_detected", whimper_triggered, whimper_score,
                 {"heuristic": "yamnet_babble_and_person_present"}),
            ]

            active_risks: list[RiskSignal] = []
            events_to_publish: list[RiskSignal] = []
            for ev_type, triggered, conf, meta in event_inputs:
                sig = transition(ev_type, triggered, event_states, now, conf, meta)
                if sig is not None:
                    events_to_publish.append(sig)
                if triggered:
                    active_risks.append(RiskSignal(ev_type, conf, meta))

            for s in events_to_publish:
                phase = s.metadata.get("phase")
                dur = s.metadata.get("duration_s", 0.0)
                print(f"[EVENT] {s.type} phase={phase} conf={s.confidence:.2f} dur={dur:.1f}s")
                payload = build_payload(s, device_serial)
                if payload is not None:
                    publisher.publish(payload)

            debug = {
                "persons": len(persons),
                "faces": len(faces),
                "cause": cause if suf_active else "-",
                "face_src": face_src,
                # 임계 이상 얼굴 키포인트 수 — kp 환각으로 위험이 취소될 때
                # face_kp_min_visible을 어디까지 올려야 하는지 판단용
                "face_kp": pose_face_kp_count(main_pose, face_kp_conf_threshold),
                "entry": entry_src,
                "flags": (",".join(k for k in ("out_of_view", "active_motion", "side_lying")
                                   if label_diag.get(k)) or "-"),
                "suf_elapsed": f"{suf_tracker.elapsed(now):.1f}s",
                "clm_elapsed": f"{clm_tracker.elapsed(now):.1f}s",
                "cry_score": f"{cry_score:.2f}" if audio_on else "off",
                "cry_elapsed": f"{cry_tracker.elapsed(now):.1f}s",
                "motion": f"{suf_motion:.3f}",
                "head": ("T" if head_present else "F") if head_present is not None else "-",
                # head 미검출 원인 판별용: 임계(0.25) 미달 conf가 보이면 임계 하향 후보.
                # head_c=- 인데 head_any에 값이 있으면 subject bbox 게이트가 버린 것
                "head_c": f"{head_best_conf:.2f}" if head_best_conf is not None else "-",
                "head_any": f"{head_any_conf:.2f}" if head_any_conf is not None else "-",
                "torso": "T" if suf_torso else "F",
                "roi_in": f"{suf_roiin:.2f}" if suf_roiin is not None else "-",
                "side": (f"{max(side_lying_features(main_pose)[:2]):.2f}"
                         if main_pose is not None else "-"),
                "latch": f"{now - suf_latch_since:.0f}s" if suf_latch_since > 0 else "-",
                # climbing 오발(누운 자세에서 발동) 튜닝용: standing_y_margin/rail_band_px 기준값
                "clm_m": clm_diag.get("standing_margin", "-"),
                "rail_d": clm_diag.get("rail_dist", "-"),
            }
            draw_overlay(frame, persons, faces, main_pose, safe_polygon,
                         active_risks, debug, clm_cfg["wrist_conf_threshold"], p_conf)
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                new_poly = select_polygon(cam, window)
                if new_poly is not None:
                    safe_polygon = new_poly
                    clamped_once = False  # 다음 프레임에 재clamp
                    if cfg["auto_roi"].get("detector", "manual") == "manual":
                        save_polygon(ROI_PATH, safe_polygon)
                        print(f"[ROI] 재정의·저장: {safe_polygon}")
    finally:
        audio.stop()
        publisher.stop()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
