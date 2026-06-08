"""measurement spike — head 검출이 자세별로 갈리는지 확인 (throwaway, 통합 아님).

crowdhuman_yolov5m(person+head)로 head 박스만 그려, 인형을 세 자세
(엎드림 / 얼굴만 천 / 이불 전체)로 두고 head conf 구간이 갈리는지 눈으로 잰다.
교차참조로 pose 몸통 키포인트 conf(어깨·엉덩이)도 같이 띄운다.

실행: python scripts/head_probe.py   (카메라 GUI라 `!`로 직접)
키: q 종료
"""
from pathlib import Path
from time import time

import cv2
import torch
import yaml

from vision.pose import PoseDetector

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "crowdhuman_yolov5m.pt"
_TORSO_KP = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").open())
    cam_index = cfg["camera"]["index"]

    head = torch.hub.load("ultralytics/yolov5", "custom", path=str(MODEL),
                          trust_repo=True, verbose=False)
    head.conf = 0.10  # 약한 검출도 보려고 낮게
    pose = PoseDetector(cfg["models"]["pose"])

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit(f"camera {cam_index} 열기 실패")

    last_log = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det = head(rgb).xyxy[0].cpu().numpy()  # [x1,y1,x2,y2,conf,cls]
        heads = [d for d in det if int(d[5]) == 1]
        heads.sort(key=lambda d: d[4], reverse=True)

        for d in heads:
            x1, y1, x2, y2, conf = d[0], d[1], d[2], d[3], d[4]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        head_max = heads[0][4] if heads else 0.0

        poses = pose.detect(frame)
        torso_n = 0
        if poses:
            kp = poses[0].keypoints
            torso_n = sum(1 for n in _TORSO_KP if kp[n][2] >= 0.5)

        cv2.putText(frame, f"head_max:{head_max:.2f}  n_head:{len(heads)}  torso:{torso_n}/4",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        now = time()
        if now - last_log > 0.5:
            print(f"head_max={head_max:.2f}  n_head={len(heads)}  torso={torso_n}/4")
            last_log = now

        cv2.imshow("head_probe", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
