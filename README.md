# AI 파트 — 영유아 안전 모니터링

담당: 강신우. 상세 계획은 `../AI_프로토타입_계획.md` 참조.

## 환경

- Ubuntu 22.04, Python 3.10
- CPU-only 노트북 (내장그래픽)

## 설치 (권장: venv)

```bash
cd /home/kangsinu/종설/ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`tensorflow`는 Week 3부터 실제로 사용. 초기 설치가 부담되면 `requirements.txt`에서 Week 1 섹션(`ultralytics`, `opencv-python`)만 먼저 설치해도 됨.

## 실행

```bash
python main.py
```

- 창이 뜨고 웹캠 프레임에 person bbox(초록) + face bbox(노랑) + pose 뼈대(파랑) + safe ROI(연빨강) + climb_rail ROI(주황)가 그려짐
- 첫 실행 시 `yolov8n.pt`, `yolov8n-pose.pt`, `models/face_detection_yunet_2023mar.onnx` 자동 다운로드
- 키
  - `q` 종료
  - `r` safe_roi 재선택
  - `c` climb_rail ROI 추가 (최대 4개, 화면에 rail0~rail3 표시)
  - `x` climb_rail ROI 전체 초기화
- HUD에 `cry_score` (off = 마이크 없음), `cry_elapsed` 표시
- 첫 실행 시 YAMNet 모델 자동 다운로드 (~200MB, TFHub 캐시)

## 위험 판정 규칙 (v1)

1. **suffocation_risk**: person 안에 face가 없는 상태 지속 → pose 키포인트 가시수로 `cause` 분기
   - 어깨·엉덩이 4개 중 3개 이상 보임 → `flipped` (뒤집힘)
   - 전부 안 보임 → `blanket` (이불덮힘)
   - 그 중간 → `unknown`
2. **climbing_risk**: pose wrist가 난간 ROI(`climb_rails`, 최대 4개 중 하나) 내부이면서 서있음 자세(hip_y − shoulder_y ≥ margin) 지속 → 침대 난간 손 걸침
3. **roi_exit_risk**: person 중심이 안전 ROI(`safe`) 밖 (낙상·이탈 일반)

판정은 지수이동평균으로 스무딩된 좌표 + `DurationTracker` grace로 프레임 단위 튐을 완화. 동일 타입 이벤트는 `dispatcher.cooldown_s`(30초) 동안 재 dispatch 안 됨.

## 진행 상황

- [x] Week 1: 웹캠 + YOLOv8n 렌더 루프 (2026-04-23 확인)
- [x] Week 2 재구성: 얼굴 미탐 + ROI 이탈 2규칙 (v0)
- [x] Week 2 v1: pose + cause 분기 + 판정 안정화
- [x] Week 3: YAMNet 울음 감지 + person AND 조건 (v2)
- [ ] Week 4: 이벤트 디스패처 + REST 연동
- [ ] Week 5: 통합 테스트·튜닝
- [ ] Week 6: 데모 준비
