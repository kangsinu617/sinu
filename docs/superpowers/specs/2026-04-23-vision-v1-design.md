# 영상 파이프라인 v1 — 판정 고도화 설계

- 작성일: 2026-04-23
- 작성자: 강신우
- 범위: 영상(vision) 파이프라인 v0 → v1 고도화. 음성(audio)·서버 연동(events)은 별개 스펙.
- 전제: 노트북 CPU-only 실행, 학습 없이 사전학습 모델만 사용, 영유아 환경 가정(이불·침대 난간 등)
- v0 스냅샷: `/home/kangsinu/종설/ai/v0/`

---

## 1. 배경과 목적

v0는 "person bbox 안에 face bbox 없음 N초" + "person 중심이 safe ROI 이탈" 두 규칙으로 위험을 판정했다. 실측 결과 다음이 드러났다.

- 판정이 프레임 단위 튐에 매우 민감 (grace 없는 DurationTracker + 위치 스무딩 없음)
- ROI 이탈 하나가 낙상·등반·단순 이동을 모두 통칭 → 원인 구분 불가
- "뒤집힘"과 "이불덮힘"이 동일한 `face_missing_risk`로 통합되어 정보 손실

v1 목표:
1. 판정 안정화 (튐 제거)
2. 가구 등반을 "침대 난간에 올라선 영유아" 자세/위치 조합으로 세분화
3. 질식 위험을 뒤집힘·이불덮힘·불명으로 세분화 (단일 이벤트 + cause 메타데이터)

## 2. 주요 설계 결정 (합의 완료)

| 축 | 결정 |
|---|---|
| 모델 추가 | **허용**. YOLOv8n-pose, COCO 가구 클래스 사용 가능. 단, 키포인트 가시성을 판정의 **필수 전제**로 삼지 않음 (메모리 지침 준수) |
| 질식 이벤트 표현 | 단일 이벤트 `suffocation_risk` + `metadata.cause ∈ {"flipped","blanket","unknown"}` |
| 가구 등반 | 가구 자동 탐지 **미사용**. 대신 **"난간 영역 ROI + pose 서있음 자세"** AND 조건 |
| 판정 안정화 | `DurationTracker.grace_s` + 좌표 `EMA` smoother |

## 3. 모듈 구조

```
vision/
├── person.py       (v0 유지) YOLOv8n person 탐지
├── face.py         (v0 유지) YuNet 얼굴 탐지
├── pose.py         (신규) YOLOv8n-pose 래퍼, Person bbox와 매칭된 keypoint 반환
├── smoothing.py    (신규) EMA 좌표 스무더
├── tracker.py      (확장) DurationTracker + grace_s
└── heuristics.py   (재작성) 3종 판정 함수
main.py             (확장) pose·스무더 초기화, 두 개 ROI 관리, 확장 HUD, 3종 이벤트 dispatch
config.yaml         (확장) suffocation/climbing/roi_exit/stability 섹션
tests/test_heuristics.py (신규) 각 판정 함수·Tracker·EMA 단위 테스트
```

## 4. 프레임당 데이터 흐름

```
frame ─┬─> person_det  ──> [Person]
       ├─> face_det    ──> [Face]
       └─> pose_det    ──> [Pose]       (keypoints: shoulder·hip·ankle …)

main_person = argmax(area)[Person]
main_pose   = argmax(IoU(person.bbox, pose.bbox))  # IoU 0이면 None

center_ema.update(main_person.center)
ankle_ema.update(main_pose.ankle_xy or main_person.bbox_bottom)

heuristics.evaluate_suffocation(main_person, faces, main_pose)
  → (active, cause, diag)
heuristics.evaluate_climbing(ankle_ema.value, main_pose, climb_roi)
  → (active, diag)
heuristics.evaluate_roi_exit(center_ema.value, safe_roi)
  → (active, diag)

sustained = [Tracker.update(active, now) for each]
active_risks = RiskSignal들 (HUD 빨강 표시용)

for s in active_risks:
    if now - last_event_ts[s.type] >= cooldown_s:
        print/dispatch, last_event_ts[s.type] = now
```

HUD에 그리는 빨강 텍스트는 `active_risks` 기반 (쿨다운과 무관 — v0 fix 시 합의).
콘솔·서버 전송은 쿨다운 필터링된 결과만.

## 5. 판정 로직 상세

### 5.1 `evaluate_suffocation(person, faces, pose) → (active, cause, diag)`

1. `person is None` → (False, None, _)
2. person.bbox 안에 face 존재 → (False, None, _) *(정상)*
3. face 없음 → `active=True`. 어깨·엉덩이 4개 키포인트 중 confidence ≥ `keypoint_conf_threshold`인 개수 `v`:
   - `v ≥ flipped_min_visible` → `cause="flipped"`
   - `v ≤ blanket_max_visible` → `cause="blanket"`
   - 그 외 → `cause="unknown"`

*pose=None*인 경우는 `v=0` → `blanket_max_visible=0`이면 `cause="blanket"`로 귀결.

### 5.2 `evaluate_climbing(smoothed_ankle, pose, climb_roi) → (active, diag)`

1. `pose is None` → False
2. smoothed_ankle이 `climb_roi` 밖 → False
3. **서있음 판정**: shoulder·hip 양쪽 중 최소 한 쪽씩이라도 conf ≥ `keypoint_conf_threshold`여야 성립. 그렇지 않으면 False. 성립 시 `hip_y − shoulder_y ≥ standing_y_margin` 아니면 False
4. 셋 다 통과 → True

*ankle 좌표 계산* (main.py에서 수행 → `smoothed_ankle`에 반영):
- 좌/우 ankle 중 conf ≥ `ankle_conf_threshold`인 것들의 y 평균
- 둘 다 실패 → `main_person.bbox` 하단 중심 y로 fallback (pose가 있어도 ankle 미탐 가능하므로)
*hip_y, shoulder_y*: 좌우 평균 (단일 실패 시 반대편 값 사용).

### 5.3 `evaluate_roi_exit(smoothed_center, safe_roi) → (active, diag)`

v0 그대로. `smoothed_center`가 `safe_roi` 밖이면 True. `min_duration_s=0`이고 튐 완화는 Tracker grace와 EMA로 처리.

### 5.4 `DurationTracker(required_s, grace_s).update(cond, now) → bool`

상태: `start_ts`, `last_true_ts` (둘 다 Optional[float])

- `cond=True`:
  - `start_ts is None` → `start_ts=now`
  - `last_true_ts=now`
  - return `(now - start_ts) ≥ required_s`
- `cond=False`:
  - `last_true_ts is None` → return False
  - `(now - last_true_ts) > grace_s` → 리셋(`start_ts=None`, `last_true_ts=None`), return False
  - 그 외 → 상태 유지, return `(now - start_ts) ≥ required_s`

즉 조건이 grace_s 이내에서만 잠깐 False가 되어도 누적 카운트 유지.

### 5.5 `EMA(alpha).update(x, y) → (x_s, y_s)`

`value = (x, y)` (첫 호출) 또는 `(α*new + (1-α)*prev)`. 프레임 단위 호출.

## 6. 이벤트 스키마

```python
@dataclass
class RiskSignal:
    type: str
    confidence: float
    metadata: dict
```

| type | metadata 필드 |
|---|---|
| `suffocation_risk` | `cause` (`flipped`/`blanket`/`unknown`), `visible_keypoints` (어깨·엉덩이 4개 중 conf threshold 통과한 개수, 0~4), `face_in_p`, `heuristic` |
| `climbing_risk` | `zone` (`"crib_rail"`), `ankle` (EMA smoothed (x,y)), `standing_margin` (`hip_y - shoulder_y` 측정값), `heuristic` |
| `roi_exit_risk` | `center` (EMA smoothed (x,y)), `roi`, `heuristic` |

- `confidence`: `main_person.confidence` pass-through. 규칙별 가중은 Week 4 서버 payload 설계 시 재검토
- 쿨다운은 **`type`별 공유** (cause가 바뀌어도 30초 내엔 재알림 안 함)

콘솔 포맷:
```
[EVENT] suffocation_risk cause=flipped conf=0.85 {"visible_keypoints":4, ...}
```

## 7. config.yaml 확장 초안

```yaml
camera:
  index: 0

models:
  person: yolov8n.pt
  face:
    score_threshold: 0.6
  pose: yolov8n-pose.pt

rois:
  safe:       { x1: 80, y1: 60, x2: 560, y2: 420 }
  climb_rail: { x1: 80, y1: 40, x2: 560, y2: 80 }   # 난간 상단 좁은 띠

heuristics:
  suffocation:
    min_duration_s: 5.0
    keypoint_conf_threshold: 0.5
    flipped_min_visible: 3     # 4개 중 3 이상 보이면 flipped
    blanket_max_visible: 0     # 전부 안 보이면 blanket
  climbing:
    min_duration_s: 2.0
    ankle_conf_threshold: 0.5
    standing_y_margin: 20      # px (절대값)
  roi_exit:
    min_duration_s: 0.0        # 즉시 판정, grace로 튐 완화

stability:
  grace_s: 0.5
  ema_alpha: 0.4

dispatcher:
  cooldown_s: 30.0
```

파라미터 값은 모두 초기 추정. Week 5 튜닝 단계에서 재조정 예정.

## 8. HUD 및 키 바인딩

### HUD

- person bbox (초록)
- face bbox (노랑)
- **pose 키포인트 점·뼈대 선** (파랑) *(신규)*
- `safe_roi` (연빨강, 실선)
- `climb_rail` (주황, 점선) *(신규)*
- 좌상단 debug: `persons`, `faces`, `face_in_p`, `visible_kp`, `ankle_ema`, `cause`, `face_elapsed`, `climb_elapsed`, `roi_exit`, `center`
- 활성 위험 빨강 텍스트에 `[suffocation_risk/flipped]`처럼 cause 병기

### 키 바인딩

| 키 | 동작 |
|---|---|
| `q` | 종료 |
| `r` | `safe_roi` 재선택 |
| `c` | `climb_rail` 재선택 *(신규)* |

## 9. 엣지 케이스

- **person 여러 명**: 가장 큰 bbox 선택 (v0 `main_person` 유지)
- **pose ↔ person 매칭**: IoU 최대 pose 연관. IoU=0이면 `main_pose=None`
- **pose=None + face 없음**: `cause="blanket"` 또는 `"unknown"` (파라미터에 따름). 놓치지 않음
- **ROI 기본값이 프레임 크기 초과**: 첫 프레임에서 프레임 크기로 클램핑 + 1회 경고 로그
- **EMA 첫 프레임**: 첫 값 그대로 초기화

## 10. 테스트 전략

### 단위 테스트 (`tests/test_heuristics.py`)

- mock `Person` / `Face` / `Pose` 데이터로 각 `evaluate_*` 경계:
  - `evaluate_suffocation`: face 있음/없음 × 키포인트 0/2/4개 보이는 케이스
  - `evaluate_climbing`: ankle ROI 안/밖 × 서있음 성립/불성립 × pose 없음
  - `evaluate_roi_exit`: center ROI 안/밖 × person 없음
- `DurationTracker`: grace 이내 튐 통과, grace 초과 리셋, required 경과 True 반환
- `EMA`: 첫 프레임 초기화, 수렴 값

### 수동 시나리오 (웹캠)

1. 얼굴 가리고 몸 노출 5초 → `suffocation_risk cause=flipped`
2. 얼굴·몸 전체 옷/담요로 덮기 5초 → `suffocation_risk cause=blanket`
3. 일어서서 난간 ROI에 발 올리기 2초 → `climbing_risk`
4. safe_roi 벗어나기 → `roi_exit_risk`
5. 각 이벤트 타입 쿨다운 30초 준수 확인

### 성능 체크

- YOLOv8n + YOLOv8n-pose + YuNet 동시 추론 CPU fps 측정. 목표 ≥ 5 fps
- 미달 시 `config.models.pose_skip_frames` 옵션 추가하여 N프레임마다만 pose 추론 (v1 초기엔 매 프레임, 추가 구현은 필요 시)

## 11. 리스크 및 후속 과제

| 리스크 | 대응 |
|---|---|
| pose 매 프레임 CPU 부담 | `pose_skip_frames` 옵션 예비. 실측 후 도입 |
| 성인 웹캠 테스트와 영유아 실환경 차이 | 실환경 샘플 확보 전까지 파라미터는 보수적으로(민감도 낮게) 세팅 |
| "서있음" 기준이 해상도 의존 (px 절대값) | 초기엔 단순. 필요 시 bbox 높이 비율로 변경 |
| climb_rail ROI 수동 설정 부담 | Week 후반 시연용 config 프리셋 두 개(침대/책상) 미리 작성 |

## 12. 스코프 외 (v1에서 하지 않음)

- 가구 자동 탐지, 다중 ROI(2개 초과), 키포인트 추적(시계열 궤적 기반 등반 판정)
- Kalman 필터
- 이벤트를 서버로 POST하는 HTTP 클라이언트 (Week 4)
- 음성 파이프라인 (Week 3)
