# AI 파트 — 영유아 안전 모니터링

침대(또는 놀이 영역)를 카메라로 지켜보며 **낙상·기어오름·질식(덮임)·영역 이탈·울음**을 감지하고 전송한다.

## 환경

- Ubuntu 22.04, Python 3.10
- CPU-only 노트북 (내장그래픽)에서 동작 확인
- (이후 Jetson orin nano 이식 및 GPU 사용 계획)

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행

```bash
python main.py
```

- 창이 뜨고 웹캠 프레임에 person bbox(초록) + face bbox(노랑) + pose 뼈대(파랑) + safe ROI(연빨강) 폴리곤이 그려짐
- 첫 실행 시 `yolov8n.pt`, `yolov8n-pose.pt`, `models/face_detection_yunet_2023mar.onnx`, YAMNet(~200MB) 자동 다운로드
- 키
  - `q` 종료
  - `r` ROI(안전 영역) 재정의 — 네 꼭짓점 다시 클릭
- HUD에 `cry_score` / `babble_score`와 각 휴리스틱 진단(diag) 표시 (audio off = 마이크 없음)

## ROI 설정 (안전 영역 지정)

안전 영역은 **4점 폴리곤** `[TL, TR, BR, BL]`로 정의한다. `config.yaml`의 `auto_roi.detector`로 방식 선택:

| 방식 | 설명 |
|---|---|
| `manual` | 첫 실행 시 마우스로 네 꼭짓점 클릭 → `saved_roi.json`에 저장 후 재사용. `r` 키로 재정의 |

- 클릭 선택 중: 좌클릭=점 추가(최대 4), `r`=초기화, `Enter`/`c`=확정, `q`/`ESC`=취소
- 검출 실패 시 `fallback_polygon` 사용
- `saved_roi.json`은 **카메라·침대 위치에 종속**

## 위험 판정 규칙

### 영상 (YOLOv8n + pose)

| 이벤트 | 조건 | 지속 시간 |
|---|---|---|
| `fall_risk` | person 중심 y 하강 속도 ≥ 200px/s (**raw bbox center**, EMA 미적용) | 0.3초 |
| `climbing_risk` | wrist가 ROI 변(난간)에 `rail_band_px` 이내 + 서있음 자세 | 2초 |
| `suffocation_risk` | 아래 두 경로 중 하나 | 5초 |
| `roi_exit_risk` | person 중심이 안전 ROI 밖 | 즉시(grace 0.5s) |

**suffocation_risk 원인 분기** (검출기 신뢰도·색이 아니라 텍스처로 구분. ROI 안에 있었고 face가 최근 `face_memory_s` 보인 적 있을 때만 판정):
- 공통 트리거: subject(pose 우선, 없으면 person) bbox에서 face가 안 보이는 상태가 지속.
- 원인은 subject bbox 안쪽의 **회색조 엣지 밀도(edge_density, 색 무관)**로 가른다 — 엎드린 인형은 person으로 거의 안 잡히지만 pose로는 잘 잡히고, 팔다리·옷·얼굴 윤곽으로 엣지가 많다(실측 0.046~0.159). 천에 덮이면 매끈한 표면이라 엣지가 적다(0.012~0.042). 색과 무관하게 갈려 `flipped_edge_threshold`(0.044)로 분리.
- `flipped`: 엎드림. subject 있음 + edge_density ≥ 임계(구조 있는 몸 노출) → 얼굴이 매트 쪽을 향함. 서버 이벤트 `PRONE_SUFFOCATION`(DANGER).
- `face_covered`: edge_density < 임계(매끈한 천), 또는 subject가 아예 없음(몸·머리까지 완전히 파묻혀 검출 붕괴). 서버 이벤트 `BLANKET_SUFFOCATION`(DANGER).
- **오탐 가드 1 — 정상 누움인데 face 미검출**: 카메라 각도상 YuNet이 얼굴을 못 잡아도, 천장을 보고 누우면(supine) pose가 얼굴 키포인트(코·눈·귀)를 높은 conf로 잡는다(실측 5/5, 0.93~0.98). 엎드리면(prone) 얼굴이 매트를 향해 죽는다(실측 1/5, nose 0.08). 그래서 `face_visible`을 **YuNet 검출 OR pose 얼굴 키포인트(`face_kp_conf_threshold` 이상이 `face_kp_min_visible`개 이상)**로 본다. 정상 누움은 `face_detected`로 빠져 위험 아님, 엎드림은 그대로 `flipped` 발송. 단 보조(OR) 신호로만 쓰므로 이불덮힘(키포인트도 안 보임)은 영향받지 않는다.
- **오탐 가드 2 — 발만 보임(out_of_view)**: `flipped` 후보라도 subject bbox의 ROI 포함율(`roi_containment`)이 `out_of_view_roi_threshold`(0.72) 미만이면 인형이 카메라 각도 안에 제대로 안 잡힌 상태(발만 보임 등)다. 엎드림 정탐은 포함율이 높고(실측 88%) 이 케이스는 낮아(68%) 갈린다. 이때는 위험이 아니라 `out_of_view`로 처리해 `PRONE` 오발송을 막는다. `flipped` 분기에만 적용(face_covered는 별개 경로).
- 한계 1: 주름·무늬가 심한 천은 엣지가 올라가 `flipped`로 튈 수 있음(5초 지속 트리거가 산발적 튐은 흡수).
- 한계 2: **얼굴에만 이불을 덮으면** 몸·팔다리 텍스처가 살아 있어 edge가 높아 `flipped`로 오판함 → `BLANKET`이어야 할 게 `PRONE`으로 발송. 얼굴 위치를 모르는(얼굴 미검출) 단일 프레임 텍스처의 구조적 한계라 못 고침. 데모에선 이불을 상체까지 덮어 `face_covered`로 가게 연출해 회피.

> 두 원인은 **서버 이벤트가 분리됨**: `flipped`→`PRONE_SUFFOCATION`, `face_covered`→`BLANKET_SUFFOCATION` (둘 다 DANGER).
> **알려진 한계:** 보호자가 아기를 손으로 들어올려 ROI 밖으로 빼면 `roi_exit`가 경계를 못 잡아 `face_covered` 오탐이 날 수 있음. 안전 우선(false positive < false negative) 원칙으로 그대로 둠.

### 음성 (YAMNet AudioSet)

| 이벤트 | YAMNet 클래스 | score 임계값 | 지속 시간 |
|---|---|---|---|
| `cry_detected` | Baby cry, Crying | 0.3 | 1초 |
| `babble_detected` | Babbling | 0.25 | 2초 |

- 두 이벤트 모두 **person이 화면에 있을 때만** 판정

### 서버 이벤트 매핑 (MQTT)

| 내부 신호 | eventType | severity |
|---|---|---|
| `fall_risk` | `FALL` | DANGER |
| `suffocation_risk` (cause=`face_covered`) | `BLANKET_SUFFOCATION` | DANGER |
| `suffocation_risk` (cause=`flipped`) | `PRONE_SUFFOCATION` | DANGER |
| `climbing_risk` | `CLIMBING` | CAUTION |
| `roi_exit_risk` | `ROI_EXIT` | CAUTION |
| `cry_detected` | `CRYING` | CAUTION |
| `babble_detected` | `WHINING` | INFO |

- 위험 시작/종료를 각각 publish (`phase`: START / END), payload에 `duration_s`·`startedAt`·`endedAt` 포함
- 토픽·디바이스 시리얼은 `config.yaml`의 `mqtt` 섹션에서 설정

### 공통

- 좌표 스무딩: 지수이동평균(α=0.4) — 단, **fall은 raw center 사용**
- 순간 튐 완화: `DurationTracker` grace 0.5초
- 알림 폭주 방지: 동일 이벤트 타입 30초 쿨다운


## 진행 상황

- [x] 1: 웹캠 + YOLOv8n 렌더 루프
- [x] 2: pose + ROI 기반 영상 휴리스틱 (suffocation/climbing/roi_exit)
- [x] 3: fall_risk 분리 + YAMNet 울음·옹알이 감지 + duration_s 전송
- [x] 4: 서버 연동 (MQTT)
- [x] 5: 폴리곤 ROI 전환 + 질식 사라짐 추적 재설계 + 실물(인형) 검증
- [ ] 6: 데모 준비 / Jetson 이식
