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
- 첫 실행 시 `yolo26n.pt`, `yolo26n-pose.pt`, `models/face_detection_yunet_2023mar.onnx`, YAMNet(~200MB) 자동 다운로드
- 키
  - `q` 종료
  - `r` ROI(안전 영역) 재정의 — 네 꼭짓점 다시 클릭
- HUD에 핵심 상태(persons/faces/cause/suf·clm_elapsed/cry/motion/head/torso/roi_in) 표시 (audio off = 마이크 없음)

## ROI 설정 (안전 영역 지정)

안전 영역은 **4점 폴리곤** `[TL, TR, BR, BL]`로 정의한다. `config.yaml`의 `auto_roi.detector`로 방식 선택:

| 방식 | 설명 |
|---|---|
| `manual` | 첫 실행 시 마우스로 네 꼭짓점 클릭 → `saved_roi.json`에 저장 후 재사용. `r` 키로 재정의 |

- 클릭 선택 중: 좌클릭=점 추가(최대 4), `r`=초기화, `Enter`/`c`=확정, `q`/`ESC`=취소
- 검출 실패 시 `fallback_polygon` 사용
- `saved_roi.json`은 **카메라·침대 위치에 종속**

## 위험 판정 규칙

### 영상 (YOLO26n + pose)

| 이벤트 | 조건 | 지속 시간 |
|---|---|---|
| `fall_risk` | 1.5초 윈도우 내 순 하강 ≥ 80px (**raw bbox center**, EMA 미적용) | — |
| `climbing_risk` | wrist가 ROI 변(난간)에 `rail_band_px` 이내 + 서있음 자세 | 2초 |
| `suffocation_risk` | 아래 두 경로 중 하나 | 5초 |
| `roi_exit_risk` | person 중심이 안전 ROI 밖 | 즉시(grace 0.5s) |

**suffocation_risk 원인 분기** (검출기 신뢰도·색이 아니라 pose 몸통 키포인트 가시성으로 구분. ROI 안에 있었고 face가 최근 `face_memory_s` 보인 적 있을 때만 판정):
- 공통 트리거: subject(pose 우선, 없으면 person) bbox에서 face가 안 보이는 상태가 지속.
- 원인은 pose의 **몸통 키포인트(어깨·엉덩이) 가시성**으로 가른다 — 엎드린 인형은 등을 카메라로 향해 어깨·엉덩이가 잘 잡히고(실측 4/4, conf 0.95~0.99), 천에 덮이면 키포인트가 0.0으로 죽는다(0/4). 구간이 압도적으로 멀어 `torso_kp_conf_threshold`(0.5) 이상이 `torso_kp_min_visible`(2)개 이상이면 `flipped`로 분리. (이전엔 회색조 엣지밀도 `edge_density`로 갈랐으나 prone 0.046~0.159 vs covered 0.012~0.042로 구간이 붙어 있어 교체 — 천 주름·무늬에 흔들리던 오탐도 함께 사라짐.)
- `flipped`: 엎드림. subject 있음 + 몸통 키포인트 노출(torso_visible) → 얼굴이 매트 쪽을 향함. 서버 이벤트 `PRONE_SUFFOCATION`(DANGER).
- `face_covered`: 몸통 키포인트가 천에 덮여 소실(torso_visible False), 또는 subject가 아예 없음(몸·머리까지 완전히 파묻혀 검출 붕괴). 서버 이벤트 `BLANKET_SUFFOCATION`(DANGER).
- **오탐 가드 1 — 정상 누움인데 face 미검출**: 카메라 각도상 YuNet이 얼굴을 못 잡아도, 천장을 보고 누우면(supine) pose가 얼굴 키포인트(코·눈·귀)를 높은 conf로 잡는다(실측 5/5, 0.93~0.98). 엎드리면(prone) 얼굴이 매트를 향해 죽는다(실측 1/5, nose 0.08). 그래서 `face_visible`을 **YuNet face가 person 안에 검출 OR ROI 안에 face 검출 OR pose 얼굴 키포인트(`face_kp_conf_threshold` 이상이 `face_kp_min_visible`개 이상)**로 본다. 정상 누움은 `face_detected`로 빠져 위험 아님, 엎드림은 그대로 `flipped` 발송. **ROI 안 face 항을 둔 이유**: 몸만 덮여 YOLO person이 실패해도(=`p` None) 얼굴이 노출돼 있으면 안전이어야 하는데, person bbox에만 묶으면 "얼굴만 보임"이 질식으로 오탐난다(얼굴 노출=안전 원칙). 단 셋 다 보조(OR) 신호라 이불덮힘(얼굴까지 가려짐)은 영향받지 않는다.
- **오탐 가드 2 — 발만 보임(out_of_view)**: subject bbox의 ROI 포함율(`roi_containment`)이 `out_of_view_roi_threshold`(0.72) 미만이면 인형이 카메라 각도 안에 제대로 안 잡힌 상태(발만 보임 등)다. 엎드림 정탐은 포함율이 높고(실측 88%) 이 케이스는 낮아(68%) 갈린다. 이때는 위험이 아니라 `out_of_view`로 처리한다. **`flipped`/`face_covered` 공통 적용**(torso로 원인을 가르기 전에 시야 밖을 먼저 거른다) — 단, subject 자체가 아예 없는 완전 파묻힘은 이 가드보다 앞서 `face_covered`로 반환해, ROI 포함율 가드가 진짜 질식을 `out_of_view`로 놓치지 않게 한다.
- **오탐 가드 3 — 능동적 움직임(active_motion)**: `flipped` 후보라도 subject 영역의 프레임 간 활동량(회색조 절대차 평균/255이 `motion_threshold`(0.02) 이상)이 크면 살아 움직이는 중(배 시간·능동적 버둥거림)이라 무반응 질식이 아니다 → `active_motion`(위험 아님). 인형(정지)·천 덮인 무반응은 활동량이 낮아 그대로 위험으로 남는다. `flipped` 분기 전용(face_covered는 천에 덮인 채 버둥거려도 위험할 수 있어 면제 안 함). 단일 프레임 신호라 산발적이어도 5초 지속 트리거가 연속 정지만 채택하므로 가끔의 움직임으로도 카운트가 리셋된다. **인형은 정지 상태라 데모에서는 이 경로가 나타나지 않고, 실제 영아 적용 시 무반응만 위험으로 좁히는 가드다.** HUD의 `motion` 값으로 임계 측정·튜닝 필요.
- 한계: **얼굴에만 이불을 덮으면** 몸통 키포인트(어깨·엉덩이)가 살아 있어 `flipped`로 오판함 → `BLANKET`이어야 할 게 `PRONE`으로 발송. 얼굴 위치를 모르는(얼굴 미검출) 단일 프레임의 구조적 한계라 못 고침. 데모에선 이불을 상체까지 덮어 키포인트를 죽여 `face_covered`로 가게 연출해 회피.

> 두 원인은 **서버 이벤트가 분리됨**: `flipped`→`PRONE_SUFFOCATION`, `face_covered`→`BLANKET_SUFFOCATION` (둘 다 DANGER).
> **알려진 한계:** 보호자가 아기를 손으로 들어올려 ROI 밖으로 빼면 `roi_exit`가 경계를 못 잡아 `face_covered` 오탐이 날 수 있음. 안전 우선(false positive < false negative) 원칙으로 그대로 둠.

### 음성 (YAMNet AudioSet)

| 이벤트 | YAMNet 클래스 | score 임계값 | 지속 시간 |
|---|---|---|---|
| `cry_detected` | Baby cry, Crying | 0.3 | 1초 |
| `babble_detected` | Babbling | 0.25 | 2초 |

- **person이 화면에 있을 때만** 판정

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

- [x] 1: 웹캠 + YOLO26n 렌더 루프
- [x] 2: pose + ROI 기반 영상 휴리스틱 (suffocation/climbing/roi_exit)
- [x] 3: fall_risk 분리 + YAMNet 울음 감지 + duration_s 전송
- [x] 4: 서버 연동 (MQTT)
- [x] 5: 폴리곤 ROI 전환 + 질식 사라짐 추적 재설계 + 실물(인형) 검증
- [x] 6: 데모 준비 완료 / Jetson 이식 예정
