# Audio v2 — YAMNet 울음 감지 설계

## Goal

v1 영상 파이프라인에 YAMNet 기반 음성 분류를 추가해 아기 울음을 감지하고, person이 영상에서 검출된 상태에서만 `cry_detected` 이벤트를 발송한다.

## Architecture

오디오 스레드가 shared state를 갱신하고, 기존 메인 루프가 매 프레임 읽는 구조. 영상 파이프라인(v1)은 변경 없이 유지된다.

```
[마이크]
  └─ sounddevice InputStream callback
       └─ sample_buffer (16kHz, 1ch)
            └─ 0.96s 청크 완성 시 YAMNet 추론
                 └─ cry 클래스 점수 추출 → 슬라이딩 윈도우 평균
                      └─ shared state: (cry_active, cry_score)  ← Lock 보호

[메인 루프 — 매 프레임]
  └─ get_state() → cry_active
       └─ p is not None AND cry_active
            └─ cry_tracker.update() → cry_detected 이벤트
```

## 파일 구조

| 파일 | 역할 |
|---|---|
| `audio/__init__.py` | 빈 패키지 파일 |
| `audio/yamnet_classifier.py` | YAMNet 로드, 마이크 캡처, shared state 갱신 |
| `config.yaml` | `audio:` 섹션 추가 |
| `main.py` | AudioClassifier 시작/종료, cry 이벤트 통합 |
| `requirements.txt` | `tensorflow`, `tensorflow-hub`, `sounddevice` 추가 |
| `tests/test_yamnet_classifier.py` | 슬라이딩 윈도우·임계값 단위 테스트 |

## 컴포넌트 상세

### `audio/yamnet_classifier.py`

```python
class AudioClassifier:
    def __init__(self, sample_rate, chunk_duration_s, window_chunks, score_threshold): ...
    def start(self) -> bool          # 마이크 없으면 False 반환, 영상은 계속 동작
    def stop(self) -> None
    def get_state(self) -> tuple[bool, float]   # (cry_active, cry_score)
```

- TFHub YAMNet(`https://tfhub.dev/google/yamnet/1`) 로드
- 관심 클래스: AudioSet 클래스명에 `"cry"` 또는 `"infant"` 포함하는 것만 필터
- 슬라이딩 윈도우: 최근 `window_chunks`개 청크의 cry 점수 평균
- `window_mean ≥ score_threshold` → `cry_active = True`
- shared state는 `threading.Lock`으로 보호

### `config.yaml` 추가 섹션

```yaml
audio:
  sample_rate: 16000
  chunk_duration_s: 0.96      # YAMNet 입력 단위
  window_chunks: 3             # 슬라이딩 윈도우 (~2.9초)
  score_threshold: 0.3
  min_duration_s: 1.0          # cry_tracker 판정까지 최소 지속 시간
```

### `main.py` 변경

- 루프 시작 전 `AudioClassifier.start()` 호출
- 매 프레임 `get_state()` → `cry_active, cry_score` 읽기
- `p is not None and cry_active` 조건 → `cry_tracker.update()`
- `cry_tracker`가 duration 충족 시 `cry_detected` 이벤트 발송
- `finally` 블록에서 `classifier.stop()`

## 에러 처리

| 상황 | 처리 |
|---|---|
| 마이크 미연결 | `start()` → `False`, `cry_active` 항상 `False`, 영상 정상 동작 |
| TFHub 다운로드 실패 | 예외 캐치 → 오디오 비활성화, 경고 출력 |
| 스레드 예외 | 스레드 내부에서 잡아 `cry_active=False`로 유지, 메인 루프에 영향 없음 |

## 테스트 전략

### 단위 테스트 (`tests/test_yamnet_classifier.py`)

- 슬라이딩 윈도우 평균 계산 정확성
- `score ≥ threshold` → `cry_active=True`
- `score < threshold` → `cry_active=False`
- 청크 수가 `window_chunks`보다 적을 때 평균 계산

### 수동 통합 테스트

- 아기 울음 오디오 클립 재생 → `[cry_detected]` 이벤트 확인
- 무음 상태에서 오탐 없는지 확인
- 마이크 없이 실행해도 영상 파이프라인 정상 동작 확인

## 범위 밖 (v2에서 안 하는 것)

- `Screaming`, `Babbling` 등 추가 클래스 (v3 이후)
- REST API 연동 / 이벤트 디스패처 (별도 태스크)
- 영상·음성 이벤트 AND 조건 고도화
