# Audio v2 — YAMNet 울음 감지 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** YAMNet 기반 아기 울음 감지를 v1 영상 파이프라인에 추가한다. 울음 + person 동시 감지 시 `cry_detected` 이벤트 발송.

**Architecture:** sounddevice 콜백이 16kHz 오디오를 0.96s 청크로 축적 → YAMNet 추론 → 슬라이딩 윈도우 평균 → shared state. 메인 루프는 매 프레임 `get_state()`로 읽어 person AND 조건으로 `cry_detected` 이벤트를 생성한다. 마이크 없이도 영상 파이프라인은 정상 동작.

**Tech Stack:** tensorflow 2.21, tensorflow-hub 0.16, sounddevice 0.5, numpy (기존 의존), YAMNet TFHub 모델

---

## 파일 구조

| 파일 | 작업 | 역할 |
|---|---|---|
| `audio/__init__.py` | 신규 | 빈 패키지 파일 |
| `audio/yamnet_classifier.py` | 신규 | YAMNet 로드, 마이크 캡처, shared state 갱신 |
| `tests/test_yamnet_classifier.py` | 신규 | 슬라이딩 윈도우·임계값 단위 테스트 |
| `config.yaml` | 수정 | `audio:` 섹션 추가 |
| `main.py` | 수정 | AudioClassifier 시작/종료, cry tracker 추가 |

`requirements.txt`는 이미 `tensorflow`, `tensorflow-hub`, `sounddevice`가 포함돼 있어 변경 불필요.

---

### Task 1: config.yaml — audio 섹션 추가

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: `audio:` 섹션을 config.yaml 하단에 추가**

`dispatcher:` 블록 아래에 추가:

```yaml
audio:
  sample_rate: 16000
  chunk_duration_s: 0.96      # YAMNet 입력 단위 (초)
  window_chunks: 3             # 슬라이딩 윈도우 크기 (~2.9초)
  score_threshold: 0.3         # cry_active=True 임계값
  min_duration_s: 1.0          # cry_tracker 판정까지 최소 지속 시간
```

- [ ] **Step 2: 파싱 확인**

```bash
python3 -c "import yaml; cfg=yaml.safe_load(open('config.yaml')); print(cfg['audio'])"
```

Expected:
```
{'sample_rate': 16000, 'chunk_duration_s': 0.96, 'window_chunks': 3, 'score_threshold': 0.3, 'min_duration_s': 1.0}
```

- [ ] **Step 3: 커밋**

```bash
git add config.yaml
git commit -m "chore: config.yaml audio 섹션 추가"
```

---

### Task 2: AudioClassifier 스켈레톤 + 슬라이딩 윈도우 TDD

YAMNet·sounddevice 없이 테스트 가능한 로직(`_window_mean`, `get_state`)을 TDD로 구현한다.

**Files:**
- Create: `audio/__init__.py`
- Create: `audio/yamnet_classifier.py`
- Create: `tests/test_yamnet_classifier.py`

- [ ] **Step 1: `audio/__init__.py` 생성**

내용은 비워둔다:
```python
```

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/test_yamnet_classifier.py`:

```python
from audio.yamnet_classifier import AudioClassifier

CFG = {
    "sample_rate": 16000,
    "chunk_duration_s": 0.96,
    "window_chunks": 3,
    "score_threshold": 0.3,
    "min_duration_s": 1.0,
}


def test_window_mean_single_chunk():
    clf = AudioClassifier(CFG)
    mean = clf._window_mean(0.5)
    assert mean == 0.5


def test_window_mean_averages_three_chunks():
    clf = AudioClassifier(CFG)
    clf._window_mean(0.2)
    clf._window_mean(0.4)
    mean = clf._window_mean(0.6)
    assert abs(mean - 0.4) < 1e-6


def test_window_capped_at_window_chunks():
    # 5청크 추가, 마지막 3개만 유효: [0.0, 0.0, 0.9] → mean = 0.3
    clf = AudioClassifier(CFG)
    for _ in range(5):
        clf._window_mean(0.0)
    mean = clf._window_mean(0.9)
    assert abs(mean - 0.3) < 1e-6


def test_window_below_threshold_cry_inactive():
    clf = AudioClassifier(CFG)
    clf._window_mean(0.1)
    clf._window_mean(0.1)
    clf._update_state(clf._window_mean(0.1))
    active, score = clf.get_state()
    assert active is False
    assert score < 0.3


def test_window_above_threshold_cry_active():
    clf = AudioClassifier(CFG)
    clf._window_mean(0.5)
    clf._window_mean(0.5)
    clf._update_state(clf._window_mean(0.5))
    active, score = clf.get_state()
    assert active is True
    assert score >= 0.3


def test_get_state_initial():
    clf = AudioClassifier(CFG)
    active, score = clf.get_state()
    assert active is False
    assert score == 0.0
```

- [ ] **Step 3: 테스트 실패 확인**

```bash
source .venv/bin/activate && python -m pytest tests/test_yamnet_classifier.py -v
```

Expected: `ModuleNotFoundError: No module named 'audio.yamnet_classifier'`

- [ ] **Step 4: `audio/yamnet_classifier.py` 스켈레톤 구현**

```python
import threading


class AudioClassifier:
    def __init__(self, cfg: dict) -> None:
        self._sr: int = cfg["sample_rate"]
        self._chunk_samples: int = int(self._sr * cfg["chunk_duration_s"])
        self._window_chunks: int = cfg["window_chunks"]
        self._threshold: float = cfg["score_threshold"]
        self._scores: list[float] = []
        self._lock = threading.Lock()
        self._cry_active: bool = False
        self._cry_score: float = 0.0
        self._stream = None
        self._model = None
        self._cry_indices: list[int] = []

    def _window_mean(self, score: float) -> float:
        self._scores.append(score)
        if len(self._scores) > self._window_chunks:
            self._scores.pop(0)
        return sum(self._scores) / len(self._scores)

    def _update_state(self, mean: float) -> None:
        with self._lock:
            self._cry_active = mean >= self._threshold
            self._cry_score = mean

    def get_state(self) -> tuple[bool, float]:
        with self._lock:
            return self._cry_active, self._cry_score

    def start(self) -> bool:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
python -m pytest tests/test_yamnet_classifier.py -v
```

Expected: 6 passed

- [ ] **Step 6: 전체 테스트 회귀 확인**

```bash
python -m pytest -q
```

Expected: 41 passed

- [ ] **Step 7: 커밋**

```bash
git add audio/__init__.py audio/yamnet_classifier.py tests/test_yamnet_classifier.py
git commit -m "feat: AudioClassifier 스켈레톤 + 슬라이딩 윈도우 TDD"
```

---

### Task 3: YAMNet 로드 + 마이크 캡처 구현

`start()` / `stop()` 과 오디오 콜백을 구현한다. YAMNet·sounddevice가 필요하므로 단위 테스트 없이 구현 후 수동 연기 테스트.

**Files:**
- Modify: `audio/yamnet_classifier.py`

- [ ] **Step 1: import 추가 및 `_buf` 필드 추가**

`audio/yamnet_classifier.py` 상단과 `__init__`을 아래로 교체:

```python
import threading

import numpy as np


class AudioClassifier:
    def __init__(self, cfg: dict) -> None:
        self._sr: int = cfg["sample_rate"]
        self._chunk_samples: int = int(self._sr * cfg["chunk_duration_s"])
        self._window_chunks: int = cfg["window_chunks"]
        self._threshold: float = cfg["score_threshold"]
        self._scores: list[float] = []
        self._buf: np.ndarray = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._cry_active: bool = False
        self._cry_score: float = 0.0
        self._stream = None
        self._model = None
        self._cry_indices: list[int] = []
```

- [ ] **Step 2: `_load_model()` 구현 추가**

`get_state()` 아래에 추가:

```python
    def _load_model(self) -> bool:
        try:
            import csv
            import tensorflow_hub as hub
            self._model = hub.load("https://tfhub.dev/google/yamnet/1")
            class_map_path = self._model.class_map_path().numpy().decode()
            with open(class_map_path) as f:
                reader = csv.DictReader(f)
                self._cry_indices = [
                    int(row["index"]) for row in reader
                    if "cry" in row["display_name"].lower()
                    or "infant" in row["display_name"].lower()
                ]
            print(f"[Audio] YAMNet 로드 완료, cry 클래스 {len(self._cry_indices)}개")
            return True
        except Exception as e:
            print(f"[Audio] 모델 로드 실패: {e}")
            return False
```

- [ ] **Step 3: `_process_chunk()` + `_audio_callback()` 구현 추가**

`_load_model()` 아래에 추가:

```python
    def _process_chunk(self, chunk: np.ndarray) -> None:
        scores, _, _ = self._model(chunk)          # scores: (N_frames, 521)
        cry_score = float(
            np.max(scores.numpy()[:, self._cry_indices])
        )
        mean = self._window_mean(cry_score)
        self._update_state(mean)

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        mono = indata[:, 0].astype(np.float32)
        self._buf = np.concatenate([self._buf, mono])
        while len(self._buf) >= self._chunk_samples:
            chunk = self._buf[:self._chunk_samples]
            self._buf = self._buf[self._chunk_samples:]
            self._process_chunk(chunk)
```

- [ ] **Step 4: `start()` / `stop()` 구현**

`NotImplementedError` 두 줄을 아래로 교체:

```python
    def start(self) -> bool:
        import sounddevice as sd
        try:
            sd.query_devices(kind="input")
        except Exception:
            print("[Audio] 마이크 없음 — 음성 감지 비활성화")
            return False
        if not self._load_model():
            return False
        self._stream = sd.InputStream(
            samplerate=self._sr,
            channels=1,
            callback=self._audio_callback,
        )
        self._stream.start()
        print("[Audio] 마이크 스트림 시작")
        return True

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
```

- [ ] **Step 5: 기존 단위 테스트 회귀 확인**

```bash
python -m pytest tests/test_yamnet_classifier.py -v
```

Expected: 6 passed (스켈레톤 로직은 그대로이므로 변화 없음)

- [ ] **Step 6: 커밋**

```bash
git add audio/yamnet_classifier.py
git commit -m "feat: YAMNet 로드 + 마이크 캡처 구현"
```

---

### Task 4: main.py 통합 — cry tracker + 이벤트

**Files:**
- Modify: `main.py`

- [ ] **Step 1: import 추가**

`main.py` 상단 import 블록에 추가:

```python
from audio.yamnet_classifier import AudioClassifier
```

- [ ] **Step 2: `main()` 함수 — AudioClassifier 초기화 + cry tracker 추가**

`stab_cfg = cfg["stability"]` 아래에 추가:

```python
    aud_cfg = cfg["audio"]
    audio = AudioClassifier(aud_cfg)
    audio_on = audio.start()

    cry_tracker = DurationTracker(aud_cfg["min_duration_s"], stab_cfg["grace_s"])
```

- [ ] **Step 3: 메인 루프 — cry 조건 판정 추가**

`exit_active, exit_diag = evaluate_roi_exit(...)` 줄 바로 아래에 추가:

```python
            cry_raw, cry_score = audio.get_state() if audio_on else (False, 0.0)
            cry_condition = cry_raw and p is not None
```

- [ ] **Step 4: 메인 루프 — cry_detected 이벤트 추가**

`if exit_tracker.update(exit_active, now):` 블록 바로 아래에 추가:

```python
            if cry_tracker.update(cry_condition, now):
                active_risks.append(RiskSignal(
                    "cry_detected",
                    cry_score,
                    {"heuristic": "yamnet_cry_and_person_present"},
                ))
```

- [ ] **Step 5: debug dict — cry 항목 추가**

`debug = { ... }` 딕셔너리에 항목 추가:

```python
                "cry_score": f"{cry_score:.2f}" if audio_on else "off",
                "cry_elapsed": f"{cry_tracker.elapsed(now):.1f}s",
```

- [ ] **Step 6: finally 블록 — audio.stop() 추가**

`finally:` 블록에 추가:

```python
        if audio_on:
            audio.stop()
```

기존 `finally:` 블록:
```python
    finally:
        cam.release()
        cv2.destroyAllWindows()
```

변경 후:
```python
    finally:
        if audio_on:
            audio.stop()
        cam.release()
        cv2.destroyAllWindows()
```

- [ ] **Step 7: 전체 테스트 회귀 확인**

```bash
python -m pytest -q
```

Expected: 41 passed

- [ ] **Step 8: 커밋**

```bash
git add main.py
git commit -m "feat: main.py cry_detected 이벤트 통합"
```

---

### Task 5: README 업데이트

**Files:**
- Modify: `README.md`

- [ ] **Step 1: README 업데이트**

`## 진행 상황` 섹션을 아래로 교체:

```markdown
## 진행 상황

- [x] Week 1: 웹캠 + YOLOv8n 렌더 루프 (2026-04-23 확인)
- [x] Week 2 재구성: 얼굴 미탐 + ROI 이탈 2규칙 (v0)
- [x] Week 2 v1: pose + cause 분기 + 판정 안정화
- [x] Week 3: YAMNet 울음 감지 + person AND 조건 (v2)
- [ ] Week 4: 이벤트 디스패처 + REST 연동
- [ ] Week 5: 통합 테스트·튜닝
- [ ] Week 6: 데모 준비
```

`## 실행` 섹션의 키 목록에 추가:

```markdown
- HUD에 `cry_score` (off = 마이크 없음), `cry_elapsed` 표시
- 첫 실행 시 YAMNet 모델 자동 다운로드 (~200MB, TFHub 캐시)
```

- [ ] **Step 2: 커밋**

```bash
git add README.md
git commit -m "docs: README v2 반영"
```

---

## 수동 통합 테스트

코드 완성 후 아래 순서로 직접 확인:

1. `python main.py` 실행 → 터미널에 `[Audio] YAMNet 로드 완료, cry 클래스 N개` 출력 확인
2. HUD에 `cry_score: 0.00`, `cry_elapsed: 0.0s` 표시 확인
3. 아기 울음 오디오 클립을 스피커로 재생 + 카메라 앞에 서기
   → `cry_score` 상승, 1초 후 터미널에 `[EVENT] cry_detected` 출력 확인
4. 마이크 없이 실행 → `[Audio] 마이크 없음` 출력, HUD에 `cry_score: off`, 영상 정상 동작 확인
