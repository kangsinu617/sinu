import threading

import numpy as np


class AudioClassifier:
    def __init__(self, cfg: dict) -> None:
        self._sr: int = cfg["sample_rate"]
        self._chunk_samples: int = int(self._sr * cfg["chunk_duration_s"])
        self._window_chunks: int = cfg["window_chunks"]
        self._threshold: float = cfg["score_threshold"]
        self._scores: list[float] = []  # audio callback thread only — no lock needed
        self._buf: np.ndarray = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._cry_active: bool = False
        self._cry_score: float = 0.0
        self._stream = None
        self._model = None
        self._cry_indices: list[int] = []  # populated by _load_model

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
            if not self._cry_indices:
                print("[Audio] cry/infant 클래스를 CSV에서 찾지 못함 — 모델 로드 실패")
                return False
            print(f"[Audio] YAMNet 로드 완료, cry 클래스 {len(self._cry_indices)}개")
            return True
        except Exception as e:
            print(f"[Audio] 모델 로드 실패: {e}")
            return False

    def _process_chunk(self, chunk: np.ndarray) -> None:
        # TODO: offload to a worker thread — TF inference blocks the PortAudio callback
        try:
            scores, _, _ = self._model(chunk)      # scores: (N_frames, 521)
            cry_score = float(
                np.max(scores.numpy()[:, self._cry_indices])
            )
        except Exception as e:
            print(f"[Audio] 추론 오류: {e}")
            return
        mean = self._window_mean(cry_score)
        self._update_state(mean)

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        mono = indata[:, 0].astype(np.float32)
        self._buf = np.concatenate([self._buf, mono])
        while len(self._buf) >= self._chunk_samples:
            chunk = self._buf[:self._chunk_samples]
            self._buf = self._buf[self._chunk_samples:]
            self._process_chunk(chunk)

    def start(self) -> bool:
        import sounddevice as sd
        try:
            sd.query_devices(kind="input")
        except Exception:
            print("[Audio] 마이크 없음 — 음성 감지 비활성화")
            return False
        if not self._load_model():
            return False
        try:
            self._stream = sd.InputStream(
                samplerate=self._sr,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as e:
            print(f"[Audio] 스트림 시작 실패: {e}")
            self._stream = None
            return False
        print("[Audio] 마이크 스트림 시작")
        return True

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
