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
