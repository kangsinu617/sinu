"""MQTT publish + 실패 시 로컬 큐.

paho-mqtt 자체 자동 재연결을 활용. publish 실패하거나 연결 안 되어 있으면
JSONL 파일에 append. 연결되면 큐를 flush.
"""
import json
import threading
from pathlib import Path
from typing import Optional

import paho.mqtt.client as mqtt


class MqttPublisher:
    def __init__(
        self,
        host: str,
        port: int,
        topic: str,
        queue_path: str = "events_queue.jsonl",
    ) -> None:
        self._host = host
        self._port = port
        self._topic = topic
        self._queue_path = Path(queue_path)
        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._lock = threading.Lock()

    def _on_connect(self, client, userdata, flags, rc, properties=None) -> None:
        self._connected = rc == 0
        if self._connected:
            print(f"[MQTT] 연결됨: {self._host}:{self._port}")
            self._flush_queue()
        else:
            print(f"[MQTT] 연결 실패 rc={rc}")

    def _on_disconnect(self, client, userdata, *args) -> None:
        self._connected = False
        print("[MQTT] 연결 끊김")

    def start(self) -> bool:
        try:
            self._client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.connect_async(self._host, self._port, keepalive=60)
            self._client.loop_start()
            print(f"[MQTT] 시작 — {self._host}:{self._port} 토픽 '{self._topic}'")
            return True
        except Exception as e:
            print(f"[MQTT] 시작 실패: {e}")
            self._client = None
            return False

    def publish(self, payload: dict) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        if self._client is None or not self._connected:
            self._enqueue(line)
            return
        result = self._client.publish(self._topic, line, qos=1)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            self._enqueue(line)

    def _enqueue(self, line: str) -> None:
        with self._lock:
            with self._queue_path.open("a") as f:
                f.write(line + "\n")

    def _flush_queue(self) -> None:
        if not self._queue_path.exists():
            return
        with self._lock:
            with self._queue_path.open("r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if not lines:
                self._queue_path.unlink()
                return
            for line in lines:
                self._client.publish(self._topic, line, qos=1)
            self._queue_path.unlink()
            print(f"[MQTT] 큐 {len(lines)}건 flush")

    def stop(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
