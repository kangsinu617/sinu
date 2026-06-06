import json
from unittest.mock import MagicMock

import paho.mqtt.client as mqtt
import pytest

from events.mqtt_client import MqttPublisher


@pytest.fixture
def queue_file(tmp_path):
    return tmp_path / "events_queue.jsonl"


def _publisher(queue_file, connected: bool = False, client=None):
    pub = MqttPublisher("localhost", 1883, "test/topic", queue_path=str(queue_file))
    pub._connected = connected
    pub._client = client
    return pub


def test_publish_when_disconnected_enqueues(queue_file):
    pub = _publisher(queue_file, connected=False, client=None)
    pub.publish({"eventType": "FALL", "duration": 1})
    assert queue_file.exists()
    line = queue_file.read_text().strip()
    assert json.loads(line)["eventType"] == "FALL"


def test_publish_when_connected_calls_client(queue_file):
    client = MagicMock()
    client.publish.return_value.rc = mqtt.MQTT_ERR_SUCCESS
    pub = _publisher(queue_file, connected=True, client=client)
    pub.publish({"eventType": "CRYING", "duration": 2})
    client.publish.assert_called_once()
    topic, payload = client.publish.call_args.args[:2]
    assert topic == "test/topic"
    assert json.loads(payload)["eventType"] == "CRYING"
    assert not queue_file.exists()


def test_publish_failure_falls_back_to_queue(queue_file):
    client = MagicMock()
    client.publish.return_value.rc = mqtt.MQTT_ERR_NO_CONN
    pub = _publisher(queue_file, connected=True, client=client)
    pub.publish({"eventType": "CLIMBING"})
    assert queue_file.exists()
    assert json.loads(queue_file.read_text().strip())["eventType"] == "CLIMBING"


def test_flush_queue_publishes_all_lines_and_removes_file(queue_file):
    queue_file.write_text(
        json.dumps({"eventType": "FALL"}) + "\n"
        + json.dumps({"eventType": "CRYING"}) + "\n"
    )
    client = MagicMock()
    pub = _publisher(queue_file, connected=True, client=client)
    pub._flush_queue()
    assert client.publish.call_count == 2
    assert not queue_file.exists()


def test_flush_queue_noop_when_no_file(queue_file):
    client = MagicMock()
    pub = _publisher(queue_file, connected=True, client=client)
    pub._flush_queue()
    client.publish.assert_not_called()


def test_enqueue_appends_multiple(queue_file):
    pub = _publisher(queue_file, connected=False, client=None)
    pub.publish({"eventType": "FALL"})
    pub.publish({"eventType": "CRYING"})
    lines = queue_file.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["eventType"] == "FALL"
    assert json.loads(lines[1])["eventType"] == "CRYING"
