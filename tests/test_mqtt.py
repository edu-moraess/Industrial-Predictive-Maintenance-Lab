import json
from unittest.mock import patch

from mqtt.client import MQTTTelemetryListener


def test_mqtt_payload_parsing():
    """Ensure MQTT payload is parsed and persisted via repository."""
    listener = MQTTTelemetryListener()

    sample_payload = json.dumps({
        "machine_id": "TEST_ESP32",
        "temperature": 55.0,
        "vibration": 2.1,
        "current": 15.0,
        "rpm": 1800.0,
        "noise": 70.0,
        "state": "OPERATIONAL",
    }).encode("utf-8")

    class MockMsg:
        topic = "industrial/telemetry/TEST_ESP32"
        payload = sample_payload

    with patch.object(listener.repo, "upsert_machine") as mock_upsert, \
         patch.object(listener.repo, "save_sensor_reading") as mock_save:
        listener._on_message(None, None, MockMsg())
        mock_upsert.assert_called_once_with(
            machine_id="TEST_ESP32", status="OPERATIONAL"
        )
        mock_save.assert_called_once()


def test_mqtt_listener_import():
    """Smoke: MQTT listener instantiates without requiring a live broker."""
    listener = MQTTTelemetryListener()
    assert listener.broker
    assert listener.topic
    assert listener.repo is not None
