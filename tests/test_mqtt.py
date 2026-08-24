import json
import pytest
from mqtt.client import MQTTTelemetryListener

def test_mqtt_payload_parsing(mocker):
    """Garante que a mensagem recepcionada do MQTT é interpretada e salva corretamente."""
    listener = MQTTTelemetryListener()
    
    # Mock do repositório para evitar gravação real durante o teste unitário
    mock_upsert = mocker.patch.object(listener.repo, 'upsert_machine')
    mock_save = mocker.patch.object(listener.repo, 'save_sensor_reading')
    
    sample_payload = json.dumps({
        "machine_id": "TEST_ESP32",
        "temperature": 55.0,
        "vibration": 2.1,
        "current": 15.0,
        "rpm": 1800.0,
        "noise": 70.0,
        "state": "OPERATIONAL"
    }).encode("utf-8")

    class MockMsg:
        topic = "industrial/telemetry/TEST_ESP32"
        payload = sample_payload

    listener._on_message(None, None, MockMsg())

    mock_upsert.assert_called_once_with(machine_id="TEST_ESP32", status="OPERATIONAL")
    mock_save.assert_called_once()
