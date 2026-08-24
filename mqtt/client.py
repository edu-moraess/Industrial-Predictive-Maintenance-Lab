import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import logging
import paho.mqtt.client as mqtt
from database.repository import DatabaseRepository

logging.basicConfig(level=logging.INFO)

class MQTTTelemetryListener:
    """
    Cliente MQTT responsável por assinar tópicos de telemetria do ESP32/Broker
    e persistir os dados no banco relacional.
    """
    def __init__(self, broker: str = "broker.hivemq.com", port: int = 1883, topic: str = "industrial/telemetry/#"):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.repo = DatabaseRepository()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            logging.info(f"Conectado com sucesso ao Broker MQTT ({self.broker}:{self.port})")
            client.subscribe(self.topic)
            logging.info(f"Inscrito no tópico: {self.topic}")
        else:
            logging.error(f"Falha na conexão MQTT. Código de retorno: {rc}")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            machine_id = payload.get("machine_id", "ESP32_UNKNOWN")
            
            # Registra/Atualiza máquina e salva leitura
            self.repo.upsert_machine(machine_id=machine_id, status=payload.get("state", "OPERATIONAL"))
            self.repo.save_sensor_reading(payload)
            
            logging.info(f" Telemetria MQTT processada de [{machine_id}] via tópico [{msg.topic}]")
        except Exception as e:
            logging.error(f"Erro ao processar mensagem MQTT do tópico {msg.topic}: {e}")

    def start(self):
        """Inicia o escutador em modo assíncrono não-bloqueante."""
        self.client.connect(self.broker, self.port, 60)
        self.client.loop_start()

    def stop(self):
        """Encerra a conexão com o broker."""
        self.client.loop_stop()
        self.client.disconnect()

if __name__ == "__main__":
    listener = MQTTTelemetryListener()
    listener.start()
    print("Escutando mensagens MQTT. Pressione Ctrl+C para encerrar...")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        listener.stop()
