import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import time
import random
from datetime import datetime
import paho.mqtt.client as mqtt

def run_esp32_simulator(broker="broker.hivemq.com", port=1883, machine_id="ESP32_PHYSICAL_01"):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(broker, port, 60)
    
    topic = f"industrial/telemetry/{machine_id}"
    print(f"Simulador ESP32 Iniciado. Publicando em: {topic}")
    
    try:
        while True:
            # Emulação de leitura de sensores físicos (ex: MPU6050 + DS18B20)
            telemetry_payload = {
                "machine_id": machine_id,
                "timestamp": datetime.now().isoformat(),
                "temperature": round(random.uniform(40.0, 75.0), 2),
                "vibration": round(random.uniform(0.5, 6.0), 2),
                "current": round(random.uniform(10.0, 25.0), 2),
                "rpm": round(random.uniform(1700.0, 1850.0), 1),
                "noise": round(random.uniform(60.0, 85.0), 1),
                "state": "OPERATIONAL"
            }
            
            client.publish(topic, json.dumps(telemetry_payload))
            print(f"[ESP32 >> MQTT] Enviado: Temp={telemetry_payload['temperature']}°C | Vib={telemetry_payload['vibration']}mm/s")
            time.sleep(2.0)
            
    except KeyboardInterrupt:
        client.disconnect()
        print("Simulador ESP32 finalizado.")

if __name__ == "__main__":
    run_esp32_simulator()
