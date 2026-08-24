from typing import List, Dict, Any, Optional
from database.models import get_db_connection, init_db

class DatabaseRepository:
    """Repositório responsável por todas as operações de leitura e escrita no SQLite."""
    
    def __init__(self):
        init_db()
        
    def upsert_machine(self, machine_id: str, name: str = "Industrial Machine", status: str = "NORMAL"):
        """Insere ou atualiza o registro de uma máquina."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO machines (machine_id, name, status)
                VALUES (?, ?, ?)
                ON CONFLICT(machine_id) DO UPDATE SET status = excluded.status
            """, (machine_id, name, status))
            conn.commit()

    def save_sensor_reading(self, reading: Dict[str, Any]):
        """Salva uma leitura de telemetria de sensores."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sensor_readings (timestamp, machine_id, state, failure_mode, temperature, vibration, current, rpm, noise)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                reading["timestamp"],
                reading["machine_id"],
                reading["state"],
                reading["failure_mode"],
                reading["temperature"],
                reading["vibration"],
                reading["current"],
                reading["rpm"],
                reading["noise"]
            ))
            conn.commit()

    def get_latest_reading(self, machine_id: str) -> Optional[Dict[str, Any]]:
        """Retorna a última leitura de sensores de uma máquina específica."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM sensor_readings 
                WHERE machine_id = ? 
                ORDER BY timestamp DESC LIMIT 1
            """, (machine_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_historical_readings(self, machine_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retorna leituras históricas de uma máquina para análise e gráficos."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM sensor_readings 
                WHERE machine_id = ? 
                ORDER BY timestamp DESC LIMIT ?
            """, (machine_id, limit))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
    def get_machines(self) -> List[Dict[str, Any]]:
        """Retorna todas as máquinas cadastradas."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM machines")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
