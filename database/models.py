import sqlite3
from contextlib import contextmanager

DB_NAME = "industrial_lab.db"

@contextmanager
def get_db_connection():
    """Context manager para conexões seguras com o SQLite."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Cria as tabelas necessárias no SQLite se não existirem."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Tabela de Máquinas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS machines (
                machine_id TEXT PRIMARY KEY,
                name TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabela de Leituras de Sensores
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                machine_id TEXT,
                state TEXT,
                failure_mode TEXT,
                temperature REAL,
                vibration REAL,
                current REAL,
                rpm REAL,
                noise REAL,
                FOREIGN KEY (machine_id) REFERENCES machines(machine_id)
            )
        """)
        
        # Tabela de Anomalias
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                machine_id TEXT,
                anomaly_score REAL,
                is_anomaly BOOLEAN,
                FOREIGN KEY (machine_id) REFERENCES machines(machine_id)
            )
        """)
        
        # Tabela de Predições (Health Score, RUL, etc.)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                machine_id TEXT,
                health_score REAL,
                risk_level TEXT,
                failure_type TEXT,
                rul_hours REAL,
                FOREIGN KEY (machine_id) REFERENCES machines(machine_id)
            )
        """)
        
        # Tabela de Eventos de Manutenção
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                machine_id TEXT,
                description TEXT,
                FOREIGN KEY (machine_id) REFERENCES machines(machine_id)
            )
        """)
        
        conn.commit()
