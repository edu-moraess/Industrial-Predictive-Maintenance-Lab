import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

# Allow override for tests / containers; default next to project root
_DEFAULT_DB = Path(__file__).resolve().parent.parent / "industrial_lab.db"
DB_NAME = os.environ.get("IPML_DB_PATH", str(_DEFAULT_DB))

@contextmanager
def get_db_connection():
    """Context manager for safe SQLite connections."""
    conn = sqlite3.connect(DB_NAME, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Create required SQLite tables if they do not exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS machines (
                machine_id TEXT PRIMARY KEY,
                name TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
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
