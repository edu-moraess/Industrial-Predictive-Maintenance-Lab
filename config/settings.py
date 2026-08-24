from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Industrial Predictive Maintenance Lab"
    VERSION: str = "0.1.0"
    
    # Parâmetros base da simulação (Valores normais de operação)
    BASE_TEMP: float = 45.0      # Graus Celsius
    BASE_VIBRATION: float = 2.5  # mm/s
    BASE_CURRENT: float = 15.0   # Amperes
    BASE_RPM: float = 1800.0     # Rotações por minuto
    BASE_NOISE: float = 65.0     # Decibéis
    
    class Config:
        env_file = ".env"

settings = Settings()
