from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "Industrial Predictive Maintenance Lab"
    VERSION: str = "1.0.0"

    # Nominal operating baselines (synthetic reference units)
    BASE_TEMP: float = 45.0       # °C
    BASE_VIBRATION: float = 2.5   # mm/s
    BASE_CURRENT: float = 15.0    # A
    BASE_RPM: float = 1800.0      # RPM
    BASE_NOISE: float = 65.0      # dB


settings = Settings()
