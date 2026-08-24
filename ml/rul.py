from typing import Dict, Any

class RULEstimator:
    """
    Estimador experimental de Vida Útil Restante (Remaining Useful Life - RUL).
    
    DISCLAIMER:
    RUL is an experimental estimate based on synthetic degradation data
    and is not a real industrial prediction.
    """
    
    MAX_RUL_HOURS: float = 720.0  # Equivale a ~30 dias de operação saudável contínua

    @classmethod
    def estimate(cls, health_score: float, telemetry_data: Dict[str, Any] = None) -> float:
        """
        Estima as horas restantes de operação segura com base no Health Score
        e na aceleração das tendências dos sensores.
        """
        if telemetry_data is None:
            telemetry_data = {}

        if health_score <= 0.0:
            return 0.0

        # Fator de aceleração de desgaste baseado nas tendências de vibração e temperatura
        vib_trend = max(0.0, telemetry_data.get('vib_trend', 0.0))
        temp_trend = max(0.0, telemetry_data.get('temp_trend', 0.0))
        
        # Coeficiente de aceleração do desgaste (1.0 representa desgaste nominal)
        degradation_speed = 1.0 + (vib_trend * 2.0) + (temp_trend * 1.5)

        # Projeção linear atenuada pela velocidade de degradação
        raw_rul = (health_score / 100.0) * cls.MAX_RUL_HOURS / degradation_speed
        
        return round(max(0.0, raw_rul), 1)
