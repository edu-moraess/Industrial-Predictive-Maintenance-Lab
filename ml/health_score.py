from typing import Dict, Any, Tuple
from config.settings import settings

class HealthScoreCalculator:
    """
    Calcula o índice de saúde (0 a 100) e o nível de risco de uma máquina.
    Leva em consideração as leituras atuais, tendências e pontuação de anomalia.
    """

    @staticmethod
    def calculate(telemetry_data: Dict[str, Any]) -> Tuple[float, str]:
        """
        Recebe um dicionário com os dados mais recentes (incluindo features e anomalias)
        e retorna o Health Score (0-100) e o Risk Level.
        """
        # Extrai os valores com fallback seguro para os valores base
        vibration = telemetry_data.get('vibration', settings.BASE_VIBRATION)
        temperature = telemetry_data.get('temperature', settings.BASE_TEMP)
        anomaly_score = telemetry_data.get('anomaly_score', 0.0)
        
        # O score começa em 100 (Perfeito)
        score = 100.0
        
        # 1. Penalidade por Anomalia (peso alto: até 40 pontos perdidos)
        # Se o modelo de ML detectou que o estado é muito anômalo, a saúde cai drasticamente.
        score -= (anomaly_score * 40.0)
        
        # 2. Penalidade por Vibração (peso médio: baseado no desvio percentual)
        # Exemplo: Se a vibração dobrou (desvio de 100%), perde 20 pontos.
        vib_deviation = max(0.0, (vibration - settings.BASE_VIBRATION) / settings.BASE_VIBRATION)
        score -= min(30.0, vib_deviation * 20.0)
        
        # 3. Penalidade por Temperatura (peso médio)
        temp_deviation = max(0.0, (temperature - settings.BASE_TEMP) / settings.BASE_TEMP)
        score -= min(30.0, temp_deviation * 25.0)

        # 4. Ajustes extras por features de tendência (se disponíveis)
        vib_trend = telemetry_data.get('vib_trend', 0.0)
        if vib_trend > 0.5:
            score -= 5.0 # Penaliza se a vibração estiver em tendência clara de alta
            
        # Garante que o score fique entre 0 e 100
        final_score = max(0.0, min(100.0, score))
        
        # Classificação de Risco baseada nos requisitos
        risk_level = HealthScoreCalculator._determine_risk_level(final_score)
        
        return round(final_score, 1), risk_level

    @staticmethod
    def _determine_risk_level(score: float) -> str:
        """Determina a categoria de risco a partir do score numérico."""
        if score >= 90:
            return "HEALTHY"
        elif score >= 70:
            return "NORMAL"
        elif score >= 50:
            return "WARNING"
        elif score >= 25:
            return "CRITICAL"
        else:
            return "FAILURE RISK"
