import numpy as np

class SensorSimulator:
    """Simula as leituras individuais dos sensores com ruído gaussiano."""
    
    @staticmethod
    def generate_reading(base_value: float, noise_std: float, trend: float = 0.0) -> float:
        """
        Gera uma leitura de sensor.
        :param base_value: Valor nominal do sensor.
        :param noise_std: Desvio padrão do ruído (simula a precisão do sensor).
        :param trend: Valor adicionado/subtraído progressivamente (simula degradação).
        """
        noise = np.random.normal(0, noise_std)
        return max(0.0, base_value + trend + noise)  # Garante que não tenhamos valores negativos absurdos
