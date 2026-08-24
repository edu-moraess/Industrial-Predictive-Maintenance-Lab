import pandas as pd
import numpy as np
from typing import List, Dict, Any

class FeatureEngineer:
    """
    Processa os dados brutos de telemetria e extrai features estatísticas e 
    temporais cruciais para detecção de anomalias e Machine Learning.
    """
    
    @staticmethod
    def process_telemetry(raw_data: List[Dict[str, Any]], window_size: int = 5) -> pd.DataFrame:
        """
        Recebe uma lista de dicionários (dados brutos do banco) e retorna 
        um DataFrame Pandas com as novas features.
        
        Args:
            raw_data: Lista de dicionários representando as leituras dos sensores.
            window_size: Tamanho da janela para cálculo das médias móveis.
        """
        if not raw_data:
            return pd.DataFrame()

        # Converte para DataFrame
        df = pd.DataFrame(raw_data)
        
        # Garante a ordenação correta pelo tempo para não bagunçar as janelas (rolling windows)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(by='timestamp').reset_index(drop=True)

        # 1. Rolling Statistics (Média Móvel e Desvio Padrão)
        # min_periods=1 garante que os primeiros registros não fiquem como NaN (Not a Number)
        
        # Vibração
        df['vib_rolling_mean'] = df['vibration'].rolling(window=window_size, min_periods=1).mean()
        df['vib_rolling_std'] = df['vibration'].rolling(window=window_size, min_periods=1).std().fillna(0)
        
        # Temperatura
        df['temp_rolling_mean'] = df['temperature'].rolling(window=window_size, min_periods=1).mean()
        df['temp_rolling_std'] = df['temperature'].rolling(window=window_size, min_periods=1).std().fillna(0)

        # Corrente
        df['curr_rolling_mean'] = df['current'].rolling(window=window_size, min_periods=1).mean()
        df['curr_rolling_std'] = df['current'].rolling(window=window_size, min_periods=1).std().fillna(0)

        # 2. Rate of Change (Taxa de Variação) - Ajuda a detectar picos súbitos
        df['vib_roc'] = df['vibration'].diff().fillna(0)
        df['temp_roc'] = df['temperature'].diff().fillna(0)

        # 3. Tendências (Valor Atual vs Média Móvel) - Ajuda a ver se está subindo ou descendo a longo prazo
        df['vib_trend'] = df['vibration'] - df['vib_rolling_mean']
        df['temp_trend'] = df['temperature'] - df['temp_rolling_mean']

        # 4. Sensor Correlations (Correlações entre sensores)
        # Quando vibração e temperatura sobem juntas, é um forte sinal mecânico.
        epsilon = 1e-6 # Evita divisão por zero
        df['vib_temp_ratio'] = df['vibration'] / (df['temperature'] + epsilon)

        return df
