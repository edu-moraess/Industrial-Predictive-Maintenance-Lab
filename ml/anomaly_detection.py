import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, List

class AnomalyDetector:
    """
    Detector de anomalias utilizando o algoritmo Isolation Forest.
    Processa dados brutos e features engenheiradas para identificar divergências operacionais.
    """
    
    FEATURE_COLUMNS = [
        'temperature', 'vibration', 'current', 'rpm', 'noise',
        'vib_rolling_mean', 'vib_rolling_std', 'temp_rolling_mean', 'temp_roc', 'vib_trend'
    ]

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.is_fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtra as colunas esperadas e trata valores nulos."""
        available_cols = [col for col in self.FEATURE_COLUMNS if col in df.columns]
        features_df = df[available_cols].copy()
        return features_df.fillna(0)

    def train(self, df: pd.DataFrame) -> None:
        """Treina o modelo Isolation Forest com base em um histórico de dados."""
        features = self._prepare_features(df)
        if len(features) > 0:
            self.model.fit(features)
            self.is_fitted = True

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Avalia um conjunto de dados e retorna o DataFrame com as colunas
        'is_anomaly' (bool) e 'anomaly_score' (float entre 0 e 1).
        """
        if not self.is_fitted:
            self.train(df)

        features = self._prepare_features(df)
        df_result = df.copy()

        # IsolationForest: 1 para normal, -1 para anomalia
        predictions = self.model.predict(features)
        
        # Decision function: quanto menor o valor, mais anômalo é o ponto
        scores = self.model.decision_function(features)
        
        # Normalização do anomaly_score entre 0.0 (normal) e 1.0 (altamente anômalo)
        normalized_scores = 1.0 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

        df_result['is_anomaly'] = predictions == -1
        df_result['anomaly_score'] = np.round(normalized_scores, 4)

        return df_result
