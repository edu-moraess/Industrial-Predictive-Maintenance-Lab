import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Dict, Any, Tuple, List
from simulator.failures import FailureMode

class FailureClassifier:
    """
    Classificador supervisionado para diagnóstico da causa raiz de falhas industriais.
    Nota: Treinado em dados sintéticos para fins experimentais.
    """
    
    FEATURE_COLUMNS = [
        'temperature', 'vibration', 'current', 'rpm', 'noise',
        'vib_rolling_mean', 'vib_rolling_std', 'temp_rolling_mean', 
        'curr_rolling_mean', 'vib_roc', 'temp_roc', 'vib_trend', 'vib_temp_ratio'
    ]

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight='balanced'
        )
        self.is_fitted = False
        self.classes_ = [e.value for e in FailureMode]

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        available_cols = [col for col in self.FEATURE_COLUMNS if col in df.columns]
        return df[available_cols].fillna(0)

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Treina o classificador e retorna as métricas de treino.
        Espera que o DataFrame contenha a coluna rotulada 'failure_mode'.
        """
        if 'failure_mode' not in df.columns or df.empty:
            raise ValueError("O DataFrame de treino deve conter a coluna 'failure_mode'.")

        X = self._prepare_features(df)
        y = df['failure_mode']

        self.model.fit(X, y)
        self.is_fitted = True
        self.classes_ = list(self.model.classes_)

        y_pred = self.model.predict(X)

        return {
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "report": classification_report(y, y_pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, y_pred, labels=self.classes_).tolist()
        }

    def predict(self, df: pd.DataFrame) -> Tuple[List[str], List[Dict[str, float]]]:
        """
        Retorna uma tupla contendo:
        1. Lista de falhas previstas por linha.
        2. Lista de dicionários com a distribuição de probabilidade de cada falha.
        """
        if not self.is_fitted:
            raise RuntimeError("O modelo precisa ser treinado antes de executar predições.")

        X = self._prepare_features(df)
        predictions = self.model.predict(X)
        probabilities_matrix = self.model.predict_proba(X)

        prob_list = []
        for row in probabilities_matrix:
            prob_dict = {
                cls_name: round(float(prob * 100), 2)
                for cls_name, prob in zip(self.classes_, row)
            }
            prob_list.append(prob_dict)

        return list(predictions), prob_list

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, Any]:
        """Avalia a precisão, recall, F1-score e matriz de confusão em dados de teste."""
        if not self.is_fitted:
            raise RuntimeError("O modelo precisa ser treinado antes da avaliação.")

        X_test = self._prepare_features(df_test)
        y_test = df_test['failure_mode']

        y_pred = self.model.predict(X_test)

        return {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=self.classes_).tolist()
        }
