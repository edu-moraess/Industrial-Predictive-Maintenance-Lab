import pytest
import pandas as pd
from ml.failure_classifier import FailureClassifier
from features.engineering import FeatureEngineer
from database.repository import DatabaseRepository
from simulator.generator import DataGenerator

def test_failure_classifier_pipeline():
    # 1. Gera dataset sintético diversificado para treino e teste
    repo = DatabaseRepository()
    generator = DataGenerator(repo)
    machine_id = "MACHINE_CLASSIFIER_TEST"

    # Gera histórico contendo diferentes estados operacionais
    generator.generate_historical_dataset(machine_id, hours=10, frequency_minutes=5)
    raw_data = repo.get_historical_readings(machine_id, limit=200)

    # 2. Transforma dados em features
    df_features = FeatureEngineer.process_telemetry(raw_data)

    # 3. Treina o modelo
    classifier = FailureClassifier()
    metrics = classifier.train(df_features)

    assert classifier.is_fitted is True
    assert metrics["accuracy"] >= 0.70
    assert "report" in metrics
    assert "confusion_matrix" in metrics

    # 4. Executa predição em tempo de execução
    predictions, probabilities = classifier.predict(df_features.tail(5))

    assert len(predictions) == 5
    assert len(probabilities) == 5
    assert isinstance(probabilities[0], dict)
    
    # A soma das probabilidades de todas as classes deve fechar em ~100%
    first_prob_sum = sum(probabilities[0].values())
    assert 99.0 <= first_prob_sum <= 101.0

def test_classifier_unfitted_raise():
    """Garante que tentar prever sem treinar dispara exceção controlada."""
    classifier = FailureClassifier()
    df_dummy = pd.DataFrame([{"temperature": 45.0, "vibration": 2.5}])

    with pytest.raises(RuntimeError):
        classifier.predict(df_dummy)
