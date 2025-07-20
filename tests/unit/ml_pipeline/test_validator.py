import pytest
import numpy as np
from src.ml_pipeline.validation.validator import ModelValidator

def test_calculate_regression_metrics():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    metrics = ModelValidator.calculate_regression_metrics(y_true, y_pred)
    assert 'mse' in metrics
    assert 'mae' in metrics
    assert 'r2_score' in metrics
    assert metrics['mse'] == pytest.approx(0.01)

def test_calculate_classification_metrics():
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])
    metrics = ModelValidator.calculate_classification_metrics(y_true, y_pred)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert metrics['accuracy'] == pytest.approx(0.8)

def test_run_backtest():
    predictions = np.array([0.6, 0.3, 0.8, 0.2, 0.9])
    actual_targets = np.array([1, 0, 1, 0, 1])
    results = ModelValidator.run_backtest(predictions, actual_targets)
    assert 'directional_accuracy' in results
    assert 'total_predictions' in results
    assert results['directional_accuracy'] == pytest.approx(1.0) # (0.6->1, 0.3->0, 0.8->1, 0.2->0, 0.9->1)
