import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

class ModelValidator:
    """Model validation and backtesting"""
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate regression metrics"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }

    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate classification metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }

    @staticmethod
    def run_backtest(predictions: np.ndarray, actual_targets: np.ndarray) -> dict:
        """Run a simplified backtesting simulation for directional accuracy"""
        # Assuming predictions are probabilities/scores and actual_targets are binary (0 or 1)
        predicted_direction = (predictions > 0.5).astype(int).flatten()

        correct_predictions = np.sum(actual_targets == predicted_direction)
        accuracy = correct_predictions / len(actual_targets)

        return {
            'directional_accuracy': accuracy,
            'total_predictions': len(actual_targets)
        }