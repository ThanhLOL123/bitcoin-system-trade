import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.ml_pipeline.training.trainer import Trainer

@pytest.fixture
def trainer():
    with patch('src.ml_pipeline.utils.data_utils.DataUtils') as MockDataUtils,
         patch('src.feature_engineering.feature_store.FeatureStore') as MockFeatureStore,
         patch('mlflow.start_run'),
         patch('mlflow.sklearn.log_model'),
         patch('mlflow.pytorch.log_model'),
         patch('mlflow.xgboost.log_model'),
         patch('mlflow.log_metric'),
         patch('mlflow.log_params'),
         patch('optuna.create_study') as MockOptunaStudy,
         patch('torch.cuda.is_available', return_value=False): # Force CPU for testing

        mock_data_utils_instance = MockDataUtils.return_value
        mock_data_utils_instance.split_data.return_value = (pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series())
        mock_data_utils_instance.scale_features.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_data_utils_instance.handle_missing_values.side_effect = lambda df: df

        mock_feature_store_instance = MockFeatureStore.return_value
        mock_feature_store_instance.load_features_range.return_value = pd.DataFrame({
            'price_usd': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            'feature2': [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        })

        mock_optuna_study_instance = MockOptunaStudy.return_value
        mock_optuna_study_instance.best_params = {'hidden_size': 64, 'num_layers': 1, 'lr': 0.001, 'd_model': 64, 'nhead': 4}
        mock_optuna_study_instance.optimize = MagicMock()

        return Trainer()

def test_trainer_initialization(trainer):
    assert trainer.data_utils is not None
    assert trainer.feature_store is not None

def test_trainer_train_empty_features(trainer):
    trainer.feature_store.load_features_range.return_value = pd.DataFrame()
    trainer.train()
    # Assert that no models were trained or MLflow calls were made
    mlflow.start_run.assert_not_called()

def test_trainer_train_models(trainer):
    # Mock data for split_data and scale_features
    mock_X_train = pd.DataFrame({'feature1': [1,2,3,4,5], 'feature2': [5,4,3,2,1]})
    mock_X_test = pd.DataFrame({'feature1': [6,7], 'feature2': [1,2]})
    mock_y_train = pd.Series([0,1,0,1,0])
    mock_y_test = pd.Series([1,0])

    trainer.data_utils.split_data.return_value = (mock_X_train, mock_X_test, mock_y_train, mock_y_test)
    trainer.data_utils.scale_features.return_value = (mock_X_train.values, mock_X_test.values)

    trainer.train()

    # Assert that MLflow calls were made for each model
    assert mlflow.sklearn.log_model.call_count == 3 # LR, RF, MLP
    assert mlflow.xgboost.log_model.call_count == 1
    assert mlflow.pytorch.log_model.call_count == 2 # LSTM, Transformer

    # Assert that optuna optimize was called for XGBoost, LSTM, and Transformer
    assert trainer.optuna.create_study.call_count == 3
