import pandas as pd
import numpy as np
from src.ml_pipeline.utils.data_utils import DataUtils

def test_split_data():
    df = pd.DataFrame({'feature1': np.random.rand(100), 'feature2': np.random.rand(100), 'target': np.random.randint(0, 2, 100)})
    X_train, X_test, y_train, y_test = DataUtils.split_data(df)
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20

def test_scale_features():
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [10, 20, 30]})
    X_test = pd.DataFrame({'feature1': [4, 5], 'feature2': [40, 50]})
    X_train_scaled, X_test_scaled = DataUtils.scale_features(X_train, X_test)
    assert X_train_scaled.shape == (3, 2)
    assert X_test_scaled.shape == (2, 2)
    assert np.isclose(X_train_scaled.mean(), 0.0, atol=1e-9).all()
    assert np.isclose(X_train_scaled.std(), 1.0, atol=1e-9).all()

def test_handle_missing_values():
    df = pd.DataFrame({'col1': [1, 2, np.nan], 'col2': [10, np.nan, 30]})
    df_filled = DataUtils.handle_missing_values(df.copy())
    assert df_filled.isnull().sum().sum() == 0
    assert df_filled.loc[2, 'col1'] == 1.5
    assert df_filled.loc[1, 'col2'] == 20.0
