import pandas as pd
import os
import pytest
from src.feature_engineering.feature_store import FeatureStore

@pytest.fixture
def feature_store(tmp_path):
    return FeatureStore(base_path=tmp_path)

def test_save_and_load_features(feature_store):
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    date = "2023-01-01"
    feature_store.save_features(df, date)
    loaded_df = feature_store.load_features(date)
    pd.testing.assert_frame_equal(df, loaded_df)

def test_load_non_existent_features(feature_store):
    date = "2023-01-02"
    loaded_df = feature_store.load_features(date)
    assert loaded_df.empty

def test_load_features_range(feature_store):
    df1 = pd.DataFrame({'col1': [1, 2]})
    df2 = pd.DataFrame({'col1': [3, 4]})
    df3 = pd.DataFrame({'col1': [5, 6]})

    feature_store.save_features(df1, "2023-01-01")
    feature_store.save_features(df2, "2023-01-02")
    feature_store.save_features(df3, "2023-01-03")

    loaded_df = feature_store.load_features_range("2023-01-01", "2023-01-03")
    expected_df = pd.concat([df1, df2, df3]).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected_df, loaded_df)

def test_load_features_range_empty(feature_store):
    loaded_df = feature_store.load_features_range("2023-01-01", "2023-01-03")
    assert loaded_df.empty
