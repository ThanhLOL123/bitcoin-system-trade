import pandas as pd
import numpy as np
from src.feature_engineering.technical_indicators import TechnicalIndicators

def test_calculate_moving_averages():
    df = pd.DataFrame({'price_usd': np.random.rand(200) * 100})
    df_result = TechnicalIndicators.calculate_moving_averages(df.copy())
    assert 'sma_20' in df_result.columns
    assert 'ema_12' in df_result.columns

def test_calculate_momentum_indicators():
    df = pd.DataFrame({'price_usd': np.random.rand(200) * 100})
    df_result = TechnicalIndicators.calculate_momentum_indicators(df.copy())
    assert 'rsi_14' in df_result.columns
    assert 'macd' in df_result.columns

def test_calculate_volatility_indicators():
    df = pd.DataFrame({'price_usd': np.random.rand(200) * 100, 'high': np.random.rand(200) * 100 + 10, 'low': np.random.rand(200) * 100 - 10})
    df_result = TechnicalIndicators.calculate_volatility_indicators(df.copy())
    assert 'bb_upper' in df_result.columns
    assert 'atr_14' in df_result.columns

def test_calculate_volume_indicators():
    df = pd.DataFrame({'price_usd': np.random.rand(200) * 100, 'volume_24h': np.random.rand(200) * 1000})
    df_result = TechnicalIndicators.calculate_volume_indicators(df.copy())
    assert 'volume_sma_20' in df_result.columns
    assert 'obv' in df_result.columns
