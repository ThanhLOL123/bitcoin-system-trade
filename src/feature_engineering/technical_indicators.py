import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Calculate technical indicators for Bitcoin price data"""
    
    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various moving averages"""
        df = df.copy()
        
        # Simple Moving Averages
        for period in [20, 50, 200]:
            df[f'sma_{period}'] = df['price_usd'].rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in [12, 26]:
            df[f'ema_{period}'] = df['price_usd'].ewm(span=period, adjust=False).mean()
        
        return df

    @staticmethod
    def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        df = df.copy()
        
        # RSI (Relative Strength Index)
        delta = df['price_usd'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        ema_12 = df['price_usd'].ewm(span=12, adjust=False).mean()
        ema_26 = df['price_usd'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df

    @staticmethod
    def calculate_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators"""
        df = df.copy()
        
        # Bollinger Bands
        window = 20
        std_dev = df['price_usd'].rolling(window=window).std()
        df['bb_middle'] = df['price_usd'].rolling(window=window).mean()
        df['bb_upper'] = df['bb_middle'] + (std_dev * 2)
        df['bb_lower'] = df['bb_middle'] - (std_dev * 2)
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['price_usd'].shift())
        low_close = np.abs(df['low'] - df['price_usd'].shift())
        tr = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close}).max(axis=1)
        df['atr_14'] = tr.ewm(span=14, adjust=False).mean()
        
        return df

    @staticmethod
    def calculate_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        df = df.copy()
        
        if 'volume_24h' not in df.columns:
            return df

        # Volume Simple Moving Average
        df['volume_sma_20'] = df['volume_24h'].rolling(window=20).mean()
        
        # OBV (On-Balance Volume)
        obv = [0]
        for i in range(1, len(df)):
            if df['price_usd'].iloc[i] > df['price_usd'].iloc[i-1]:
                obv.append(obv[-1] + df['volume_24h'].iloc[i])
            elif df['price_usd'].iloc[i] < df['price_usd'].iloc[i-1]:
                obv.append(obv[-1] - df['volume_24h'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        
        return df