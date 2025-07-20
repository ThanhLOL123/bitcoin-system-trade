import pytest
from unittest.mock import patch, MagicMock
from src.data_collector.crawlers.yahoo_crawler import YahooCrawler
from datetime import datetime
import pandas as pd

@pytest.fixture
def yahoo_crawler():
    with patch('redis.Redis') as mock_redis:
        mock_redis_instance = mock_redis.return_value
        mock_redis_instance.setex = MagicMock()
        mock_redis_instance.get = MagicMock(return_value=None)
        return YahooCrawler()

def test_get_current_price_success(yahoo_crawler):
    mock_history_data = pd.DataFrame({
        'Open': [100],
        'High': [110],
        'Low': [90],
        'Close': [105],
        'Volume': [100000]
    }, index=[pd.Timestamp.now()])

    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker_instance = mock_ticker.return_value
        mock_ticker_instance.history.return_value = mock_history_data
        
        result = yahoo_crawler.get_current_price()
        assert result is not None
        assert result['price_usd'] == 105
        assert result['source'] == 'yahoo_finance'

def test_get_current_price_empty_history(yahoo_crawler):
    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker_instance = mock_ticker.return_value
        mock_ticker_instance.history.return_value = pd.DataFrame()
        
        result = yahoo_crawler.get_current_price()
        assert result is None

def test_transform_data(yahoo_crawler):
    raw_data = [{'price_usd': 100, 'timestamp': '2023-01-01T00:00:00'}]
    transformed_data = yahoo_crawler.transform_data(raw_data)
    assert transformed_data == raw_data
