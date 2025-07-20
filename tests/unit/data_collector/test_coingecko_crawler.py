import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from src.data_collector.crawlers.coingecko_crawler import CoinGeckoCrawler
from datetime import datetime

@pytest.fixture
def coingecko_crawler():
    with patch('redis.Redis') as mock_redis:
        mock_redis_instance = mock_redis.return_value
        mock_redis_instance.setex = AsyncMock()
        mock_redis_instance.get = AsyncMock(return_value=None)
        return CoinGeckoCrawler()

@pytest.mark.asyncio
async def test_get_current_price_success(coingecko_crawler):
    mock_response_data = {
        "bitcoin": {
            "usd": 50000,
            "usd_market_cap": 900000000000,
            "usd_24h_vol": 50000000000,
            "usd_24h_change": 2.5,
            "last_updated_at": 1678886400
        }
    }
    with patch.object(coingecko_crawler, 'make_request', new=AsyncMock(return_value=mock_response_data)):
        result = await coingecko_crawler.get_current_price()
        assert result['price_usd'] == 50000
        assert result['source'] == 'coingecko'
        coingecko_crawler.cache_data.assert_called_once()

@pytest.mark.asyncio
async def test_get_ohlc_data_success(coingecko_crawler):
    mock_response_data = [
        [1678886400000, 49000, 51000, 48000, 50000], # timestamp, open, high, low, close
        [1678890000000, 50000, 50500, 49500, 50200]
    ]
    with patch.object(coingecko_crawler, 'make_request', new=AsyncMock(return_value=mock_response_data)):
        result = await coingecko_crawler.get_ohlc_data(days=1)
        assert len(result) == 2
        assert result[0]['price_usd'] == 50000
        assert result[0]['source'] == 'coingecko'

@pytest.mark.asyncio
async def test_collect_data(coingecko_crawler):
    with patch.object(coingecko_crawler, 'get_current_price', new=AsyncMock(return_value={"data": "current"})):
        with patch.object(coingecko_crawler, 'get_ohlc_data', new=AsyncMock(return_value=[{"data": "ohlc1"}, {"data": "ohlc2"}])):
            result = await coingecko_crawler.collect_data()
            assert len(result) == 3
            assert {"data": "current"} in result
            assert {"data": "ohlc1"} in result
            assert {"data": "ohlc2"} in result

def test_transform_data(coingecko_crawler):
    raw_data = [
        {'price_usd': 100, 'timestamp': '2023-01-01T00:00:00'},
        {'price_usd': 200, 'timestamp': '2023-01-01T01:00:00'}
    ]
    transformed_data = coingecko_crawler.transform_data(raw_data)
    assert len(transformed_data) == 2
    assert 'processed_at' in transformed_data[0]
