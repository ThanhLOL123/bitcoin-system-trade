import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from src.data_collector.base_crawler import BaseCrawler, CrawlerConfig

class ConcreteCrawler(BaseCrawler):
    async def collect_data(self):
        return []

    def transform_data(self, raw_data):
        return raw_data

@pytest.fixture
def crawler_config():
    return CrawlerConfig(rate_limit=60, retry_attempts=1, retry_delay=0)

@pytest.fixture
def concrete_crawler(crawler_config):
    with patch('redis.Redis') as mock_redis:
        mock_redis_instance = mock_redis.return_value
        mock_redis_instance.setex = AsyncMock()
        mock_redis_instance.get = AsyncMock(return_value=None)
        return ConcreteCrawler(crawler_config)

@pytest.mark.asyncio
async def test_rate_limit_wait(concrete_crawler):
    concrete_crawler.last_request_time = 0
    await concrete_crawler.rate_limit_wait()
    assert concrete_crawler.last_request_time > 0

@pytest.mark.asyncio
async def test_make_request_success(concrete_crawler):
    with patch('aiohttp.ClientSession.request') as mock_request:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": "test"})
        mock_request.return_value.__aenter__.return_value = mock_response
        
        result = await concrete_crawler.make_request("http://test.com")
        assert result == {"data": "test"}

@pytest.mark.asyncio
async def test_make_request_429_retry(concrete_crawler):
    concrete_crawler.config.retry_attempts = 2
    concrete_crawler.config.retry_delay = 0.01
    with patch('aiohttp.ClientSession.request') as mock_request:
        mock_response_429 = AsyncMock()
        mock_response_429.status = 429
        mock_response_success = AsyncMock()
        mock_response_success.status = 200
        mock_response_success.json = AsyncMock(return_value={"data": "retried"})

        mock_request.side_effect = [
            AsyncMock(return_value=mock_response_429),
            AsyncMock(return_value=mock_response_success)
        ]
        mock_request.return_value.__aenter__.side_effect = [
            mock_response_429,
            mock_response_success
        ]
        
        result = await concrete_crawler.make_request("http://test.com")
        assert result == {"data": "retried"}
        assert mock_request.call_count == 2

@pytest.mark.asyncio
async def test_cache_data(concrete_crawler):
    await concrete_crawler.cache_data("key", {"test": "data"}, 60)
    concrete_crawler.redis_client.setex.assert_called_once_with("key", 60, '{"test": "data"}')

@pytest.mark.asyncio
async def test_get_cached_data(concrete_crawler):
    concrete_crawler.redis_client.get.return_value = '{"test": "cached"}'
    result = await concrete_crawler.get_cached_data("key")
    assert result == {"test": "cached"}
    concrete_crawler.redis_client.get.assert_called_once_with("key")
