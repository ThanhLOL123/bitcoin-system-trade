import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from src.data_collector.crawlers.news_crawler import NewsCrawler
from datetime import datetime

@pytest.fixture
def news_crawler():
    with patch('redis.Redis') as mock_redis:
        mock_redis_instance = mock_redis.return_value
        mock_redis_instance.setex = AsyncMock()
        mock_redis_instance.get = AsyncMock(return_value=None)
        return NewsCrawler()

@pytest.mark.asyncio
async def test_scrape_headlines_success(news_crawler):
    mock_html = """
    <html><body>
        <h1>Headline 1</h1>
        <h1>Headline 2</h1>
    </body></html>
    """
    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = mock_html
        mock_get.return_value.__aenter__.return_value = mock_response
        
        headlines = await news_crawler.scrape_headlines("TestSource", "http://test.com")
        assert len(headlines) == 2
        assert headlines[0]['title'] == "Headline 1"
        assert headlines[1]['source'] == "news"
        assert headlines[1]['platform'] == "TestSource"

@pytest.mark.asyncio
async def test_scrape_headlines_http_error(news_crawler):
    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_get.return_value.__aenter__.return_value = mock_response
        
        headlines = await news_crawler.scrape_headlines("TestSource", "http://test.com")
        assert headlines is None

@pytest.mark.asyncio
async def test_collect_data(news_crawler):
    with patch.object(news_crawler, 'scrape_headlines', new=AsyncMock(side_effect=[
        [{'title': 'CoinDesk News'}],
        [{'title': 'Cointelegraph News'}]
    ])):
        collected_data = await news_crawler.collect_data()
        assert len(collected_data) == 2
        assert collected_data[0]['title'] == "CoinDesk News"
        assert collected_data[1]['title'] == "Cointelegraph News"

def test_transform_data(news_crawler):
    raw_data = [{'title': 'Test', 'source': 'news'}]
    transformed_data = news_crawler.transform_data(raw_data)
    assert transformed_data == raw_data
