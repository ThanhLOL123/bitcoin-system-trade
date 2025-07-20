import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.data_collector.scheduler import DataCollectionScheduler

@pytest.fixture
def scheduler():
    with patch('src.data_collector.crawlers.coingecko_crawler.CoinGeckoCrawler') as MockCoinGeckoCrawler,
         patch('src.data_collector.crawlers.yahoo_crawler.YahooCrawler') as MockYahooCrawler,
         patch('src.data_collector.crawlers.reddit_crawler.RedditCrawler') as MockRedditCrawler,
         patch('src.data_collector.crawlers.news_crawler.NewsCrawler') as MockNewsCrawler,
         patch('src.data_collector.kafka_producer.KafkaProducerWrapper') as MockKafkaProducerWrapper:
        
        mock_coingecko_crawler_instance = MockCoinGeckoCrawler.return_value
        mock_coingecko_crawler_instance.__aenter__ = AsyncMock(return_value=mock_coingecko_crawler_instance)
        mock_coingecko_crawler_instance.__aexit__ = AsyncMock()
        mock_coingecko_crawler_instance.collect_data = AsyncMock(return_value=[{"coingecko": "data"}])
        mock_coingecko_crawler_instance.transform_data = MagicMock(side_effect=lambda x: x)

        mock_yahoo_crawler_instance = MockYahooCrawler.return_value
        mock_yahoo_crawler_instance.__aenter__ = AsyncMock(return_value=mock_yahoo_crawler_instance)
        mock_yahoo_crawler_instance.__aexit__ = AsyncMock()
        mock_yahoo_crawler_instance.collect_data = AsyncMock(return_value=[{"yahoo": "data"}])
        mock_yahoo_crawler_instance.transform_data = MagicMock(side_effect=lambda x: x)

        mock_reddit_crawler_instance = MockRedditCrawler.return_value
        mock_reddit_crawler_instance.__aenter__ = AsyncMock(return_value=mock_reddit_crawler_instance)
        mock_reddit_crawler_instance.__aexit__ = AsyncMock()
        mock_reddit_crawler_instance.collect_data = AsyncMock(return_value=[{"reddit": "data"}])
        mock_reddit_crawler_instance.transform_data = MagicMock(side_effect=lambda x: x)

        mock_news_crawler_instance = MockNewsCrawler.return_value
        mock_news_crawler_instance.__aenter__ = AsyncMock(return_value=mock_news_crawler_instance)
        mock_news_crawler_instance.__aexit__ = AsyncMock()
        mock_news_crawler_instance.collect_data = AsyncMock(return_value=[{"news": "data"}])
        mock_news_crawler_instance.transform_data = MagicMock(side_effect=lambda x: x)

        mock_kafka_producer_wrapper_instance = MockKafkaProducerWrapper.return_value
        mock_kafka_producer_wrapper_instance.send_message = MagicMock()
        mock_kafka_producer_wrapper_instance.close = MagicMock()

        reddit_config = {"client_id": "test", "client_secret": "test", "user_agent": "test"}
        return DataCollectionScheduler(['localhost:9092'], reddit_config)

@pytest.mark.asyncio
async def test_collect_coingecko_data(scheduler):
    await scheduler.collect_coingecko_data()
    scheduler.coingecko_crawler.collect_data.assert_called_once()
    scheduler.kafka_producer.send_message.assert_called_once_with('bitcoin-prices', {'coingecko': 'data'}, None)

@pytest.mark.asyncio
async def test_collect_yahoo_data(scheduler):
    await scheduler.collect_yahoo_data()
    scheduler.yahoo_crawler.collect_data.assert_called_once()
    scheduler.kafka_producer.send_message.assert_called_once_with('bitcoin-prices', {'yahoo': 'data'}, None)

@pytest.mark.asyncio
async def test_collect_reddit_data(scheduler):
    await scheduler.collect_reddit_data()
    scheduler.reddit_crawler.collect_data.assert_called_once()
    scheduler.kafka_producer.send_message.assert_called_once_with('market-sentiment', {'reddit': 'data'}, None)

@pytest.mark.asyncio
async def test_collect_news_data(scheduler):
    await scheduler.collect_news_data()
    scheduler.news_crawler.collect_data.assert_called_once()
    scheduler.kafka_producer.send_message.assert_called_once_with('market-sentiment', {'news': 'data'}, None)

@pytest.mark.asyncio
async def test_start_and_stop_scheduler(scheduler):
    # Run for a short period to test start/stop logic
    with patch('asyncio.sleep', new=AsyncMock()) as mock_sleep:
        mock_sleep.side_effect = asyncio.CancelledError # To stop the loop quickly
        
        try:
            await scheduler.start()
        except asyncio.CancelledError:
            pass

        assert scheduler.is_running is False
        scheduler.kafka_producer.close.assert_called_once()
