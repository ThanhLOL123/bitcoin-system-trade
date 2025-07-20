# src/data_collector/scheduler.py
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict

from .crawlers.coingecko_crawler import CoinGeckoCrawler
from .crawlers.yahoo_crawler import YahooCrawler
from .crawlers.reddit_crawler import RedditCrawler
from .crawlers.news_crawler import NewsCrawler
from .kafka_producer import KafkaProducerWrapper

logger = logging.getLogger(__name__)

class DataCollectionScheduler:
    """Schedules and runs data collection crawlers"""
    def __init__(self, kafka_bootstrap_servers: List[str], reddit_config: Dict):
        self.kafka_producer = KafkaProducerWrapper(kafka_bootstrap_servers)
        self.coingecko_crawler = CoinGeckoCrawler()
        self.yahoo_crawler = YahooCrawler()
        self.reddit_crawler = RedditCrawler(**reddit_config)
        self.news_crawler = NewsCrawler()
        self.is_running = False

    async def start(self):
        """Start the data collection scheduler"""
        self.is_running = True
        logger.info("Data collection scheduler started.")
        while self.is_running:
            await asyncio.gather(
                self.collect_coingecko_data(),
                self.collect_yahoo_data(),
                self.collect_reddit_data(),
                self.collect_news_data()
            )
            # Schedule next run (e.g., every 5 minutes for sentiment data)
            await asyncio.sleep(300)

    async def stop(self):
        """Stop the data collection scheduler"""
        self.is_running = False
        self.kafka_producer.close()
        logger.info("Data collection scheduler stopped.")

    async def collect_coingecko_data(self):
        """Collect data from CoinGecko and send to Kafka"""
        logger.info("Collecting data from CoinGecko...")
        try:
            async with self.coingecko_crawler as crawler:
                raw_data = await crawler.collect_data()
                transformed_data = crawler.transform_data(raw_data)
                for data_point in transformed_data:
                    key = data_point.get('timestamp')
                    self.kafka_producer.send_message('bitcoin-prices', data_point, key)
            logger.info(f"Successfully collected and sent {len(transformed_data)} data points from CoinGecko.")
        except Exception as e:
            logger.error(f"Error collecting CoinGecko data: {e}")

    async def collect_yahoo_data(self):
        """Collect data from Yahoo Finance and send to Kafka"""
        logger.info("Collecting data from Yahoo Finance...")
        try:
            async with self.yahoo_crawler as crawler:
                raw_data = await crawler.collect_data()
                transformed_data = crawler.transform_data(raw_data)
                for data_point in transformed_data:
                    key = data_point.get('timestamp')
                    self.kafka_producer.send_message('bitcoin-prices', data_point, key)
            logger.info(f"Successfully collected and sent {len(transformed_data)} data points from Yahoo Finance.")
        except Exception as e:
            logger.error(f"Error collecting Yahoo Finance data: {e}")

    async def collect_reddit_data(self):
        """Collect data from Reddit and send to Kafka"""
        logger.info("Collecting data from Reddit...")
        try:
            async with self.reddit_crawler as crawler:
                raw_data = await crawler.collect_data()
                transformed_data = crawler.transform_data(raw_data)
                for data_point in transformed_data:
                    key = data_point.get('timestamp')
                    self.kafka_producer.send_message('market-sentiment', data_point, key)
            logger.info(f"Successfully collected and sent {len(transformed_data)} data points from Reddit.")
        except Exception as e:
            logger.error(f"Error collecting Reddit data: {e}")

    async def collect_news_data(self):
        """Collect data from news sources and send to Kafka"""
        logger.info("Collecting data from news sources...")
        try:
            async with self.news_crawler as crawler:
                raw_data = await crawler.collect_data()
                transformed_data = crawler.transform_data(raw_data)
                for data_point in transformed_data:
                    key = data_point.get('timestamp')
                    self.kafka_producer.send_message('market-sentiment', data_point, key)
            logger.info(f"Successfully collected and sent {len(transformed_data)} data points from news sources.")
        except Exception as e:
            logger.error(f"Error collecting news data: {e}")

async def main():
    logging.basicConfig(level=logging.INFO)
    # Assuming Kafka bootstrap servers are available via environment variable or config
    kafka_servers = ['localhost:9092'] # Replace with actual config loading
    reddit_config = {
        "client_id": "your-reddit-client-id",
        "client_secret": "your-reddit-client-secret",
        "user_agent": "bitcoin-trader-v1.0"
    }
    scheduler = DataCollectionScheduler(kafka_servers, reddit_config)
    try:
        await scheduler.start()
    except asyncio.CancelledError:
        logger.info("Scheduler task cancelled.")
    finally:
        await scheduler.stop()

if __name__ == "__main__":
    asyncio.run(main())

