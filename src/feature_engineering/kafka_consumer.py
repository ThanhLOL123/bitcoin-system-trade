import asyncio
import json
import logging
import pandas as pd
from kafka import KafkaConsumer
from datetime import datetime

from .technical_indicators import TechnicalIndicators
from .sentiment_analysis import SentimentAnalysis
from .feature_store import FeatureStore

logger = logging.getLogger(__name__)

class FeatureEngineeringConsumer:
    """Kafka consumer for feature engineering"""
    
    def __init__(self, kafka_bootstrap_servers: list[str]):
        self.consumer = KafkaConsumer(
            'bitcoin-prices', 'market-sentiment',
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='earliest'
        )
        self.tech_indicators = TechnicalIndicators()
        self.sentiment_analyzer = SentimentAnalysis()
        self.feature_store = FeatureStore()
        self.price_data = []
        self.sentiment_data = []

    async def start(self):
        """Start consuming messages from Kafka"""
        logger.info("Feature engineering consumer started.")
        for message in self.consumer:
            if message.topic == 'bitcoin-prices':
                self.price_data.append(message.value)
            elif message.topic == 'market-sentiment':
                self.sentiment_data.append(message.value)
            
            # Process data in batches
            if len(self.price_data) >= 100:
                self.process_features()
                self.price_data = []
                self.sentiment_data = []

    def process_features(self):
        """Process collected data to generate and store features"""
        logger.info("Processing features...")
        price_df = pd.DataFrame(self.price_data)
        sentiment_df = pd.DataFrame(self.sentiment_data)

        # Calculate technical indicators
        price_df = self.tech_indicators.calculate_moving_averages(price_df)
        price_df = self.tech_indicators.calculate_momentum_indicators(price_df)
        price_df = self.tech_indicators.calculate_volatility_indicators(price_df)
        price_df = self.tech_indicators.calculate_volume_indicators(price_df)

        # Calculate sentiment scores
        if not sentiment_df.empty:
            sentiment_df['sentiment_scores'] = sentiment_df['title'].apply(self.sentiment_analyzer.get_sentiment_scores)
            # Further sentiment feature engineering can be done here

        # Merge dataframes and save features
        # This is a simplified merge, a more robust solution would be needed
        if not price_df.empty:
            today = datetime.now().strftime('%Y-%m-%d')
            self.feature_store.save_features(price_df, today)
            logger.info(f"Successfully processed and saved {len(price_df)} data points.")

async def main():
    logging.basicConfig(level=logging.INFO)
    kafka_servers = ['localhost:9092']
    consumer = FeatureEngineeringConsumer(kafka_servers)
    await consumer.start()

if __name__ == "__main__":
    asyncio.run(main())