import pytest
import asyncio
import os
from unittest.mock import patch, AsyncMock
from src.data_collector.scheduler import DataCollectionScheduler
from src.feature_engineering.kafka_consumer import FeatureEngineeringConsumer
from src.feature_engineering.feature_store import FeatureStore
import pandas as pd
from datetime import datetime, timedelta

# Mock KafkaProducerWrapper to prevent actual Kafka connection during integration tests
@pytest.fixture(autouse=True)
def mock_kafka_producer_wrapper():
    with patch('src.data_collector.kafka_producer.KafkaProducerWrapper') as MockKafkaProducerWrapper:
        mock_instance = MockKafkaProducerWrapper.return_value
        mock_instance.send_message = AsyncMock()
        mock_instance.close = MagicMock()
        yield mock_instance

# Mock KafkaConsumer to control message flow
@pytest.fixture
def mock_kafka_consumer():
    with patch('kafka.KafkaConsumer') as MockKafkaConsumer:
        mock_instance = MockKafkaConsumer.return_value
        mock_instance.__iter__.return_value = [] # No messages by default
        yield mock_instance

@pytest.fixture
def feature_store_instance(tmp_path):
    return FeatureStore(base_path=tmp_path)

@pytest.mark.asyncio
async def test_data_pipeline_integration(mock_kafka_producer_wrapper, mock_kafka_consumer, feature_store_instance):
    # 1. Setup Data Collector
    reddit_config = {"client_id": "test", "client_secret": "test", "user_agent": "test"}
    data_scheduler = DataCollectionScheduler(['localhost:9092'], reddit_config)

    # Mock collect_data methods to return dummy data
    with patch.object(data_scheduler.coingecko_crawler, 'collect_data', new=AsyncMock(return_value=[{'timestamp': datetime.now().isoformat(), 'price_usd': 50000, 'volume_24h': 10000}])):
        with patch.object(data_scheduler.yahoo_crawler, 'collect_data', new=AsyncMock(return_value=[{'timestamp': datetime.now().isoformat(), 'price_usd': 50000, 'volume_24h': 10000}])):
            with patch.object(data_scheduler.reddit_crawler, 'collect_data', new=AsyncMock(return_value=[{'timestamp': datetime.now().isoformat(), 'title': 'good news'}])):
                with patch.object(data_scheduler.news_crawler, 'collect_data', new=AsyncMock(return_value=[{'timestamp': datetime.now().isoformat(), 'title': 'breaking news'}])):

                    # 2. Simulate data collection and sending to Kafka (mocked)
                    await data_scheduler.collect_coingecko_data()
                    await data_scheduler.collect_yahoo_data()
                    await data_scheduler.collect_reddit_data()
                    await data_scheduler.collect_news_data()

                    # Verify data collector sent messages
                    assert mock_kafka_producer_wrapper.send_message.call_count == 4

                    # 3. Simulate Kafka Consumer receiving messages
                    # Create mock messages that the consumer would receive
                    mock_messages = []
                    for call_args in mock_kafka_producer_wrapper.send_message.call_args_list:
                        topic, data, key = call_args.args
                        mock_message = MagicMock()
                        mock_message.topic = topic
                        mock_message.value = data
                        mock_messages.append(mock_message)
                    
                    mock_kafka_consumer.__iter__.return_value = mock_messages

                    # 4. Setup Feature Engineering Consumer
                    fe_consumer = FeatureEngineeringConsumer(['localhost:9092'])
                    fe_consumer.feature_store = feature_store_instance # Use the mocked feature store

                    # Run the consumer for a short period to process messages
                    # We need to manually trigger process_features as it's batch-based
                    await fe_consumer.start() # This will populate price_data and sentiment_data
                    fe_consumer.process_features()

                    # 5. Verify features are stored in FeatureStore
                    today_date_str = datetime.now().strftime('%Y-%m-%d')
                    stored_features = feature_store_instance.load_features(today_date_str)
                    
                    assert not stored_features.empty
                    assert 'price_usd' in stored_features.columns
                    assert 'sma_20' in stored_features.columns # Verify technical indicators
                    # Add more assertions to verify sentiment features if they were merged

                    print(f"Integration test passed. Stored features: {stored_features.shape}")
