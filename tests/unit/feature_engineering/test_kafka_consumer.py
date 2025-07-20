import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.feature_engineering.kafka_consumer import FeatureEngineeringConsumer
import json
import pandas as pd

@pytest.fixture
def fe_consumer():
    with patch('kafka.KafkaConsumer') as MockKafkaConsumer,
         patch('src.feature_engineering.technical_indicators.TechnicalIndicators') as MockTechnicalIndicators,
         patch('src.feature_engineering.sentiment_analysis.SentimentAnalysis') as MockSentimentAnalysis,
         patch('src.feature_engineering.feature_store.FeatureStore') as MockFeatureStore:
        
        mock_consumer_instance = MockKafkaConsumer.return_value
        mock_consumer_instance.__iter__.return_value = [] # No messages by default

        mock_tech_indicators_instance = MockTechnicalIndicators.return_value
        mock_tech_indicators_instance.calculate_moving_averages = MagicMock(side_effect=lambda df: df)
        mock_tech_indicators_instance.calculate_momentum_indicators = MagicMock(side_effect=lambda df: df)
        mock_tech_indicators_instance.calculate_volatility_indicators = MagicMock(side_effect=lambda df: df)
        mock_tech_indicators_instance.calculate_volume_indicators = MagicMock(side_effect=lambda df: df)

        mock_sentiment_analysis_instance = MockSentimentAnalysis.return_value
        mock_sentiment_analysis_instance.get_sentiment_scores = MagicMock(return_value={'compound': 0.5})

        mock_feature_store_instance = MockFeatureStore.return_value
        mock_feature_store_instance.save_features = MagicMock()

        return FeatureEngineeringConsumer(['localhost:9092'])

@pytest.mark.asyncio
async def test_process_features_price_data(fe_consumer):
    fe_consumer.price_data = [{'price_usd': 100, 'volume_24h': 1000, 'timestamp': '2023-01-01'}] * 100
    fe_consumer.process_features()
    fe_consumer.tech_indicators.calculate_moving_averages.assert_called_once()
    fe_consumer.feature_store.save_features.assert_called_once()
    assert fe_consumer.price_data == []

@pytest.mark.asyncio
async def test_process_features_sentiment_data(fe_consumer):
    fe_consumer.price_data = [{'price_usd': 100, 'volume_24h': 1000, 'timestamp': '2023-01-01'}] * 100
    fe_consumer.sentiment_data = [{'title': 'good news', 'timestamp': '2023-01-01'}]
    fe_consumer.process_features()
    fe_consumer.sentiment_analyzer.get_sentiment_scores.assert_called_once()

@pytest.mark.asyncio
async def test_start_consuming(fe_consumer):
    mock_message_price = MagicMock()
    mock_message_price.topic = 'bitcoin-prices'
    mock_message_price.value = {'price_usd': 100, 'volume_24h': 1000, 'timestamp': '2023-01-01'}

    mock_message_sentiment = MagicMock()
    mock_message_sentiment.topic = 'market-sentiment'
    mock_message_sentiment.value = {'title': 'good news', 'timestamp': '2023-01-01'}

    # Simulate 100 price messages and 1 sentiment message
    fe_consumer.consumer.__iter__.return_value = [mock_message_price] * 100 + [mock_message_sentiment]

    with patch.object(fe_consumer, 'process_features', MagicMock()) as mock_process_features:
        # Run start for a short duration to process messages
        async def _run_once():
            for _ in range(101): # Process 101 messages
                next(fe_consumer.consumer.__iter__()) # Simulate consuming a message
            fe_consumer.process_features()

        await asyncio.wait_for(fe_consumer.start(), timeout=0.1) # Use a short timeout
        mock_process_features.assert_called_once()
