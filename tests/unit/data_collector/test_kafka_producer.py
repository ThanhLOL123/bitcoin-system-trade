import pytest
from unittest.mock import MagicMock, patch
from src.data_collector.kafka_producer import KafkaProducerWrapper

@pytest.fixture
def kafka_producer_wrapper():
    with patch('kafka.KafkaProducer') as MockKafkaProducer:
        mock_producer_instance = MockKafkaProducer.return_value
        mock_producer_instance.send = MagicMock()
        mock_producer_instance.flush = MagicMock()
        mock_producer_instance.close = MagicMock()
        return KafkaProducerWrapper(['localhost:9092'])

def test_send_message(kafka_producer_wrapper):
    topic = "test_topic"
    data = {"key": "value"}
    key = "test_key"
    
    kafka_producer_wrapper.send_message(topic, data, key)
    
    kafka_producer_wrapper.producer.send.assert_called_once_with(
        topic,
        value={'key': 'value'},
        key=b'test_key'
    )
    kafka_producer_wrapper.producer.flush.assert_called_once_with(timeout=10)

def test_close_producer(kafka_producer_wrapper):
    kafka_producer_wrapper.close()
    kafka_producer_wrapper.producer.close.assert_called_once()
