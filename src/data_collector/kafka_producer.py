# src/data_collector/kafka_producer.py
import json
import logging
from typing import Dict, Optional

from kafka import KafkaProducer

logger = logging.getLogger(__name__)

class KafkaProducerWrapper:
    """Wrapper for Kafka Producer"""
    def __init__(self, bootstrap_servers: List[str]):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
            key_serializer=lambda k: str(k).encode('utf-8') if k else None
        )

    def send_message(self, topic: str, data: Dict, key: Optional[str] = None):
        """Send data to Kafka topic"""
        try:
            future = self.producer.send(topic, value=data, key=key)
            self.producer.flush(timeout=10)
            logger.debug(f"Sent data to {topic}: {key}")
        except Exception as e:
            logger.error(f"Failed to send to Kafka: {e}")

    def close(self):
        """Close the producer"""
        self.producer.close()
