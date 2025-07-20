# src/data_collector/base_crawler.py
import asyncio
import aiohttp
import time
import random
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
from fake_useragent import UserAgent
import json
import redis

logger = logging.getLogger(__name__)

@dataclass
class CrawlerConfig:
    """Configuration for crawlers"""
    rate_limit: int = 60  # requests per minute
    retry_attempts: int = 3
    retry_delay: float = 1.0
    timeout: int = 30
    use_proxy: bool = False
    respect_robots: bool = True

class BaseCrawler(ABC):
    """Base class for all data crawlers"""
    
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.ua = UserAgent()
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request_time = 0
        
        # Redis for caching and rate limiting
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            headers=self.get_headers()
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def get_headers(self) -> Dict[str, str]:
        """Generate headers with random user agent"""
        return {
            'User-Agent': self.ua.random,
            'Accept': 'application/json, text/html, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    async def rate_limit_wait(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 60 / self.config.rate_limit  # seconds between requests
        
        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    async def make_request(self, url: str, method: str = 'GET', **kwargs) -> Optional[Dict]:
        """Make HTTP request with retry logic"""
        await self.rate_limit_wait()
        
        for attempt in range(self.config.retry_attempts):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limited
                        wait_time = self.config.retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        
            except Exception as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
        return None
    
    def cache_data(self, key: str, data: Any, ttl: int = 3600):
        """Cache data in Redis"""
        try:
            self.redis_client.setex(key, ttl, json.dumps(data, default=str))
        except Exception as e:
            logger.error(f"Failed to cache data: {e}")
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data from Redis"""
        try:
            cached = self.redis_client.get(key)
            return json.loads(cached) if cached else None
        except Exception as e:
            logger.error(f"Failed to get cached data: {e}")
            return None
    
    @abstractmethod
    async def collect_data(self) -> List[Dict]:
        """Collect data from source - implemented by subclasses"""
        pass
    
    @abstractmethod
    def transform_data(self, raw_data: List[Dict]) -> List[Dict]:
        """Transform raw data to standard format"""
        pass
