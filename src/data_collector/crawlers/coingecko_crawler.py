# src/data_collector/crawlers/coingecko_crawler.py
import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from ..base_crawler import BaseCrawler, CrawlerConfig

class CoinGeckoCrawler(BaseCrawler):
    """CoinGecko API crawler for Bitcoin price data"""
    
    def __init__(self):
        # CoinGecko free tier: 50 requests/minute
        config = CrawlerConfig(rate_limit=45, timeout=30)
        super().__init__(config)
        
        self.base_url = "https://api.coingecko.com/api/v3"
        self.endpoints = {
            'current_price': '/simple/price',
            'market_data': '/coins/bitcoin/market_chart',
            'ohlc': '/coins/bitcoin/ohlc'
        }
    
    async def collect_data(self) -> List[Dict]:
        """Collect current Bitcoin data from CoinGecko"""
        data_points = []
        
        # Get current price data
        current_data = await self.get_current_price()
        if current_data:
            data_points.append(current_data)
        
        # Get historical OHLC data (last 24 hours)
        ohlc_data = await self.get_ohlc_data(days=1)
        if ohlc_data:
            data_points.extend(ohlc_data)
        
        return data_points
    
    async def get_current_price(self) -> Optional[Dict]:
        """Get current Bitcoin price and market data"""
        cache_key = f"coingecko:current:{datetime.now().strftime('%Y%m%d%H%M')}"
        cached = self.get_cached_data(cache_key)
        if cached:
            return cached
        
        url = f"{self.base_url}{self.endpoints['current_price']}"
        params = {
            'ids': 'bitcoin',
            'vs_currencies': 'usd',
            'include_market_cap': 'true',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true',
            'include_last_updated_at': 'true'
        }
        
        response = await self.make_request(url, params=params)
        if not response or 'bitcoin' not in response:
            return None
        
        bitcoin_data = response['bitcoin']
        
        current_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': 'BTC',
            'price_usd': bitcoin_data.get('usd'),
            'market_cap': bitcoin_data.get('usd_market_cap'),
            'volume_24h': bitcoin_data.get('usd_24h_vol'),
            'price_change_24h': bitcoin_data.get('usd_24h_change'),
            'source': 'coingecko',
            'data_type': 'current_price'
        }
        
        # Cache for 1 minute
        self.cache_data(cache_key, current_data, ttl=60)
        
        return current_data
    
    async def get_ohlc_data(self, days: int = 1) -> List[Dict]:
        """Get OHLC data for specified days"""
        url = f"{self.base_url}{self.endpoints['ohlc']}"
        params = {
            'vs_currency': 'usd',
            'days': days
        }
        
        response = await self.make_request(url, params=params)
        if not response or not isinstance(response, list):
            return []
        
        ohlc_data = []
        for entry in response:
            # [timestamp, open, high, low, close]
            ohlc_data.append({
                'timestamp': datetime.fromtimestamp(entry[0] / 1000).isoformat(),
                'symbol': 'BTC',
                'price_usd': entry[4], # Close price
                'open': entry[1],
                'high': entry[2],
                'low': entry[3],
                'source': 'coingecko',
                'data_type': 'ohlc'
            })
        return ohlc_data

    def transform_data(self, raw_data: List[Dict]) -> List[Dict]:
        """Transform raw data to standard format"""
        transformed_data = []
        for data_point in raw_data:
            # Example transformation: add a processed timestamp
            data_point['processed_at'] = datetime.now().isoformat()
            transformed_data.append(data_point)
        return transformed_data
