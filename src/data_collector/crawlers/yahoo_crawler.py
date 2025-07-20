import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import yfinance as yf

from ..base_crawler import BaseCrawler, CrawlerConfig

class YahooCrawler(BaseCrawler):
    """Yahoo Finance crawler for Bitcoin price data"""
    
    def __init__(self):
        config = CrawlerConfig(rate_limit=10, timeout=30) # Yahoo Finance has stricter rate limits
        super().__init__(config)
        
    async def collect_data(self) -> List[Dict]:
        """Collect current Bitcoin data from Yahoo Finance"""
        data_points = []
        
        # Get current price data
        current_data = self.get_current_price()
        if current_data:
            data_points.append(current_data)
        
        return data_points
    
    def get_current_price(self) -> Optional[Dict]:
        """Get current Bitcoin price and market data from yfinance"""
        try:
            btc = yf.Ticker("BTC-USD")
            hist = btc.history(period="1d")
            if hist.empty:
                return None

            latest = hist.iloc[-1]
            return {
                'timestamp': datetime.now().isoformat(),
                'symbol': 'BTC',
                'price_usd': latest['Close'],
                'open': latest['Open'],
                'high': latest['High'],
                'low': latest['Low'],
                'volume_24h': latest['Volume'],
                'source': 'yahoo_finance',
                'data_type': 'current_price'
            }
        except Exception as e:
            self.logger.error(f"Error fetching data from Yahoo Finance: {e}")
            return None

    def transform_data(self, raw_data: List[Dict]) -> List[Dict]:
        """Transform raw data to standard format"""
        return raw_data
