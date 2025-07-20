import asyncio
import praw
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from ..base_crawler import BaseCrawler, CrawlerConfig

class RedditCrawler(BaseCrawler):
    """Reddit crawler for sentiment data"""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        config = CrawlerConfig(rate_limit=30, timeout=60)
        super().__init__(config)
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )

    async def collect_data(self) -> List[Dict]:
        """Collect recent Bitcoin-related submissions from Reddit"""
        data_points = []
        subreddits = ["bitcoin", "cryptocurrency"]
        for sub in subreddits:
            subreddit = self.reddit.subreddit(sub)
            for submission in subreddit.hot(limit=25):
                data_points.append({
                    'timestamp': datetime.fromtimestamp(submission.created_utc).isoformat(),
                    'source': 'reddit',
                    'platform': sub,
                    'title': submission.title,
                    'content_snippet': submission.selftext[:200],
                    'url': submission.url,
                    'author': submission.author.name if submission.author else "[deleted]",
                    'mention_count': 1,
                    'engagement_score': submission.score,
                    'data_type': 'submission'
                })
        return data_points

    def transform_data(self, raw_data: List[Dict]) -> List[Dict]:
        """Transform raw data to standard format"""
        return raw_data
