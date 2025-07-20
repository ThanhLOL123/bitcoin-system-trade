import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.data_collector.crawlers.reddit_crawler import RedditCrawler
from datetime import datetime

@pytest.fixture
def reddit_crawler():
    with patch('redis.Redis') as mock_redis:
        mock_redis_instance = mock_redis.return_value
        mock_redis_instance.setex = AsyncMock()
        mock_redis_instance.get = AsyncMock(return_value=None)
        
        with patch('praw.Reddit') as mock_praw:
            mock_reddit_instance = mock_praw.return_value
            mock_subreddit = MagicMock()
            mock_reddit_instance.subreddit.return_value = mock_subreddit
            
            # Mock submission objects
            mock_submission1 = MagicMock()
            mock_submission1.created_utc = datetime.now().timestamp()
            mock_submission1.title = "Test Title 1"
            mock_submission1.selftext = "Test selftext 1. This is a longer text that should be truncated."
            mock_submission1.url = "http://example.com/1"
            mock_submission1.author = MagicMock(name="author1")
            mock_submission1.score = 100

            mock_submission2 = MagicMock()
            mock_submission2.created_utc = (datetime.now() - timedelta(hours=1)).timestamp()
            mock_submission2.title = "Test Title 2"
            mock_submission2.selftext = "Test selftext 2."
            mock_submission2.url = "http://example.com/2"
            mock_submission2.author = None # Mock deleted author
            mock_submission2.score = 50

            mock_subreddit.hot.return_value = [mock_submission1, mock_submission2]
            
            return RedditCrawler(client_id="test_id", client_secret="test_secret", user_agent="test_agent")

@pytest.mark.asyncio
async def test_collect_data_success(reddit_crawler):
    data = await reddit_crawler.collect_data()
    assert len(data) == 4 # 2 subreddits * 2 submissions
    assert data[0]['title'] == "Test Title 1"
    assert data[0]['content_snippet'] == "Test selftext 1. This is a longer text that should be truncated."[:200]
    assert data[1]['author'] == "[deleted]"
    assert data[0]['source'] == 'reddit'

def test_transform_data(reddit_crawler):
    raw_data = [{'title': 'Test', 'score': 10}]
    transformed_data = reddit_crawler.transform_data(raw_data)
    assert transformed_data == raw_data
