from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

class SentimentAnalysis:
    """Perform sentiment analysis on text data"""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def get_sentiment_scores(self, text: str) -> dict:
        """Get sentiment scores for a given text"""
        vader_scores = self.analyzer.polarity_scores(text)
        blob = TextBlob(text)
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'blob_polarity': blob.sentiment.polarity,
            'blob_subjectivity': blob.sentiment.subjectivity
        }