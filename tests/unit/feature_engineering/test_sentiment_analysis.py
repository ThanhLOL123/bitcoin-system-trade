from src.feature_engineering.sentiment_analysis import SentimentAnalysis

def test_get_sentiment_scores():
    analyzer = SentimentAnalysis()
    text = "This is a wonderful movie!"
    scores = analyzer.get_sentiment_scores(text)
    assert 'vader_compound' in scores
    assert 'blob_polarity' in scores
    assert scores['vader_compound'] > 0
    assert scores['blob_polarity'] > 0

def test_get_sentiment_scores_negative():
    analyzer = SentimentAnalysis()
    text = "This is a terrible movie!"
    scores = analyzer.get_sentiment_scores(text)
    assert scores['vader_compound'] < 0
    assert scores['blob_polarity'] < 0

def test_get_sentiment_scores_neutral():
    analyzer = SentimentAnalysis()
    text = "This is a movie."
    scores = analyzer.get_sentiment_scores(text)
    assert scores['vader_compound'] == 0
    assert scores['blob_polarity'] == 0
