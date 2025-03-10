import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import logging
import re

# Download NLTK data on first run
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def initialize_sentiment_analyzer():
    """
    Initializes and returns a sentiment analyzer.
    
    Returns:
        SentimentIntensityAnalyzer: NLTK's VADER sentiment analyzer
    """
    return SentimentIntensityAnalyzer()

def analyze_news_sentiment(news_list, analyzer=None):
    """
    Analyzes sentiment of news articles.
    
    Args:
        news_list (list): List of dictionaries with news title and other info
        analyzer (SentimentIntensityAnalyzer, optional): Initialized sentiment analyzer
        
    Returns:
        dict: Dictionary with overall sentiment score and individual article scores
    """
    if not news_list:
        return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0, 'articles': []}
    
    if analyzer is None:
        analyzer = initialize_sentiment_analyzer()
    
    article_sentiments = []
    total_compound = 0
    total_pos = 0
    total_neg = 0
    total_neu = 0
    
    for news in news_list:
        # Analyze sentiment of the news title
        title = news['title']
        sentiment = analyzer.polarity_scores(title)
        
        article_sentiments.append({
            'title': title,
            'link': news['link'],
            'date': news['date'],
            'sentiment': sentiment
        })
        
        # Add to totals
        total_compound += sentiment['compound']
        total_pos += sentiment['pos']
        total_neg += sentiment['neg']
        total_neu += sentiment['neu']
    
    # Calculate averages
    n = len(news_list)
    avg_compound = total_compound / n
    avg_pos = total_pos / n
    avg_neg = total_neg / n
    avg_neu = total_neu / n
    
    return {
        'compound': avg_compound,
        'pos': avg_pos,
        'neu': avg_neu,
        'neg': avg_neg,
        'articles': article_sentiments
    }

def combine_sentiment_scores(news_sentiment, technical_sentiment):
    """
    Combines news sentiment and technical indicators sentiment.
    
    Args:
        news_sentiment (dict): News sentiment scores
        technical_sentiment (dict): Technical indicators sentiment scores
        
    Returns:
        dict: Combined sentiment scores
    """
    try:
        # News sentiment is in range [-1, 1] from VADER
        news_score = news_sentiment.get('compound', 0)
        
        # Technical sentiment is already in range [-1, 1]
        tech_score = technical_sentiment.get('overall', 0)
        
        # Weights for combining scores (can be adjusted)
        news_weight = 0.4
        tech_weight = 0.6
        
        # Combine scores
        combined_score = (news_score * news_weight) + (tech_score * tech_weight)
        
        # Scale to range [-1, 1] (though it should already be in that range)
        combined_score = max(min(combined_score, 1), -1)
        
        # Determine sentiment category
        if combined_score >= 0.5:
            category = "Very Bullish"
        elif combined_score >= 0.2:
            category = "Bullish"
        elif combined_score > -0.2:
            category = "Neutral"
        elif combined_score > -0.5:
            category = "Bearish"
        else:
            category = "Very Bearish"
        
        return {
            'score': combined_score,
            'category': category,
            'news_score': news_score,
            'technical_score': tech_score
        }
    
    except Exception as e:
        logging.error(f"Error combining sentiment scores: {e}")
        return {
            'score': 0,
            'category': "Neutral",
            'news_score': 0,
            'technical_score': 0
        }
