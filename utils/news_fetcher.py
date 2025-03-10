import feedparser
import requests
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
import re

def fetch_google_news(symbol, company_name=None, max_results=5):
    """
    Fetches news articles for a given stock symbol from Google News RSS feed.
    
    Args:
        symbol (str): Stock symbol
        company_name (str, optional): Company name for better search results
        max_results (int): Maximum number of news articles to return
        
    Returns:
        list: List of dictionaries with news title, link, and date
    """
    try:
        # Remove suffix like '.NS' for better search results
        clean_symbol = symbol.split('.')[0]
        
        # Create search query
        if company_name:
            query = f"{company_name} OR {clean_symbol} stock"
        else:
            query = f"{clean_symbol} stock"
        
        # Format query for URL
        query = query.replace(' ', '+')
        
        # Build Google News RSS feed URL
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        
        # Parse RSS feed
        feed = feedparser.parse(url)
        
        news_list = []
        for i, entry in enumerate(feed.entries):
            if i >= max_results:
                break
            
            # Extract date and convert to datetime object
            pub_date = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %Z")
            
            news_list.append({
                'title': entry.title,
                'link': entry.link,
                'date': pub_date
            })
        
        return news_list
    
    except Exception as e:
        logging.error(f"Error fetching news for {symbol}: {e}")
        return []

def clean_news_text(text):
    """
    Cleans news text by removing unwanted characters and formatting.
    
    Args:
        text (str): News text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def fetch_news_with_cache(symbols, company_names=None, max_results_per_symbol=5, cache_duration_hours=24):
    """
    Fetches news for multiple symbols with caching to avoid repeated API calls.
    
    Args:
        symbols (list): List of stock symbols
        company_names (dict, optional): Dictionary mapping symbols to company names
        max_results_per_symbol (int): Maximum number of news articles per symbol
        cache_duration_hours (int): News cache duration in hours
        
    Returns:
        dict: Dictionary with stock symbols as keys and their news as values
    """
    try:
        # Initialize news dictionary
        news_dict = {}
        
        for symbol in symbols:
            # Clean symbol for display (remove '.NS' suffix)
            display_symbol = symbol.split('.')[0]
            
            # Get company name if available
            company_name = None
            if company_names and display_symbol in company_names:
                company_name = company_names[display_symbol]
            
            # Fetch news
            news = fetch_google_news(symbol, company_name, max_results_per_symbol)
            
            # Add to dictionary
            if news:
                news_dict[symbol] = news
            
            # Sleep to avoid rate limiting
            time.sleep(0.5)
        
        return news_dict
    
    except Exception as e:
        logging.error(f"Error fetching news for multiple symbols: {e}")
        return {}
