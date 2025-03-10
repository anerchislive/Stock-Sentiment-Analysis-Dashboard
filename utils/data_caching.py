"""
Module for handling data caching and background data fetching.
"""
import os
import time
import threading
import logging
import joblib
import pandas as pd
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache file paths
STOCK_DATA_CACHE = os.path.join(CACHE_DIR, 'stock_data.joblib')
NEWS_CACHE = os.path.join(CACHE_DIR, 'news_data.joblib')
TECHNICAL_INDICATORS_CACHE = os.path.join(CACHE_DIR, 'technical_indicators.joblib')
SENTIMENT_RESULTS_CACHE = os.path.join(CACHE_DIR, 'sentiment_results.joblib')
PREDICTIONS_CACHE = os.path.join(CACHE_DIR, 'predictions.joblib')

# Cache expiration time (in seconds)
CACHE_EXPIRY = {
    'stock_data': 3600,  # 1 hour
    'news_data': 7200,   # 2 hours
    'predictions': 86400 # 24 hours
}

# Global variable to track if the background thread is running
_background_thread_running = False

def save_to_cache(data, cache_file):
    """Save data to cache file."""
    try:
        joblib.dump(data, cache_file)
        logger.info(f"Saved data to cache: {cache_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving data to cache: {e}")
        return False

def load_from_cache(cache_file, max_age=None):
    """Load data from cache file if it exists and is not expired."""
    if not os.path.exists(cache_file):
        return None
    
    # Check if cache is expired
    if max_age:
        file_time = os.path.getmtime(cache_file)
        if time.time() - file_time > max_age:
            logger.info(f"Cache expired: {cache_file}")
            return None
    
    try:
        data = joblib.load(cache_file)
        logger.info(f"Loaded data from cache: {cache_file}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from cache: {e}")
        return None

def is_market_open():
    """Check if the market is currently open (US market hours approximation)."""
    now = datetime.now()
    
    # Weekend check
    if now.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
        return False
    
    # US Market hours (9:30 AM to 4:00 PM Eastern Time)
    # Simplified approximation - adjust as needed for proper timezone handling
    market_open_hour, market_open_minute = 9, 30
    market_close_hour, market_close_minute = 16, 0
    
    current_time = now.hour * 60 + now.minute  # Convert to minutes
    market_open_time = market_open_hour * 60 + market_open_minute
    market_close_time = market_close_hour * 60 + market_close_minute
    
    return market_open_time <= current_time <= market_close_time

def get_update_frequency():
    """Determine how frequently to update data based on market conditions."""
    if is_market_open():
        return {
            'stock_data': 900,  # 15 minutes when market is open
            'news_data': 1800,  # 30 minutes when market is open
            'predictions': 3600  # 1 hour when market is open
        }
    else:
        return CACHE_EXPIRY  # Use standard expiry times when market is closed

def background_data_fetching(symbols, stop_event=None):
    """
    Function to run in background thread for fetching and updating data.
    
    Args:
        symbols (list): List of stock symbols to fetch data for
        stop_event (threading.Event, optional): Event to signal thread to stop
    """
    from utils.stock_data import fetch_stock_data
    from utils.news_fetcher import fetch_news_with_cache
    from utils.technical_indicators import calculate_technical_indicators, get_technical_sentiment
    from utils.sentiment_analysis import initialize_sentiment_analyzer, analyze_news_sentiment, combine_sentiment_scores
    from utils.ml_prediction import train_models_if_needed, make_predictions
    
    if stop_event is None:
        stop_event = threading.Event()
    
    logger.info("Starting background data fetching thread")
    
    while not stop_event.is_set():
        try:
            update_frequency = get_update_frequency()
            
            # Check if stock data cache needs update
            stock_data = load_from_cache(STOCK_DATA_CACHE, update_frequency['stock_data'])
            if stock_data is None:
                logger.info("Fetching fresh stock data...")
                stock_data = fetch_stock_data(symbols, period="1mo")
                save_to_cache(stock_data, STOCK_DATA_CACHE)
            
            # Calculate and cache technical indicators
            technical_indicators = {}
            for symbol, data in stock_data.items():
                if not data.empty:
                    technical_indicators[symbol] = calculate_technical_indicators(data)
            save_to_cache(technical_indicators, TECHNICAL_INDICATORS_CACHE)
            
            # Check if news data cache needs update
            news_data = load_from_cache(NEWS_CACHE, update_frequency['news_data'])
            if news_data is None:
                logger.info("Fetching fresh news data...")
                company_names = {}  # Could populate this from stock info if needed
                news_data = fetch_news_with_cache(symbols, company_names)
                save_to_cache(news_data, NEWS_CACHE)
            
            # Calculate and cache sentiment
            analyzer = initialize_sentiment_analyzer()
            sentiment_results = {}
            
            for symbol in symbols:
                if symbol in stock_data and not stock_data[symbol].empty:
                    # Get technical sentiment
                    tech_sentiment = get_technical_sentiment(technical_indicators.get(symbol, pd.DataFrame()))
                    
                    # Get news sentiment
                    symbol_news = news_data.get(symbol, [])
                    news_sentiment = analyze_news_sentiment(symbol_news, analyzer)
                    
                    # Combine sentiments
                    combined_sentiment = combine_sentiment_scores(news_sentiment, tech_sentiment)
                    
                    # Store results
                    sentiment_results[symbol] = {
                        'news': news_sentiment,
                        'technical': tech_sentiment,
                        'combined': combined_sentiment
                    }
            
            save_to_cache(sentiment_results, SENTIMENT_RESULTS_CACHE)
            
            # Update ML predictions if needed
            predictions = load_from_cache(PREDICTIONS_CACHE, update_frequency['predictions'])
            if predictions is None:
                logger.info("Generating fresh ML predictions...")
                train_models_if_needed(stock_data)
                predictions = make_predictions(stock_data, technical_indicators)
                save_to_cache(predictions, PREDICTIONS_CACHE)
            
            # Sleep before next update cycle - use shorter interval during market hours
            sleep_time = 300 if is_market_open() else 1800  # 5 min if market open, 30 min otherwise
            logger.info(f"Background fetch complete, sleeping for {sleep_time} seconds")
            
            # Break sleep into smaller chunks to check stop_event more frequently
            for _ in range(int(sleep_time / 10)):
                if stop_event.is_set():
                    break
                time.sleep(10)
                
        except Exception as e:
            logger.error(f"Error in background data fetching: {e}")
            time.sleep(60)  # Sleep for a minute if there's an error
    
    logger.info("Background data fetching thread stopped")

def start_background_fetching(symbols):
    """Start the background data fetching thread if not already running."""
    global _background_thread_running
    
    if _background_thread_running:
        logger.info("Background fetching already running")
        return
    
    stop_event = threading.Event()
    thread = threading.Thread(
        target=background_data_fetching,
        args=(symbols, stop_event),
        daemon=True  # Make thread daemon so it exits when main program exits
    )
    thread.start()
    
    _background_thread_running = True
    logger.info("Started background data fetching thread")
    
    return stop_event

def get_latest_data(symbols):
    """
    Get the latest data from cache, or fetch it if not available.
    This function is used to quickly load the dashboard with the latest available data.
    
    Args:
        symbols (list): List of stock symbols
        
    Returns:
        tuple: (stock_data, technical_indicators, news_data, sentiment_results, predictions)
    """
    # Try to load from cache first
    stock_data = load_from_cache(STOCK_DATA_CACHE)
    technical_indicators = load_from_cache(TECHNICAL_INDICATORS_CACHE)
    news_data = load_from_cache(NEWS_CACHE)
    sentiment_results = load_from_cache(SENTIMENT_RESULTS_CACHE)
    predictions = load_from_cache(PREDICTIONS_CACHE)
    
    # Start background fetching if needed
    if not _background_thread_running:
        start_background_fetching(symbols)
    
    # If any data is missing, we need to fetch it synchronously for the first load
    if stock_data is None or technical_indicators is None or news_data is None or sentiment_results is None:
        from utils.stock_data import fetch_stock_data
        from utils.news_fetcher import fetch_news_with_cache
        from utils.technical_indicators import calculate_technical_indicators, get_technical_sentiment
        from utils.sentiment_analysis import initialize_sentiment_analyzer, analyze_news_sentiment, combine_sentiment_scores
        
        if stock_data is None:
            logger.info("No stock data in cache, fetching now...")
            stock_data = fetch_stock_data(symbols, period="1mo")
            save_to_cache(stock_data, STOCK_DATA_CACHE)
        
        if technical_indicators is None:
            logger.info("Calculating technical indicators...")
            technical_indicators = {}
            for symbol, data in stock_data.items():
                if not data.empty:
                    technical_indicators[symbol] = calculate_technical_indicators(data)
            save_to_cache(technical_indicators, TECHNICAL_INDICATORS_CACHE)
        
        if news_data is None:
            logger.info("No news data in cache, fetching now...")
            company_names = {}  # Could populate this from stock info if needed
            news_data = fetch_news_with_cache(symbols, company_names)
            save_to_cache(news_data, NEWS_CACHE)
        
        if sentiment_results is None:
            logger.info("Calculating sentiment results...")
            analyzer = initialize_sentiment_analyzer()
            sentiment_results = {}
            
            for symbol in symbols:
                if symbol in stock_data and not stock_data[symbol].empty:
                    # Get technical sentiment
                    tech_sentiment = get_technical_sentiment(technical_indicators.get(symbol, pd.DataFrame()))
                    
                    # Get news sentiment
                    symbol_news = news_data.get(symbol, [])
                    news_sentiment = analyze_news_sentiment(symbol_news, analyzer)
                    
                    # Combine sentiments
                    combined_sentiment = combine_sentiment_scores(news_sentiment, tech_sentiment)
                    
                    # Store results
                    sentiment_results[symbol] = {
                        'news': news_sentiment,
                        'technical': tech_sentiment,
                        'combined': combined_sentiment
                    }
            
            save_to_cache(sentiment_results, SENTIMENT_RESULTS_CACHE)
    
    # Handle predictions
    if predictions is None:
        # Don't block initial page load with ML predictions
        # They'll be calculated in the background thread
        predictions = {}
    
    return stock_data, technical_indicators, news_data, sentiment_results, predictions