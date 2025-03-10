import pandas as pd
import numpy as np
import ta
import logging

def calculate_technical_indicators(stock_data):
    """
    Calculates various technical indicators for the given stock data.
    
    Args:
        stock_data (DataFrame): Historical stock price data
        
    Returns:
        DataFrame: DataFrame with added technical indicators
    """
    if stock_data.empty:
        return pd.DataFrame()
    
    try:
        # Make a copy to avoid modifying the original data
        df = stock_data.copy()
        
        # Add RSI (Relative Strength Index)
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        
        # Add MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        
        # Add Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        df['BB_Mid'] = bollinger.bollinger_mavg()
        
        # Add Moving Averages
        df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
        df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
        
        # Add Volume indicators
        df['Volume_EMA'] = ta.trend.EMAIndicator(df['Volume'], window=20).ema_indicator()
        
        # Add ATR (Average True Range)
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        return df
    
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")
        return stock_data

def get_technical_sentiment(indicators_df):
    """
    Analyzes technical indicators to determine sentiment.
    
    Args:
        indicators_df (DataFrame): DataFrame with technical indicators
        
    Returns:
        dict: Dictionary with sentiment scores for different indicators and overall technical sentiment
    """
    try:
        if indicators_df.empty:
            return {'overall': 0, 'details': {}}
        
        # Get the latest values
        latest = indicators_df.iloc[-1]
        
        # RSI sentiment
        if latest['RSI'] > 70:
            rsi_sentiment = -0.5  # Overbought
        elif latest['RSI'] < 30:
            rsi_sentiment = 0.5  # Oversold
        elif latest['RSI'] > 50:
            rsi_sentiment = 0.3  # Bullish
        else:
            rsi_sentiment = -0.3  # Bearish
        
        # MACD sentiment
        if latest['MACD'] > latest['MACD_Signal']:
            macd_sentiment = 0.4  # Bullish
        else:
            macd_sentiment = -0.4  # Bearish
        
        # Bollinger Bands sentiment
        if latest['Close'] > latest['BB_High']:
            bb_sentiment = -0.4  # Overbought
        elif latest['Close'] < latest['BB_Low']:
            bb_sentiment = 0.4  # Oversold
        else:
            bb_sentiment = 0  # Neutral
        
        # Moving Averages sentiment
        if latest['Close'] > latest['SMA_20'] and latest['SMA_20'] > latest['SMA_50']:
            ma_sentiment = 0.5  # Strong bullish
        elif latest['Close'] > latest['SMA_20']:
            ma_sentiment = 0.3  # Bullish
        elif latest['Close'] < latest['SMA_20'] and latest['SMA_20'] < latest['SMA_50']:
            ma_sentiment = -0.5  # Strong bearish
        elif latest['Close'] < latest['SMA_20']:
            ma_sentiment = -0.3  # Bearish
        else:
            ma_sentiment = 0  # Neutral
        
        # Price trend
        price_trend = indicators_df['Close'].iloc[-5:].pct_change().mean() * 10
        price_sentiment = min(max(price_trend, -0.5), 0.5)  # Limit to range [-0.5, 0.5]
        
        # Calculate overall sentiment (weighted average)
        weights = {
            'rsi': 0.2,
            'macd': 0.2,
            'bollinger': 0.15,
            'moving_avg': 0.25,
            'price_trend': 0.2
        }
        
        overall_sentiment = (
            rsi_sentiment * weights['rsi'] +
            macd_sentiment * weights['macd'] +
            bb_sentiment * weights['bollinger'] +
            ma_sentiment * weights['moving_avg'] +
            price_sentiment * weights['price_trend']
        )
        
        # Normalize to range [-1, 1]
        overall_sentiment = min(max(overall_sentiment, -1), 1)
        
        return {
            'overall': overall_sentiment,
            'details': {
                'rsi': rsi_sentiment,
                'macd': macd_sentiment,
                'bollinger': bb_sentiment,
                'moving_avg': ma_sentiment,
                'price_trend': price_sentiment
            }
        }
    
    except Exception as e:
        logging.error(f"Error calculating technical sentiment: {e}")
        return {'overall': 0, 'details': {}}
