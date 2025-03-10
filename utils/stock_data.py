import pandas as pd
import yfinance as yf
import os
import logging

def read_stock_symbols(csv_file_path):
    """
    Reads stock symbols from a CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file containing stock symbols
        
    Returns:
        list: List of stock symbols
    """
    try:
        df = pd.read_csv(csv_file_path)
        # Assuming the first column contains the stock symbols
        symbols = df.iloc[:, 0].tolist()
        # Remove any empty strings or NaN values
        symbols = [str(symbol) for symbol in symbols if pd.notna(symbol) and symbol]
        # Add '.NS' suffix for NSE stocks (assuming Indian stocks)
        symbols = [f"{symbol}.NS" for symbol in symbols]
        return symbols
    except Exception as e:
        logging.error(f"Error reading stock symbols: {e}")
        return []

def fetch_stock_data(symbols, period="1mo", interval="1d"):
    """
    Fetches stock data for given symbols using Yahoo Finance API.
    
    Args:
        symbols (list): List of stock symbols
        period (str): Time period for data (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y')
        interval (str): Data interval (e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d')
        
    Returns:
        dict: Dictionary with stock symbols as keys and their data as values
    """
    stock_data = {}
    
    for symbol in symbols:
        try:
            data = yf.Ticker(symbol).history(period=period, interval=interval)
            if not data.empty:
                stock_data[symbol] = data
            else:
                logging.warning(f"No data found for {symbol}")
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
    
    return stock_data

def get_stock_info(symbol):
    """
    Fetches detailed information about a stock.
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        dict: Dictionary with stock information
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return info
    except Exception as e:
        logging.error(f"Error fetching info for {symbol}: {e}")
        return {}
