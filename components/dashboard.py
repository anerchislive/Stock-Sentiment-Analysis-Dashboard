import streamlit as st
import pandas as pd
import os
import logging
import time
from utils.stock_data import read_stock_symbols, fetch_stock_data, get_stock_info
from utils.technical_indicators import calculate_technical_indicators, get_technical_sentiment
from utils.news_fetcher import fetch_news_with_cache
from utils.sentiment_analysis import initialize_sentiment_analyzer, analyze_news_sentiment, combine_sentiment_scores
from utils.data_caching import get_latest_data, start_background_fetching
from utils.ml_prediction import train_models_if_needed, make_predictions
from components.overview import render_overview
from components.stock_detail import render_stock_detail
from components.detailed_table import render_detailed_table
from components.ml_predictions_table import render_ml_predictions_table

def setup_dashboard():
    """
    Sets up the main dashboard layout and navigation.
    """
    # Set page title and configuration
    st.set_page_config(
        page_title="Stock Sentiment Analysis Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    # Title and description
    st.title("📊 Stock Sentiment Analysis Dashboard")
    st.markdown(
        """
        This dashboard analyzes sentiment for stocks based on news and technical indicators.
        Select a stock from the sidebar for detailed analysis.
        """
    )
    
    # Initialize session state if not already done
    if 'stock_data' not in st.session_state:
        st.session_state['stock_data'] = {}
    
    if 'news_data' not in st.session_state:
        st.session_state['news_data'] = {}
    
    if 'sentiment_results' not in st.session_state:
        st.session_state['sentiment_results'] = {}
    
    if 'technical_indicators' not in st.session_state:
        st.session_state['technical_indicators'] = {}
    
    # Read stock symbols from CSV
    csv_path = "attached_assets/MW-SECURITIES-IN-F&O-11-Mar-2025.csv"
    symbols = read_stock_symbols(csv_path)
    
    if not symbols:
        st.error("Failed to load stock symbols from CSV file.")
        return
    
    # Create sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Add a button to refresh data
        if st.button("Refresh Data"):
            with st.spinner("Fetching latest data..."):
                load_data(symbols)
        
        # Add a select box to choose a view
        # Clean symbol names for display
        display_symbols = [symbol.split('.')[0] for symbol in symbols]
        
        view_options = ["Overview", "Detailed Table"] + display_symbols
        selected_view = st.selectbox(
            "Select view or stock for analysis:",
            view_options
        )
        
        # Convert back to the original symbol format if needed
        if selected_view not in ["Overview", "Detailed Table"]:
            selected_symbol = f"{selected_view}.NS"  # Add suffix back for stock
        else:
            selected_symbol = selected_view
        
        st.info(f"Total stocks: {len(symbols)}")
        
        # Information section
        st.markdown("""
        ### About This Dashboard
        
        This dashboard combines:
        - News sentiment analysis
        - Technical indicators
        - Market data from Yahoo Finance
        
        For a comprehensive view of stock sentiment.
        """)
    
    # Load data if not already loaded
    if not st.session_state['stock_data']:
        with st.spinner("Loading data for the first time..."):
            load_data(symbols)
    
    # Create a layout with two columns
    # Left column for ML predictions, right column for main content
    left_col, right_col = st.columns([1, 2])
    
    # Left column always shows ML predictions
    with left_col:
        # Check if predictions are in session state
        if 'predictions' not in st.session_state:
            st.session_state['predictions'] = {}
            
        render_ml_predictions_table(
            st.session_state['stock_data'],
            st.session_state['predictions']
        )
    
    # Right column shows the selected view
    with right_col:
        # Main content based on selection
        if selected_symbol == "Overview":
            render_overview(
                st.session_state['stock_data'],
                st.session_state['sentiment_results']
            )
        elif selected_symbol == "Detailed Table":
            render_detailed_table(
                st.session_state['stock_data'],
                st.session_state['news_data'],
                st.session_state['sentiment_results'],
                st.session_state['technical_indicators']
            )
        else:
            # Check if the selected symbol exists in our data
            if selected_symbol in st.session_state['stock_data']:
                render_stock_detail(
                    selected_symbol,
                    st.session_state['stock_data'].get(selected_symbol, pd.DataFrame()),
                    st.session_state['news_data'].get(selected_symbol, []),
                    st.session_state['technical_indicators'].get(selected_symbol, pd.DataFrame()),
                    st.session_state['sentiment_results'].get(selected_symbol, {})
                )
            else:
                # Extract symbol name without suffix for display
                display_symbol = selected_symbol.split('.')[0]
                st.warning(f"No data available for {display_symbol}")

def load_data(symbols):
    """
    Loads and processes data for all symbols.
    Uses cached data when available for faster loading,
    and triggers background data fetching for up-to-date data.
    
    Args:
        symbols (list): List of stock symbols
    """
    try:
        with st.spinner("Fetching latest data..."):
            # Get the latest data from cache (or fetch it if not available)
            stock_data, technical_indicators, news_data, sentiment_results, predictions = get_latest_data(symbols)
            
            # Update session state with the latest data
            st.session_state['stock_data'] = stock_data
            st.session_state['technical_indicators'] = technical_indicators
            st.session_state['news_data'] = news_data
            st.session_state['sentiment_results'] = sentiment_results
            
            # Initialize ML predictions in session state if needed
            if 'predictions' not in st.session_state:
                st.session_state['predictions'] = {}
            
            # Update predictions with any that were already cached
            if predictions:
                st.session_state['predictions'] = predictions
            
            # Start background data fetching to keep data updated
            # This runs in a separate thread and doesn't block the UI
            start_background_fetching(symbols)
            
            st.success("Data loaded successfully!")
            
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        st.error(f"An error occurred while loading data: {str(e)}")
