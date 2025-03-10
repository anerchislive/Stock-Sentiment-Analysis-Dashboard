import streamlit as st
import pandas as pd
from datetime import datetime

def render_detailed_table(stock_data, news_data, sentiment_results, technical_indicators):
    """
    Renders a full-width detailed table with all sentiment metrics and news for each stock.
    
    Args:
        stock_data (dict): Dictionary with stock symbols as keys and their data as values
        news_data (dict): Dictionary with stock symbols as keys and their news as values
        sentiment_results (dict): Dictionary with stock symbols as keys and sentiment analysis results as values
        technical_indicators (dict): Dictionary with stock symbols as keys and technical indicators as values
    """
    st.header("Detailed Stock Analysis Table")
    
    if not sentiment_results:
        st.warning("No sentiment data available. Please check if stock symbols were loaded correctly.")
        return
    
    # Create a dataframe from all available data
    detailed_data = []
    
    for symbol, sentiment in sentiment_results.items():
        # Clean the symbol for display (remove suffix like '.NS')
        display_symbol = symbol.split('.')[0]
        
        # Get the latest price if available
        latest_close = None
        if symbol in stock_data and not stock_data[symbol].empty:
            latest_data = stock_data[symbol].iloc[-1]
            latest_close = latest_data['Close']
        
        # Get technical sentiment details
        tech_sentiment = sentiment.get('technical', {})
        tech_details = tech_sentiment.get('details', {})
        
        # Get the most recent news article
        recent_news = ""
        news_date = ""
        if symbol in news_data and news_data[symbol]:
            most_recent = news_data[symbol][0]  # First article is most recent
            recent_news = most_recent['title']
            news_date = most_recent['date'].strftime("%Y-%m-%d %H:%M") if isinstance(most_recent['date'], datetime) else most_recent['date']
        
        detailed_data.append({
            'Stock': display_symbol,
            'Latest Close': latest_close,
            'Sentiment Score': sentiment.get('combined', {}).get('score', 0),
            'Technical Sentiment': tech_sentiment.get('overall', 0),
            'RSI Sentiment': tech_details.get('rsi', 0),
            'MACD Sentiment': tech_details.get('macd', 0),
            'BB Sentiment': tech_details.get('bollinger', 0),
            'MA Sentiment': tech_details.get('moving_avg', 0),
            'News Sentiment': sentiment.get('news', {}).get('compound', 0),
            'Recent News Articles': recent_news,
            'Published': news_date
        })
    
    if not detailed_data:
        st.warning("No data available for the detailed table.")
        return
    
    df_detailed = pd.DataFrame(detailed_data)
    
    # Format the price
    if 'Latest Close' in df_detailed.columns:
        df_detailed['Latest Close'] = df_detailed['Latest Close'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    # Format sentiment scores
    sentiment_cols = [
        'Sentiment Score', 'Technical Sentiment', 'RSI Sentiment', 
        'MACD Sentiment', 'BB Sentiment', 'MA Sentiment', 'News Sentiment'
    ]
    for col in sentiment_cols:
        if col in df_detailed.columns:
            df_detailed[col] = df_detailed[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    # Add conditional styling based on sentiment values
    def highlight_sentiment(val):
        """Highlight cells based on sentiment value"""
        if isinstance(val, str) and val != "N/A":
            try:
                num_val = float(val)
                if num_val >= 0.5:
                    return 'background-color: #CCFFCC'  # Light green
                elif num_val >= 0.2:
                    return 'background-color: #E6FFE6'  # Very light green
                elif num_val <= -0.5:
                    return 'background-color: #FFCCCC'  # Light red
                elif num_val <= -0.2:
                    return 'background-color: #FFE6E6'  # Very light red
            except ValueError:
                pass
        return ''
    
    # Apply styling
    styled_df = df_detailed.style.map(
        highlight_sentiment, 
        subset=sentiment_cols
    )
    
    # Display the styled dataframe with full width
    st.dataframe(styled_df, use_container_width=True)