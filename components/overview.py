import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def render_overview(stock_data, sentiment_results):
    """
    Renders the overview section of the dashboard with all stocks.
    
    Args:
        stock_data (dict): Dictionary with stock symbols as keys and their data as values
        sentiment_results (dict): Dictionary with stock symbols as keys and sentiment analysis results as values
    """
    st.header("Stock Sentiment Overview")
    
    if not sentiment_results:
        st.warning("No sentiment data available. Please check if stock symbols were loaded correctly.")
        return
    
    # Create a dataframe from sentiment results for the overview table
    overview_data = []
    
    for symbol, sentiment in sentiment_results.items():
        # Clean the symbol for display (remove suffix like '.NS')
        display_symbol = symbol.split('.')[0]
        
        # Get the latest price if available
        latest_price = None
        price_change = None
        if symbol in stock_data and not stock_data[symbol].empty:
            latest_data = stock_data[symbol].iloc[-1]
            prev_data = stock_data[symbol].iloc[-2] if len(stock_data[symbol]) > 1 else latest_data
            latest_price = latest_data['Close']
            price_change = (latest_price - prev_data['Close']) / prev_data['Close'] * 100
        
        overview_data.append({
            'Symbol': display_symbol,
            'Latest Price': latest_price,
            'Price Change %': price_change,
            'Sentiment Score': sentiment['combined']['score'],
            'Sentiment': sentiment['combined']['category'],
            'News Sentiment': sentiment['news']['compound'],
            'Technical Sentiment': sentiment['technical']['overall']
        })
    
    if not overview_data:
        st.warning("No data available for overview.")
        return
    
    df_overview = pd.DataFrame(overview_data)
    
    # Sort by sentiment score (descending)
    df_overview = df_overview.sort_values('Sentiment Score', ascending=False)
    
    # Add styling to the overview table
    def highlight_sentiment(val):
        """Highlight cells based on sentiment value"""
        if isinstance(val, (int, float)):
            if val >= 0.5:
                return 'background-color: #CCFFCC'  # Light green
            elif val >= 0.2:
                return 'background-color: #E6FFE6'  # Very light green
            elif val <= -0.5:
                return 'background-color: #FFCCCC'  # Light red
            elif val <= -0.2:
                return 'background-color: #FFE6E6'  # Very light red
        return ''
    
    # Format the price and percentage change
    df_overview['Latest Price'] = df_overview['Latest Price'].map('{:.2f}'.format)
    df_overview['Price Change %'] = df_overview['Price Change %'].map('{:.2f}%'.format)
    
    # Apply styling
    styled_df = df_overview.style.applymap(highlight_sentiment, subset=['Sentiment Score', 'News Sentiment', 'Technical Sentiment'])
    
    # Display the styled dataframe
    st.dataframe(styled_df, width=1000)
    
    # Create sentiment distribution chart
    st.subheader("Sentiment Distribution")
    
    # Count stocks in each sentiment category
    sentiment_counts = df_overview['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    # Order the categories from most negative to most positive
    category_order = ["Very Bearish", "Bearish", "Neutral", "Bullish", "Very Bullish"]
    sentiment_counts['Sentiment'] = pd.Categorical(
        sentiment_counts['Sentiment'], 
        categories=category_order, 
        ordered=True
    )
    sentiment_counts = sentiment_counts.sort_values('Sentiment')
    
    # Set colors for each category
    colors = {
        "Very Bearish": "#FF4136",
        "Bearish": "#FF851B",
        "Neutral": "#FFDC00",
        "Bullish": "#2ECC40",
        "Very Bullish": "#01FF70"
    }
    
    # Create bar chart
    fig = px.bar(
        sentiment_counts,
        x='Sentiment',
        y='Count',
        color='Sentiment',
        color_discrete_map=colors,
        title="Distribution of Stock Sentiment"
    )
    
    fig.update_layout(
        xaxis_title="Sentiment Category",
        yaxis_title="Number of Stocks",
        legend_title="Sentiment",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top 5 Bullish and Bearish stocks
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 5 Bullish Stocks")
        top_bullish = df_overview.nlargest(5, 'Sentiment Score')
        st.table(top_bullish[['Symbol', 'Sentiment Score', 'Sentiment']])
    
    with col2:
        st.subheader("Top 5 Bearish Stocks")
        top_bearish = df_overview.nsmallest(5, 'Sentiment Score')
        st.table(top_bearish[['Symbol', 'Sentiment Score', 'Sentiment']])
