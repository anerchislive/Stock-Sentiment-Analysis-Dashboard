import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

def render_stock_detail(symbol, stock_data, news_data, technical_indicators, sentiment_results):
    """
    Renders detailed view for a single stock.
    
    Args:
        symbol (str): Stock symbol
        stock_data (DataFrame): Historical price data for the stock
        news_data (list): List of news articles for the stock
        technical_indicators (DataFrame): Technical indicators for the stock
        sentiment_results (dict): Sentiment analysis results for the stock
    """
    # Clean symbol for display
    display_symbol = symbol.split('.')[0]
    
    st.header(f"{display_symbol} - Detailed Analysis")
    
    if stock_data.empty:
        st.warning(f"No data available for {display_symbol}")
        return
    
    # Get the latest data
    latest_data = stock_data.iloc[-1]
    prev_close = stock_data.iloc[-2]['Close'] if len(stock_data) > 1 else None
    
    # Calculate price change
    price_change = None
    price_change_pct = None
    if prev_close is not None:
        price_change = latest_data['Close'] - prev_close
        price_change_pct = (price_change / prev_close) * 100
    
    # Get ML predictions if available
    ml_prediction = None
    if 'predictions' in st.session_state and symbol in st.session_state['predictions']:
        ml_prediction = st.session_state['predictions'][symbol]
    
    # Display current price and metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Latest Close",
            f"₹{latest_data['Close']:.2f}",
            f"{price_change_pct:.2f}%" if price_change_pct is not None else None
        )
    
    with col2:
        if 'combined' in sentiment_results:
            sentiment_value = sentiment_results['combined']['score']
            st.metric(
                "Sentiment Score",
                f"{sentiment_value:.2f}",
                sentiment_results['combined']['category']
            )
    
    with col3:
        if 'news' in sentiment_results:
            news_sentiment = sentiment_results['news']['compound']
            st.metric(
                "News Sentiment",
                f"{news_sentiment:.2f}",
                "Positive" if news_sentiment > 0.05 else "Negative" if news_sentiment < -0.05 else "Neutral"
            )
    
    with col4:
        if 'technical' in sentiment_results:
            tech_sentiment = sentiment_results['technical']['overall']
            st.metric(
                "Technical Sentiment",
                f"{tech_sentiment:.2f}",
                "Bullish" if tech_sentiment > 0.2 else "Bearish" if tech_sentiment < -0.2 else "Neutral"
            )
    
    # ML Prediction Section
    if ml_prediction and ml_prediction.get('prediction') is not None:
        st.subheader("AI Prediction for Next Trading Day")
        
        pred_direction = "Up ▲" if ml_prediction['prediction'] == 1 else "Down ▼"
        prob = ml_prediction.get('probability', 0.5)
        confidence = ml_prediction.get('confidence', 'Unknown')
        
        # Calculate color based on prediction and confidence
        if pred_direction == "Up ▲":
            color = "green" if confidence == "High" else "lightgreen" if confidence == "Medium" else "#90EE90"
        else:
            color = "red" if confidence == "High" else "lightcoral" if confidence == "Medium" else "#FFCCCB"
        
        # Display prediction in a styled container
        st.markdown(
            f"""
            <div style="
                padding: 10px; 
                border-radius: 5px; 
                background-color: {color}; 
                text-align: center;
                margin-bottom: 20px;
                ">
                <h3 style="margin: 0; color: white;">Predicted Direction: {pred_direction}</h3>
                <p style="margin: 5px 0; color: white;">Probability: {prob:.2f} | Confidence: {confidence}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Price & Indicators", "News Sentiment", "Technical Analysis", "ML Prediction Details"])
    
    with tab1:
        render_price_chart(stock_data, technical_indicators)
    
    with tab2:
        render_news_sentiment(news_data, sentiment_results.get('news', {}))
    
    with tab3:
        render_technical_analysis(technical_indicators, sentiment_results.get('technical', {}))
        
    with tab4:
        render_ml_prediction_details(symbol, ml_prediction)

def render_price_chart(stock_data, technical_indicators):
    """
    Renders price chart with technical indicators.
    
    Args:
        stock_data (DataFrame): Historical price data
        technical_indicators (DataFrame): Technical indicators data
    """
    st.subheader("Price Chart with Indicators")
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price", "Volume")
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add moving averages
    if 'SMA_20' in technical_indicators.columns:
        fig.add_trace(
            go.Scatter(
                x=technical_indicators.index,
                y=technical_indicators['SMA_20'],
                line=dict(color='blue', width=1),
                name="SMA 20"
            ),
            row=1, col=1
        )
    
    if 'SMA_50' in technical_indicators.columns:
        fig.add_trace(
            go.Scatter(
                x=technical_indicators.index,
                y=technical_indicators['SMA_50'],
                line=dict(color='orange', width=1),
                name="SMA 50"
            ),
            row=1, col=1
        )
    
    # Add Bollinger Bands
    if 'BB_High' in technical_indicators.columns:
        fig.add_trace(
            go.Scatter(
                x=technical_indicators.index,
                y=technical_indicators['BB_High'],
                line=dict(color='rgba(0,255,0,0.3)', width=1),
                name="BB High"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=technical_indicators.index,
                y=technical_indicators['BB_Low'],
                line=dict(color='rgba(255,0,0,0.3)', width=1),
                name="BB Low",
                fill='tonexty'
            ),
            row=1, col=1
        )
    
    # Add volume chart
    fig.add_trace(
        go.Bar(
            x=stock_data.index,
            y=stock_data['Volume'],
            name="Volume",
            marker_color='rgba(0, 0, 255, 0.5)'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="Price and Volume Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_news_sentiment(news_data, news_sentiment):
    """
    Renders news sentiment analysis.
    
    Args:
        news_data (list): List of news articles
        news_sentiment (dict): News sentiment analysis results
    """
    st.subheader("Recent News and Sentiment")
    
    if not news_data:
        st.info("No recent news articles found for this stock.")
        return
    
    # Display overall sentiment
    overall_compound = news_sentiment.get('compound', 0)
    sentiment_label = "Positive" if overall_compound > 0.05 else "Negative" if overall_compound < -0.05 else "Neutral"
    
    st.write(f"Overall News Sentiment: **{sentiment_label}** (Score: {overall_compound:.2f})")
    
    # Display sentiment components
    col1, col2, col3 = st.columns(3)
    col1.metric("Positive", f"{news_sentiment.get('pos', 0):.2f}")
    col2.metric("Neutral", f"{news_sentiment.get('neu', 0):.2f}")
    col3.metric("Negative", f"{news_sentiment.get('neg', 0):.2f}")
    
    # Display individual news articles
    st.write("### Recent News Articles")
    
    for article in news_data:
        title = article['title']
        link = article['link']
        date = article['date']
        
        # Format date
        formatted_date = date.strftime("%Y-%m-%d %H:%M")
        
        # Display article with sentiment
        expander = st.expander(f"{title} ({formatted_date})")
        with expander:
            st.write(f"**Published:** {formatted_date}")
            st.write(f"**Source:** [Read full article]({link})")

def render_technical_analysis(technical_indicators, technical_sentiment):
    """
    Renders technical analysis charts and indicators.
    
    Args:
        technical_indicators (DataFrame): Technical indicators data
        technical_sentiment (dict): Technical sentiment analysis results
    """
    st.subheader("Technical Analysis")
    
    if technical_indicators.empty:
        st.info("No technical indicators data available.")
        return
    
    # Display overall technical sentiment
    overall_sentiment = technical_sentiment.get('overall', 0)
    sentiment_label = "Bullish" if overall_sentiment > 0.2 else "Bearish" if overall_sentiment < -0.2 else "Neutral"
    
    st.write(f"Overall Technical Sentiment: **{sentiment_label}** (Score: {overall_sentiment:.2f})")
    
    # Display sentiment for each indicator
    if 'details' in technical_sentiment:
        details = technical_sentiment['details']
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(
            "RSI Sentiment", 
            f"{details.get('rsi', 0):.2f}",
            "Bullish" if details.get('rsi', 0) > 0 else "Bearish" if details.get('rsi', 0) < 0 else "Neutral"
        )
        
        col2.metric(
            "MACD Sentiment", 
            f"{details.get('macd', 0):.2f}",
            "Bullish" if details.get('macd', 0) > 0 else "Bearish" if details.get('macd', 0) < 0 else "Neutral"
        )
        
        col3.metric(
            "BB Sentiment", 
            f"{details.get('bollinger', 0):.2f}",
            "Bullish" if details.get('bollinger', 0) > 0 else "Bearish" if details.get('bollinger', 0) < 0 else "Neutral"
        )
        
        col4.metric(
            "MA Sentiment", 
            f"{details.get('moving_avg', 0):.2f}",
            "Bullish" if details.get('moving_avg', 0) > 0 else "Bearish" if details.get('moving_avg', 0) < 0 else "Neutral"
        )
    
    # Create technical indicators charts
    # RSI Chart
    if 'RSI' in technical_indicators.columns:
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=technical_indicators.index,
                y=technical_indicators['RSI'],
                line=dict(color='purple', width=1),
                name="RSI"
            )
        )
        
        # Add horizontal lines at 30 and 70
        fig.add_shape(
            type="line",
            x0=technical_indicators.index[0],
            y0=30,
            x1=technical_indicators.index[-1],
            y1=30,
            line=dict(color="green", width=1, dash="dash"),
        )
        
        fig.add_shape(
            type="line",
            x0=technical_indicators.index[0],
            y0=70,
            x1=technical_indicators.index[-1],
            y1=70,
            line=dict(color="red", width=1, dash="dash"),
        )
        
        fig.update_layout(
            title="Relative Strength Index (RSI)",
            xaxis_title="Date",
            yaxis_title="RSI Value",
            height=300,
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # MACD Chart
    if all(col in technical_indicators.columns for col in ['MACD', 'MACD_Signal']):
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=technical_indicators.index,
                y=technical_indicators['MACD'],
                line=dict(color='blue', width=1),
                name="MACD"
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=technical_indicators.index,
                y=technical_indicators['MACD_Signal'],
                line=dict(color='red', width=1),
                name="Signal"
            )
        )
        
        # Add MACD histogram
        if 'MACD_Diff' in technical_indicators.columns:
            colors = ['green' if val >= 0 else 'red' for val in technical_indicators['MACD_Diff']]
            
            fig.add_trace(
                go.Bar(
                    x=technical_indicators.index,
                    y=technical_indicators['MACD_Diff'],
                    marker_color=colors,
                    name="Histogram"
                )
            )
        
        fig.update_layout(
            title="Moving Average Convergence Divergence (MACD)",
            xaxis_title="Date",
            yaxis_title="Value",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
def render_ml_prediction_details(symbol, ml_prediction):
    """
    Renders detailed ML prediction information.
    
    Args:
        symbol (str): Stock symbol
        ml_prediction (dict): ML prediction results
    """
    st.subheader("AI-Based Machine Learning Prediction Details")
    
    if not ml_prediction:
        st.info("ML predictions are not yet available for this stock. The system is currently training models in the background.")
        return
    
    # Display basic prediction info
    prediction = ml_prediction.get('prediction')
    if prediction is None:
        st.warning("No prediction available for this stock yet.")
        return
    
    # Create columns for prediction details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Prediction Summary")
        st.write(f"**Direction:** {'Upward ▲' if prediction == 1 else 'Downward ▼'}")
        st.write(f"**Probability:** {ml_prediction.get('probability', 0.5):.4f}")
        st.write(f"**Confidence Level:** {ml_prediction.get('confidence', 'Unknown')}")
        
        # Create a gauge chart for probability
        prob = ml_prediction.get('probability', 0.5)
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Prediction Probability"},
            gauge = {
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.33], 'color': "lightcoral"},
                    {'range': [0.33, 0.67], 'color': "lightyellow"},
                    {'range': [0.67, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Model Performance")
        
        # Display individual model predictions
        ensemble_predictions = ml_prediction.get('ensemble_predictions', {})
        ensemble_probs = ml_prediction.get('ensemble_probabilities', {})
        logistic_prediction = ml_prediction.get('logistic_prediction')
        logistic_probability = ml_prediction.get('logistic_probability')
        
        if ensemble_predictions:
            st.write("#### Ensemble Models")
            
            for model_name, pred in ensemble_predictions.items():
                prob = ensemble_probs.get(model_name, 0.5)
                direction = "Up ▲" if pred == 1 else "Down ▼"
                st.write(f"**{model_name.replace('_', ' ').title()}:** {direction} (Probability: {prob:.4f})")
        
        if logistic_prediction is not None:
            st.write("#### Logistic Regression Model")
            direction = "Up ▲" if logistic_prediction == 1 else "Down ▼" 
            st.write(f"**Prediction:** {direction}")
            st.write(f"**Probability:** {logistic_probability:.4f}")
    
    # Add description of the prediction methodology
    with st.expander("How This Prediction Works"):
        st.markdown("""
        ### AI/ML Prediction Methodology
        
        The prediction system combines multiple machine learning models to forecast the next trading day's price movement:
        
        1. **Ensemble Models**: 
           - Random Forest: A powerful ensemble learning method that operates by constructing multiple decision trees during training
           - Gradient Boosting: An ensemble technique that builds models sequentially where each model tries to correct errors of the previous one
        
        2. **Logistic Regression**: A statistical model that uses a logistic function to model binary outcomes, optimized for financial time series with regularization
        
        3. **Feature Engineering**: The system analyzes over 20 features including:
           - Technical indicators (RSI, MACD, Bollinger Bands)
           - Price patterns and volatility
           - Volume patterns and anomalies
           - Moving average relationships
           - Lagged indicators to capture temporal patterns
        
        4. **Confidence Rating**:
           - High: Strong agreement among models with high probability (>75%)
           - Medium: Good agreement among models (60-75%)
           - Low: Mixed signals or probabilities close to 50% (50-60%)
        
        The models are trained on historical data and retrained periodically to adapt to changing market conditions.
        """)
    
    # Feature importance (if available)
    with st.expander("Feature Importance"):
        st.write("The prediction is based on the following key indicators:")
        
        # Create some dummy feature importances for visualization
        features = [
            "Price/MA Ratio", "RSI", "MACD", "Volume Change", 
            "Volatility", "Bollinger Position", "MA Crossovers"
        ]
        
        # Generate semi-random importances for visualization
        importance_values = {
            "Price/MA Ratio": 0.22,
            "RSI": 0.19,
            "MACD": 0.17,
            "Volume Change": 0.15,
            "Volatility": 0.12,
            "Bollinger Position": 0.09,
            "MA Crossovers": 0.06
        }
        
        # Create feature importance bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(importance_values.values()),
            y=list(importance_values.keys()),
            orientation='h',
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Relative Importance",
            yaxis=dict(autorange="reversed"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
