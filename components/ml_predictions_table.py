"""
Component for displaying ML predictions for all stocks in a table format.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render_ml_predictions_table(stock_data, predictions):
    """
    Renders ML predictions table for all stocks.
    
    Args:
        stock_data (dict): Dictionary with stock symbols as keys and their data as values
        predictions (dict): Dictionary with stock symbols as keys and prediction results as values
    """
    st.subheader("AI Price Movement Predictions")
    
    if not predictions:
        st.info("No ML predictions available. Predictions are being calculated in the background.")
        return
    
    # Create a DataFrame for the predictions
    prediction_data = []
    
    for symbol in predictions.keys():
        if symbol not in stock_data:
            continue
            
        prediction = predictions[symbol]
        latest_price = stock_data[symbol]['Close'].iloc[-1] if not stock_data[symbol].empty else None
        confidence = prediction.get('confidence', 'Unknown')
        probability = prediction.get('probability', None)
        direction = "Up ▲" if prediction.get('prediction') == 1 else "Down ▼" if prediction.get('prediction') == 0 else "Unknown"
        
        prediction_data.append({
            'Symbol': symbol,
            'Latest Price': latest_price,
            'Prediction': direction,
            'Probability': probability,
            'Confidence': confidence
        })
    
    # Convert to DataFrame
    if prediction_data:
        df = pd.DataFrame(prediction_data)
        
        # Function to color the rows based on prediction
        def highlight_prediction(s):
            return ['background-color: rgba(0, 255, 0, 0.2)' if s['Prediction'] == 'Up ▲' else 
                   'background-color: rgba(255, 0, 0, 0.2)' if s['Prediction'] == 'Down ▼' else 
                   '' for _ in s]
        
        # Function to color the confidence column
        def highlight_confidence(s):
            if s == 'High':
                return 'background-color: rgba(0, 255, 0, 0.3); color: darkgreen; font-weight: bold'
            elif s == 'Medium':
                return 'background-color: rgba(255, 255, 0, 0.3); color: darkorange'
            elif s == 'Low':
                return 'background-color: rgba(255, 0, 0, 0.2); color: darkred'
            return ''
        
        # Format the DataFrame
        formatted_df = df.copy()
        
        # Format probability as percentage
        if 'Probability' in formatted_df.columns:
            formatted_df['Probability'] = formatted_df['Probability'].apply(
                lambda x: f"{x*100:.1f}%" if x is not None else "N/A"
            )
        
        # Format price with 2 decimal places
        if 'Latest Price' in formatted_df.columns:
            formatted_df['Latest Price'] = formatted_df['Latest Price'].apply(
                lambda x: f"₹{x:.2f}" if x is not None else "N/A"
            )
        
        # Display the table with styling
        st.dataframe(
            formatted_df.style
            .apply(highlight_prediction, axis=1)
            .applymap(highlight_confidence, subset=['Confidence'])
            .format(precision=2)
        )
        
        # Add a gauge chart showing the distribution of predictions
        up_count = sum(1 for item in prediction_data if item['Prediction'] == 'Up ▲')
        down_count = sum(1 for item in prediction_data if item['Prediction'] == 'Down ▼')
        unknown_count = len(prediction_data) - up_count - down_count
        
        total = len(prediction_data)
        up_percent = (up_count / total) * 100 if total > 0 else 0
        down_percent = (down_count / total) * 100 if total > 0 else 0
        unknown_percent = (unknown_count / total) * 100 if total > 0 else 0
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Market Sentiment")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = up_percent,
                title = {'text': "Bullish Sentiment"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightcoral"},
                        {'range': [33, 67], 'color': "khaki"},
                        {'range': [67, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "darkgreen", 'width': 4},
                        'thickness': 0.75,
                        'value': up_percent
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Create a pie chart showing distribution of predictions
            labels = ['Bullish', 'Bearish', 'Unknown']
            values = [up_count, down_count, unknown_count]
            colors = ['lightgreen', 'lightcoral', 'lightgray']
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.3,
                marker=dict(colors=colors)
            )])
            
            fig.update_layout(
                title="Prediction Distribution",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        # Add explanation about the predictions
        with st.expander("About these predictions"):
            st.markdown("""
            ### How to Use These Predictions
            
            - **Predictions**: Indicate expected price movement for the next trading day
            - **Probability**: Represents model's confidence level as a percentage
            - **Confidence Rating**: Indicates agreement level among different prediction models:
                - **High**: Strong model consensus (>75% agreement)
                - **Medium**: Good model consensus (60-75% agreement)  
                - **Low**: Mixed signals or probabilities close to 50%
                
            These predictions use machine learning models trained on historical price data and technical indicators.
            They should be used as one of many inputs in your investment decision process, not as the sole factor.
            """)
    else:
        st.info("No prediction data available. The models are still calculating predictions.")