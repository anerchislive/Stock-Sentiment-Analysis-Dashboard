"""
Module for machine learning-based stock price prediction using scikit-learn.
"""
import os
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Constants
PREDICTION_HORIZON = 1  # Predict 1 day ahead
MODEL_RETRAIN_DAYS = 15  # Retrain models every 15 days

def create_features(data):
    """
    Create features for machine learning from stock data and technical indicators.
    
    Args:
        data (DataFrame): DataFrame with stock price and technical indicators
        
    Returns:
        DataFrame: DataFrame with features for machine learning
    """
    # Create a copy of the data
    features = data.copy()
    
    # Add price-based features
    features['return_1d'] = features['Close'].pct_change(1)
    features['return_5d'] = features['Close'].pct_change(5)
    features['return_10d'] = features['Close'].pct_change(10)
    features['volatility_5d'] = features['return_1d'].rolling(window=5).std()
    features['volatility_10d'] = features['return_1d'].rolling(window=10).std()
    
    # Add volume-based features
    features['volume_change_1d'] = features['Volume'].pct_change(1)
    features['volume_change_5d'] = features['Volume'].pct_change(5)
    features['volume_ma5'] = features['Volume'].rolling(window=5).mean()
    features['volume_ma10'] = features['Volume'].rolling(window=10).mean()
    features['volume_ratio'] = features['Volume'] / features['volume_ma5']
    
    # Add price pattern features
    features['price_ma_ratio_5d'] = features['Close'] / features['Close'].rolling(window=5).mean()
    features['price_ma_ratio_10d'] = features['Close'] / features['Close'].rolling(window=10).mean()
    
    # Calculate target variable (1 if price goes up next day, 0 otherwise)
    features['target'] = (features['Close'].shift(-PREDICTION_HORIZON) > features['Close']).astype(int)
    
    # Drop NaN values
    features = features.dropna()
    
    return features

def prepare_sequence_data(features):
    """
    Prepare sequence data for prediction models.
    This is a simplified version that doesn't create actual sequences
    but extracts lagged features to capture temporal patterns.
    
    Args:
        features (DataFrame): DataFrame with features
        
    Returns:
        tuple: (X, y, scaler, feature_columns)
    """
    # Select features for modeling
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'BB_Mid',
        'SMA_20', 'SMA_50', 'EMA_20',
        'return_1d', 'return_5d', 'return_10d',
        'volume_change_1d', 'volume_ma5', 'price_ma_ratio_5d'
    ]
    
    # Ensure all selected features exist in the DataFrame
    available_columns = [col for col in feature_columns if col in features.columns]
    
    # Create lagged features to capture time series patterns
    lag_features = features[available_columns].copy()
    
    # Add 1-day and 2-day lagged features for key indicators
    for col in ['Close', 'Volume', 'RSI', 'MACD']:
        if col in lag_features.columns:
            lag_features[f'{col}_lag1'] = lag_features[col].shift(1)
            lag_features[f'{col}_lag2'] = lag_features[col].shift(2)
    
    # Drop rows with NaN values
    lag_features = lag_features.dropna()
    
    # Get the target variable
    y = features.loc[lag_features.index, 'target']
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(lag_features)
    
    return X, y, scaler, lag_features.columns.tolist()

def build_logistic_regression():
    """
    Build a logistic regression model for stock prediction.
    
    Returns:
        LogisticRegression: Scikit-learn logistic regression model
    """
    return LogisticRegression(
        C=0.1,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )

def build_ensemble_model():
    """
    Build ensemble model with Random Forest and Gradient Boosting.
    
    Returns:
        dict: Dictionary with trained models
    """
    return {
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }

def train_models(stock_data, symbol):
    """
    Train ML models for stock prediction.
    
    Args:
        stock_data (DataFrame): Historical stock data with technical indicators
        symbol (str): Stock symbol
        
    Returns:
        dict: Dictionary with trained models and training information
    """
    logger.info(f"Training ML models for {symbol}")
    
    # Create features
    features = create_features(stock_data)
    
    if len(features) < 60:  # Minimum data requirement (arbitrary threshold)
        logger.warning(f"Not enough data for {symbol} to train ML models")
        return None
    
    # Split data for traditional ML models
    # Use earlier data for training, keep most recent for validation
    train_size = int(0.8 * len(features))
    X_train = features.iloc[:train_size].drop(['target', 'Date'], axis=1, errors='ignore')
    y_train = features.iloc[:train_size]['target']
    X_valid = features.iloc[train_size:].drop(['target', 'Date'], axis=1, errors='ignore')
    y_valid = features.iloc[train_size:]['target']
    
    # Train ensemble models
    ensemble_models = build_ensemble_model()
    ensemble_results = {}
    
    for name, model in ensemble_models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_valid)
        
        accuracy = accuracy_score(y_valid, predictions)
        precision = precision_score(y_valid, predictions, zero_division=0)
        recall = recall_score(y_valid, predictions, zero_division=0)
        f1 = f1_score(y_valid, predictions, zero_division=0)
        
        ensemble_results[name] = {
            'model': model,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        }
        
        logger.info(f"{symbol} - {name} model - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
    
    # Prepare sequence data with lagged features
    X, y, scaler, feature_columns = prepare_sequence_data(features)
    
    if len(X) < 50:  # Minimum data points for advanced models
        logger.warning(f"Not enough data for {symbol} to train logistic regression model")
        logistic_model = None
    else:
        # Split data for logistic regression
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Build and train logistic regression model
        logistic_model = build_logistic_regression()
        logistic_model.fit(X_train, y_train)
        
        # Evaluate logistic regression model
        predictions = logistic_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        logger.info(f"{symbol} - Logistic Regression model - Accuracy: {accuracy:.4f}")
    
    # Save models
    model_info = {
        'ensemble_models': ensemble_results,
        'logistic_model': logistic_model if logistic_model is not None else None,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'training_date': datetime.now(),
        'data_length': len(features)
    }
    
    # Save models to disk
    try:
        os.makedirs(os.path.join(MODELS_DIR, symbol), exist_ok=True)
        
        # Save ensemble models
        for name, model_data in ensemble_results.items():
            model_path = os.path.join(MODELS_DIR, symbol, f"{name}.joblib")
            joblib.dump(model_data['model'], model_path)
        
        # Save logistic regression model
        if logistic_model is not None:
            logistic_path = os.path.join(MODELS_DIR, symbol, "logistic_model.joblib")
            joblib.dump(logistic_model, logistic_path)
        
        # Save scaler and feature info
        metadata_path = os.path.join(MODELS_DIR, symbol, "metadata.joblib")
        joblib.dump({
            'scaler': scaler,
            'feature_columns': feature_columns,
            'training_date': datetime.now(),
            'data_length': len(features)
        }, metadata_path)
        
        logger.info(f"Saved models for {symbol} to {os.path.join(MODELS_DIR, symbol)}")
    except Exception as e:
        logger.error(f"Error saving models for {symbol}: {e}")
    
    return model_info

def load_models(symbol):
    """
    Load trained ML models for stock prediction.
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        dict: Dictionary with loaded models
    """
    model_dir = os.path.join(MODELS_DIR, symbol)
    
    if not os.path.exists(model_dir):
        logger.warning(f"No models found for {symbol}")
        return None
    
    try:
        # Load metadata
        metadata_path = os.path.join(model_dir, "metadata.joblib")
        metadata = joblib.load(metadata_path)
        
        # Load ensemble models
        ensemble_models = {}
        for model_name in ['random_forest', 'gradient_boosting']:
            model_path = os.path.join(model_dir, f"{model_name}.joblib")
            if os.path.exists(model_path):
                ensemble_models[model_name] = joblib.load(model_path)
        
        # Load logistic regression model
        logistic_path = os.path.join(model_dir, "logistic_model.joblib")
        logistic_model = joblib.load(logistic_path) if os.path.exists(logistic_path) else None
        
        return {
            'ensemble_models': ensemble_models,
            'logistic_model': logistic_model,
            'metadata': metadata,
            'training_date': metadata.get('training_date')
        }
    except Exception as e:
        logger.error(f"Error loading models for {symbol}: {e}")
        return None

def models_need_retraining(symbol):
    """
    Check if models need retraining.
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        bool: True if models need retraining, False otherwise
    """
    model_dir = os.path.join(MODELS_DIR, symbol)
    
    if not os.path.exists(model_dir):
        return True
    
    try:
        # Check metadata for training date
        metadata_path = os.path.join(model_dir, "metadata.joblib")
        if not os.path.exists(metadata_path):
            return True
        
        metadata = joblib.load(metadata_path)
        training_date = metadata.get('training_date')
        
        if training_date is None:
            return True
        
        # Check if models were trained more than MODEL_RETRAIN_DAYS ago
        days_since_training = (datetime.now() - training_date).days
        return days_since_training > MODEL_RETRAIN_DAYS
    except Exception as e:
        logger.error(f"Error checking if models need retraining for {symbol}: {e}")
        return True

def train_models_if_needed(stock_data):
    """
    Train models for each stock if needed.
    
    Args:
        stock_data (dict): Dictionary with stock symbols as keys and their data as values
        
    Returns:
        dict: Dictionary with symbols as keys and model info as values
    """
    model_info = {}
    
    for symbol, data in stock_data.items():
        if data.empty:
            continue
        
        if models_need_retraining(symbol):
            logger.info(f"Models for {symbol} need retraining")
            model_info[symbol] = train_models(data, symbol)
        else:
            logger.info(f"Using existing models for {symbol}")
            model_info[symbol] = load_models(symbol)
    
    return model_info

def make_prediction(model_info, latest_data):
    """
    Make prediction for a single stock.
    
    Args:
        model_info (dict): Dictionary with model information
        latest_data (DataFrame): Latest stock data with technical indicators
        
    Returns:
        dict: Dictionary with prediction information
    """
    if model_info is None or latest_data.empty:
        return {
            'prediction': None,
            'probability': None,
            'confidence': 'Unknown',
            'ensemble_predictions': {}
        }
    
    try:
        # Create features
        features = create_features(latest_data)
        if features.empty:
            return {
                'prediction': None,
                'probability': None,
                'confidence': 'Unknown',
                'ensemble_predictions': {}
            }
        
        # Get the latest data point (excluding target)
        latest_features = features.iloc[-1:].drop(['target', 'Date'], axis=1, errors='ignore')
        
        # Make predictions with ensemble models
        ensemble_predictions = {}
        ensemble_probabilities = {}
        
        for name, model in model_info.get('ensemble_models', {}).items():
            pred = model.predict(latest_features)[0]
            
            # Get probability if possible
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(latest_features)[0][1]  # Probability of positive class
            else:
                prob = float(pred)
                
            ensemble_predictions[name] = int(pred)
            ensemble_probabilities[name] = prob
        
        # Make prediction with logistic regression model
        logistic_prediction = None
        logistic_probability = None
        
        if model_info.get('logistic_model') is not None and model_info.get('metadata') is not None:
            # Prepare data for logistic regression
            metadata = model_info['metadata']
            scaler = metadata['scaler']
            feature_columns = metadata['feature_columns']
            
            # Get the latest data point
            latest_point = features.iloc[-1:].drop(['target', 'Date'], axis=1, errors='ignore')
            
            # Ensure all required columns exist
            available_columns = [col for col in feature_columns if col in latest_point.columns]
            
            # Scale the data
            scaled_point = scaler.transform(latest_point[available_columns])
            
            # Make prediction
            logistic_prediction = int(model_info['logistic_model'].predict(scaled_point)[0])
            
            # Get probability if possible
            if hasattr(model_info['logistic_model'], 'predict_proba'):
                logistic_probability = model_info['logistic_model'].predict_proba(scaled_point)[0][1]
            else:
                logistic_probability = float(logistic_prediction)
        
        # Combine predictions
        all_probabilities = list(ensemble_probabilities.values())
        if logistic_probability is not None:
            all_probabilities.append(logistic_probability)
        
        # Calculate average probability
        avg_probability = sum(all_probabilities) / len(all_probabilities) if all_probabilities else None
        
        # Calculate consensus prediction
        if avg_probability is not None:
            consensus_prediction = 1 if avg_probability > 0.5 else 0
            
            # Determine confidence level
            if abs(avg_probability - 0.5) > 0.25:
                confidence = 'High'
            elif abs(avg_probability - 0.5) > 0.15:
                confidence = 'Medium'
            else:
                confidence = 'Low'
        else:
            consensus_prediction = None
            confidence = 'Unknown'
        
        return {
            'prediction': consensus_prediction,
            'probability': avg_probability,
            'confidence': confidence,
            'ensemble_predictions': ensemble_predictions,
            'logistic_prediction': logistic_prediction,
            'ensemble_probabilities': ensemble_probabilities,
            'logistic_probability': logistic_probability
        }
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return {
            'prediction': None,
            'probability': None,
            'confidence': 'Error',
            'error': str(e)
        }

def make_predictions(stock_data, technical_indicators):
    """
    Make predictions for all stocks.
    
    Args:
        stock_data (dict): Dictionary with stock symbols as keys and their data as values
        technical_indicators (dict): Dictionary with stock symbols as keys and technical indicators as values
        
    Returns:
        dict: Dictionary with stock symbols as keys and prediction results as values
    """
    predictions = {}
    
    for symbol in stock_data.keys():
        if symbol not in technical_indicators or technical_indicators[symbol].empty:
            continue
        
        # Load models
        model_info = load_models(symbol)
        
        if model_info is None:
            logger.warning(f"No models available for {symbol}, training now")
            model_info = train_models(stock_data[symbol], symbol)
            
        if model_info is not None:
            # Make prediction
            latest_data = technical_indicators[symbol]
            prediction = make_prediction(model_info, latest_data)
            predictions[symbol] = prediction
    
    return predictions