"""
Random Forest model module for sales forecasting.
This module implements the core RandomForest-based forecasting functionality.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import pickle
import logging
import argparse
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('rf_model')

# Import settings if available
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from config.settings import (
        COMBINED_DATA_FILE, RF_MODEL_FILE, RF_MODEL_FEATURES_FILE,
        RF_FORECASTS_FILE, STATIC_DIR
    )
except ImportError:
    # Default paths for backward compatibility
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    COMBINED_DATA_FILE = os.path.join(ROOT_DIR, "combined_pizza_data.csv")
    RF_MODEL_FILE = os.path.join(ROOT_DIR, "models", "rf_model.pkl")
    RF_MODEL_FEATURES_FILE = os.path.join(ROOT_DIR, "models", "rf_model_features.pkl")
    RF_FORECASTS_FILE = os.path.join(ROOT_DIR, "rf_forecasts.csv")
    STATIC_DIR = os.path.join(ROOT_DIR, "static")


# Define ensemble predict function outside class for pickling support
def ensemble_predict(models, ensemble_weights, X_input, add_noise=False):
    """
    Weighted ensemble prediction with optional noise component and robust NaN handling
    
    Args:
        models (dict): Dictionary of named models
        ensemble_weights (dict): Dictionary of weights for each model
        X_input (DataFrame): Input features
        add_noise (bool): Whether to add noise for stochastic predictions
        
    Returns:
        ndarray: Predicted values
    """
    predictions = {}
    
    # Handle potential NaN values in input features
    X_clean = X_input.copy()
    nan_cols = X_clean.columns[X_clean.isna().any()].tolist()
    if nan_cols:
        # For each column with NaNs, apply appropriate imputation
        for col in nan_cols:
            if X_clean[col].isna().all():
                X_clean.loc[:, col] = 0
            else:
                # Use median for robustness against outliers
                X_clean.loc[:, col] = X_clean[col].fillna(X_clean[col].median())
    
    # Make predictions with each model in the ensemble
    for name, model in models.items():
        if model is not None:  # Check if model exists
            try:
                predictions[name] = model.predict(X_clean)
            except Exception as e:
                # If prediction fails, log and continue with other models
                logger.warning(f"Error predicting with {name} model: {e}")
                continue
    
    # Check if we have any valid predictions
    if not predictions:
        logger.error("No valid predictions from any models in ensemble")
        return np.zeros(X_input.shape[0])  # Return zeros as fallback
    
    # Weighted average of predictions
    weighted_pred = np.zeros(X_input.shape[0])
    total_weight = 0
    
    for name, preds in predictions.items():
        if name in ensemble_weights and ensemble_weights[name] > 0:
            weighted_pred += preds * ensemble_weights[name]
            total_weight += ensemble_weights[name]
    
    # If we have some valid weights, normalize by total weight
    if total_weight > 0:
        weighted_pred /= total_weight
        
    # Add stochastic noise component if requested
    if add_noise:
        # Extract randomness amplitude if available in X_input
        if 'Randomness_Amplitude' in X_input.columns:
            noise_scale = X_input['Randomness_Amplitude'].values
        else:
            # Default small noise (1-3% random variation)
            noise_scale = np.random.uniform(0.01, 0.03, size=X_input.shape[0])
            
        # Generate noise based on the prediction magnitude
        noise = np.random.normal(0, weighted_pred * noise_scale)
        weighted_pred += noise
        
    # Ensure predictions are non-negative
    weighted_pred = np.maximum(0, weighted_pred)
    
    return weighted_pred


# Create a model-like object that wraps the ensemble
# Moved outside of function for pickling support
class EnsembleModel:
    """
    Wrapper class for ensemble of models that provides a scikit-learn-like interface
    """
    def __init__(self, models, weights):
        """
        Initialize the ensemble model
        
        Args:
            models (dict): Dictionary of named models
            weights (dict): Dictionary of weights for each model
        """
        self.models = models
        self.weights = weights
        # Get feature importances from main model
        self.feature_importances_ = models['rf_standard'].feature_importances_
        # Store stochastic parameters
        self.stochastic_parameters = {
            'noise_base_level': 0.02,  # Base noise level (2%)
            'seasonal_amplitude': 0.03,  # Seasonal component (3%)
            'trend_factor': 0.005,     # Gradual trend component (0.5%)
            'last_prediction_time': None,  # Track time for auto-correlation
            'last_noise_value': 0.0,   # Previous noise value for auto-correlation
            'correlation_factor': 0.7   # Auto-correlation factor (0-1)
        }
    
    def predict(self, X, add_noise=False):
        """
        Make predictions with the ensemble model
        
        Args:
            X (DataFrame): Input features
            add_noise (bool): Whether to add stochastic noise
            
        Returns:
            ndarray: Predicted values
        """
        predictions = ensemble_predict(self.models, self.weights, X, add_noise)
        
        # Apply additional stochasticity if requested
        if add_noise:
            # Generate realistic noise with temporal correlation
            current_time = time.time()
            
            # Initialize tracking values if first prediction
            if self.stochastic_parameters['last_prediction_time'] is None:
                self.stochastic_parameters['last_prediction_time'] = current_time
                self.stochastic_parameters['last_noise_value'] = np.random.normal(0, 0.01, size=len(predictions))
            
            # Calculate time-based correlation factor
            time_diff = current_time - self.stochastic_parameters['last_prediction_time']
            correlation_decay = max(0, self.stochastic_parameters['correlation_factor'] * np.exp(-time_diff / 10))
            
            # Generate new noise with temporal correlation to previous noise
            base_noise = np.random.normal(0, self.stochastic_parameters['noise_base_level'], size=len(predictions))
            correlated_noise = correlation_decay * self.stochastic_parameters['last_noise_value'] + \
                              (1 - correlation_decay) * base_noise
            
            # Add seasonal component based on calendar info if available in X
            if 'Month_Sin' in X.columns and 'Day_Sin' in X.columns:
                seasonal_component = self.stochastic_parameters['seasonal_amplitude'] * \
                                   (X['Month_Sin'].values + X['Day_Sin'].values)  # Combine monthly and daily patterns
            else:
                seasonal_component = np.zeros(len(predictions))  # Fallback if no calendar features
            
            # Combine noise components and scale by prediction magnitude
            combined_noise = (correlated_noise + seasonal_component) * predictions
            predictions += combined_noise
            
            # Ensure non-negative values
            predictions = np.maximum(0, predictions)
            
            # Update tracking values for next prediction
            self.stochastic_parameters['last_prediction_time'] = current_time
            self.stochastic_parameters['last_noise_value'] = correlated_noise
        
        return predictions


def prepare_features(df):
    """
    Prepare features for the forecasting model with enhanced seasonality,
    stochastic behavior, and recent data emphasis
    
    Args:
        df (DataFrame): Input data
        
    Returns:
        DataFrame: Processed features for modeling
    """
    logger.info("Preparing features for RF model...")
    
    # Create copy to avoid modifying the original
    df_features = df.copy()
    
    # Convert Date to datetime if it's not already
    if 'Date' in df_features.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_features['Date']):
            df_features['Date'] = pd.to_datetime(df_features['Date'])
        
        # Add recency features - days since most recent date in dataset
        max_date = df_features['Date'].max()
        df_features['Days_Since_Last'] = (max_date - df_features['Date']).dt.days
        
        # Add extremely recent data flags (last 7 days)
        df_features['Last_7_Days'] = (df_features['Days_Since_Last'] <= 7).astype(int)
        df_features['Last_14_Days'] = (df_features['Days_Since_Last'] <= 14).astype(int)
        df_features['Last_30_Days'] = (df_features['Days_Since_Last'] <= 30).astype(int)
        
        # Add day of year and expand window features
        df_features['Day_Of_Year'] = df_features['Date'].dt.dayofyear
        
        # Add week number and month end/start indicators
        df_features['Week_Of_Year'] = df_features['Date'].dt.isocalendar().week
        df_features['Is_Month_Start'] = df_features['Date'].dt.is_month_start.astype(int)
        df_features['Is_Month_End'] = df_features['Date'].dt.is_month_end.astype(int)
        df_features['Is_Quarter_Start'] = df_features['Date'].dt.is_quarter_start.astype(int)
        df_features['Is_Quarter_End'] = df_features['Date'].dt.is_quarter_end.astype(int)
        
        # Add time windows for sequential patterns (last 90/180 days)
        df_features['Last_90_Days'] = (df_features['Days_Since_Last'] <= 90).astype(int)
        df_features['Last_180_Days'] = (df_features['Days_Since_Last'] <= 180).astype(int)
    
    # Make sure these columns are passed through to the final dataset
    # This is important for the recent data RF model
    
    # One-hot encode product
    product_dummies = pd.get_dummies(df_features['Item'], prefix='Product')
    df_features = pd.concat([df_features, product_dummies], axis=1)
    
    # One-hot encode weather
    weather_dummies = pd.get_dummies(df_features['Weather'], prefix='Weather')
    df_features = pd.concat([df_features, weather_dummies], axis=1)
    
    # Enhanced cyclical time encoding for multiple seasonality patterns
    # Daily seasonality (day of week) with harmonics for better cyclical capture
    df_features['Day_Sin'] = np.sin(df_features['Day_Of_Week'] * (2 * np.pi / 7))
    df_features['Day_Cos'] = np.cos(df_features['Day_Of_Week'] * (2 * np.pi / 7))
    # Add second harmonic for more nuanced daily patterns (captures twice-weekly patterns)
    df_features['Day_Sin_2'] = np.sin(df_features['Day_Of_Week'] * (4 * np.pi / 7))
    df_features['Day_Cos_2'] = np.cos(df_features['Day_Of_Week'] * (4 * np.pi / 7))
    
    # Weekly seasonality (week of month)
    df_features['Week_Of_Month'] = df_features['Date'].dt.day.apply(lambda d: (d - 1) // 7 + 1)
    df_features['Week_Sin'] = np.sin(df_features['Week_Of_Month'] * (2 * np.pi / 5))  # Assuming max 5 weeks in a month
    df_features['Week_Cos'] = np.cos(df_features['Week_Of_Month'] * (2 * np.pi / 5))
    
    # Monthly seasonality with harmonics
    df_features['Month_Sin'] = np.sin(df_features['Month'] * (2 * np.pi / 12))
    df_features['Month_Cos'] = np.cos(df_features['Month'] * (2 * np.pi / 12))
    # Add quarter harmonic (captures seasonal patterns every 3 months)
    df_features['Quarter_Sin'] = np.sin(df_features['Month'] * (2 * np.pi / 3))
    df_features['Quarter_Cos'] = np.cos(df_features['Month'] * (2 * np.pi / 3))
    
    # Yearly seasonality features
    df_features['Day_Of_Year'] = df_features['Date'].dt.dayofyear
    df_features['Year_Sin'] = np.sin(df_features['Day_Of_Year'] * (2 * np.pi / 366))  # Using 366 for leap years
    df_features['Year_Cos'] = np.cos(df_features['Day_Of_Year'] * (2 * np.pi / 366))
    # Add half-year harmonic
    df_features['Half_Year_Sin'] = np.sin(df_features['Day_Of_Year'] * (4 * np.pi / 366))
    df_features['Half_Year_Cos'] = np.cos(df_features['Day_Of_Year'] * (4 * np.pi / 366))
    
    # Weekend indicator
    df_features['Is_Weekend'] = df_features['Day_Of_Week'].apply(lambda d: 1 if d >= 5 else 0)  # 5=Sat, 6=Sun
    
    # Day type indicators (more granular than weekend/weekday)
    # Create day type dummies directly without string conversion
    # This prevents 'could not convert string to float' errors
    for day in range(7):
        df_features[f'Day_{day}'] = (df_features['Day_Of_Week'] == day).astype(int)
    
    # Special events indicator (major shopping seasons)
    df_features['Is_Special_Event'] = 0
    
    # Black Friday period (late Nov)
    black_friday_mask = (df_features['Month'] == 11) & (df_features['Day'] >= 20) & (df_features['Day'] <= 30)
    df_features.loc[black_friday_mask, 'Is_Special_Event'] = 1
    
    # Christmas shopping period (Dec 1-24)
    christmas_mask = (df_features['Month'] == 12) & (df_features['Day'] <= 24)
    df_features.loc[christmas_mask, 'Is_Special_Event'] = 1
    
    # Summer sales (July)
    summer_mask = (df_features['Month'] == 7)
    df_features.loc[summer_mask, 'Is_Special_Event'] = 1
    
    # Back to school (Aug 15-Sep 15)
    back_to_school_mask = ((df_features['Month'] == 8) & (df_features['Day'] >= 15)) | ((df_features['Month'] == 9) & (df_features['Day'] <= 15))
    df_features.loc[back_to_school_mask, 'Is_Special_Event'] = 1
    
    # Month-based events (more granular special periods)
    # Valentine's Day period
    valentine_mask = (df_features['Month'] == 2) & (df_features['Day'] >= 1) & (df_features['Day'] <= 14)
    df_features['Is_Valentine'] = 0
    df_features.loc[valentine_mask, 'Is_Valentine'] = 1
    
    # Independence Day period
    july4_mask = (df_features['Month'] == 7) & (df_features['Day'] >= 1) & (df_features['Day'] <= 7)
    df_features['Is_July4th'] = 0
    df_features.loc[july4_mask, 'Is_July4th'] = 1
    
    # Halloween period
    halloween_mask = (df_features['Month'] == 10) & (df_features['Day'] >= 15) & (df_features['Day'] <= 31)
    df_features['Is_Halloween'] = 0
    df_features.loc[halloween_mask, 'Is_Halloween'] = 1
    
    # Thanksgiving period (approximate)
    thanksgiving_mask = (df_features['Month'] == 11) & (df_features['Day'] >= 15) & (df_features['Day'] <= 30)
    df_features['Is_Thanksgiving'] = 0
    df_features.loc[thanksgiving_mask, 'Is_Thanksgiving'] = 1
    
    # Promotion indicator
    if 'Promotion' in df_features.columns:
        df_features['Has_Promotion'] = df_features['Promotion'].astype(int)
    else:
        # Create a default Has_Promotion column with all zeros if Promotion doesn't exist
        logger.info("'Promotion' column not found, creating default Has_Promotion with all zeros")
        df_features['Has_Promotion'] = 0
    
    # Stock level indicators
    if 'Stock_Level' in df_features.columns and 'Recent_Daily_Sales' in df_features.columns:
        # Convert stock to weeks of supply
        df_features['Stock_Weeks'] = df_features['Stock_Level'] / (df_features['Recent_Daily_Sales'] * 7)
        df_features['Stock_Weeks'] = df_features['Stock_Weeks'].replace([np.inf, -np.inf], 4)  # Cap at 4 weeks
        df_features['Stock_Weeks'] = df_features['Stock_Weeks'].fillna(2)  # Fill NaNs with middle value
        
        # Categorize stock levels
        df_features['Low_Stock'] = (df_features['Stock_Weeks'] < 1).astype(int)
        df_features['Med_Stock'] = ((df_features['Stock_Weeks'] >= 1) & (df_features['Stock_Weeks'] <= 3)).astype(int)
        df_features['High_Stock'] = (df_features['Stock_Weeks'] > 3).astype(int)
    
    # Price indicators
    if 'Price' in df_features.columns:
        # Calculate relative price compared to product average
        df_features['Avg_Product_Price'] = df_features.groupby('Item')['Price'].transform('mean')
        df_features['Price_Ratio'] = df_features['Price'] / df_features['Avg_Product_Price']
        df_features['Price_Ratio'] = df_features['Price_Ratio'].fillna(1.0)  # Fill NaNs with neutral value
        
        # Low/High price indicators
        df_features['Low_Price'] = (df_features['Price_Ratio'] < 0.9).astype(int)
        df_features['High_Price'] = (df_features['Price_Ratio'] > 1.1).astype(int)
        
    # Weather impact indicators
    df_features['Rain_Impact'] = ((df_features['Weather'] == 'Heavy Rain') | (df_features['Weather'] == 'Storm')).astype(int)
    df_features['Snow_Impact'] = (df_features['Weather'] == 'Snow').astype(int)
    
    # Recent sales trend indicators
    for col in df_features.columns:
        if col.startswith('Sales_Avg_') or col.startswith('Sales_Lag_'):
            df_features[col] = df_features[col].fillna(0)
    
    # If lagged features exist, create trend indicators
    if all(f'Sales_Lag_{lag}' in df_features.columns for lag in [1, 7]):
        # Recent trend (1-day vs 7-day average)
        df_features['Recent_Trend'] = df_features['Sales_Lag_1'] / (df_features['Sales_Lag_7'] + 1e-5)
        df_features['Recent_Trend'] = df_features['Recent_Trend'].replace([np.inf, -np.inf], 1.0)
        df_features['Recent_Trend'] = df_features['Recent_Trend'].fillna(1.0)
        
        # Trend indicators
        df_features['Up_Trend'] = (df_features['Recent_Trend'] > 1.1).astype(int)
        df_features['Down_Trend'] = (df_features['Recent_Trend'] < 0.9).astype(int)
    
    # Add randomness amplitude for controlled stochasticity
    df_features['Randomness_Amplitude'] = np.random.uniform(0.01, 0.03, size=len(df_features))
    
    logger.info(f"Feature preparation complete - {df_features.shape[1]} features created")
    return df_features


def train_random_forest_model(data, features=None, target='Sales'):
    """
    Train a Random Forest model with specified features
    
    Args:
        data (DataFrame): Training data
        features (list): List of feature columns
        target (str): Target column name
        
    Returns:
        tuple: (model, features list)
    """
    logger.info(f"Training Random Forest model on {len(data)} rows")
    
    # Prepare features if not provided
    if features is None:
        # Prepare all features
        df_features = prepare_features(data)
        
        # Remove non-feature columns
        exclude_cols = ['Date', 'Sales', 'Item', 'Store_Id', 'Product', 'Size', 
                        'Weather', 'Holiday_Name', 'Stock_Status', 'Stock_Level',
                        'Price', 'Units_Purchased', 'Profit', 'Retail_Revenue', 
                        'Purchase_Cost', 'Purchase_Retail_Value', 'Units_Sold', 'Cost']
        
        features = [col for col in df_features.columns if col not in exclude_cols]
        X = df_features[features]
        y = df_features[target]
    else:
        # Use provided features
        X = data[features]
        y = data[target]
    
    # Enhanced NaN handling with more sophisticated imputation strategies
    logger.info("Enhanced handling of NaN values in features...")
    nan_cols = X.columns[X.isna().any()].tolist()
    
    if nan_cols:
        logger.warning(f"Found NaN values in columns: {nan_cols}")
        logger.info("Applying advanced imputation strategies")
        
        # Group columns by type for appropriate imputation strategies
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns
        date_cols = X.select_dtypes(include=['datetime64']).columns
        
        # For each column with NaNs, apply appropriate imputation
        for col in nan_cols:
            # Check if column is completely NaN
            if X[col].isna().all():
                X.loc[:, col] = 0
                logger.info(f"Column {col} was all NaN - filled with zeros")
                continue
                
            # Use different strategies based on column type
            if col in numeric_cols:
                # For numeric columns, try more advanced strategies
                if 'Day_Of_Week' in X.columns and col not in ['Day_Of_Week']:
                    # For time-series data, impute with same day of week mean when possible
                    try:
                        # Group by day of week and impute
                        day_means = X.groupby('Day_Of_Week')[col].transform('mean')
                        # Fill remaining NaNs with column mean
                        X.loc[:, col] = X[col].fillna(day_means).fillna(X[col].mean())
                        logger.info(f"Imputed {col} with day-of-week means")
                    except Exception as e:
                        # Fallback to regular mean imputation
                        X.loc[:, col] = X[col].fillna(X[col].mean())
                        logger.info(f"Fallback imputation for {col}: {str(e)}")
                else:
                    # Use median for robustness against outliers instead of mean
                    X.loc[:, col] = X[col].fillna(X[col].median())
                    logger.info(f"Imputed {col} with column median")
            
            elif col in categorical_cols:
                # For categorical, use mode (most common value)
                X.loc[:, col] = X[col].fillna(X[col].mode()[0])
                logger.info(f"Imputed {col} with most frequent value")
                
            elif col in date_cols:
                # For date columns, use previous or next date
                X.loc[:, col] = X[col].fillna(method='ffill').fillna(method='bfill')
                logger.info(f"Imputed {col} with forward/backward fill")
                
            else:
                # Default fallback
                X.loc[:, col] = X[col].fillna(X[col].mean())
                logger.info(f"Imputed {col} with column mean (default strategy)")
    
    # Add outlier detection and handling for time series data
    logger.info("Detecting and handling outliers in time series...")
    
    # Only apply to numeric columns and only if we have enough data
    if len(X) > 30:
        for col in X.select_dtypes(include=['int64', 'float64']).columns:
            # Skip columns that are indicators or binary
            if X[col].nunique() <= 2 or col in ['Day_Of_Week', 'Month', 'Year', 'Day']:
                continue
                
            # Calculate rolling statistics if this is time series data
            if 'Date' in data.columns:
                try:
                    # Sort by date if possible
                    temp_data = pd.concat([data['Date'], X[col]], axis=1).sort_values('Date')
                    temp_data['rolling_median'] = temp_data[col].rolling(window=7, min_periods=3).median()
                    temp_data['rolling_std'] = temp_data[col].rolling(window=7, min_periods=3).std()
                    
                    # Identify outliers (3 std from rolling median)
                    temp_data['is_outlier'] = np.abs(temp_data[col] - temp_data['rolling_median']) > 3 * temp_data['rolling_std']
                    
                    # Replace outliers with rolling median
                    outlier_indices = temp_data['is_outlier'] & ~temp_data['rolling_median'].isna()
                    if outlier_indices.sum() > 0:
                        # Map back to original indices
                        original_indices = temp_data[outlier_indices].index
                        X.loc[original_indices, col] = temp_data.loc[outlier_indices, 'rolling_median']
                        logger.info(f"Replaced {outlier_indices.sum()} outliers in {col} with rolling medians")
                except Exception as e:
                    logger.warning(f"Could not apply time series outlier detection to {col}: {str(e)}")
    
    # Sample weights to emphasize recent data
    weights = np.ones(len(data))
    if 'Days_Since_Last' in X.columns:
        # Exponential decay weights - more recent data gets higher weight
        recency_weights = np.exp(-0.03 * X['Days_Since_Last'])  # 0.03 decay factor
        weights = weights * recency_weights
    
    # Initialize the ensemble components
    ensemble_models = {}
    
    # Train standard RF model
    logger.info("Training standard RF model...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
        oob_score=True  # Enable out-of-bag scoring for better model validation
    )
    rf_model.fit(X, y, sample_weight=weights)
    ensemble_models['rf_standard'] = rf_model
    logger.info(f"Standard RF model OOB score: {rf_model.oob_score_:.4f}")
    
    # Train specialized RF for recent data
    # First check if Last_30_Days exists in the data dataframe
    if 'Last_30_Days' in data.columns:
        logger.info("Training recent data RF model...")
        recent_mask = data['Last_30_Days'] == 1
        if recent_mask.sum() > 100:  # Only if we have enough recent data
            X_recent = X[recent_mask]
            y_recent = y[recent_mask]
            rf_recent = RandomForestRegressor(
                n_estimators=50,
                max_depth=10, 
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42,
                oob_score=True  # Enable out-of-bag scoring
            )
            rf_recent.fit(X_recent, y_recent)
            ensemble_models['rf_recent'] = rf_recent
            logger.info(f"Recent data RF model OOB score: {rf_recent.oob_score_:.4f}")
        else:
            ensemble_models['rf_recent'] = None
    
    # Train specialized RF for promotion data
    # First check if Has_Promotion exists in either dataframe
    has_promotion_column = 'Has_Promotion' in X.columns and 'Has_Promotion' in data.columns
    
    if has_promotion_column:
        logger.info("Training promotion RF model...")
        promo_mask = data['Has_Promotion'] == 1
        if promo_mask.sum() > 100:  # Only if we have enough promotion data
            X_promo = X[promo_mask]
            y_promo = y[promo_mask]
            rf_promo = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                min_samples_split=3,
                min_samples_leaf=2,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42,
                oob_score=True  # Enable out-of-bag scoring
            )
            rf_promo.fit(X_promo, y_promo)
            ensemble_models['rf_promo'] = rf_promo
            logger.info(f"Promotion RF model OOB score: {rf_promo.oob_score_:.4f}")
        else:
            ensemble_models['rf_promo'] = None
    else:
        logger.warning("'Has_Promotion' column not found, skipping promotion-specific model")
        
    # Add time series specific model when appropriate
    if 'Date' in data.columns and len(data) > 100:
        logger.info("Training time series specific RF model...")
        
        # Use the same X, y but with additional time-based sampling
        # Sort data by date if it's a DataFrame with Date column
        if isinstance(data, pd.DataFrame) and 'Date' in data.columns:
            sorted_indices = data.sort_values('Date').index
            X_ts = X.loc[sorted_indices].copy()
            y_ts = y.loc[sorted_indices].copy()
            
            # Enhanced time-based feature importance
            time_weights = np.linspace(0.5, 1.0, len(y_ts))  # Gradually increasing weights over time
            
            # Build time-series focused random forest with temporal awareness
            rf_ts = RandomForestRegressor(
                n_estimators=75,
                max_depth=12,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                n_jobs=-1,
                random_state=42
            )
            rf_ts.fit(X_ts, y_ts, sample_weight=time_weights)
            ensemble_models['rf_timeseries'] = rf_ts
            logger.info(f"Time series RF model OOB score: {rf_ts.oob_score_:.4f}")
        else:
            ensemble_models['rf_timeseries'] = None
    
    # Define ensemble weights based on model performance if available
    ensemble_weights = {
        'rf_standard': 0.6,
        'rf_recent': 0.2 if 'rf_recent' in ensemble_models and ensemble_models['rf_recent'] is not None else 0,
        'rf_promo': 0.1 if 'rf_promo' in ensemble_models and ensemble_models['rf_promo'] is not None else 0,
        'rf_timeseries': 0.1 if 'rf_timeseries' in ensemble_models and ensemble_models['rf_timeseries'] is not None else 0
    }
    
    # Dynamically adjust weights based on OOB scores if available
    oob_scores = {}
    for name, model in ensemble_models.items():
        if model is not None and hasattr(model, 'oob_score_'):
            oob_scores[name] = max(0.01, model.oob_score_)  # Ensure non-zero weight
    
    # If we have OOB scores for all models, use them to determine weights
    if len(oob_scores) >= 2:  # Only adjust if we have multiple models with OOB scores
        # Normalize OOB scores to weights
        total_oob = sum(oob_scores.values())
        if total_oob > 0:
            for name in oob_scores:
                ensemble_weights[name] = oob_scores[name] / total_oob
            logger.info(f"Adjusted ensemble weights based on OOB scores: {ensemble_weights}")
    
    # Normalize weights for any missing models
    weight_sum = sum(ensemble_weights.values())
    if weight_sum > 0:
        for k in ensemble_weights:
            ensemble_weights[k] /= weight_sum
    
    # Create ensemble model
    ensemble = EnsembleModel(ensemble_models, ensemble_weights)
    
    logger.info("Random Forest model training complete")
    return ensemble, features


def create_forecasts(model, features, forecast_df, days_to_forecast=30, add_noise=True, use_bootstrap=True):
    """
    Create forecasts for future dates with improved uncertainty estimation
    
    Args:
        model: Trained model
        features (list): Feature column names
        forecast_df (DataFrame): Base data for forecasting
        days_to_forecast (int): Number of days to forecast
        add_noise (bool): Whether to add stochastic noise to forecasts
        use_bootstrap (bool): Whether to use bootstrap for better confidence intervals
        
    Returns:
        DataFrame: Forecast results
    """
    logger.info(f"Creating forecasts for {days_to_forecast} days with {'bootstrapping' if use_bootstrap else 'standard'} uncertainty estimation")
    
    # Generate future dates
    start_date = forecast_df['Date'].max() + timedelta(days=1)
    future_dates = pd.date_range(start=start_date, periods=days_to_forecast)
    
    # Get unique product/store combinations
    products = forecast_df[['Store_Id', 'Item', 'Product', 'Size']].drop_duplicates()
    
    # Create future dataframe with all products and dates
    future_rows = []
    for _, product in products.iterrows():
        for date in future_dates:
            future_row = {
                'Date': date,
                'Store_Id': product['Store_Id'],
                'Item': product['Item'],
                'Product': product['Product'],
                'Size': product['Size'],
                'Day_Of_Week': date.dayofweek,
                'Month': date.month,
                'Year': date.year,
                'Day': date.day
            }
            future_rows.append(future_row)
    
    future_df = pd.DataFrame(future_rows)
    
    # Get the most recent values for each product
    latest_values = forecast_df.sort_values('Date').groupby(['Store_Id', 'Item']).last().reset_index()
    
    # Merge recent values with future dataframe for continuity
    value_columns = ['Price', 'Stock_Level', 'Recent_Daily_Sales', 'Weather']
    for col in value_columns:
        if col in latest_values.columns:
            future_df = pd.merge(
                future_df,
                latest_values[['Store_Id', 'Item', col]],
                on=['Store_Id', 'Item'],
                how='left'
            )
    
    # Add lag features from the most recent actual data
    lag_features = [col for col in latest_values.columns if col.startswith('Sales_Lag_') or col.startswith('Sales_Avg_')]
    for col in lag_features:
        if col in latest_values.columns:
            future_df = pd.merge(
                future_df,
                latest_values[['Store_Id', 'Item', col]],
                on=['Store_Id', 'Item'],
                how='left'
            )
    
    # Add exponentially weighted moving averages for trend detection
    if 'Sales' in latest_values.columns:
        try:
            # Group by product/store and calculate EWMAs with different decay rates
            ewma_df = pd.DataFrame()
            for (store, item), group in forecast_df.groupby(['Store_Id', 'Item']):
                if len(group) > 5:  # Need minimum data points
                    group = group.sort_values('Date')
                    
                    # Calculate exponential moving averages with different spans
                    group_ewma = pd.DataFrame({
                        'Store_Id': store,
                        'Item': item,
                        'Sales_EWMA_7': group['Sales'].ewm(span=7, min_periods=1).mean().values[-1],
                        'Sales_EWMA_14': group['Sales'].ewm(span=14, min_periods=1).mean().values[-1],
                        'Sales_EWMA_30': group['Sales'].ewm(span=30, min_periods=1).mean().values[-1]
                    }, index=[0])
                    ewma_df = pd.concat([ewma_df, group_ewma], ignore_index=True)
            
            if not ewma_df.empty:
                # Merge EWMAs into future_df
                future_df = pd.merge(
                    future_df,
                    ewma_df,
                    on=['Store_Id', 'Item'],
                    how='left'
                )
                logger.info("Added exponential weighted moving averages for trend detection")
        except Exception as e:
            logger.warning(f"Could not add EWMA features: {str(e)}")
    
    # Prepare features for the future data
    future_features = prepare_features(future_df)
    
    # Ensure all required features are present
    for feature in features:
        if feature not in future_features.columns:
            future_features[feature] = 0
    
    # Check for NaN values in prediction features and handle them
    nan_prediction_cols = future_features[features].columns[future_features[features].isna().any()].tolist()
    if nan_prediction_cols:
        logger.warning(f"Found NaN values in prediction features: {nan_prediction_cols}")
        logger.info("Imputing NaN values for prediction")
        
        # Apply similar imputation logic as in training
        numeric_cols = future_features.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = future_features.select_dtypes(include=['object', 'category', 'bool']).columns
        date_cols = future_features.select_dtypes(include=['datetime64']).columns
        
        for col in nan_prediction_cols:
            if col in numeric_cols:
                if future_features[col].isna().all():
                    future_features.loc[:, col] = 0
                elif 'Day_Of_Week' in future_features.columns and col not in ['Day_Of_Week']:
                    # Try day-of-week based imputation
                    try:
                        day_means = future_features.groupby('Day_Of_Week')[col].transform('mean')
                        future_features.loc[:, col] = future_features[col].fillna(day_means).fillna(future_features[col].median())
                    except:
                        future_features.loc[:, col] = future_features[col].fillna(future_features[col].median())
                else:
                    future_features.loc[:, col] = future_features[col].fillna(future_features[col].median())
            elif col in categorical_cols:
                future_features.loc[:, col] = future_features[col].fillna(future_features[col].mode()[0] if not future_features[col].mode().empty else 0)
            elif col in date_cols:
                future_features.loc[:, col] = future_features[col].fillna(method='ffill').fillna(method='bfill')
            else:
                future_features.loc[:, col] = future_features[col].fillna(0)
    
    # Make predictions
    future_features['Forecast'] = model.predict(future_features[features], add_noise=add_noise)
    
    # Create output dataframe with essential columns
    forecast_output = future_features[['Date', 'Store_Id', 'Item', 'Forecast']].copy()
    forecast_output['Date'] = pd.to_datetime(forecast_output['Date'])
    
    # Add product names and other helpful columns
    forecast_output = pd.merge(
        forecast_output,
        products[['Store_Id', 'Item', 'Product', 'Size']],
        on=['Store_Id', 'Item'],
        how='left'
    )
    
    # Add additional info
    forecast_output['Forecast_Generated'] = datetime.now()
    forecast_output['Days_In_Future'] = (forecast_output['Date'] - start_date).dt.days + 1
    
    # Add improved confidence intervals with bootstrap or advanced approximation
    if use_bootstrap and hasattr(model, 'models') and 'rf_standard' in model.models:
        logger.info("Using bootstrap for uncertainty estimation")
        # Use bootstrap predictions from individual trees in the forest
        try:
            # Get the main random forest model
            rf = model.models['rf_standard']
            
            # Generate predictions from each tree in the forest
            bootstrap_preds = np.array([tree.predict(future_features[features]) for tree in rf.estimators_])
            
            # Calculate statistics from bootstrap predictions
            forecast_output['Std_Dev'] = np.std(bootstrap_preds, axis=0)
            
            # Scale std dev based on days in future (uncertainty increases over time)
            time_scaling = 1.0 + 0.02 * forecast_output['Days_In_Future']  # 2% increase per day
            forecast_output['Std_Dev'] = forecast_output['Std_Dev'] * time_scaling
            
            # Calculate percentile-based bounds (more robust than normal approximation)
            lower_percentile = np.percentile(bootstrap_preds, 2.5, axis=0)
            upper_percentile = np.percentile(bootstrap_preds, 97.5, axis=0)
            
            forecast_output['Lower_Bound'] = np.maximum(0, lower_percentile)
            forecast_output['Upper_Bound'] = upper_percentile
            
            logger.info("Successfully created bootstrap-based confidence intervals")
        except Exception as e:
            logger.warning(f"Bootstrap confidence interval generation failed: {e}")
            # Fall back to standard approximation
            forecast_output['Std_Dev'] = forecast_output['Forecast'] * (0.1 + 0.005 * forecast_output['Days_In_Future'])
            forecast_output['Lower_Bound'] = np.maximum(0, forecast_output['Forecast'] - 1.96 * forecast_output['Std_Dev'])
            forecast_output['Upper_Bound'] = forecast_output['Forecast'] + 1.96 * forecast_output['Std_Dev']
    else:
        # Use enhanced approximation with time-dependent scaling
        logger.info("Using enhanced approximation for uncertainty estimation")
        base_uncertainty = 0.08  # 8% base uncertainty
        daily_increase = 0.004   # 0.4% increase per day
        max_uncertainty = 0.40   # Cap at 40%
        
        # Calculate increasing uncertainty over forecast horizon
        uncertainty = np.minimum(base_uncertainty + daily_increase * forecast_output['Days_In_Future'], max_uncertainty)
        forecast_output['Std_Dev'] = forecast_output['Forecast'] * uncertainty
        
        # More conservative bounds at 95% confidence interval
        forecast_output['Lower_Bound'] = np.maximum(0, forecast_output['Forecast'] - 1.96 * forecast_output['Std_Dev'])
        forecast_output['Upper_Bound'] = forecast_output['Forecast'] + 1.96 * forecast_output['Std_Dev']
    
    logger.info(f"Created forecasts for {len(forecast_output)} product-days")
    return forecast_output


def save_model(model, features, model_file=RF_MODEL_FILE, feature_file=RF_MODEL_FEATURES_FILE):
    """
    Save the model and features to disk
    
    Args:
        model: Trained model
        features (list): Feature columns
        model_file (str): Path to save the model
        feature_file (str): Path to save the features
    """
    logger.info(f"Saving model to {model_file}")
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        
        # Save model
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
            
        # Save features
        with open(feature_file, 'wb') as f:
            pickle.dump(features, f)
            
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {e}")


def load_model(model_file=RF_MODEL_FILE, feature_file=RF_MODEL_FEATURES_FILE):
    """
    Load the model and features from disk
    
    Args:
        model_file (str): Path to the model file
        feature_file (str): Path to the features file
        
    Returns:
        tuple: (model, features)
    """
    logger.info(f"Loading model from {model_file}")
    try:
        # Load model
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
            
        # Load features
        with open(feature_file, 'rb') as f:
            features = pickle.load(f)
            
        logger.info("Model loaded successfully")
        return model, features
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None


def save_forecasts(forecasts, output_file=RF_FORECASTS_FILE):
    """
    Save forecasts to CSV
    
    Args:
        forecasts (DataFrame): Forecast data
        output_file (str): Path to save the CSV
    """
    logger.info(f"Saving forecasts to {output_file}")
    try:
        forecasts.to_csv(output_file, index=False)
        logger.info(f"Saved forecasts to {output_file}")
    except Exception as e:
        logger.error(f"Error saving forecasts: {e}")


def plot_feature_importance(model, features, output_file=None):
    """
    Plot feature importance
    
    Args:
        model: Trained model
        features (list): Feature names
        output_file (str): Path to save the plot
    """
    logger.info("Plotting feature importance")
    try:
        # Get feature importance from the model
        importances = model.feature_importances_
        
        # Sort features by importance
        sorted_idx = np.argsort(importances)
        
        # Plot the top 20 features
        plt.figure(figsize=(10, 12))
        plt.barh(np.array(features)[sorted_idx][-20:], importances[sorted_idx][-20:])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Features by Importance')
        
        # Save or show
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {output_file}")
        else:
            plt.show()
        
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting feature importance: {e}")


def plot_forecast(forecasts, history=None, store_id=None, item=None, output_file=None):
    """
    Plot forecast with optional history
    
    Args:
        forecasts (DataFrame): Forecast data
        history (DataFrame): Historical data (optional)
        store_id: Store ID to filter (optional)
        item: Item ID to filter (optional)
        output_file (str): Path to save the plot
    """
    logger.info("Creating forecast plot")
    try:
        plt.figure(figsize=(12, 6))
        
        # Filter data if requested
        plot_forecasts = forecasts.copy()
        plot_history = None if history is None else history.copy()
        
        if store_id is not None:
            plot_forecasts = plot_forecasts[plot_forecasts['Store_Id'] == store_id]
            if plot_history is not None:
                plot_history = plot_history[plot_history['Store_Id'] == store_id]
        
        if item is not None:
            plot_forecasts = plot_forecasts[plot_forecasts['Item'] == item]
            if plot_history is not None:
                plot_history = plot_history[plot_history['Item'] == item]
        
        # Plot history if available
        if plot_history is not None:
            # Group by date if multiple products
            if len(plot_history['Item'].unique()) > 1:
                history_agg = plot_history.groupby('Date')['Sales'].sum().reset_index()
                plt.plot(history_agg['Date'], history_agg['Sales'], 'b-', label='Historical Sales')
            else:
                plt.plot(plot_history['Date'], plot_history['Sales'], 'b-', label='Historical Sales')
        
        # Plot forecasts
        # Group by date if multiple products
        if len(plot_forecasts['Item'].unique()) > 1:
            forecast_agg = plot_forecasts.groupby('Date').agg({
                'Forecast': 'sum',
                'Lower_Bound': 'sum',
                'Upper_Bound': 'sum'
            }).reset_index()
            
            plt.plot(forecast_agg['Date'], forecast_agg['Forecast'], 'r-', label='Forecast')
            plt.fill_between(
                forecast_agg['Date'],
                forecast_agg['Lower_Bound'],
                forecast_agg['Upper_Bound'],
                color='r', alpha=0.2, label='95% Confidence Interval'
            )
        else:
            plt.plot(plot_forecasts['Date'], plot_forecasts['Forecast'], 'r-', label='Forecast')
            plt.fill_between(
                plot_forecasts['Date'],
                plot_forecasts['Lower_Bound'],
                plot_forecasts['Upper_Bound'],
                color='r', alpha=0.2, label='95% Confidence Interval'
            )
        
        # Add labels and legend
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title('Sales Forecast' + 
                 (f' for Store {store_id}' if store_id else '') + 
                 (f' Item {item}' if item else ''))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            logger.info(f"Saved forecast plot to {output_file}")
        else:
            plt.show()
        
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting forecast: {e}")


def run_rf_forecasting(data_file=COMBINED_DATA_FILE, use_existing=False, days_to_forecast=30):
    """
    Main function to run the RF forecasting process
    
    Args:
        data_file (str): Path to the data file
        use_existing (bool): Whether to use an existing model
        days_to_forecast (int): Number of days to forecast
        
    Returns:
        DataFrame: Forecast results
    """
    logger.info(f"Starting RF forecasting process with {'existing' if use_existing else 'new'} model")
    
    try:
        # Load data
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        logger.info(f"Loaded data with {len(df)} rows from {data_file}")
        
        # Get or train model
        if use_existing and os.path.exists(RF_MODEL_FILE) and os.path.exists(RF_MODEL_FEATURES_FILE):
            logger.info("Loading existing model...")
            model, features = load_model()
            if model is None:
                logger.warning("Failed to load existing model, training new model instead")
                model, features = train_random_forest_model(df)
                save_model(model, features)
        else:
            logger.info("Training new model...")
            model, features = train_random_forest_model(df)
            save_model(model, features)
        
        # Create forecasts
        forecasts = create_forecasts(model, features, df, days_to_forecast=days_to_forecast)
        
        # Save forecasts
        save_forecasts(forecasts)
        
        # Create plots
        os.makedirs(os.path.join(STATIC_DIR, 'images'), exist_ok=True)
        plot_feature_importance(model, features, os.path.join(STATIC_DIR, 'images', 'rf_feature_importance.png'))
        plot_forecast(forecasts, df, output_file=os.path.join(STATIC_DIR, 'images', 'rf_demand_forecast.png'))
        
        return forecasts
    
    except Exception as e:
        logger.error(f"Error in RF forecasting process: {e}", exc_info=True)
        raise


def main():
    """
    Main function to run when script is called directly
    """
    parser = argparse.ArgumentParser(description='Random Forest model training and forecasting')
    parser.add_argument('--use-existing', action='store_true', help='Use existing model instead of retraining')
    parser.add_argument('--days', type=int, default=30, help='Number of days to forecast')
    parser.add_argument('--data-file', type=str, default=COMBINED_DATA_FILE, help='Path to input data file')
    args = parser.parse_args()
    
    run_rf_forecasting(
        data_file=args.data_file,
        use_existing=args.use_existing,
        days_to_forecast=args.days
    )
    
    logger.info("RF forecasting completed successfully")


if __name__ == "__main__":
    main()