import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from datetime import datetime, timedelta

# Define ensemble predict function outside class for pickling support
def ensemble_predict(models, ensemble_weights, X_input, add_noise=False):
    """Weighted ensemble prediction with optional noise component"""
    predictions = {}
    for name, model in models.items():
        if model is not None:  # Check if model exists
            predictions[name] = model.predict(X_input)
    
    # Weighted average of predictions
    weighted_pred = np.zeros(X_input.shape[0])
    for name, preds in predictions.items():
        weighted_pred += preds * ensemble_weights[name]
        
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
    def __init__(self, models, weights):
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
    """Prepare features for the forecasting model with enhanced seasonality, stochastic behavior, and recent data emphasis"""
    
    # Create copy to avoid modifying the original
    df_features = df.copy()
    
    # Convert Date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df_features['Date']):
        df_features['Date'] = pd.to_datetime(df_features['Date'])
    
    # Add recency features - days since most recent date in dataset
    max_date = df_features['Date'].max()
    df_features['Days_Since_Last'] = (max_date - df_features['Date']).dt.days
    
    # Add extremely recent data flags (last 7 days)
    df_features['Last_7_Days'] = (df_features['Days_Since_Last'] <= 7).astype(int)
    df_features['Last_14_Days'] = (df_features['Days_Since_Last'] <= 14).astype(int)
    df_features['Last_30_Days'] = (df_features['Days_Since_Last'] <= 30).astype(int)
    
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
    df_features.loc[valentine_mask, 'Is_Valentine'] = 1
    df_features['Is_Valentine'] = df_features['Is_Valentine'].fillna(0)
    
    # Independence Day period
    july4_mask = (df_features['Month'] == 7) & (df_features['Day'] >= 1) & (df_features['Day'] <= 7)
    df_features.loc[july4_mask, 'Is_July4th'] = 1
    df_features['Is_July4th'] = df_features['Is_July4th'].fillna(0)
    
    # Halloween period
    halloween_mask = (df_features['Month'] == 10) & (df_features['Day'] >= 15) & (df_features['Day'] <= 31)
    df_features.loc[halloween_mask, 'Is_Halloween'] = 1
    df_features['Is_Halloween'] = df_features['Is_Halloween'].fillna(0)
    
    # Add volatility features (rolling standard deviation)
    sales_series = df_features.groupby(['Store_Id', 'Item'])['Sales'].transform(lambda x: x.rolling(window=7, min_periods=1).std())
    df_features['Rolling_Std_7'] = sales_series.fillna(0)
    
    sales_series = df_features.groupby(['Store_Id', 'Item'])['Sales'].transform(lambda x: x.rolling(window=28, min_periods=1).std())
    df_features['Rolling_Std_28'] = sales_series.fillna(0)
    
    # Add coefficient of variation (normalized volatility)
    sales_series_mean = df_features.groupby(['Store_Id', 'Item'])['Sales'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df_features['CV_7'] = (df_features['Rolling_Std_7'] / sales_series_mean.replace(0, np.nan)).fillna(0)
    
    sales_series_mean = df_features.groupby(['Store_Id', 'Item'])['Sales'].transform(lambda x: x.rolling(window=28, min_periods=1).mean())
    df_features['CV_28'] = (df_features['Rolling_Std_28'] / sales_series_mean.replace(0, np.nan)).fillna(0)
    
    # Add historical randomness features for stochastic behavior
    # Calculate historical day-of-week random component
    df_features['DOW_Random'] = 0.0
    for store_id, item_id in df_features.groupby(['Store_Id', 'Item']).groups:
        for dow in range(7):
            # Get mean and std for this day of week
            dow_data = df_features[(df_features['Store_Id'] == store_id) & 
                                 (df_features['Item'] == item_id) & 
                                 (df_features['Day_Of_Week'] == dow)]
            if len(dow_data) >= 3:  # Ensure enough data points
                dow_mean = dow_data['Sales'].mean()
                dow_std = max(1.0, dow_data['Sales'].std())  # Minimum std of 1 to avoid division by zero
                
                # Calculate normalized random component for this day of week
                dow_indices = dow_data.index
                df_features.loc[dow_indices, 'DOW_Random'] = (dow_data['Sales'] - dow_mean) / dow_std
    
    # Calculate momentum features (rate of change)
    for store_id, item_group in df_features.groupby(['Store_Id', 'Item']):
        sorted_group = item_group.sort_values('Date')
        
        # Calculate short-term and medium-term momentum
        df_features.loc[(df_features['Store_Id'] == store_id[0]) & 
                       (df_features['Item'] == store_id[1]), 'Momentum_7d'] = \
            sorted_group['Sales'].rolling(window=7).mean().pct_change(periods=7).fillna(0)
            
        df_features.loc[(df_features['Store_Id'] == store_id[0]) & 
                       (df_features['Item'] == store_id[1]), 'Momentum_28d'] = \
            sorted_group['Sales'].rolling(window=28).mean().pct_change(periods=28).fillna(0)
    
    # Add lag features specifically focused on very recent history
    # Group by store and product
    for store_id, item_group in df_features.groupby(['Store_Id', 'Item']):
        # Sort by date within each group
        sorted_group = item_group.sort_values('Date')
        
        # Create lag features for recent days (1, 2, 3, 7, 14, 28 days ago)
        for lag in [1, 2, 3, 7, 14, 28]:
            col_name = f'Sales_Lag_{lag}'
            df_features.loc[(df_features['Store_Id'] == store_id[0]) & 
                           (df_features['Item'] == store_id[1]), col_name] = \
                sorted_group['Sales'].shift(lag)
        
        # Create lag features for same day of week (1, 2, 3, 4 weeks ago)
        for lag in [1, 2, 3, 4]:
            col_name = f'Sales_Lag_DOW_{lag}w'
            df_features.loc[(df_features['Store_Id'] == store_id[0]) & 
                           (df_features['Item'] == store_id[1]), col_name] = \
                sorted_group['Sales'].shift(7 * lag)
        
        # Create exponentially weighted moving averages with different spans
        for span in [7, 14, 28]:
            col_name = f'Sales_EWMA_{span}'
            ewma = sorted_group['Sales'].ewm(span=span, adjust=False).mean()
            df_features.loc[(df_features['Store_Id'] == store_id[0]) & 
                           (df_features['Item'] == store_id[1]), col_name] = ewma
        
        # Create heavily weighted recent average (70% yesterday, 30% rest of week)
        df_features.loc[(df_features['Store_Id'] == store_id[0]) & 
                       (df_features['Item'] == store_id[1]), 'Sales_Recent_Weighted'] = \
            sorted_group['Sales'].shift(1) * 0.7 + \
            (sorted_group['Sales'].shift(2) + sorted_group['Sales'].shift(3) + \
             sorted_group['Sales'].shift(4) + sorted_group['Sales'].shift(5) + \
             sorted_group['Sales'].shift(6) + sorted_group['Sales'].shift(7)) * 0.05
             
        # Calculate differenced series for stationarity (day-to-day changes)
        df_features.loc[(df_features['Store_Id'] == store_id[0]) & 
                       (df_features['Item'] == store_id[1]), 'Sales_Diff_1d'] = \
            sorted_group['Sales'].diff().fillna(0)
            
        # Calculate week-over-week differenced series
        df_features.loc[(df_features['Store_Id'] == store_id[0]) & 
                       (df_features['Item'] == store_id[1]), 'Sales_Diff_7d'] = \
            (sorted_group['Sales'] - sorted_group['Sales'].shift(7)).fillna(0)
    
    # Fill NaN values for all calculated features
    lag_cols = [f'Sales_Lag_{lag}' for lag in [1, 2, 3, 7, 14, 28]]
    dow_lag_cols = [f'Sales_Lag_DOW_{lag}w' for lag in [1, 2, 3, 4]]
    ewma_cols = [f'Sales_EWMA_{span}' for span in [7, 14, 28]]
    diff_cols = ['Sales_Diff_1d', 'Sales_Diff_7d']
    momentum_cols = ['Momentum_7d', 'Momentum_28d']
    
    for col in lag_cols + dow_lag_cols + ewma_cols + diff_cols + momentum_cols + ['Sales_Recent_Weighted']:
        df_features[col] = df_features[col].fillna(0)
    
    # Add interaction terms between important features for non-linear patterns
    df_features['Price_Promotion_Interaction'] = df_features['Price'] * df_features['Promotion']
    df_features['Promo_Weekend_Interaction'] = df_features['Promotion'] * df_features['Is_Weekend']
    df_features['Price_Weekend_Interaction'] = df_features['Price'] * df_features['Is_Weekend']
    
    # Add stock-related interactions
    df_features['Stock_Price_Ratio'] = (df_features['Stock_Level'] / df_features['Price']).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Randomness amplitude based on historical patterns (for stochastic behavior)
    for store_id, item_group in df_features.groupby(['Store_Id', 'Item']):
        # Calculate the historical coefficient of variation (standard deviation / mean)
        if len(item_group) > 0:
            cv = item_group['Sales'].std() / item_group['Sales'].mean() if item_group['Sales'].mean() > 0 else 0
            # Apply a small amount of noise proportional to the historical variation
            random_component = np.random.normal(loc=1.0, scale=max(0.01, min(cv/10, 0.1)), size=len(item_group))
            # Store the randomness amplitude for use in prediction
            df_features.loc[(df_features['Store_Id'] == store_id[0]) & 
                           (df_features['Item'] == store_id[1]), 'Randomness_Amplitude'] = random_component
    
    # Enhanced features to use in the model with recent data emphasis and stochastic components
    feature_cols = [
        'Price', 'Promotion',
        # Daily seasonality
        'Day_Sin', 'Day_Cos', 'Day_Sin_2', 'Day_Cos_2',
        # Weekly seasonality
        'Week_Sin', 'Week_Cos',
        # Monthly and quarterly seasonality
        'Month_Sin', 'Month_Cos', 'Quarter_Sin', 'Quarter_Cos',
        # Yearly seasonality
        'Year_Sin', 'Year_Cos', 'Half_Year_Sin', 'Half_Year_Cos',
        # Special events and holidays
        'Is_Holiday', 'Is_Weekend', 'Is_Special_Event', 
        'Is_Valentine', 'Is_July4th', 'Is_Halloween',
        # Volatility indicators
        'Rolling_Std_7', 'Rolling_Std_28', 'CV_7', 'CV_28',
        # Stock related features
        'Stock_Level', 'Weeks_Of_Stock', 'Stock_Price_Ratio',
        # Stochastic components
        'DOW_Random', 'Randomness_Amplitude',
        # Momentum features
        'Momentum_7d', 'Momentum_28d',
        # Interaction terms
        'Price_Promotion_Interaction', 'Promo_Weekend_Interaction', 'Price_Weekend_Interaction',
        # Recency features - these will help heavily weight recent data
        'Days_Since_Last', 'Last_7_Days', 'Last_14_Days', 'Last_30_Days'
    ]
    
    # Add product and weather dummy columns
    feature_cols.extend([col for col in df_features.columns if col.startswith('Product_')])
    feature_cols.extend([col for col in df_features.columns if col.startswith('Weather_')])
    # Add day type dummies (for 7 days of the week)
    feature_cols.extend([f'Day_{i}' for i in range(7)])
    
    # Add all lag features to feature columns
    feature_cols.extend(lag_cols)
    feature_cols.extend(dow_lag_cols)
    feature_cols.extend(ewma_cols)
    feature_cols.extend(diff_cols)
    feature_cols.extend(['Sales_Recent_Weighted'])
    
    return df_features, feature_cols

def train_forecast_model(df):
    """Train a forecasting model using ensemble techniques for improved accuracy and stochastic behavior"""
    
    # Prepare features with enhanced seasonality and stochastic components
    df_features, feature_cols = prepare_features(df)
    
    # Define target
    target = 'Sales'
    
    # Filter out any rows with NaN in feature columns
    df_features = df_features.dropna(subset=feature_cols + [target])
    
    # Split into features and target
    X = df_features[feature_cols]
    y = df_features[target]
    
    # Create sample weights that heavily favor recent data
    # The more recent the data, the higher the weight
    # This creates near-overfitting on recent data as requested
    max_date = df_features['Date'].max()
    days_since_last = (max_date - df_features['Date']).dt.days
    
    # Create exponential decay weights with very strong recency bias
    # Formula: 0.85^days_since ensures very recent days have much higher weight
    # For example: today=1.0, yesterday=0.85, week ago=0.32, month ago=0.008
    sample_weights = 0.85 ** days_since_last
    
    # Boost the most recent 7 days by 10x - extreme emphasis on recent data
    sample_weights[days_since_last <= 7] *= 10.0
    
    # Ensure no 0 values in predictions by giving even more weight to the last value
    # This ensures the model will heavily prioritize matching the most recent value
    sample_weights[days_since_last == 0] *= 5.0  # 50x weight for the most recent day
    
    # Use time-based split instead of random split
    # This ensures recent data is in the test set
    cutoff_date = max_date - pd.Timedelta(days=14)  # Use last 2 weeks as test
    train_mask = df_features['Date'] < cutoff_date
    test_mask = df_features['Date'] >= cutoff_date
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    train_weights = sample_weights[train_mask]
    
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    # Create a heterogeneous ensemble of models for improved accuracy and stochastic behavior
    print("Training ensemble of Random Forest models for improved accuracy and stochastic behavior...")
    
    # Model 1: Standard Random Forest with extreme recency bias
    print("Training base Random Forest model...")
    rf_standard = RandomForestRegressor(
        n_estimators=300,  # More trees for better predictions
        max_depth=25,     # Deeper trees to enable near-memorization of recent data
        min_samples_split=2, # Allow smaller splits to memorize recent patterns
        min_samples_leaf=1,  # Allow leaf nodes with single samples for near-overfitting
        max_features='sqrt',  # Better feature selection
        bootstrap=True,    # Use bootstrap samples
        random_state=42,
        n_jobs=-1,
        oob_score=True,    # Use out-of-bag samples to estimate model performance
        warm_start=True    # Allow further fitting if needed
    )
    
    # Fit with sample weights to heavily favor recent data
    rf_standard.fit(X_train, y_train, sample_weight=train_weights)
    
    # Model 2: Random Forest with deeper trees and different feature subset for diversity
    print("Training deep Random Forest model with alternative feature selection...")
    rf_deep = RandomForestRegressor(
        n_estimators=250,
        max_depth=30,    # Even deeper trees
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.7,  # Different feature selection strategy
        bootstrap=True,
        random_state=101, # Different random seed for diversity
        n_jobs=-1
    )
    rf_deep.fit(X_train, y_train, sample_weight=train_weights)
    
    # Model 3: Random Forest with more randomness for stochastic behavior
    print("Training diverse Random Forest model with increased randomness...")
    rf_random = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=3,  # Slightly less overfitting
        min_samples_leaf=2,   # Slightly less overfitting
        max_features=0.5,     # Even more randomness in feature selection
        bootstrap=True,
        random_state=202,     # Different random seed
        n_jobs=-1
    )
    rf_random.fit(X_train, y_train, sample_weight=train_weights * 0.8)  # Less extreme weights
    
    # Model 4: Random Forest focused on very recent data only
    print("Training specialized Random Forest for very recent data patterns...")
    # Only use data from last 90 days for this model
    recent_cutoff = max_date - pd.Timedelta(days=90)
    recent_mask = (df_features['Date'] >= recent_cutoff) & train_mask
    X_recent = X[recent_mask]
    y_recent = y[recent_mask]
    weights_recent = sample_weights[recent_mask]
    
    if len(X_recent) > 50:  # Only if we have enough recent data
        rf_recent = RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=303,
            n_jobs=-1
        )
        rf_recent.fit(X_recent, y_recent, sample_weight=weights_recent)
    else:
        rf_recent = None
        print("Not enough recent data for specialized model")
    
    # Create a dictionary of models in the ensemble
    models = {
        'rf_standard': rf_standard,
        'rf_deep': rf_deep,
        'rf_random': rf_random
    }
    if rf_recent is not None:
        models['rf_recent'] = rf_recent
    
    # Set ensemble weights that favor the standard model but include others for stochasticity
    ensemble_weights = {
        'rf_standard': 0.6,   # Primary model gets highest weight
        'rf_deep': 0.2,       # Deep model helps with complex patterns
        'rf_random': 0.1,     # Random model adds stochasticity
        'rf_recent': 0.1 if rf_recent is not None else 0  # Recent model if available
    }
    
    # Normalize weights if the recent model isn't available
    if rf_recent is None:
        total = sum(ensemble_weights.values())
        ensemble_weights = {k: v/total for k, v in ensemble_weights.items()}
    
    # We'll use the ensemble_predict function defined at the module level
    # This makes it picklable
    
    # Create an EnsembleModel instance
    # The class is defined outside this function for pickling support
        
    # Create the ensemble model
    ensemble_model = EnsembleModel(models, ensemble_weights)
    
    # Evaluate the ensemble model
    y_pred_train = ensemble_model.predict(X_train)
    y_pred_test = ensemble_model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print("Ensemble model evaluation:")
    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Also evaluate individual models for comparison
    print("\nIndividual model performance:")
    for name, model in models.items():
        if model is not None:
            y_pred = model.predict(X_test)
            model_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"{name} Test RMSE: {model_rmse:.2f}")
    
    # Feature importance (use the main model's feature importances)
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': ensemble_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return ensemble_model, df_features, feature_cols, feature_importance, (X_test, y_test, y_pred_test)

def generate_demand_forecast(df, model, feature_cols, days_to_forecast=30, add_stochastic_variation=True):
    """Generate demand forecast with emphasis on recent data to avoid zero predictions
    
    Args:
        df: The input dataframe with historical data
        model: The trained forecast model
        feature_cols: List of feature columns used by the model
        days_to_forecast: Number of days to forecast
        add_stochastic_variation: Whether to add stochastic variation to predictions
    """
    
    # Prepare features with recency emphasis
    df_features, _ = prepare_features(df)
    
    # Get the latest date in the dataset
    latest_date = df['Date'].max()
    
    # Create a DataFrame for future dates
    future_dates = pd.date_range(start=latest_date + timedelta(days=1),
                              end=latest_date + timedelta(days=days_to_forecast))
    
    # Create a DataFrame to store forecasts
    forecast_data = []
    
    # Get all unique store-product combinations
    store_products = df[['Store_Id', 'Item']].drop_duplicates()
    
    # Loop through each store-product and forecast
    for _, row in store_products.iterrows():
        store_id = row['Store_Id']
        item = row['Item']
        
        # Get the latest data for this store-product
        product_data = df_features[(df_features['Store_Id'] == store_id) & 
                                   (df_features['Item'] == item)].sort_values('Date', ascending=False)
        
        # Skip if no data
        if len(product_data) == 0:
            continue
            
        # Use the most recent row as a template
        template_row = product_data.iloc[0].copy()
        
        # Get product details
        product_details = df[(df['Store_Id'] == store_id) & (df['Item'] == item)].iloc[0]
        product_name = product_details['Product']
        product_size = product_details['Size']
        
        # Get recent sales patterns
        recent_sales = product_data.sort_values('Date')
        if len(recent_sales) > 30:
            recent_sales = recent_sales.tail(30)
        
        # Calculate recent stock metrics
        current_stock = recent_sales['Stock_Level'].iloc[-1]
        avg_sales = recent_sales['Sales'].mean()
        
        # Create forecast for each future date
        for i, future_date in enumerate(future_dates):
            forecast_row = template_row.copy()
            
            # Update date-related features
            forecast_row['Date'] = future_date
            forecast_row['Day_Of_Week'] = future_date.dayofweek
            forecast_row['Month'] = future_date.month
            forecast_row['Year'] = future_date.year
            forecast_row['Day'] = future_date.day
            
            # Update enhanced cyclical encodings
            forecast_row['Day_Sin'] = np.sin(forecast_row['Day_Of_Week'] * (2 * np.pi / 7))
            forecast_row['Day_Cos'] = np.cos(forecast_row['Day_Of_Week'] * (2 * np.pi / 7))
            
            # Week of month (1-5)
            week_of_month = ((future_date.day - 1) // 7) + 1
            forecast_row['Week_Of_Month'] = week_of_month
            forecast_row['Week_Sin'] = np.sin(week_of_month * (2 * np.pi / 5))
            forecast_row['Week_Cos'] = np.cos(week_of_month * (2 * np.pi / 5))
            
            forecast_row['Month_Sin'] = np.sin(forecast_row['Month'] * (2 * np.pi / 12))
            forecast_row['Month_Cos'] = np.cos(forecast_row['Month'] * (2 * np.pi / 12))
            
            # Yearly seasonality
            day_of_year = future_date.timetuple().tm_yday
            forecast_row['Day_Of_Year'] = day_of_year
            forecast_row['Year_Sin'] = np.sin(day_of_year * (2 * np.pi / 366))
            forecast_row['Year_Cos'] = np.cos(day_of_year * (2 * np.pi / 366))
            
            # Weekend indicator
            forecast_row['Is_Weekend'] = 1 if future_date.weekday() >= 5 else 0
            
            # Holiday check
            month_day = f"{future_date.month:02d}-{future_date.day:02d}"
            holidays = {
                '01-01': 'New Year',
                '02-14': 'Valentine', 
                '07-04': 'Independence Day',
                '10-31': 'Halloween',
                '11-25': 'Thanksgiving',
                '12-25': 'Christmas',
                '04-15': 'Easter'
            }
            forecast_row['Is_Holiday'] = 1 if month_day in holidays else 0
            
            # Special event indicator
            forecast_row['Is_Special_Event'] = 0
            
            # Black Friday period (late Nov)
            if future_date.month == 11 and 20 <= future_date.day <= 30:
                forecast_row['Is_Special_Event'] = 1
            # Christmas shopping period (Dec 1-24)
            elif future_date.month == 12 and future_date.day <= 24:
                forecast_row['Is_Special_Event'] = 1
            # Summer sales (July)
            elif future_date.month == 7:
                forecast_row['Is_Special_Event'] = 1
            # Back to school (Aug 15-Sep 15)
            elif (future_date.month == 8 and future_date.day >= 15) or \
                 (future_date.month == 9 and future_date.day <= 15):
                forecast_row['Is_Special_Event'] = 1
            
            # Weather - use normal weather for forecasting
            for col in feature_cols:
                if col.startswith('Weather_'):
                    weather_type = col.replace('Weather_', '')
                    forecast_row[col] = 1 if weather_type == 'Normal' else 0
            
            # Update stock level projection based on previous day's forecast
            if i == 0:
                # First day of forecast
                projected_stock = current_stock
            else:
                # Use previous day's ending stock
                projected_stock = prev_projected_stock
            
            # No purchases assumed in the forecast period
            forecast_row['Stock_Level'] = projected_stock
            
            # Update weeks of stock
            if avg_sales > 0:
                weeks_of_stock = projected_stock / (avg_sales * 7)
                forecast_row['Weeks_Of_Stock'] = min(4, weeks_of_stock)  # Cap at 4 weeks
            else:
                forecast_row['Weeks_Of_Stock'] = 4  # Default to 4 weeks if no sales
                
            # Add volatility features for forecasting (use averages from historical data)
            forecast_row['Rolling_Std_7'] = recent_sales['Rolling_Std_7'].mean() if 'Rolling_Std_7' in recent_sales.columns else 0
            forecast_row['Rolling_Std_28'] = recent_sales['Rolling_Std_28'].mean() if 'Rolling_Std_28' in recent_sales.columns else 0
            
            # Add recency features for forecasting
            # For forecasts, Days_Since_Last is the forecast day number
            forecast_row['Days_Since_Last'] = i
            forecast_row['Last_7_Days'] = 1 if i < 7 else 0
            forecast_row['Last_14_Days'] = 1 if i < 14 else 0
            forecast_row['Last_30_Days'] = 1
            
            # Add lag features to ensure model has recent data for predictions
            # These are crucial for avoiding zero predictions
            if i == 0:
                # For first forecast day, use actual most recent values
                forecast_row['Sales_Lag_1'] = recent_sales['Sales'].iloc[-1] if len(recent_sales) > 0 else 0
                forecast_row['Sales_Lag_2'] = recent_sales['Sales'].iloc[-2] if len(recent_sales) > 1 else 0
                forecast_row['Sales_Lag_3'] = recent_sales['Sales'].iloc[-3] if len(recent_sales) > 2 else 0
                forecast_row['Sales_Lag_7'] = recent_sales['Sales'].iloc[-7] if len(recent_sales) > 6 else 0
                
                # EWMA and weighted recent values from historical data
                forecast_row['Sales_EWMA_7'] = recent_sales['Sales_EWMA_7'].iloc[-1] if 'Sales_EWMA_7' in recent_sales.columns and len(recent_sales) > 0 else 0
                forecast_row['Sales_Recent_Weighted'] = recent_sales['Sales_Recent_Weighted'].iloc[-1] if 'Sales_Recent_Weighted' in recent_sales.columns and len(recent_sales) > 0 else 0
            else:
                # For subsequent forecasts, use previous forecasts as lags
                # This ensures continuity in the forecast
                forecast_row['Sales_Lag_1'] = forecast_data[-1]['Predicted_Demand'] if len(forecast_data) > 0 else 0
                forecast_row['Sales_Lag_2'] = forecast_data[-2]['Predicted_Demand'] if len(forecast_data) > 1 else 0
                forecast_row['Sales_Lag_3'] = forecast_data[-3]['Predicted_Demand'] if len(forecast_data) > 2 else 0
                forecast_row['Sales_Lag_7'] = forecast_data[-7]['Predicted_Demand'] if len(forecast_data) > 6 else 0
                
                # Calculate EWMA and weighted recent values based on forecasts
                if len(forecast_data) > 0:
                    recent_preds = [fd['Predicted_Demand'] for fd in forecast_data[-7:]] if len(forecast_data) > 0 else [0]
                    
                    # Calculate EWMA
                    alpha = 2.0 / (7 + 1)
                    ewma = sum(alpha * (1 - alpha)**i * val for i, val in enumerate(reversed(recent_preds[:7])))
                    forecast_row['Sales_EWMA_7'] = ewma
                    
                    # Calculate heavily weighted recent average
                    if len(forecast_data) >= 7:
                        forecast_row['Sales_Recent_Weighted'] = forecast_data[-1]['Predicted_Demand'] * 0.7 + \
                            sum(forecast_data[-i-2]['Predicted_Demand'] * 0.05 for i in range(6) if len(forecast_data) > i+1)
                    else:
                        forecast_row['Sales_Recent_Weighted'] = forecast_data[-1]['Predicted_Demand'] if len(forecast_data) > 0 else 0
                else:
                    forecast_row['Sales_EWMA_7'] = 0
                    forecast_row['Sales_Recent_Weighted'] = 0
            
            # Make prediction
            X_forecast = pd.DataFrame([forecast_row[feature_cols]])
            predicted_sales = max(0, model.predict(X_forecast, add_noise=add_stochastic_variation)[0])
            
            # Update projected stock for next day
            prev_projected_stock = max(0, projected_stock - predicted_sales)
            
            # Add to forecast data
            forecast_data.append({
                'Date': future_date,
                'Store_Id': store_id,
                'Item': item,
                'Product': product_name,
                'Size': product_size,
                'Predicted_Demand': round(predicted_sales),
                'Projected_Stock': round(projected_stock),
                'Stock_Status': 'Low' if forecast_row['Weeks_Of_Stock'] < 1 else 'Adequate' if forecast_row['Weeks_Of_Stock'] <= 3 else 'Excess'
            })
    
    return pd.DataFrame(forecast_data)

def calculate_order_recommendations(forecast_df, current_stock_df=None):
    """Calculate order recommendations based on the forecast and current stock levels"""
    # If no current stock data is provided, use the first day's projected stock
    if current_stock_df is None:
        # Get the first day's projected stock for each store-item
        current_stock = forecast_df.groupby(['Store_Id', 'Item']).first().reset_index()[['Store_Id', 'Item', 'Projected_Stock']]
        current_stock.rename(columns={'Projected_Stock': 'Current_Stock'}, inplace=True)
    else:
        current_stock = current_stock_df[['Store_Id', 'Item', 'Current_Stock']]
    
    # Calculate average daily demand for each store-item
    avg_demand = forecast_df.groupby(['Store_Id', 'Item'])['Predicted_Demand'].mean().reset_index()
    avg_demand.rename(columns={'Predicted_Demand': 'Avg_Daily_Demand'}, inplace=True)
    
    # Create a recommendations dataframe
    recommendations = current_stock.merge(avg_demand, on=['Store_Id', 'Item'])
    
    # Calculate weeks of stock
    recommendations['Weeks_Of_Stock'] = recommendations['Current_Stock'] / (recommendations['Avg_Daily_Demand'] * 7)
    recommendations['Weeks_Of_Stock'] = recommendations['Weeks_Of_Stock'].replace([np.inf, -np.inf], 4)
    recommendations['Weeks_Of_Stock'] = recommendations['Weeks_Of_Stock'].fillna(4)
    
    # Set target stock levels (2 weeks of inventory)
    recommendations['Target_Stock_Weeks'] = 2
    recommendations['Min_Stock_Weeks'] = 1
    recommendations['Target_Stock_Units'] = recommendations['Avg_Daily_Demand'] * 7 * recommendations['Target_Stock_Weeks']
    
    # Calculate recommended order quantity
    # If we have less than 1 week of stock, order enough to get to 2 weeks
    # If we have 1-3 weeks of stock, don't order
    # If we have more than 3 weeks of stock, don't order
    conditions = [
        (recommendations['Weeks_Of_Stock'] < recommendations['Min_Stock_Weeks']),  # Less than 1 week
        (recommendations['Weeks_Of_Stock'] >= recommendations['Min_Stock_Weeks']) & 
        (recommendations['Weeks_Of_Stock'] <= 3),  # 1-3 weeks
        (recommendations['Weeks_Of_Stock'] > 3)  # More than 3 weeks
    ]
    
    values = [
        # If low stock, order to reach target stock level
        recommendations['Target_Stock_Units'] - recommendations['Current_Stock'],
        # If adequate stock, don't order
        0,
        # If excess stock, don't order
        0
    ]
    
    recommendations['Recommended_Order_Quantity'] = np.select(conditions, values, default=0)
    recommendations['Recommended_Order_Quantity'] = recommendations['Recommended_Order_Quantity'].apply(lambda x: max(0, round(x)))
    
    # Add product details and recommendation justification
    product_details = forecast_df[['Store_Id', 'Item', 'Product', 'Size']].drop_duplicates()
    recommendations = recommendations.merge(product_details, on=['Store_Id', 'Item'])
    
    # Add stock status and recommendation reason
    conditions = [
        (recommendations['Weeks_Of_Stock'] < recommendations['Min_Stock_Weeks']),
        (recommendations['Weeks_Of_Stock'] >= recommendations['Min_Stock_Weeks']) & 
        (recommendations['Weeks_Of_Stock'] <= 3),
        (recommendations['Weeks_Of_Stock'] > 3)
    ]
    
    statuses = ['Low', 'Adequate', 'Excess']
    recommendations['Stock_Status'] = np.select(conditions, statuses, default='Adequate')
    
    reasons = [
        f"Low stock - less than {recommendations['Min_Stock_Weeks']} week of inventory",
        f"Adequate stock - between {recommendations['Min_Stock_Weeks']} and 3 weeks of inventory",
        "Excess stock - more than 3 weeks of inventory"
    ]
    recommendations['Recommendation_Reason'] = np.select(conditions, reasons, default="Adequate stock")
    
    # Calculate profit impact of recommendations
    profit_margins = forecast_df[['Store_Id', 'Item', 'Product']].drop_duplicates()
    
    # Add dummy profit margin data (would be replaced with actual data)
    np.random.seed(42)
    profit_margins['Unit_Profit'] = np.random.uniform(1.0, 5.0, size=len(profit_margins))
    
    recommendations = recommendations.merge(profit_margins[['Store_Id', 'Item', 'Unit_Profit']], on=['Store_Id', 'Item'])
    recommendations['Profit_Impact'] = recommendations['Recommended_Order_Quantity'] * recommendations['Unit_Profit']
    
    return recommendations

def save_model(model, feature_cols, filepath='models/rf_model.pkl'):
    """Save the trained Random Forest model"""
    import pickle
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the model
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature columns
    with open(filepath.replace('.pkl', '_features.pkl'), 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print(f"Model saved to {filepath}")

def load_model(filepath='models/rf_model.pkl'):
    """Load a trained Random Forest model"""
    import pickle
    
    # Load model
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    # Load feature columns
    with open(filepath.replace('.pkl', '_features.pkl'), 'rb') as f:
        feature_cols = pickle.load(f)
    
    print(f"Model loaded from {filepath}")
    return model, feature_cols

def generate_feature_importance_chart(feature_importance):
    """Generate feature importance chart"""
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 20 Features Affecting Sales Predictions')
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('static/images', exist_ok=True)
    plt.savefig('static/images/rf_feature_importance.png')
    plt.close()
    
    return 'static/images/rf_feature_importance.png'

def generate_sales_forecast_chart(forecast_df):
    """Generate sales forecast chart for top products"""
    # Get top 9 products by forecast volume
    top_products = forecast_df.groupby(['Store_Id', 'Item', 'Product'])['Predicted_Demand'].sum().reset_index().sort_values('Predicted_Demand', ascending=False).head(9)
    
    plt.figure(figsize=(16, 10))
    
    for i, (_, row) in enumerate(top_products.iterrows()):
        if i < 9:  # Limit to 9 products for readability
            plt.subplot(3, 3, i+1)
            
            # Plot product forecast
            product_forecast = forecast_df[(forecast_df['Store_Id'] == row['Store_Id']) & 
                                           (forecast_df['Item'] == row['Item'])]
            plt.plot(product_forecast['Date'], product_forecast['Predicted_Demand'], 'b-', marker='o')
            plt.title(f"{row['Product']} (Store {row['Store_Id']})")
            plt.xticks(rotation=45, fontsize=8)
            plt.grid(True, alpha=0.3)
            if i % 3 == 0:
                plt.ylabel('Predicted Demand (units)')
            plt.tight_layout()
    
    plt.suptitle('30-Day Demand Forecast by Product (Random Forest Model)', y=1.05)
    plt.tight_layout()
    
    # Save figure
    os.makedirs('static/images', exist_ok=True)
    plt.savefig('static/images/rf_demand_forecast.png')
    plt.close()
    
    return 'static/images/rf_demand_forecast.png'

def generate_stock_projection_chart(forecast_df):
    """Generate stock projection chart for top products"""
    # Get top 9 products by forecast volume
    top_products = forecast_df.groupby(['Store_Id', 'Item', 'Product'])['Predicted_Demand'].sum().reset_index().sort_values('Predicted_Demand', ascending=False).head(9)
    
    plt.figure(figsize=(16, 10))
    
    for i, (_, row) in enumerate(top_products.iterrows()):
        if i < 9:  # Limit to 9 products for readability
            plt.subplot(3, 3, i+1)
            
            # Plot product stock projection
            product_forecast = forecast_df[(forecast_df['Store_Id'] == row['Store_Id']) & 
                                           (forecast_df['Item'] == row['Item'])]
            plt.plot(product_forecast['Date'], product_forecast['Projected_Stock'], 'g-', marker='o')
            plt.title(f"{row['Product']} (Store {row['Store_Id']})")
            plt.xticks(rotation=45, fontsize=8)
            plt.grid(True, alpha=0.3)
            if i % 3 == 0:
                plt.ylabel('Projected Stock (units)')
            plt.tight_layout()
    
    plt.suptitle('30-Day Stock Projection by Product', y=1.05)
    plt.tight_layout()
    
    # Save figure
    os.makedirs('static/images', exist_ok=True)
    plt.savefig('static/images/rf_stock_projection.png')
    plt.close()
    
    return 'static/images/rf_stock_projection.png'

def generate_order_recommendation_chart(recommendations):
    """Generate order recommendation chart for products with recommended orders"""
    # Filter to products with recommended orders
    order_recs = recommendations[recommendations['Recommended_Order_Quantity'] > 0].sort_values('Recommended_Order_Quantity', ascending=False).head(20)
    
    plt.figure(figsize=(14, 8))
    x = range(len(order_recs))
    plt.bar(x, order_recs['Recommended_Order_Quantity'], color='blue', alpha=0.7)
    plt.xticks(x, [f"{row['Product'][:15]}... (Store {row['Store_Id']})" for _, row in order_recs.iterrows()], 
               rotation=45, ha='right')
    plt.ylabel('Recommended Order Quantity')
    plt.title('Top Product Order Recommendations')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    os.makedirs('static/images', exist_ok=True)
    plt.savefig('static/images/rf_order_recommendations.png')
    plt.close()
    
    return 'static/images/rf_order_recommendations.png'

def main(use_existing_model=False):
    """Main function to run the RF model with the new data
    
    Args:
        use_existing_model (bool): If True, uses a previously saved model instead of retraining
    """
    # Load the integrated dataset
    df = pd.read_csv('combined_pizza_data.csv')
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    if use_existing_model:
        try:
            print("Loading existing Random Forest model...")
            model, feature_cols = load_model()
            # Prepare features for consistency
            df_features, _ = prepare_features(df)
            feature_importance = None
            test_data = None
            print("Existing model loaded successfully!")
        except Exception as e:
            print(f"Error loading existing model: {e}")
            print("Falling back to training a new model...")
            use_existing_model = False
    
    if not use_existing_model:
        print("Training Random Forest model on new pizza data...")
        model, df_features, feature_cols, feature_importance, test_data = train_forecast_model(df)
        
        # Save the model
        save_model(model, feature_cols)
    
    # Generate forecasts
    print("Generating demand forecasts...")
    forecast_df = generate_demand_forecast(df, model, feature_cols, days_to_forecast=30, add_stochastic_variation=True)
    forecast_df.to_csv('rf_forecasts.csv', index=False)
    
    # Calculate order recommendations
    print("Calculating order recommendations...")
    recommendations = calculate_order_recommendations(forecast_df)
    recommendations.to_csv('rf_recommendations.csv', index=False)
    
    # Generate charts (only if we have feature importance data)
    print("Generating visualization charts...")
    if feature_importance is not None:
        generate_feature_importance_chart(feature_importance)
    generate_sales_forecast_chart(forecast_df)
    generate_stock_projection_chart(forecast_df)
    generate_order_recommendation_chart(recommendations)
    
    print("Random Forest model processing completed!")

if __name__ == '__main__':
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Random Forest Model for Pizza Sales Forecasting')
    parser.add_argument('--use-existing', action='store_true',
                        help='Use existing saved model instead of retraining')
    
    args = parser.parse_args()
    
    # Run main function with command line arguments
    main(use_existing_model=args.use_existing)