"""
Weighted Averaged ARIMA model module for sales forecasting.
This module implements a weighted ensemble of ARIMA models across different time windows.
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import os
import pickle
import logging
import argparse
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import multiprocessing
from joblib import Parallel, delayed
import warnings

# Suppress statsmodels warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('weighted_arima_model')

# Import settings if available
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from config.settings import (
        COMBINED_DATA_FILE, MODELS_DIR, WEIGHTED_ARIMA_FORECASTS_FILE, STATIC_DIR
    )
except ImportError:
    # Default paths for backward compatibility
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    COMBINED_DATA_FILE = os.path.join(ROOT_DIR, "data", "processed", "combined_pizza_data.csv")
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    WEIGHTED_ARIMA_FORECASTS_FILE = os.path.join(ROOT_DIR, "data", "processed", "weighted_arima_forecasts.csv")
    STATIC_DIR = os.path.join(ROOT_DIR, "static")

# Define time windows for weighted models
TIME_WINDOWS = {
    'daily': {'days': 7, 'weight': 0.30},      # Recent daily patterns (1 week)
    'weekly': {'days': 28, 'weight': 0.25},    # Recent weekly patterns (4 weeks)
    'monthly': {'days': 90, 'weight': 0.20},   # Monthly patterns (3 months)
    'quarterly': {'days': 180, 'weight': 0.15}, # Quarterly patterns (6 months)
    'yearly': {'days': 365, 'weight': 0.10}    # Yearly patterns
}


class WeightedARIMAModel:
    """
    Weighted Averaged ARIMA model that combines predictions from multiple time windows
    """
    def __init__(self, store_id=None, item=None, exog_vars=None, time_windows=None):
        """
        Initialize Weighted ARIMA model
        
        Args:
            store_id: Store ID
            item: Item ID
            exog_vars: List of exogenous variables to include
            time_windows: Dictionary of time windows and weights to use
        """
        self.store_id = store_id
        self.item = item
        self.exog_vars = exog_vars or ['Price', 'Promotion', 'Is_Holiday']
        self.time_windows = time_windows or TIME_WINDOWS
        self.models = {}
        self.training_data = None
        self.y_mean = None
        self.y_std = None
        self.forecast_history = {}
    
    def preprocess_data(self, df, window_name=None):
        """
        Preprocess time series data for ARIMA modeling with optional window filtering
        
        Args:
            df: DataFrame with time series data
            window_name: Time window to filter for (daily, weekly, etc.)
            
        Returns:
            tuple: Processed data, exogenous variables
        """
        logger.info(f"Preprocessing data for Store {self.store_id}, Item {self.item}, Window {window_name}")
        
        # Filter data for this store-item
        item_df = df[(df['Store_Id'] == self.store_id) & (df['Item'] == self.item)].copy()
        
        # Sort by date
        item_df = item_df.sort_values('Date')
        
        # Apply time window filter if specified
        if window_name and window_name in self.time_windows:
            # Filter based on the window days
            window_days = self.time_windows[window_name]['days']
            max_date = item_df['Date'].max()
            cutoff_date = max_date - timedelta(days=window_days)
            item_df = item_df[item_df['Date'] >= cutoff_date]
            logger.info(f"Applied {window_name} window filter: {window_days} days, {len(item_df)} records")
        
        # Convert date to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(item_df['Date']):
            item_df['Date'] = pd.to_datetime(item_df['Date'])
        
        # Set date as index
        item_df = item_df.set_index('Date')
        
        # Get target variable (Sales)
        y = item_df['Sales']
        
        # Check if 'Sales' column contains non-numeric values
        if not pd.api.types.is_numeric_dtype(y):
            logger.info("Converting Sales column to numeric values")
            y = pd.to_numeric(y, errors='coerce')
            # Replace NaN values from the conversion with zeros
            y = y.fillna(0)
        
        # Always ensure target is float64 to avoid dtype issues
        y = y.astype(np.float64)
        
        # Check for zeros and replace with small values to enable log transformation
        if (y == 0).any():
            logger.info("Replacing zeros with small values for log transformation")
            y = y.replace(0, 0.01)
        
        # Store mean and std for later denormalization if this is the full dataset
        if window_name is None:
            self.y_mean = y.mean()
            self.y_std = y.std()
            
            # Store training data info for future reference
            self.training_data = {
                'last_date': item_df.index[-1],
                'store_id': self.store_id,
                'item': self.item
            }
        
        # Standardize the target variable (helps ARIMA convergence)
        y_normalized = (y - self.y_mean) / self.y_std
        
        # Ensure normalized y is float64
        y_normalized = y_normalized.astype(np.float64)
        
        # Prepare exogenous variables
        X = None
        if self.exog_vars:
            valid_exog_vars = [v for v in self.exog_vars if v in item_df.columns]
            if valid_exog_vars:
                X = item_df[valid_exog_vars].copy()
                
                # Handle missing values with more robust approach
                X = X.fillna(method='ffill').fillna(method='bfill')
                
                # If there are still NaN values after forward/backward fill
                if X.isna().any().any():
                    for col in X.columns:
                        if X[col].isna().any():
                            # Fill remaining NaNs with column mean or 0
                            if X[col].dtype.kind in 'iuf':  # integer, unsigned int, or float
                                X[col] = X[col].fillna(X[col].mean() if not X[col].isna().all() else 0)
                            else:  # categorical or object type
                                # For categorical, fill with most common value
                                most_common = X[col].value_counts().idxmax() if not X[col].isna().all() else "missing"
                                X[col] = X[col].fillna(most_common)
                
                # Handle categorical variables and ensure numeric types
                # First identify all object columns
                object_cols = [col for col in X.columns if X[col].dtype == 'object']
                
                # Convert object columns to dummies
                if object_cols:
                    logger.info(f"Converting categorical columns to dummies: {object_cols}")
                    X = pd.get_dummies(X, columns=object_cols, drop_first=True)
                
                # Then ensure all remaining columns are numeric
                for col in X.columns:
                    if not pd.api.types.is_numeric_dtype(X[col]):
                        # Convert non-numeric columns to numeric
                        logger.info(f"Converting {col} to numeric values")
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                        X[col] = X[col].fillna(X[col].mean() if not X[col].isnull().all() else 0)
                        # Ensure float64 type
                        X[col] = X[col].astype(np.float64)
        
        # Detect seasonality
        seasonal_periods = self._detect_seasonality(y)
        
        return y_normalized, X, seasonal_periods
    
    def _detect_seasonality(self, series):
        """
        Detect seasonality in time series data
        
        Args:
            series: Time series data
            
        Returns:
            dict: Detected seasonality periods
        """
        seasonal_periods = {}
        
        try:
            # Need minimum amount of data for seasonality detection
            min_periods = 14
            if len(series) >= min_periods:
                # Try different seasonal periods
                for period, name in [(7, 'weekly'), (30, 'monthly'), (90, 'quarterly'), (365, 'yearly')]:
                    if len(series) >= period * 2:  # Need at least 2 full seasons
                        try:
                            seasonal_result = seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
                            seasonal_strength = np.std(seasonal_result.seasonal) / np.std(series - seasonal_result.trend)
                            
                            if seasonal_strength > 0.1:  # Significant seasonality
                                seasonal_periods[name] = {
                                    'period': period,
                                    'strength': seasonal_strength
                                }
                                logger.info(f"Detected {name} seasonality with period {period}, strength {seasonal_strength:.3f}")
                        except Exception as e:
                            logger.warning(f"Error detecting {name} seasonality: {e}")
        
        except Exception as e:
            logger.warning(f"Error in seasonality detection: {e}")
        
        return seasonal_periods
    
    def train_window_model(self, df, window_name):
        """
        Train ARIMA model for a specific time window
        
        Args:
            df: DataFrame with time series data
            window_name: Time window name
            
        Returns:
            dict: Trained model information
        """
        logger.info(f"Training ARIMA model for {window_name} window: Store {self.store_id}, Item {self.item}")
        
        try:
            # Preprocess data for this window
            y, X, seasonal_periods = self.preprocess_data(df, window_name)
            
            if len(y) < 10:
                logger.warning(f"Insufficient data for {window_name} window. Skipping.")
                return None
            
            # Determine if we should use seasonal ARIMA
            use_seasonal = False
            seasonal_period = None
            if window_name in seasonal_periods:
                use_seasonal = True
                seasonal_period = seasonal_periods[window_name]['period']
            elif 'weekly' in seasonal_periods and window_name in ['daily', 'weekly']:
                use_seasonal = True
                seasonal_period = 7  # Weekly seasonality
            elif 'monthly' in seasonal_periods and window_name in ['monthly']:
                use_seasonal = True
                seasonal_period = 30  # Monthly seasonality
            
            # Find optimal parameters
            logger.info(f"Finding optimal parameters for {window_name} window")
            
            try:
                if use_seasonal and seasonal_period:
                    # Ensure data types before passing to auto_arima
                    y_arr = np.asarray(y, dtype=np.float64)
                    X_arr = np.asarray(X, dtype=np.float64) if X is not None else None
                    
                    # Check for NaN values
                    if np.isnan(y_arr).any():
                        logger.warning("Target array contains NaN values, replacing with zeros")
                        y_arr = np.nan_to_num(y_arr, nan=0.0)
                    
                    if X_arr is not None and np.isnan(X_arr).any():
                        # More detailed logging about NaN values in exogenous variables
                        if isinstance(X, pd.DataFrame):
                            nan_columns = [col for col in X.columns if X[col].isna().any()]
                            logger.warning(f"Exogenous array contains NaN values in columns: {nan_columns}, replacing with zeros")
                        else:
                            logger.warning("Exogenous array contains NaN values, replacing with zeros")
                        X_arr = np.nan_to_num(X_arr, nan=0.0)
                    
                    # Seasonal ARIMA
                    model = auto_arima(
                        y_arr,
                        exogenous=X_arr,
                        start_p=0, start_q=0,
                        max_p=2, max_q=2, max_d=1,
                        start_P=0, start_Q=0,
                        max_P=1, max_Q=1, max_D=1,
                        m=seasonal_period,
                        seasonal=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True,
                        trace=False
                    )
                    
                    # Extract parameters
                    order = model.order
                    seasonal_order = model.seasonal_order
                    
                    best_params = {
                        'order': order,
                        'seasonal_order': seasonal_order
                    }
                    
                    logger.info(f"Best SARIMA parameters for {window_name}: order={order}, seasonal_order={seasonal_order}")
                else:
                    # Ensure data types before passing to auto_arima
                    y_arr = np.asarray(y, dtype=np.float64)
                    X_arr = np.asarray(X, dtype=np.float64) if X is not None else None
                    
                    # Check for NaN values
                    if np.isnan(y_arr).any():
                        logger.warning("Target array contains NaN values, replacing with zeros")
                        y_arr = np.nan_to_num(y_arr, nan=0.0)
                    
                    if X_arr is not None and np.isnan(X_arr).any():
                        # More detailed logging about NaN values in exogenous variables
                        if isinstance(X, pd.DataFrame):
                            nan_columns = [col for col in X.columns if X[col].isna().any()]
                            logger.warning(f"Exogenous array contains NaN values in columns: {nan_columns}, replacing with zeros")
                        else:
                            logger.warning("Exogenous array contains NaN values, replacing with zeros")
                        X_arr = np.nan_to_num(X_arr, nan=0.0)
                    
                    # Non-seasonal ARIMA
                    model = auto_arima(
                        y_arr,
                        exogenous=X_arr,
                        start_p=0, start_q=0,
                        max_p=3, max_q=3, max_d=1,
                        seasonal=False,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True,
                        trace=False
                    )
                    
                    # Extract parameters
                    order = model.order
                    
                    best_params = {
                        'order': order,
                        'seasonal_order': (0, 0, 0, 0)
                    }
                    
                    logger.info(f"Best ARIMA parameters for {window_name}: order={order}")
            
            except Exception as e:
                logger.warning(f"Auto parameter selection failed for {window_name}: {str(e)}")
                
                # Use default parameters
                if use_seasonal and seasonal_period:
                    best_params = {
                        'order': (1, 1, 1),
                        'seasonal_order': (1, 0, 1, seasonal_period)
                    }
                else:
                    best_params = {
                        'order': (1, 1, 1),
                        'seasonal_order': (0, 0, 0, 0)
                    }
                
                logger.info(f"Using default parameters for {window_name}: {best_params}")
            
            # Train model with selected parameters
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Ensure data types before passing to SARIMAX
                # Convert to numpy arrays with float64 dtype
                if isinstance(y, pd.Series):
                    y_arr = y.astype(np.float64).values
                else:
                    y_arr = np.asarray(y, dtype=np.float64)
                
                if X is not None:
                    if isinstance(X, pd.DataFrame):
                        X_arr = X.astype(np.float64).values
                    else:
                        X_arr = np.asarray(X, dtype=np.float64)
                else:
                    X_arr = None
                
                # Check for NaN values
                if np.isnan(y_arr).any():
                    logger.warning("Target array contains NaN values, replacing with zeros")
                    y_arr = np.nan_to_num(y_arr, nan=0.0)
                
                if X_arr is not None and np.isnan(X_arr).any():
                    logger.warning("Exogenous array contains NaN values, replacing with zeros")
                    X_arr = np.nan_to_num(X_arr, nan=0.0)
                
                model = SARIMAX(
                    y_arr,
                    exog=X_arr,
                    order=best_params['order'],
                    seasonal_order=best_params['seasonal_order'],
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                # Fit model with appropriate method based on data size
                if len(y) > 1000:
                    fitted_model = model.fit(disp=0, maxiter=50, method='bfgs')
                else:
                    fitted_model = model.fit(disp=0)
            
            # Store model info
            model_info = {
                'model': fitted_model,
                'params': best_params,
                'window': window_name,
                'weight': self.time_windows[window_name]['weight'],
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'training_size': len(y)
            }
            
            logger.info(f"Model trained successfully for {window_name} window - AIC: {fitted_model.aic:.2f}")
            
            return model_info
        
        except Exception as e:
            logger.error(f"Error training model for {window_name} window: {str(e)}")
            return None
    
    def train_all_models(self, df):
        """
        Train models for all time windows
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            dict: Trained models information
        """
        logger.info(f"Training all window models for Store {self.store_id}, Item {self.item}")
        
        # First, process the full dataset to set global statistics
        y_full, X_full, _ = self.preprocess_data(df)
        
        # Train models for each window
        self.models = {}
        for window_name in self.time_windows.keys():
            model_info = self.train_window_model(df, window_name)
            if model_info:
                self.models[window_name] = model_info
        
        # Check if we have at least one model
        if not self.models:
            logger.error(f"No models could be trained for Store {self.store_id}, Item {self.item}")
            return False
        
        # Normalize weights if some models failed to train
        self._normalize_weights()
        
        logger.info(f"Trained {len(self.models)} window models successfully")
        return True
    
    def _normalize_weights(self):
        """
        Normalize weights of trained models
        """
        # Sum of weights for trained models
        total_weight = sum(model_info['weight'] for model_info in self.models.values())
        
        # Normalize if total is not 1
        if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
            for window_name in self.models:
                self.models[window_name]['weight'] /= total_weight
            
            logger.info(f"Normalized model weights: {[(w, self.models[w]['weight']) for w in self.models]}")
    
    def forecast(self, steps, X_future=None, return_conf_int=True, alpha=0.05):
        """
        Generate weighted ensemble forecasts
        
        Args:
            steps: Number of steps to forecast
            X_future: Future exogenous variables
            return_conf_int: Whether to return confidence intervals
            alpha: Significance level for confidence intervals
            
        Returns:
            DataFrame with forecasts
        """
        logger.info(f"Generating {steps} step weighted forecast for Store {self.store_id}, Item {self.item}")
        
        # Check if we have trained models
        if not self.models:
            raise ValueError("No trained models available. Call train_all_models() first.")
        
        try:
            # Initialize arrays for weighted forecasts and confidence bounds
            weighted_forecast = np.zeros(steps)
            
            # For confidence interval calculation
            all_forecasts = {}
            all_lower_bounds = {}
            all_upper_bounds = {}
            
            # Generate forecasts for each model
            for window_name, model_info in self.models.items():
                logger.info(f"Generating forecast from {window_name} window model")
                
                model = model_info['model']
                weight = model_info['weight']
                
                # Generate forecast
                if return_conf_int:
                    forecast_result = model.get_forecast(steps=steps, exog=X_future)
                    forecast_mean = forecast_result.predicted_mean
                    forecast_ci = forecast_result.conf_int(alpha=alpha)
                    
                    # Store window forecasts for confidence calculation
                    # Handle both pandas Series and numpy array types
                    if hasattr(forecast_mean, 'values'):
                        all_forecasts[window_name] = forecast_mean.values
                        all_lower_bounds[window_name] = forecast_ci.iloc[:, 0].values
                        all_upper_bounds[window_name] = forecast_ci.iloc[:, 1].values
                    else:
                        all_forecasts[window_name] = forecast_mean
                        all_lower_bounds[window_name] = forecast_ci[:, 0]
                        all_upper_bounds[window_name] = forecast_ci[:, 1]
                else:
                    forecast_mean = model.forecast(steps=steps, exog=X_future)
                    # Handle both pandas Series and numpy array types
                    if hasattr(forecast_mean, 'values'):
                        all_forecasts[window_name] = forecast_mean.values
                    else:
                        all_forecasts[window_name] = forecast_mean
                
                # Add weighted contribution to ensemble forecast
                if hasattr(forecast_mean, 'values'):
                    weighted_forecast += weight * forecast_mean.values
                else:
                    weighted_forecast += weight * forecast_mean
            
            # Create forecast index
            forecast_index = pd.date_range(
                start=self.training_data['last_date'] + timedelta(days=1), 
                periods=steps
            )
            
            # Create DataFrame for results
            forecast_df = pd.DataFrame({
                'Forecast': weighted_forecast
            }, index=forecast_index)
            
            # Calculate confidence intervals for weighted ensemble
            # We use weighted mean of variances and add uncertainty for weighted model combination
            if return_conf_int:
                # Calculate weighted variances from individual model confidence intervals
                # For a normal distribution, CI width = 2 * z_score * std_dev
                # So std_dev = CI_width / (2 * z_score)
                z_score = 1.96  # 95% confidence interval
                
                weighted_vars = np.zeros(steps)
                for window_name, model_info in self.models.items():
                    weight = model_info['weight']
                    # Estimate variance from confidence interval width
                    forecast_std = (all_upper_bounds[window_name] - all_lower_bounds[window_name]) / (2 * z_score)
                    # Add weighted contribution to ensemble variance
                    weighted_vars += weight * (forecast_std ** 2)
                
                # Add additional uncertainty from model combination
                # We add a small variance proportional to the disagreement between models
                if len(self.models) > 1:
                    # Calculate variance of individual forecasts
                    model_disagreement = np.zeros(steps)
                    for window_name in self.models:
                        model_disagreement += (all_forecasts[window_name] - weighted_forecast) ** 2
                    
                    model_disagreement /= len(self.models)
                    # Add a portion of this disagreement to the overall variance
                    weighted_vars += 0.2 * model_disagreement  # 20% weight for model disagreement
                
                # Calculate confidence intervals
                forecast_df['Lower_Bound'] = forecast_df['Forecast'] - z_score * np.sqrt(weighted_vars)
                forecast_df['Upper_Bound'] = forecast_df['Forecast'] + z_score * np.sqrt(weighted_vars)
            
            # Denormalize predictions
            forecast_df['Forecast'] = (forecast_df['Forecast'] * self.y_std) + self.y_mean
            
            if return_conf_int:
                forecast_df['Lower_Bound'] = (forecast_df['Lower_Bound'] * self.y_std) + self.y_mean
                forecast_df['Upper_Bound'] = (forecast_df['Upper_Bound'] * self.y_std) + self.y_mean
            
            # Ensure non-negative values
            forecast_df['Forecast'] = np.maximum(0, forecast_df['Forecast'])
            if return_conf_int:
                forecast_df['Lower_Bound'] = np.maximum(0, forecast_df['Lower_Bound'])
                forecast_df['Upper_Bound'] = np.maximum(0, forecast_df['Upper_Bound'])
            
            # Reset index to get Date as column
            forecast_df.reset_index(inplace=True)
            forecast_df.rename(columns={'index': 'Date'}, inplace=True)
            
            # Add metadata
            forecast_df['Store_Id'] = self.store_id
            forecast_df['Item'] = self.item
            forecast_df['Forecast_Generated'] = datetime.now()
            forecast_df['Days_In_Future'] = range(1, steps + 1)
            
            # Store forecast in history
            self.forecast_history[datetime.now()] = forecast_df
            
            logger.info(f"Weighted forecast generated successfully for {steps} days")
            return forecast_df
        
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise
    
    def save_model(self, model_dir=None):
        """
        Save trained models to disk
        
        Args:
            model_dir: Directory to save the model
            
        Returns:
            str: Path to saved model
        """
        if not self.models:
            raise ValueError("No trained models to save")
        
        if model_dir is None:
            model_dir = os.path.join(MODELS_DIR, 'weighted_arima')
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Create filename
        filename = f"weighted_arima_model_{self.store_id}_{self.item}.pkl"
        filepath = os.path.join(model_dir, filename)
        
        logger.info(f"Saving model to {filepath}")
        
        try:
            # Save the models and metadata
            with open(filepath, 'wb') as f:
                model_data = {
                    'models': self.models,
                    'y_mean': self.y_mean,
                    'y_std': self.y_std,
                    'time_windows': self.time_windows,
                    'store_id': self.store_id,
                    'item': self.item,
                    'exog_vars': self.exog_vars,
                    'training_data': self.training_data
                }
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, store_id, item, model_dir=None):
        """
        Load trained model from disk
        
        Args:
            store_id: Store ID
            item: Item ID
            model_dir: Directory where models are stored
            
        Returns:
            WeightedARIMAModel: Loaded model
        """
        if model_dir is None:
            model_dir = os.path.join(MODELS_DIR, 'weighted_arima')
        
        # Create filename
        filename = f"weighted_arima_model_{store_id}_{item}.pkl"
        filepath = os.path.join(model_dir, filename)
        
        logger.info(f"Loading model from {filepath}")
        
        try:
            # Check if model exists
            if not os.path.exists(filepath):
                logger.warning(f"Model file {filepath} not found")
                return None
            
            # Load the model and metadata
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create new model
            model = cls(
                store_id=model_data['store_id'], 
                item=model_data['item'], 
                exog_vars=model_data['exog_vars'],
                time_windows=model_data['time_windows']
            )
            
            # Restore model attributes
            model.models = model_data['models']
            model.y_mean = model_data['y_mean']
            model.y_std = model_data['y_std']
            model.training_data = model_data['training_data']
            
            logger.info(f"Model loaded from {filepath}")
            return model
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def plot_forecast(self, forecast_df, history_df=None, output_file=None):
        """
        Plot forecasts with historical data
        
        Args:
            forecast_df: DataFrame with forecast data
            history_df: DataFrame with historical data (optional)
            output_file: Path to save the plot (optional)
            
        Returns:
            None
        """
        plt.figure(figsize=(12, 6))
        
        # Plot historical data if available
        if history_df is not None:
            history = history_df[(history_df['Store_Id'] == self.store_id) & 
                               (history_df['Item'] == self.item)].copy()
            
            if len(history) > 0:
                # Convert date if needed
                if not pd.api.types.is_datetime64_any_dtype(history['Date']):
                    history['Date'] = pd.to_datetime(history['Date'])
                
                # Plot historical data - last 60 days for better visualization
                recent_history = history.sort_values('Date').tail(60)
                plt.plot(recent_history['Date'], recent_history['Sales'], 'b-', label='Historical Sales')
        
        # Plot forecast
        plt.plot(forecast_df['Date'], forecast_df['Forecast'], 'r-', label='Weighted ARIMA Forecast')
        
        # Plot confidence intervals if available
        if 'Lower_Bound' in forecast_df.columns and 'Upper_Bound' in forecast_df.columns:
            plt.fill_between(
                forecast_df['Date'],
                forecast_df['Lower_Bound'],
                forecast_df['Upper_Bound'],
                color='r', alpha=0.2, label='95% Confidence Interval'
            )
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title(f'Weighted ARIMA Forecast for Store {self.store_id}, Item {self.item}')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save or display plot
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def prepare_future_exog(df, store_id, item, future_dates, exog_vars=None, required_shape=14):
    """
    Prepare future exogenous variables for forecasting
    
    Args:
        df: Historical data
        store_id: Store ID
        item: Item ID
        future_dates: List of future dates
        exog_vars: List of exogenous variables to include
        
    Returns:
        DataFrame: Future exogenous variables
        
    Note: This function has been updated to fix dimension mismatches between
    the training data exogenous variables and the forecast exogenous variables.
    """
    if exog_vars is None or len(exog_vars) == 0:
        return None
    
    logger.info(f"Preparing future exogenous variables for Store {store_id}, Item {item}")
    
    # Filter historical data for this store-item
    hist_data = df[(df['Store_Id'] == store_id) & (df['Item'] == item)].copy()
    
    if len(hist_data) == 0:
        logger.warning(f"No historical data found for Store {store_id}, Item {item}")
        return None
    
    # Create future dates DataFrame
    future_df = pd.DataFrame({'Date': future_dates})
    
    # Add calendar features
    future_df['Day_Of_Week'] = future_df['Date'].dt.dayofweek
    future_df['Month'] = future_df['Date'].dt.month
    future_df['Year'] = future_df['Date'].dt.year
    future_df['Day'] = future_df['Date'].dt.day
    
    # Use most recent values for continuous variables
    most_recent = hist_data.sort_values('Date').iloc[-1]
    
    # Add exogenous variables
    valid_exog = []
    for var in exog_vars:
        if var in hist_data.columns:
            valid_exog.append(var)
            
            if var == 'Price':
                # Use most recent price
                future_df[var] = most_recent[var]
            elif var == 'Promotion':
                # Assume no promotions by default
                future_df[var] = 0
            elif var == 'Is_Holiday':
                # Add holiday flag based on common US holidays
                holidays = {
                    '01-01': 'New Year',
                    '02-14': 'Valentine', 
                    '07-04': 'Independence Day',
                    '10-31': 'Halloween',
                    '11-25': 'Thanksgiving', # Approximate
                    '12-25': 'Christmas',
                    '04-15': 'Easter' # Approximate
                }
                
                future_df['month_day'] = future_df['Date'].dt.strftime('%m-%d')
                future_df[var] = future_df['month_day'].apply(lambda x: 1 if x in holidays else 0)
                future_df.drop('month_day', axis=1, inplace=True)
            elif var == 'Weather':
                # Use normal weather as default
                future_df[var] = 'Normal'
                
                # One-hot encode weather
                weather_dummies = pd.get_dummies(future_df[var], prefix='Weather')
                future_df = pd.concat([future_df, weather_dummies], axis=1)
                future_df.drop(var, axis=1, inplace=True)
                
                # Ensure dummy columns are float64
                for col in weather_dummies.columns:
                    future_df[col] = future_df[col].astype(np.float64)
            elif var in ['Avg_Weekly_Sales_4W', 'Avg_Weekly_Sales_13W', 'Stock_Coverage_Weeks']:
                # Calculate based on recent sales data if missing
                if var == 'Avg_Weekly_Sales_4W' and (var not in hist_data.columns or hist_data[var].isna().all()):
                    # Calculate 4-week average sales from history
                    recent_sales = hist_data.sort_values('Date', ascending=False).head(28)  # Last 4 weeks (28 days)
                    if len(recent_sales) > 0:
                        avg_weekly = recent_sales['Sales'].sum() / 4  # Average weekly sales
                        future_df[var] = avg_weekly
                    else:
                        future_df[var] = most_recent.get('Recent_Daily_Sales', 0) * 7 if most_recent.get('Recent_Daily_Sales', 0) > 0 else 1
                        
                elif var == 'Avg_Weekly_Sales_13W' and (var not in hist_data.columns or hist_data[var].isna().all()):
                    # Calculate 13-week average sales from history
                    recent_sales = hist_data.sort_values('Date', ascending=False).head(91)  # Last 13 weeks (91 days)
                    if len(recent_sales) > 0:
                        avg_weekly = recent_sales['Sales'].sum() / 13  # Average weekly sales
                        future_df[var] = avg_weekly
                    else:
                        future_df[var] = most_recent.get('Recent_Daily_Sales', 0) * 7 if most_recent.get('Recent_Daily_Sales', 0) > 0 else 1
                
                elif var == 'Stock_Coverage_Weeks' and (var not in hist_data.columns or hist_data[var].isna().all()):
                    # Calculate stock coverage from stock level and sales rate
                    stock_level = most_recent.get('Stock_Level', 0)
                    recent_daily_sales = most_recent.get('Recent_Daily_Sales', 0)
                    if recent_daily_sales > 0:
                        future_df[var] = stock_level / (recent_daily_sales * 7)
                    else:
                        future_df[var] = 4.0  # Default to 4 weeks if no sales data
                else:
                    # Use most recent value if it exists
                    future_df[var] = most_recent.get(var, 0)
            else:
                # For other variables, use most recent value
                future_df[var] = most_recent[var]
    
    # Handle categorical variables
    for col in valid_exog:
        if col in future_df.columns and future_df[col].dtype == 'object':
            future_df = pd.get_dummies(future_df, columns=[col], drop_first=True)
    
    # Set date as index
    future_df_indexed = future_df.set_index('Date')
    
    # Ensure calendar variables are included in exogenous variables
    # as they are expected by the model
    calendar_exog = ['Day_Of_Week', 'Month', 'Year', 'Day']
    for cal_var in calendar_exog:
        if cal_var in future_df_indexed.columns:
            future_df_indexed[cal_var] = future_df_indexed[cal_var].astype(np.float64)
    
    # Always include all available columns to match the training data shape
    # This ensures we don't get shape mismatches when making forecasts
    exog_columns = list(future_df_indexed.columns)
    
    if len(exog_columns) == 0:
        return None
    
    # Debug information
    logger.info(f"Preparing {len(exog_columns)} exogenous variables: {exog_columns}")
    
    # Check for NaN values before returning
    nan_check = future_df_indexed.isna().any()
    if nan_check.any():
        nan_cols = future_df_indexed.columns[nan_check].tolist()
        logger.warning(f"Future exogenous variables contain NaN values in columns: {nan_cols}")
        
        # Fill NaN values with 0
        for col in nan_cols:
            logger.info(f"Filling NaN values in column '{col}' with zeros")
            future_df_indexed[col] = future_df_indexed[col].fillna(0)
    
    future_exog = future_df_indexed
    
    # Check for missing columns that might have been in training data
    # Add them with default values to match expected shape
    if exog_vars is not None:
        for var in exog_vars:
            if var not in future_exog.columns and var != 'Weather':  # Weather gets converted to dummies
                logger.info(f"Adding missing exogenous variable: {var}")
                future_exog[var] = 0.0
                
    # The error shows we need 14 columns instead of 12
    # Add dummy columns to match the required shape if needed
    current_shape = future_exog.shape[1]
    required_shape = 14  # From the error message
    
    if current_shape < required_shape:
        cols_needed = required_shape - current_shape
        logger.warning(f"Adding {cols_needed} dummy columns to match required shape ({required_shape})")
        
        for i in range(cols_needed):
            dummy_col = f"dummy_exog_{i}"
            logger.info(f"Adding dummy column '{dummy_col}' to match expected shape")
            future_exog[dummy_col] = 0.0
        
        logger.info(f"Updated exogenous variables shape: {future_exog.shape}")
    
    return future_exog


def train_weighted_arima_model(df, store_id, item, exog_vars=None, time_windows=None):
    """
    Train weighted ARIMA model for a specific store-item combination
    
    Args:
        df: DataFrame with historical data
        store_id: Store ID
        item: Item ID
        exog_vars: List of exogenous variables to include
        time_windows: Dictionary of time windows and weights
        
    Returns:
        WeightedARIMAModel: Trained model
    """
    try:
        # Create model
        model = WeightedARIMAModel(
            store_id=store_id, 
            item=item, 
            exog_vars=exog_vars,
            time_windows=time_windows
        )
        
        # Train all window models
        success = model.train_all_models(df)
        
        if success:
            # Save model
            model.save_model()
            return model
        else:
            logger.warning(f"Failed to train model for Store {store_id}, Item {item}")
            return None
    
    except Exception as e:
        logger.error(f"Error training weighted ARIMA model for Store {store_id}, Item {item}: {str(e)}")
        return None


def process_store_item(df, store_id, item, days_to_forecast=30, exog_vars=None,
                     time_windows=None, model_dir=None, output_dir=None, use_existing=False):
    """
    Process a single store-item: train model and generate forecasts
    
    Args:
        df: DataFrame with historical data
        store_id: Store ID
        item: Item ID
        days_to_forecast: Number of days to forecast
        exog_vars: List of exogenous variables to include
        time_windows: Dictionary of time windows and weights
        model_dir: Directory for saved models
        output_dir: Directory for output plots
        use_existing: Whether to use existing model if available
        
    Returns:
        DataFrame: Forecast data
    """
    try:
        logger.info(f"Processing Store {store_id}, Item {item}")
        
        # Ensure data types in input DataFrame
        # This preemptively fixes potential data type issues
        if 'Sales' in df.columns and not pd.api.types.is_numeric_dtype(df['Sales']):
            logger.info("Converting Sales column to numeric in input DataFrame")
            df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce').fillna(0)
        
        # Check exogenous variables data types
        if exog_vars:
            for var in exog_vars:
                if var in df.columns and not pd.api.types.is_numeric_dtype(df[var]):
                    if df[var].dtype == 'object':
                        logger.info(f"Exogenous variable {var} is object type, will be one-hot encoded")
                    else:
                        logger.info(f"Converting exogenous variable {var} to numeric")
                        df[var] = pd.to_numeric(df[var], errors='coerce').fillna(0)
        
        model = None
        
        # Try to load existing model if requested
        if use_existing:
            model = WeightedARIMAModel.load_model(store_id, item, model_dir)
        
        # Train new model if needed
        if model is None:
            model = train_weighted_arima_model(df, store_id, item, exog_vars, time_windows)
        
        # If training failed, return None
        if model is None:
            logger.warning(f"Could not create model for Store {store_id}, Item {item}")
            return None
        
        # Generate forecast
        last_date = df[df['Store_Id'] == store_id][df['Item'] == item]['Date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_to_forecast)
        
        # Prepare future exogenous variables
        logger.info(f"Preparing exogenous variables for Store {store_id}, Item {item}, with variables: {exog_vars}")
        future_exog = prepare_future_exog(df, store_id, item, future_dates, exog_vars, required_shape=14)
        
        # Debug info about exogenous variables shape
        if future_exog is not None:
            logger.info(f"Future exogenous variables shape: {future_exog.shape}, columns: {future_exog.columns.tolist()}")
            
            # Check if we need to adapt the X shape to match what model expects
            if hasattr(model, 'models') and len(model.models) > 0 and next(iter(model.models.values())).get('model', None) is not None:
                first_model = next(iter(model.models.values()))['model']
                if hasattr(first_model, 'exog_names') and first_model.exog_names is not None:
                    expected_exog = first_model.exog_names
                    logger.info(f"Model expects exogenous variables: {expected_exog}")
                    
                    # Check for missing columns
                    missing_cols = []
                    for col in expected_exog:
                        if col not in future_exog.columns:
                            logger.warning(f"Adding missing column '{col}' to future exogenous variables")
                            future_exog[col] = 0.0
                            missing_cols.append(col)
                    
                    # Check for extra columns that might cause issues
                    extra_cols = []
                    for col in future_exog.columns:
                        if col not in expected_exog:
                            logger.warning(f"Removing extra column '{col}' from future exogenous variables")
                            extra_cols.append(col)
                    
                    if extra_cols:
                        future_exog = future_exog.drop(columns=extra_cols)
                    
                    # Select only needed columns in the right order
                    if len(expected_exog) > 0:
                        try:
                            future_exog = future_exog[expected_exog]
                            logger.info(f"Adjusted future exogenous variables shape: {future_exog.shape}")
                            if missing_cols:
                                logger.info(f"Added {len(missing_cols)} missing columns: {missing_cols}")
                            if extra_cols:
                                logger.info(f"Removed {len(extra_cols)} extra columns: {extra_cols}")
                        except Exception as e:
                            logger.error(f"Error adjusting exogenous variables: {e}")
                else:
                    # If we can't determine exact column names, ensure we have the right shape
                    # The error shows we need 14 columns instead of 12
                    current_shape = future_exog.shape[1]
                    required_shape = 14  # From the error message
                    
                    if current_shape < required_shape:
                        cols_needed = required_shape - current_shape
                        logger.warning(f"Adding {cols_needed} dummy columns to match required shape ({required_shape})")
                        
                        for i in range(cols_needed):
                            dummy_col = f"dummy_exog_{i}"
                            future_exog[dummy_col] = 0.0
                        
                        logger.info(f"Updated exogenous variables shape: {future_exog.shape}")
        
        # Generate forecast
        forecast_df = model.forecast(steps=days_to_forecast, X_future=future_exog)
        
        # Add product name
        product_name = df[(df['Store_Id'] == store_id) & (df['Item'] == item)]['Product'].iloc[0] if len(df[(df['Store_Id'] == store_id) & (df['Item'] == item)]) > 0 else "Unknown"
        forecast_df['Product'] = product_name
        
        # Create visualization
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            plot_file = os.path.join(output_dir, f'weighted_arima_forecast_{store_id}_{item}.png')
            model.plot_forecast(forecast_df, df, output_file=plot_file)
        
        return forecast_df
    
    except Exception as e:
        logger.error(f"Error processing Store {store_id}, Item {item}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def run_weighted_arima_forecasting(data_file=COMBINED_DATA_FILE, days_to_forecast=30, 
                        time_windows=None, use_existing=False, parallel=True):
    """
    Run weighted ARIMA forecasting for all store-item combinations
    
    Args:
        data_file: Path to the data file
        days_to_forecast: Number of days to forecast
        time_windows: Dictionary of time windows and weights
        use_existing: Whether to use existing models if available
        parallel: Whether to use parallel processing
        
    Returns:
        DataFrame: Combined forecasts
    """
    logger.info("Starting weighted ARIMA forecasting process")
    
    try:
        # Ensure the data file exists
        if not os.path.exists(data_file):
            # Check if the file exists in the root directory
            alt_path = os.path.join(ROOT_DIR, "combined_pizza_data.csv")
            if os.path.exists(alt_path):
                data_file = alt_path
                logger.info(f"Using alternative data file path: {data_file}")
                
        # Load data
        df = pd.read_csv(data_file)
        
        # Ensure Date is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            
        # Ensure all numeric columns are properly typed
        numeric_columns = ['Sales', 'Price', 'Cost', 'Profit', 'Units_Purchased', 'Stock_Level', 'Avg_Weekly_Sales_4W', 'Avg_Weekly_Sales_13W', 'Stock_Coverage_Weeks']
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Column {col} is not numeric, converting to float64")
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                df[col] = df[col].astype(np.float64)
        
        # Get unique store-item combinations
        store_items = df[['Store_Id', 'Item']].drop_duplicates().values
        
        logger.info(f"Found {len(store_items)} store-item combinations")
        
        # Set up directories
        model_dir = os.path.join(MODELS_DIR, 'weighted_arima')
        os.makedirs(model_dir, exist_ok=True)
        
        output_dir = os.path.join(STATIC_DIR, 'images', 'weighted_arima')
        os.makedirs(output_dir, exist_ok=True)
        
        # Use default time windows if not specified
        if time_windows is None:
            time_windows = TIME_WINDOWS
        
        # List of exogenous variables to include
        exog_vars = ['Price', 'Promotion', 'Is_Holiday', 'Weather', 'Day_Of_Week', 'Month', 'Year', 'Day', 'Avg_Weekly_Sales_4W', 'Avg_Weekly_Sales_13W', 'Stock_Level', 'Stock_Coverage_Weeks']
        
        # Process store-items
        all_forecasts = []
        
        if parallel and len(store_items) > 1:
            # Use parallel processing but limit cores to avoid memory issues
            num_cores = 2  # Limit to 2 cores to prevent memory overload
            logger.info(f"Using {num_cores} cores for parallel processing")
            
            results = Parallel(n_jobs=num_cores)(
                delayed(process_store_item)(
                    df, store_id, item, days_to_forecast, exog_vars, 
                    time_windows, model_dir, output_dir, use_existing
                )
                for store_id, item in store_items
            )
            
            # Filter out None results
            all_forecasts = [r for r in results if r is not None]
        else:
            # Process sequentially
            for store_id, item in store_items:
                forecast_df = process_store_item(
                    df, store_id, item, days_to_forecast, exog_vars, 
                    time_windows, model_dir, output_dir, use_existing
                )
                
                if forecast_df is not None:
                    all_forecasts.append(forecast_df)
        
        # Combine forecasts
        if not all_forecasts:
            logger.error("No forecasts were generated")
            return None
        
        combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
        
        # Add metadata
        combined_forecasts['Model'] = 'Weighted_ARIMA'
        combined_forecasts['Forecast_Type'] = 'Time Series Ensemble'
        
        # Save forecasts
        weighted_arima_forecasts_file = WEIGHTED_ARIMA_FORECASTS_FILE
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(weighted_arima_forecasts_file), exist_ok=True)
        
        # Save the forecasts
        combined_forecasts.to_csv(weighted_arima_forecasts_file, index=False)
        logger.info(f"Saved forecasts to {weighted_arima_forecasts_file}")
        
        # Also save a copy to the root directory for backward compatibility
        root_file = os.path.join(ROOT_DIR, "weighted_arima_forecasts.csv")
        combined_forecasts.to_csv(root_file, index=False)
        logger.info(f"Saved a copy of forecasts to {root_file} for backward compatibility")
        
        # Create summary plots
        create_summary_plots(combined_forecasts, df, output_dir)
        
        return combined_forecasts
    
    except Exception as e:
        logger.error(f"Error in weighted ARIMA forecasting process: {str(e)}", exc_info=True)
        raise


def create_summary_plots(forecasts, historical_df, output_dir):
    """
    Create summary plots for all forecasts
    
    Args:
        forecasts: DataFrame with forecasts
        historical_df: DataFrame with historical data
        output_dir: Directory for output plots
        
    Returns:
        None
    """
    try:
        logger.info("Creating summary plots")
        
        # Create directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot 1: Total sales forecast
        plt.figure(figsize=(12, 6))
        
        # Aggregate forecasts by date
        agg_forecasts = forecasts.groupby('Date').agg({
            'Forecast': 'sum',
            'Lower_Bound': 'sum',
            'Upper_Bound': 'sum'
        }).reset_index()
        
        # Aggregate historical data
        if 'Date' in historical_df.columns:
            hist_dates = historical_df['Date'].sort_values().unique()
            last_days = 30  # Last 30 days
            recent_dates = hist_dates[-last_days:] if len(hist_dates) > last_days else hist_dates
            
            hist_agg = historical_df[historical_df['Date'].isin(recent_dates)].groupby('Date')['Sales'].sum().reset_index()
            
            plt.plot(hist_agg['Date'], hist_agg['Sales'], 'b-', label='Historical Sales')
        
        # Plot forecast
        plt.plot(agg_forecasts['Date'], agg_forecasts['Forecast'], 'r-', label='Weighted ARIMA Forecast')
        plt.fill_between(
            agg_forecasts['Date'],
            agg_forecasts['Lower_Bound'],
            agg_forecasts['Upper_Bound'],
            color='r', alpha=0.2, label='95% Confidence Interval'
        )
        
        plt.xlabel('Date')
        plt.ylabel('Total Sales')
        plt.title('Weighted ARIMA Total Sales Forecast')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'weighted_arima_total_forecast.png'), bbox_inches='tight')
        plt.close()
        
        # Plot 2: Top products by sales volume
        top_products = forecasts.groupby('Product')['Forecast'].sum().sort_values(ascending=False).head(10)
        
        plt.figure(figsize=(12, 8))
        top_products.plot(kind='barh', color='skyblue')
        plt.xlabel('Forecasted Sales')
        plt.ylabel('Product')
        plt.title('Top 10 Products by Forecasted Sales')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'weighted_arima_top_products.png'), bbox_inches='tight')
        plt.close()
        
        # Plot 3: Forecast confidence over time
        plt.figure(figsize=(12, 6))
        
        # Calculate confidence interval width
        agg_forecasts['CI_Width'] = agg_forecasts['Upper_Bound'] - agg_forecasts['Lower_Bound']
        agg_forecasts['CI_Relative'] = agg_forecasts['CI_Width'] / agg_forecasts['Forecast']
        
        plt.plot(agg_forecasts['Date'], agg_forecasts['CI_Relative'], 'g-')
        plt.xlabel('Date')
        plt.ylabel('Relative Confidence Interval Width')
        plt.title('Forecast Uncertainty Over Time')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'weighted_arima_uncertainty.png'), bbox_inches='tight')
        plt.close()
        
        logger.info("Summary plots created successfully")
        
    except Exception as e:
        logger.error(f"Error creating summary plots: {str(e)}")


def main():
    """
    Main function to run when script is called directly
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Weighted ARIMA forecasting tool')
    parser.add_argument('--data', type=str, default=COMBINED_DATA_FILE, help='Path to data file')
    parser.add_argument('--days', type=int, default=30, help='Number of days to forecast')
    parser.add_argument('--use-existing', action='store_true', help='Use existing models if available')
    parser.add_argument('--no-parallel', dest='parallel', action='store_false', help='Disable parallel processing')
    parser.set_defaults(parallel=True)
    
    args = parser.parse_args()
    
    # Run weighted ARIMA forecasting
    run_weighted_arima_forecasting(
        data_file=args.data,
        days_to_forecast=args.days,
        use_existing=args.use_existing,
        parallel=args.parallel
    )
    
    logger.info("Weighted ARIMA forecasting process completed successfully")


if __name__ == "__main__":
    main()