"""
ARIMA model module for sales forecasting.
This module implements ARIMA (AutoRegressive Integrated Moving Average) time series forecasting.
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('arima_model')

# Import settings if available
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from config.settings import (
        COMBINED_DATA_FILE, MODELS_DIR, ARIMA_FORECASTS_FILE, STATIC_DIR
    )
except ImportError:
    # Default paths for backward compatibility
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    COMBINED_DATA_FILE = os.path.join(ROOT_DIR, "data", "processed", "combined_pizza_data.csv")
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    ARIMA_FORECASTS_FILE = os.path.join(ROOT_DIR, "data", "processed", "arima_forecasts.csv")
    STATIC_DIR = os.path.join(ROOT_DIR, "static")


class ARIMAForecaster:
    """
    ARIMA model class for time series forecasting
    """
    def __init__(self, store_id=None, item=None, exog_vars=None):
        """
        Initialize ARIMA forecaster
        
        Args:
            store_id: Store ID
            item: Item ID
            exog_vars: List of exogenous variables to include
        """
        self.store_id = store_id
        self.item = item
        self.exog_vars = exog_vars or ['Price', 'Promotion', 'Is_Holiday']
        self.model = None
        self.best_params = None
        self.scaler = None
        self.training_data = None
        self.is_seasonal = False
        self.seasonal_period = None
        self.forecast_history = {}
    
    def preprocess_data(self, df):
        """
        Preprocess time series data for ARIMA modeling
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            tuple: Processed data, exogenous variables
        """
        logger.info(f"Preprocessing data for Store {self.store_id}, Item {self.item}")
        
        # Filter data for this store-item
        item_df = df[(df['Store_Id'] == self.store_id) & (df['Item'] == self.item)].copy()
        
        # Sort by date
        item_df = item_df.sort_values('Date')
        
        # Convert date to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(item_df['Date']):
            item_df['Date'] = pd.to_datetime(item_df['Date'])
        
        # Set date as index
        item_df = item_df.set_index('Date')
        
        # Get target variable (Sales) and ensure it's float64
        if 'Sales' in item_df.columns:
            y = item_df['Sales']
            # Convert to numeric explicitly if not already
            if not pd.api.types.is_numeric_dtype(y):
                y = pd.to_numeric(y, errors='coerce').fillna(0)
            # Always ensure target is float64
            y = y.astype(np.float64)
        else:
            logger.error(f"'Sales' column not found in data for Store {self.store_id}, Item {self.item}")
            raise ValueError("'Sales' column not found in data")
        
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
        
        # Store mean and std for later denormalization
        self.y_mean = y.mean()
        self.y_std = y.std()
        
        # Check if standard deviation is valid to avoid division by zero
        if self.y_std <= 0:
            self.y_std = 1.0
            logger.warning("Standard deviation is zero or negative, using default value of 1.0")
            
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
                
                # Handle missing values
                X = X.fillna(method='ffill').fillna(method='bfill')
                
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
        
        self.training_data = {
            'y': y,
            'y_normalized': y_normalized,
            'X': X,
            'last_date': item_df.index[-1]
        }
        
        # Check for seasonality
        try:
            # Use at least 2 seasons of data for reliable detection
            min_periods = 14
            if len(y) >= min_periods:
                logger.info("Detecting seasonality...")
                
                # Try different seasonal periods
                for period in [7, 14, 30]:
                    if len(y) >= period * 2:  # Need at least 2 full seasons
                        seasonal_result = seasonal_decompose(y, model='additive', period=period, extrapolate_trend='freq')
                        seasonal_strength = np.std(seasonal_result.seasonal) / np.std(y - seasonal_result.trend)
                        
                        if seasonal_strength > 0.1:  # Significant seasonality
                            self.is_seasonal = True
                            self.seasonal_period = period
                            logger.info(f"Detected seasonality with period {period}")
                            break
        except Exception as e:
            logger.warning(f"Error detecting seasonality: {e}")
        
        logger.info(f"Preprocessing complete for Store {self.store_id}, Item {self.item}")
        return y_normalized, X
    
    def find_optimal_params(self, y, X=None):
        """
        Find optimal ARIMA parameters using auto_arima
        
        Args:
            y: Target time series
            X: Exogenous variables (optional)
            
        Returns:
            dict: Best parameters
        """
        logger.info(f"Finding optimal ARIMA parameters for Store {self.store_id}, Item {self.item}")
        
        try:
            # Ensure data has proper numeric types before passing to auto_arima
            if isinstance(y, pd.Series):
                if y.dtype == 'object' or not pd.api.types.is_numeric_dtype(y):
                    logger.warning("Converting target Series from non-numeric dtype to float64")
                    y = pd.to_numeric(y, errors='coerce').fillna(0)
                # Always convert to float64 to be safe
                y = y.astype(np.float64)
            else:
                logger.warning(f"Converting target to numpy array, current type: {type(y)}")
                y = np.asarray(y, dtype=np.float64)
            
            # Check if y contains any NaN values
            if isinstance(y, pd.Series) and y.isna().any():
                logger.warning("Target series contains NaN values, replacing with zeros")
                y = y.fillna(0)
            elif isinstance(y, np.ndarray) and np.isnan(y).any():
                logger.warning("Target array contains NaN values, replacing with zeros")
                y = np.nan_to_num(y, nan=0.0)
                
            # Check exogenous variables
            if X is not None:
                if isinstance(X, pd.DataFrame):
                    # First check for object dtypes and convert whole DataFrame if needed
                    has_object_cols = any(X[col].dtype == 'object' for col in X.columns)
                    has_non_numeric = any(not pd.api.types.is_numeric_dtype(X[col]) for col in X.columns)
                    
                    if has_object_cols or has_non_numeric:
                        logger.warning("Converting exogenous variables to numeric types")
                        
                        # Process each column individually
                        for col in X.columns:
                            if X[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(X[col]):
                                # Try to convert to numeric first
                                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                        
                        # Convert entire DataFrame to float64
                        X = X.astype(np.float64)
                        
                        # Check for any remaining issues
                        if X.isna().any().any():
                            logger.warning("Exogenous variables contain NaN values after conversion, filling with zeros")
                            X = X.fillna(0)
                elif isinstance(X, np.ndarray):
                    if X.dtype == 'object' or not np.issubdtype(X.dtype, np.number):
                        logger.warning("Converting numpy array from object dtype to float64")
                        X = X.astype(np.float64)
                    
                    # Check for NaN values in numpy array
                    if np.isnan(X).any():
                        logger.warning("Exogenous numpy array contains NaN values, replacing with zeros")
                        X = np.nan_to_num(X, nan=0.0)
            
            # Use auto_arima to find optimal parameters
            if self.is_seasonal:
                # Seasonal ARIMA model
                model = auto_arima(
                    y,
                    exogenous=X,
                    start_p=0, start_q=0,
                    max_p=3, max_q=3, max_d=2,
                    start_P=0, start_Q=0,
                    max_P=2, max_Q=2, max_D=1,
                    m=self.seasonal_period,
                    seasonal=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,
                    trace=False
                )
                
                # Extract parameters
                order = model.order
                seasonal_order = model.seasonal_order
                
                self.best_params = {
                    'order': order,
                    'seasonal_order': seasonal_order
                }
                
                logger.info(f"Best SARIMA parameters: order={order}, seasonal_order={seasonal_order}")
            else:
                # Non-seasonal ARIMA model
                model = auto_arima(
                    y,
                    exogenous=X,
                    start_p=0, start_q=0,
                    max_p=5, max_q=5, max_d=2,
                    seasonal=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,
                    trace=False
                )
                
                # Extract parameters
                order = model.order
                
                self.best_params = {
                    'order': order,
                    'seasonal_order': (0, 0, 0, 0)
                }
                
                logger.info(f"Best ARIMA parameters: order={order}")
            
            return self.best_params
        
        except Exception as e:
            logger.warning(f"Auto parameter selection failed: {str(e)}")
            
            # Use default parameters
            if self.is_seasonal:
                self.best_params = {
                    'order': (1, 1, 1),
                    'seasonal_order': (1, 1, 1, self.seasonal_period or 7)
                }
            else:
                self.best_params = {
                    'order': (1, 1, 1),
                    'seasonal_order': (0, 0, 0, 0)
                }
            
            logger.info(f"Using default parameters: {self.best_params}")
            return self.best_params
    
    def train_model(self, y, X=None):
        """
        Train ARIMA model with given parameters
        
        Args:
            y: Target time series
            X: Exogenous variables (optional)
            
        Returns:
            fitted model
        """
        logger.info(f"Training ARIMA model for Store {self.store_id}, Item {self.item}")
        
        try:
            # If parameters not yet determined, find them
            if self.best_params is None:
                self.find_optimal_params(y, X)
                
            # Additional check to ensure data types are correct before passing to SARIMAX
            # Always convert to numpy arrays with float64 dtype for SARIMAX
            if isinstance(y, pd.Series):
                logger.info("Converting Series to numpy array for SARIMAX")
                y = y.astype(np.float64).values
            elif not isinstance(y, np.ndarray):
                logger.warning(f"Converting y to numpy array, current type: {type(y)}")
                y = np.asarray(y, dtype=np.float64)
            elif y.dtype != np.float64:
                logger.warning(f"Converting numpy array from {y.dtype} to float64")
                y = y.astype(np.float64)
            
            # Check for NaN values
            if np.isnan(y).any():
                logger.warning("Target array contains NaN values, replacing with zeros")
                y = np.nan_to_num(y, nan=0.0)
                
            # Similarly check exogenous variables
            if X is not None:
                if isinstance(X, pd.DataFrame):
                    # Convert DataFrame to numpy array
                    logger.info("Converting exogenous DataFrame to numpy array for SARIMAX")
                    X = X.astype(np.float64).values
                elif not isinstance(X, np.ndarray):
                    logger.warning(f"Converting exogenous variables to numpy array, current type: {type(X)}")
                    X = np.asarray(X, dtype=np.float64)
                elif X.dtype != np.float64:
                    logger.warning(f"Converting exogenous numpy array from {X.dtype} to float64")
                    X = X.astype(np.float64)
                
                # Check for NaN values
                if np.isnan(X).any():
                    logger.warning("Exogenous array contains NaN values, replacing with zeros")
                    X = np.nan_to_num(X, nan=0.0)
            
            # Create and fit model
            model = SARIMAX(
                y,
                exog=X,
                order=self.best_params['order'],
                seasonal_order=self.best_params['seasonal_order'],
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            # Suppress warnings from SARIMAX
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Fit model with appropriate method based on data size
                if len(y) > 1000:
                    fitted_model = model.fit(disp=0, maxiter=50, method='bfgs')
                else:
                    fitted_model = model.fit(disp=0)
            
            self.model = fitted_model
            
            # Store model summary metrics
            self.aic = fitted_model.aic
            self.bic = fitted_model.bic
            
            logger.info(f"Model trained successfully - AIC: {self.aic:.2f}, BIC: {self.bic:.2f}")
            
            return fitted_model
        
        except Exception as e:
            logger.error(f"Error training ARIMA model: {str(e)}")
            raise
    
    def forecast(self, steps, X_future=None, return_conf_int=True, alpha=0.05):
        """
        Generate forecasts from trained model
        
        Args:
            steps: Number of steps to forecast
            X_future: Future exogenous variables
            return_conf_int: Whether to return confidence intervals
            alpha: Significance level for confidence intervals
            
        Returns:
            DataFrame with forecasts
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        logger.info(f"Generating {steps} step forecast for Store {self.store_id}, Item {self.item}")
        
        try:
            # Generate forecast
            if return_conf_int:
                # Get forecast with confidence intervals - remove alpha parameter which causes warnings
                forecast_result = self.model.get_forecast(steps=steps, exog=X_future)
                forecast_mean = forecast_result.predicted_mean
                forecast_ci = forecast_result.conf_int(alpha=alpha)
                
                # Convert to DataFrame
                forecast_index = pd.date_range(start=self.training_data['last_date'] + timedelta(days=1), periods=steps)
                
                # Check if forecast_ci is a DataFrame or array
                if isinstance(forecast_ci, pd.DataFrame):
                    lower_bound = forecast_ci.iloc[:, 0]
                    upper_bound = forecast_ci.iloc[:, 1]
                else:
                    # If it's a numpy array, access columns directly
                    lower_bound = forecast_ci[:, 0]
                    upper_bound = forecast_ci[:, 1]
                
                forecast_df = pd.DataFrame({
                    'Forecast': forecast_mean,
                    'Lower_Bound': lower_bound,
                    'Upper_Bound': upper_bound
                }, index=forecast_index)
            else:
                # Get point forecasts only
                forecast_mean = self.model.forecast(steps=steps, exog=X_future)
                
                # Convert to DataFrame
                forecast_index = pd.date_range(start=self.training_data['last_date'] + timedelta(days=1), periods=steps)
                forecast_df = pd.DataFrame({'Forecast': forecast_mean}, index=forecast_index)
            
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
            
            # Add inventory integration - Stock projection based on forecasted sales
            if 'Stock_Level' in self.training_data and self.training_data['Stock_Level'] is not None:
                last_stock = self.training_data['Stock_Level'][-1] if len(self.training_data['Stock_Level']) > 0 else 0
                
                # Project future stock levels without new purchases
                forecast_df['Projected_Stock'] = last_stock
                for i in range(len(forecast_df)):
                    if i > 0:
                        forecast_df.loc[i, 'Projected_Stock'] = max(0, forecast_df.loc[i-1, 'Projected_Stock'] - forecast_df.loc[i-1, 'Forecast'])
            
            # Store forecast in history
            self.forecast_history[datetime.now()] = forecast_df
            
            logger.info(f"Forecast generated successfully for {steps} days")
            return forecast_df
        
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise
    
    def save_model(self, model_dir=None):
        """
        Save trained model to disk
        
        Args:
            model_dir: Directory to save the model
            
        Returns:
            str: Path to saved model
        """
        if self.model is None:
            raise ValueError("No trained model to save")
        
        if model_dir is None:
            model_dir = os.path.join(MODELS_DIR, 'arima')
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Create filename
        filename = f"arima_model_{self.store_id}_{self.item}.pkl"
        filepath = os.path.join(model_dir, filename)
        
        logger.info(f"Saving model to {filepath}")
        
        try:
            # Save the model and metadata
            with open(filepath, 'wb') as f:
                model_data = {
                    'model': self.model,
                    'best_params': self.best_params,
                    'y_mean': self.y_mean,
                    'y_std': self.y_std,
                    'is_seasonal': self.is_seasonal,
                    'seasonal_period': self.seasonal_period,
                    'store_id': self.store_id,
                    'item': self.item,
                    'exog_vars': self.exog_vars,
                    'training_data': {
                        'last_date': self.training_data['last_date'] if self.training_data else None
                    }
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
            ARIMAForecaster: Loaded model
        """
        if model_dir is None:
            model_dir = os.path.join(MODELS_DIR, 'arima')
        
        # Create filename
        filename = f"arima_model_{store_id}_{item}.pkl"
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
            
            # Create new forecaster
            forecaster = cls(store_id=model_data['store_id'], item=model_data['item'], exog_vars=model_data['exog_vars'])
            
            # Restore model attributes
            forecaster.model = model_data['model']
            forecaster.best_params = model_data['best_params']
            forecaster.y_mean = model_data['y_mean']
            forecaster.y_std = model_data['y_std']
            forecaster.is_seasonal = model_data['is_seasonal']
            forecaster.seasonal_period = model_data['seasonal_period']
            
            # Initialize training data
            forecaster.training_data = {'last_date': model_data['training_data']['last_date']}
            
            logger.info(f"Model loaded from {filepath}")
            return forecaster
        
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
        plt.plot(forecast_df['Date'], forecast_df['Forecast'], 'r-', label='ARIMA Forecast')
        
        # Plot confidence intervals if available
        if 'Lower_Bound' in forecast_df.columns and 'Upper_Bound' in forecast_df.columns:
            plt.fill_between(
                forecast_df['Date'],
                forecast_df['Lower_Bound'],
                forecast_df['Upper_Bound'],
                color='r', alpha=0.2, label='95% Confidence Interval'
            )
        
        # Add inventory projection if available
        if 'Projected_Stock' in forecast_df.columns:
            ax2 = plt.gca().twinx()  # Create second y-axis
            ax2.plot(forecast_df['Date'], forecast_df['Projected_Stock'], 'g-', label='Projected Stock')
            ax2.set_ylabel('Stock Level', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            
            # Add both legends
            lines, labels = plt.gca().get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        else:
            plt.legend()
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title(f'ARIMA Forecast for Store {self.store_id}, Item {self.item}')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save or display plot
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def prepare_future_exog(df, store_id, item, future_dates, exog_vars=None):
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
            else:
                # For other variables, use most recent value
                future_df[var] = most_recent[var]
    
    # Handle categorical variables
    for col in valid_exog:
        if col in future_df.columns and future_df[col].dtype == 'object':
            future_df = pd.get_dummies(future_df, columns=[col], drop_first=True)
    
    # Set date as index
    future_df_indexed = future_df.set_index('Date')
    
    # Filter to include only exogenous variables
    exog_columns = [c for c in future_df_indexed.columns if c not in ['Day_Of_Week', 'Month', 'Year', 'Day']]
    
    if len(exog_columns) == 0:
        return None
    
    future_exog = future_df_indexed[exog_columns]
    
    # Ensure all data is numeric
    for col in future_exog.columns:
        if future_exog[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(future_exog[col]):
            logger.info(f"Converting future exogenous variable {col} to numeric")
            future_exog[col] = pd.to_numeric(future_exog[col], errors='coerce')
            future_exog[col] = future_exog[col].fillna(future_exog[col].mean() if not future_exog[col].isnull().all() else 0)
            # Explicitly convert to float64
            future_exog[col] = future_exog[col].astype(np.float64)
    
    # Convert entire DataFrame to float64 for safety
    future_exog = future_exog.astype(np.float64)
    
    # Final check for any remaining non-numeric data
    try:
        # Test conversion to numpy array
        np.asarray(future_exog, dtype=np.float64)
    except (ValueError, TypeError) as e:
        logger.warning(f"Future exogenous data contains non-numeric values: {e}")
        # Force conversion of all columns to float64
        future_exog = future_exog.astype(np.float64)
    
    return future_exog


def train_arima_model(df, store_id, item, exog_vars=None):
    """
    Train ARIMA model for a specific store-item combination
    
    Args:
        df: DataFrame with historical data
        store_id: Store ID
        item: Item ID
        exog_vars: List of exogenous variables to include
        
    Returns:
        ARIMAForecaster: Trained model
    """
    try:
        # Create forecaster
        forecaster = ARIMAForecaster(store_id=store_id, item=item, exog_vars=exog_vars)
        
        # Preprocess data
        y, X = forecaster.preprocess_data(df)
        
        # Find optimal parameters
        forecaster.find_optimal_params(y, X)
        
        # Train model
        forecaster.train_model(y, X)
        
        # Save model
        forecaster.save_model()
        
        return forecaster
    
    except Exception as e:
        logger.error(f"Error training ARIMA model for Store {store_id}, Item {item}: {str(e)}")
        return None


def process_store_item(df, store_id, item, days_to_forecast=30, exog_vars=None, 
                     model_dir=None, output_dir=None, use_existing=False):
    """
    Process a single store-item: train model and generate forecasts
    
    Args:
        df: DataFrame with historical data
        store_id: Store ID
        item: Item ID
        days_to_forecast: Number of days to forecast
        exog_vars: List of exogenous variables to include
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
            # Ensure float64 type explicitly
            df['Sales'] = df['Sales'].astype(np.float64)
        
        # Check exogenous variables data types
        if exog_vars:
            for var in exog_vars:
                if var in df.columns and not pd.api.types.is_numeric_dtype(df[var]):
                    if df[var].dtype == 'object':
                        logger.info(f"Exogenous variable {var} is object type, will be one-hot encoded")
                    else:
                        logger.info(f"Converting exogenous variable {var} to numeric")
                        df[var] = pd.to_numeric(df[var], errors='coerce').fillna(0)
        
        forecaster = None
        
        # Try to load existing model if requested
        if use_existing:
            forecaster = ARIMAForecaster.load_model(store_id, item, model_dir)
        
        # Train new model if needed
        if forecaster is None:
            forecaster = train_arima_model(df, store_id, item, exog_vars)
        
        # If training failed, return None
        if forecaster is None:
            logger.warning(f"Could not create model for Store {store_id}, Item {item}")
            return None
        
        # Generate forecast
        last_date = df[df['Store_Id'] == store_id][df['Item'] == item]['Date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_to_forecast)
        
        # Prepare future exogenous variables
        try:
            future_exog = prepare_future_exog(df, store_id, item, future_dates, exog_vars)
            
            # Additional safety check for future_exog data types
            if future_exog is not None:
                # Ensure all columns are float64
                future_exog = future_exog.astype(np.float64)
                
                # Verify convertibility to numpy array
                try:
                    np.asarray(future_exog, dtype=np.float64)
                except Exception as e:
                    logger.warning(f"Converting future_exog failed: {e}. Setting to None.")
                    future_exog = None
                    
        except Exception as e:
            logger.error(f"Error preparing future exogenous variables: {e}")
            future_exog = None
        
        # Generate forecast
        forecast_df = forecaster.forecast(steps=days_to_forecast, X_future=future_exog)
        
        # Add product name
        product_name = df[(df['Store_Id'] == store_id) & (df['Item'] == item)]['Product'].iloc[0] if len(df[(df['Store_Id'] == store_id) & (df['Item'] == item)]) > 0 else "Unknown"
        forecast_df['Product'] = product_name
        
        # Create visualization
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            plot_file = os.path.join(output_dir, f'arima_forecast_{store_id}_{item}.png')
            forecaster.plot_forecast(forecast_df, df, output_file=plot_file)
        
        return forecast_df
    
    except Exception as e:
        logger.error(f"Error processing Store {store_id}, Item {item}: {str(e)}")
        return None


def run_arima_forecasting(data_file=COMBINED_DATA_FILE, days_to_forecast=30, 
                        use_existing=False, parallel=True):
    """
    Run ARIMA forecasting for all store-item combinations
    
    Args:
        data_file: Path to the data file
        days_to_forecast: Number of days to forecast
        use_existing: Whether to use existing models if available
        parallel: Whether to use parallel processing
        
    Returns:
        DataFrame: Combined forecasts
    """
    logger.info("Starting ARIMA forecasting process")
    
    try:
        # Load data
        df = pd.read_csv(data_file)
        
        # Ensure Date is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            
        # Ensure all numeric columns are properly typed
        numeric_columns = ['Sales', 'Price', 'Cost', 'Profit', 'Units_Purchased', 'Stock_Level']
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Column {col} is not numeric, converting to float64")
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                df[col] = df[col].astype(np.float64)
        
        # Get unique store-item combinations
        store_items = df[['Store_Id', 'Item']].drop_duplicates().values
        
        logger.info(f"Found {len(store_items)} store-item combinations")
        
        # Set up directories
        model_dir = os.path.join(MODELS_DIR, 'arima')
        os.makedirs(model_dir, exist_ok=True)
        
        output_dir = os.path.join(STATIC_DIR, 'images', 'arima')
        os.makedirs(output_dir, exist_ok=True)
        
        # List of exogenous variables to include
        exog_vars = ['Price', 'Promotion', 'Is_Holiday', 'Weather']
        
        # Ensure all columns are numeric before processing
        for col in df.columns:
            if col not in ['Date', 'Product', 'Store_Id', 'Item'] and df[col].dtype == 'object':
                try:
                    # Try to convert non-numeric columns to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    df[col] = df[col].astype(np.float64)
                    logger.info(f"Converted column {col} to numeric type")
                except Exception as e:
                    logger.warning(f"Could not convert column {col} to numeric: {e}")
                    
        # Process store-items
        all_forecasts = []
        
        if parallel and len(store_items) > 1:
            # Use parallel processing
            num_cores = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
            logger.info(f"Using {num_cores} cores for parallel processing")
            
            results = Parallel(n_jobs=num_cores)(
                delayed(process_store_item)(
                    df, store_id, item, days_to_forecast, exog_vars, 
                    model_dir, output_dir, use_existing
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
                    model_dir, output_dir, use_existing
                )
                
                if forecast_df is not None:
                    all_forecasts.append(forecast_df)
        
        # Combine forecasts
        if not all_forecasts:
            logger.error("No forecasts were generated")
            return None
        
        combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
        
        # Add metadata
        combined_forecasts['Model'] = 'ARIMA'
        combined_forecasts['Forecast_Type'] = 'Time Series'
        
        # Save forecasts
        arima_forecasts_file = os.path.join(os.path.dirname(ARIMA_FORECASTS_FILE), 'arima_forecasts.csv')
        combined_forecasts.to_csv(arima_forecasts_file, index=False)
        logger.info(f"Saved forecasts to {arima_forecasts_file}")
        
        # Create summary plots
        create_summary_plots(combined_forecasts, df, output_dir)
        
        return combined_forecasts
    
    except Exception as e:
        logger.error(f"Error in ARIMA forecasting process: {str(e)}", exc_info=True)
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
        plt.plot(agg_forecasts['Date'], agg_forecasts['Forecast'], 'r-', label='ARIMA Forecast')
        plt.fill_between(
            agg_forecasts['Date'],
            agg_forecasts['Lower_Bound'],
            agg_forecasts['Upper_Bound'],
            color='r', alpha=0.2, label='95% Confidence Interval'
        )
        
        plt.xlabel('Date')
        plt.ylabel('Total Sales')
        plt.title('ARIMA Total Sales Forecast')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'arima_total_forecast.png'), bbox_inches='tight')
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
        
        plt.savefig(os.path.join(output_dir, 'arima_top_products.png'), bbox_inches='tight')
        plt.close()
        
        logger.info("Summary plots created successfully")
        
    except Exception as e:
        logger.error(f"Error creating summary plots: {str(e)}")


def main():
    """
    Main function to run when script is called directly
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='ARIMA forecasting tool')
    parser.add_argument('--data', type=str, default=COMBINED_DATA_FILE, help='Path to data file')
    parser.add_argument('--days', type=int, default=30, help='Number of days to forecast')
    parser.add_argument('--use-existing', action='store_true', help='Use existing models if available')
    parser.add_argument('--no-parallel', dest='parallel', action='store_false', help='Disable parallel processing')
    parser.set_defaults(parallel=True)
    
    args = parser.parse_args()
    
    # Run ARIMA forecasting
    run_arima_forecasting(
        data_file=args.data,
        days_to_forecast=args.days,
        use_existing=args.use_existing,
        parallel=args.parallel
    )
    
    logger.info("ARIMA forecasting process completed successfully")


if __name__ == "__main__":
    main()