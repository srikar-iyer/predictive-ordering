import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_fix')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.weighted_arima_model import WeightedARIMAModel, prepare_future_exog

def test_fix():
    """Test the fix for the exogenous variable shape mismatch"""
    # Define paths
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "combined_pizza_data.csv")
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    
    # Load data
    if not os.path.exists(DATA_PATH):
        # Try alternate locations
        alt_paths = [
            os.path.join(ROOT_DIR, "combined_pizza_data.csv"),
            os.path.join(ROOT_DIR, "data", "combined_pizza_data.csv")
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                DATA_PATH = alt_path
                break
    
    if not os.path.exists(DATA_PATH):
        logger.error(f"Could not find data file at {DATA_PATH} or alternative locations")
        return
    
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Ensure Date column is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Use the store-item from error message
    store_id = 104.0
    item = 7570630570.0
    
    logger.info(f"Testing forecast for Store {store_id}, Item {item}")
    
    # Load model if available
    model_path = os.path.join(MODELS_DIR, 'weighted_arima', f"weighted_arima_model_{store_id}_{item}.pkl")
    model = WeightedARIMAModel.load_model(store_id, item)
    
    if model is None:
        logger.info("Model not found, creating new one")
        model = WeightedARIMAModel(store_id=store_id, item=item)
        model.train_all_models(df)
    
    # Generate future dates for forecasting
    last_date = df[(df['Store_Id'] == store_id) & (df['Item'] == item)]['Date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)
    
    # Prepare exogenous variables
    exog_vars = ['Price', 'Promotion', 'Is_Holiday', 'Weather', 'Day_Of_Week', 'Month', 'Year', 'Day', 'Avg_Weekly_Sales_4W', 'Avg_Weekly_Sales_13W', 'Stock_Level', 'Stock_Coverage_Weeks']
    
    logger.info(f"Preparing exogenous variables for Store {store_id}, Item {item}")
    future_exog = prepare_future_exog(df, store_id, item, future_dates, exog_vars)
    
    if future_exog is not None:
        logger.info(f"Future exogenous variables shape: {future_exog.shape}")
        logger.info(f"Future exogenous variables columns: {future_exog.columns.tolist()}")
    else:
        logger.error("Failed to prepare future exogenous variables")
        return
    
    # Generate forecast
    try:
        logger.info("Generating forecast with fixed model")
        forecast_df = model.forecast(steps=30, X_future=future_exog)
        logger.info(f"Forecast generated successfully with shape: {forecast_df.shape}")
        logger.info("Fix was successful\!")
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_fix()
