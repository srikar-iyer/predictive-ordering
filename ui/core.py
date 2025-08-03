"""
Core UI components and utilities for the Pizza Predictive Ordering dashboard.
This module contains shared components and utilities used across the UI.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import os
import logging
from datetime import datetime
import base64
from io import BytesIO
import pathlib
import time
from typing import Dict, List, Tuple, Optional, Union, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ui_core')

# Import settings if available
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.settings import (
        COMBINED_DATA_FILE, PYTORCH_FORECASTS_FILE, RF_FORECASTS_FILE,
        INVENTORY_RECOMMENDATIONS_FILE, OPTIMIZED_ORDERS_FILE, INVENTORY_PROJECTION_FILE,
        PRICE_ELASTICITIES_FILE, PRICE_RECOMMENDATIONS_FILE, PROFIT_IMPACT_FILE,
        PRODUCT_MIX_FILE, STATIC_DIR
    )
except ImportError:
    # Default paths for backward compatibility
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Try data/processed paths first, then fall back to root if needed
    COMBINED_DATA_FILE = os.path.join(ROOT_DIR, "data", "processed", "combined_pizza_data.csv")
    PYTORCH_FORECASTS_FILE = os.path.join(ROOT_DIR, "data", "processed", "pytorch_forecasts.csv")
    RF_FORECASTS_FILE = os.path.join(ROOT_DIR, "data", "processed", "rf_forecasts.csv")
    INVENTORY_RECOMMENDATIONS_FILE = os.path.join(ROOT_DIR, "data", "processed", "inventory", "inventory_recommendations.csv")
    OPTIMIZED_ORDERS_FILE = os.path.join(ROOT_DIR, "data", "processed", "inventory", "optimized_orders.csv")
    INVENTORY_PROJECTION_FILE = os.path.join(ROOT_DIR, "data", "processed", "inventory", "inventory_projection.csv")
    PRICE_ELASTICITIES_FILE = os.path.join(ROOT_DIR, "data", "processed", "price_elasticities.csv")
    PRICE_RECOMMENDATIONS_FILE = os.path.join(ROOT_DIR, "data", "processed", "price_recommendations.csv")
    PROFIT_IMPACT_FILE = os.path.join(ROOT_DIR, "data", "processed", "profit_impact.csv")
    PRODUCT_MIX_FILE = os.path.join(ROOT_DIR, "data", "processed", "product_mix_optimization.csv")
    
    # Fallback to root directory files if needed
    if not os.path.exists(COMBINED_DATA_FILE):
        COMBINED_DATA_FILE = os.path.join(ROOT_DIR, "combined_pizza_data.csv")
    if not os.path.exists(PYTORCH_FORECASTS_FILE):
        PYTORCH_FORECASTS_FILE = os.path.join(ROOT_DIR, "pytorch_forecasts.csv")
    if not os.path.exists(RF_FORECASTS_FILE):
        RF_FORECASTS_FILE = os.path.join(ROOT_DIR, "rf_forecasts.csv")
    if not os.path.exists(INVENTORY_RECOMMENDATIONS_FILE):
        INVENTORY_RECOMMENDATIONS_FILE = os.path.join(ROOT_DIR, "inventory_recommendations.csv")
    if not os.path.exists(OPTIMIZED_ORDERS_FILE):
        OPTIMIZED_ORDERS_FILE = os.path.join(ROOT_DIR, "optimized_orders.csv")
    if not os.path.exists(INVENTORY_PROJECTION_FILE):
        INVENTORY_PROJECTION_FILE = os.path.join(ROOT_DIR, "inventory_projection.csv")
    if not os.path.exists(PRICE_ELASTICITIES_FILE):
        PRICE_ELASTICITIES_FILE = os.path.join(ROOT_DIR, "price_elasticities.csv")
    if not os.path.exists(PRICE_RECOMMENDATIONS_FILE):
        PRICE_RECOMMENDATIONS_FILE = os.path.join(ROOT_DIR, "price_recommendations.csv")
    if not os.path.exists(PROFIT_IMPACT_FILE):
        PROFIT_IMPACT_FILE = os.path.join(ROOT_DIR, "profit_impact.csv")
    if not os.path.exists(PRODUCT_MIX_FILE):
        PRODUCT_MIX_FILE = os.path.join(ROOT_DIR, "product_mix_optimization.csv")
    STATIC_DIR = os.path.join(ROOT_DIR, "static")

# Try to import the weather service
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.services.weather_service import WeatherService
except ImportError:
    try:
        from weather_service import WeatherService
    except ImportError:
        logger.warning("Could not import WeatherService. Weather features will be disabled.")
        WeatherService = None


# Data caching system to avoid repeated file reads
_DATA_CACHE = {}
_DATA_CACHE_TIMESTAMPS = {}
DATA_CACHE_EXPIRY = 900  # 15 minutes in seconds

def load_data_with_fallback(file_path, fallback_path=None, required_columns=None, use_cache=True, cache_key=None):
    """
    Load data from a CSV file with fallback option and data validation.
    
    Args:
        file_path: Path to the primary CSV file
        fallback_path: Path to the fallback CSV file
        required_columns: List of columns that must be present in the dataframe
        use_cache: Whether to use the cache system
        cache_key: Custom cache key to use instead of file_path
        
    Returns:
        DataFrame or None
    """
    # Use the file path as the cache key if none provided
    if cache_key is None:
        cache_key = str(file_path)
    
    # Check cache first if enabled
    if use_cache and cache_key in _DATA_CACHE:
        # Check if cache is still valid
        if time.time() - _DATA_CACHE_TIMESTAMPS.get(cache_key, 0) < DATA_CACHE_EXPIRY:
            logger.info(f"Using cached data for {cache_key}")
            try:
                # Validate cached data still has required columns (schema might have changed)
                cached_df = _DATA_CACHE[cache_key]
                if required_columns:
                    missing = [col for col in required_columns if col not in cached_df.columns]
                    if not missing:
                        return cached_df
                    else:
                        logger.warning(f"Cached data for {cache_key} is missing required columns: {missing}. Reloading.")
                else:
                    return cached_df
            except Exception as cache_error:
                logger.error(f"Error validating cached data for {cache_key}: {cache_error}")
        else:
            # Cache expired
            logger.info(f"Cache expired for {cache_key}, reloading")
    
    # Convert paths to pathlib.Path objects
    try:
        file_path = pathlib.Path(file_path)
        if fallback_path:
            fallback_path = pathlib.Path(fallback_path)
    except Exception as path_error:
        logger.error(f"Invalid path format: {path_error}")
        file_path = pathlib.Path(str(file_path))
        if fallback_path:
            fallback_path = pathlib.Path(str(fallback_path))
        
    # Try to load from primary path
    df = None
    errors = []
    
    try:
        if file_path.exists():
            # Use a more robust CSV reading approach with error handling
            try:
                df = pd.read_csv(file_path, on_bad_lines='warn')
                logger.info(f"Successfully loaded {file_path} with {len(df)} rows")
            except Exception as read_error:
                # Try with different encoding if default fails
                try:
                    logger.warning(f"Retrying with UTF-8 encoding: {file_path}")
                    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                except Exception:
                    try:
                        logger.warning(f"Retrying with latin1 encoding: {file_path}")
                        df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')
                    except Exception as final_error:
                        logger.error(f"All CSV loading attempts failed for {file_path}: {final_error}")
                        errors.append(f"CSV parsing error: {final_error}")
                        df = None
            
            # Process dates if the DataFrame was loaded successfully
            if df is not None and 'Date' in df.columns:
                try:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    # Drop rows with invalid dates
                    invalid_dates = df['Date'].isna().sum()
                    if invalid_dates > 0:
                        logger.warning(f"Dropped {invalid_dates} rows with invalid dates from {file_path}")
                        df.dropna(subset=['Date'], inplace=True)
                        
                    # Check if we have any data left after dropping invalid dates
                    if len(df) == 0:
                        logger.error(f"No valid data remains after dropping rows with invalid dates from {file_path}")
                        df = None
                        errors.append("No valid data after date parsing")
                except Exception as date_error:
                    logger.error(f"Error processing dates in {file_path}: {date_error}")
            
            # Validate required columns
            if df is not None and required_columns:
                missing = [col for col in required_columns if col not in df.columns]
                if missing:
                    logger.error(f"Missing required columns in {file_path}: {missing}")
                    errors.append(f"Missing columns: {missing}")
                    
                    # Try to auto-fix common column naming issues
                    renamed = False
                    for required_col in missing:
                        # Check for alternate column names (lowercase, uppercase, underscores, spaces)
                        alternates = [
                            required_col.lower(),
                            required_col.upper(),
                            required_col.replace('_', ' '),
                            required_col.replace(' ', '_')
                        ]
                        
                        for alt in alternates:
                            if alt in df.columns:
                                logger.info(f"Auto-fixing column name: '{alt}' -> '{required_col}'")
                                df = df.rename(columns={alt: required_col})
                                renamed = True
                                break
                    
                    # If we couldn't fix all the missing columns, set df to None
                    if not renamed or any(col not in df.columns for col in required_columns):
                        df = None
        else:
            logger.warning(f"File not found: {file_path}")
            errors.append("File not found")
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        errors.append(str(e))
        df = None
    
    # Try fallback if needed
    if df is None and fallback_path:
        try:
            if fallback_path.exists():
                # Try multiple approaches for fallback file too
                try:
                    df = pd.read_csv(fallback_path, on_bad_lines='warn')
                except Exception:
                    try:
                        df = pd.read_csv(fallback_path, encoding='utf-8', on_bad_lines='skip')
                    except Exception:
                        df = pd.read_csv(fallback_path, encoding='latin1', on_bad_lines='skip')
                
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    # Drop rows with invalid dates
                    invalid_dates = df['Date'].isna().sum()
                    if invalid_dates > 0:
                        logger.warning(f"Dropped {invalid_dates} rows with invalid dates from {fallback_path}")
                        df.dropna(subset=['Date'], inplace=True)
                logger.info(f"Using fallback {fallback_path} with {len(df)} rows")
                
                # Validate required columns
                if required_columns:
                    missing = [col for col in required_columns if col not in df.columns]
                    if missing:
                        logger.error(f"Missing required columns in fallback {fallback_path}: {missing}")
                        # Try one more auto-fix attempt
                        renamed = False
                        for required_col in missing:
                            alternates = [required_col.lower(), required_col.upper(), 
                                         required_col.replace('_', ' '), required_col.replace(' ', '_')]
                            for alt in alternates:
                                if alt in df.columns:
                                    df = df.rename(columns={alt: required_col})
                                    renamed = True
                                    break
                        
                        # If still missing after auto-fix attempts, set df to None
                        if not renamed or any(col not in df.columns for col in required_columns):
                            df = None
            else:
                logger.warning(f"Fallback file not found: {fallback_path}")
        except Exception as e2:
            logger.error(f"Error loading fallback {fallback_path}: {e2}")
            df = None
    
    # Try loading from data/raw directory if both primary and fallback paths failed
    if df is None:
        try:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            raw_dir = os.path.join(root_dir, "data", "raw")
            
            # Extract the file name from the path
            file_name = os.path.basename(file_path)
            raw_file_path = os.path.join(raw_dir, file_name)
            
            if os.path.exists(raw_file_path):
                logger.info(f"Attempting to load from data/raw directory: {raw_file_path}")
                df = pd.read_csv(raw_file_path, on_bad_lines='skip')
                
                # Process dates
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df.dropna(subset=['Date'], inplace=True)
                
                logger.info(f"Successfully loaded from raw directory: {raw_file_path} with {len(df)} rows")
        except Exception as raw_error:
            logger.error(f"Error loading from raw directory: {raw_error}")
    
    # Create synthetic data as last resort if required columns are provided
    if df is None and required_columns:
        logger.warning(f"Creating synthetic data for {cache_key} as last resort")
        try:
            df = create_synthetic_data(required_columns, cache_key)
            logger.info(f"Created synthetic data with {len(df)} rows for {cache_key}")
        except Exception as e:
            logger.error(f"Failed to create synthetic data: {e}")
            # Create a minimal DataFrame with the required columns as a fallback
            try:
                logger.info("Creating minimal emergency DataFrame")
                minimal_data = {}
                for col in required_columns:
                    if col == 'Date':
                        minimal_data[col] = [pd.Timestamp.now()]
                    elif col in ['Store_Id', 'Item']:
                        minimal_data[col] = ['1']
                    else:
                        minimal_data[col] = [0]
                df = pd.DataFrame(minimal_data)
            except Exception as last_resort_error:
                logger.error(f"Could not create minimal DataFrame: {last_resort_error}")
    
    # Perform final validation and cleanup on loaded data
    if df is not None:
        # Remove duplicates if any
        original_len = len(df)
        df = df.drop_duplicates()
        if len(df) < original_len:
            logger.info(f"Removed {original_len - len(df)} duplicate rows")
        
        # Check for and fix common data issues
        try:
            # Convert numeric columns that might be strings
            for col in df.columns:
                if col not in ['Date', 'Product', 'Product_Name', 'Item_Description', 'Weather', 'Holiday_Name']:
                    try:
                        if df[col].dtype == 'object':
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception:
                        pass
            
            # Replace NaN values with appropriate defaults
            df = df.fillna({
                'Sales': 0, 
                'Price': 0, 
                'Cost': 0, 
                'Stock_Level': 0,
                'Units_Sold': 0,
                'Units_Purchased': 0
            })
        except Exception as cleanup_error:
            logger.warning(f"Error during data cleanup: {cleanup_error}")
    
    # Update cache if data was successfully loaded and caching is enabled
    if df is not None and use_cache:
        _DATA_CACHE[cache_key] = df
        _DATA_CACHE_TIMESTAMPS[cache_key] = time.time()
    
    return df


def create_synthetic_data(columns, data_type):
    """
    Create synthetic data for demo purposes when real data is unavailable.
    
    Args:
        columns: List of required columns
        data_type: Type of data to generate (used to determine appropriate values)
        
    Returns:
        DataFrame with synthetic data
    """
    logger.info(f"Generating synthetic data for {data_type} with columns: {columns}")
    rows = 100  # Default number of synthetic rows
    
    # Create a base dataframe with the required columns
    df = pd.DataFrame(columns=columns)
    
    # Generate appropriate data based on column names
    for col in columns:
        if col.lower() in ['date', 'proc_date']:
            df[col] = pd.date_range(start='2023-01-01', periods=rows)
        elif col.lower() in ['store_id', 'item', 'id']:
            df[col] = np.random.randint(1, 10, size=rows)
        elif 'price' in col.lower() or 'cost' in col.lower() or 'revenue' in col.lower() or 'retail' in col.lower():
            df[col] = np.random.uniform(5, 50, size=rows).round(2)
        elif 'unit' in col.lower() or 'sales' in col.lower() or 'quantity' in col.lower() or 'stock' in col.lower():
            df[col] = np.random.randint(0, 100, size=rows)
        elif 'product' in col.lower() or 'name' in col.lower() or 'description' in col.lower():
            products = ['Pepperoni', 'Cheese', 'Veggie', 'Supreme', 'Hawaiian']
            df[col] = np.random.choice(products, size=rows)
        else:
            # Generic numeric data for unknown columns
            df[col] = np.random.uniform(0, 10, size=rows).round(2)
    
    logger.info(f"Generated {len(df)} rows of synthetic data")
    return df


def apply_stock_adjustments(data_frame, store, product, stock_adjustment=None, adjust_clicks=None, adjustment_date=None, manual_adjustments=None):
    """
    Apply stock adjustments to a dataset in a consistent way across all visualizations
    
    Args:
        data_frame: DataFrame to adjust
        store: Store ID
        product: Product ID
        stock_adjustment: New stock level value
        adjust_clicks: Button click counter to trigger adjustment
        adjustment_date: Specific date for the adjustment
        manual_adjustments: Dictionary to store adjustments
        
    Returns:
        tuple: (Adjusted DataFrame, bool indicating if update was applied)
    """
    if data_frame is None or len(data_frame) == 0 or store is None or product is None:
        return data_frame, False
    
    # Make a copy to avoid modifying the original
    df = data_frame.copy()
    
    # Create an adjustment key from store and product
    adjustment_key = f"{store}_{product}"
    
    # Apply manual stock adjustment if provided
    # This is triggered by a click on the adjust button
    adjustment_applied = False
    
    if manual_adjustments is not None:
        # If no date provided, use the current date for single adjustments
        if adjustment_date is None:
            if stock_adjustment is not None and adjust_clicks is not None:
                # Record the adjustment in the state dictionary
                manual_adjustments[adjustment_key] = int(stock_adjustment)
                adjustment_applied = True
        
            # Apply all stored adjustments to the data
            for key, value in manual_adjustments.items():
                if value is not None:
                    # Parse the key to get store and item IDs
                    try:
                        adj_store, adj_item = key.split('_')
                        adj_store = int(adj_store)
                        adj_item = float(adj_item)  # Item might be a float in some datasets
                        
                        # Apply the adjustment to the stock level
                        stock_col = 'Stock_Level' if 'Stock_Level' in df.columns else 'Current_Stock'
                        if stock_col in df.columns:
                            # Update all records for this store-item combination
                            mask = (df['Store_Id'] == adj_store) & (df['Item'] == adj_item)
                            if len(df[mask]) > 0:
                                # Set the stock level to the adjustment value
                                df.loc[mask, stock_col] = value
                                logger.info(f"Applied stock adjustment: Store {adj_store}, Item {adj_item}, New level: {value}")
                    except Exception as e:
                        logger.error(f"Error applying stock adjustment for {key}: {str(e)}")
        else:
            # Handle date-specific adjustments
            if stock_adjustment is not None and adjust_clicks is not None:
                # Create a dictionary for this store/product if it doesn't exist
                if adjustment_key not in manual_adjustments:
                    manual_adjustments[adjustment_key] = {}
                
                # Record the adjustment for this date
                manual_adjustments[adjustment_key][adjustment_date] = int(stock_adjustment)
                adjustment_applied = True
            
            # Apply all stored date-specific adjustments
            for key, date_adjustments in manual_adjustments.items():
                if date_adjustments and isinstance(date_adjustments, dict):
                    # Parse the key to get store and item IDs
                    try:
                        adj_store, adj_item = key.split('_')
                        adj_store = int(adj_store)
                        adj_item = float(adj_item)
                        
                        # Apply each date-specific adjustment
                        for adj_date, adj_value in date_adjustments.items():
                            if adj_value is not None:
                                # Parse the date
                                parsed_date = pd.to_datetime(adj_date).date()
                                
                                # Apply the adjustment to records on or after this date
                                stock_col = 'Stock_Level' if 'Stock_Level' in df.columns else 'Current_Stock'
                                if stock_col in df.columns and 'Date' in df.columns:
                                    # Match records for this store-item on or after the adjustment date
                                    mask = (df['Store_Id'] == adj_store) & (df['Item'] == adj_item) & (df['Date'].dt.date >= parsed_date)
                                    if len(df[mask]) > 0:
                                        # Set the stock level to the adjustment value for the first matched record
                                        first_idx = df[mask].index[0]
                                        df.loc[first_idx, stock_col] = adj_value
                                        logger.info(f"Applied date-specific stock adjustment: Store {adj_store}, Item {adj_item}, Date {adj_date}, New level: {adj_value}")
                                        
                                        # Update subsequent records if needed (e.g., for projections)
                                        # This would depend on the specific business logic
                    except Exception as e:
                        logger.error(f"Error applying date-specific stock adjustment for {key}: {str(e)}")
    
    return df, adjustment_applied


def format_product_name(product_name, item_number, show_item_number=True):
    """
    Format product name with or without item number.
    
    Args:
        product_name: Raw product name
        item_number: Item number/ID
        show_item_number: Whether to include the item number
        
    Returns:
        str: Formatted product name
    """
    if show_item_number:
        return f"{product_name} ({item_number})"
    else:
        return product_name


def get_store_product_options(data_df, item_id_col='Item', store_id_col='Store_Id', product_name_col='Product'):
    """
    Generate store and product dropdown options from data.
    
    Args:
        data_df: DataFrame with store and product data
        item_id_col: Column name for item ID
        store_id_col: Column name for store ID
        product_name_col: Column name for product name
        
    Returns:
        tuple: (store_options, product_options)
    """
    if data_df is None or len(data_df) == 0:
        return [], []
    
    try:
        # Get unique stores
        stores = data_df[store_id_col].unique()
        store_options = [{'label': f'Store {s}', 'value': s} for s in sorted(stores)]
        
        # Get unique products with names
        products = data_df[[item_id_col, product_name_col]].drop_duplicates()
        product_options = [{'label': f"{row[product_name_col]} ({row[item_id_col]})", 'value': row[item_id_col]} 
                        for _, row in products.iterrows()]
        
        return store_options, product_options
    
    except Exception as e:
        logger.error(f"Error generating store/product options: {str(e)}")
        return [], []


def create_date_range_slider(id_prefix, min_date=None, max_date=None, start_date=None, end_date=None):
    """
    Create an enhanced date range slider component with better visual styling.
    
    Args:
        id_prefix: Prefix for the component ID
        min_date: Minimum date for the slider
        max_date: Maximum date for the slider
        start_date: Initial start date
        end_date: Initial end date
        
    Returns:
        dbc.Card: Date range slider component in a styled card
    """
    if min_date is None:
        min_date = datetime.now().date()
    if max_date is None:
        max_date = (datetime.now() + pd.Timedelta(days=30)).date()
    if start_date is None:
        start_date = min_date
    if end_date is None:
        end_date = max_date
    
    # Create the date range card with enhanced styling and mobile responsiveness
    return dbc.Card([
        dbc.CardHeader(
            html.H5([
                html.I(className="fas fa-calendar-alt me-2"),
                "Time Period Selection"
            ], className="m-0 fs-6 fs-md-5"),
            style={
                "backgroundColor": "white",
                "borderBottom": "1px solid #eaecef"
            }
        ),
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.I(className="fas fa-calendar me-2", style={"color": "#3498db"}),
                    "Select Date Range:"
                ], style={
                    "fontWeight": "600",
                    "marginBottom": "0.75rem",
                    "color": "#2c3e50",
                    "display": "flex",
                    "alignItems": "center",
                    "fontSize": "0.9rem"
                }),
                # Using a responsive layout for date picker
                html.Div(
                    dcc.DatePickerRange(
                        id=f"{id_prefix}-date-range",
                        min_date_allowed=min_date,
                        max_date_allowed=max_date,
                        start_date=start_date,
                        end_date=end_date,
                        display_format='YYYY-MM-DD',
                        className="shadow-sm date-picker-responsive",
                        style={
                            "zIndex": "1000",
                            "width": "100%"
                        }
                    ),
                    className="date-range-container"
                )
            ])
        ], className="px-3 py-3 px-md-4")
    ], style={
        "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.07)",
        "border": "none",
        "borderRadius": "6px"
    }, className="mb-3 mb-md-4 date-range-card")


def create_store_product_selectors(id_prefix, store_options, product_options, default_store=None, default_product=None):
    """
    Create store and product selector dropdowns with enhanced styling.
    
    Args:
        id_prefix: Prefix for component IDs
        store_options: List of store options
        product_options: List of product options
        default_store: Default store value
        default_product: Default product value
        
    Returns:
        dbc.Row: Row with store and product selector dropdowns
    """
    label_style = {
        "fontWeight": "600",
        "marginBottom": "0.5rem",
        "color": "#2c3e50",
        "display": "flex",
        "alignItems": "center",
        "fontSize": "0.9rem"
    }
    
    dropdown_style = {
        "border": "1px solid #ced4da",
        "borderRadius": "6px"
    }
    
    return dbc.Card([
        dbc.CardHeader(
            html.H5([
                html.I(className="fas fa-filter me-2"), 
                "Data Selection"
            ], className="m-0 fs-6 fs-md-5"),
            style={
                "backgroundColor": "white",
                "borderBottom": "1px solid #eaecef"
            }
        ),
        dbc.CardBody([
            dbc.Row([
                # Stack vertically on mobile, side by side on larger screens
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-store me-2", style={"color": "#3498db"}),
                        "Select Store:"
                    ], style=label_style),
                    dcc.Dropdown(
                        id=f"{id_prefix}-store-dropdown",
                        options=store_options,
                        value=default_store if default_store is not None else store_options[0]['value'] if store_options else None,
                        clearable=False,
                        style=dropdown_style,
                        className="shadow-sm mb-3 mb-md-0"
                    )
                ], xs=12, md=6, className="mb-3 mb-md-0"),
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-pizza-slice me-2", style={"color": "#e74c3c"}),
                        "Select Product:"
                    ], style=label_style),
                    dcc.Dropdown(
                        id=f"{id_prefix}-product-dropdown",
                        options=product_options,
                        value=default_product if default_product is not None else product_options[0]['value'] if product_options else None,
                        clearable=False,
                        style=dropdown_style,
                        className="shadow-sm"
                    )
                ], xs=12, md=6)
            ])
        ], className="px-3 py-3 px-md-4")
    ], style={
        "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.07)",
        "border": "none",
        "borderRadius": "6px"
    }, className="mb-3 mb-md-4 selector-card")


def create_toggle_switch(id_prefix, label, value=False, icon=None, description=None):
    """
    Create an enhanced toggle switch component with optional icon and description.
    Mobile-responsive design with better touch targets.
    
    Args:
        id_prefix: Prefix for component ID
        label: Label for the toggle switch
        value: Initial value (True/False)
        icon: Optional icon classname (e.g., "weather", "chart")
        description: Optional description text to display under the toggle
        
    Returns:
        dbc.Row: Toggle switch component
    """
    icon_map = {
        "weather": "cloud-sun",
        "chart": "chart-area",
        "history": "history",
        "confidence": "percentage",
        "item-numbers": "list-ol",
        "forecast": "chart-line",
        "settings": "cog",
        None: None  # Default case
    }
    
    icon_class = icon_map.get(icon, icon_map.get(None))
    
    # Create a more mobile-friendly toggle label with proper spacing
    toggle_label = html.Span(
        # Only include icon if icon_class is provided, otherwise just include the label span
        ([html.I(className=f"fas fa-{icon_class} me-2", style={"width": "16px"}), 
          html.Span(label, style={"fontWeight": "500", "fontSize": "0.9rem"}, className="toggle-text")] if icon_class 
         else [html.Span(label, style={"fontWeight": "500", "fontSize": "0.9rem"}, className="toggle-text")]),
        className="d-flex align-items-center")
    
    checklist = dbc.Checklist(
        options=[
            {"label": toggle_label, "value": 1},
        ],
        value=[1] if value else [],
        id=f"{id_prefix}-toggle",
        switch=True,
        className="custom-toggle-switch",
        inputClassName="toggle-input",
        labelClassName="toggle-label",
    )
    
    content = [checklist]
    
    if description:
        content.append(
            html.Small(
                description, 
                className="text-muted ms-4 mt-1 d-block small",
                style={"opacity": "0.8"}
            )
        )
    
    # Return a more mobile-friendly container
    return dbc.Row([
        dbc.Col(
            html.Div(
                content,
                className="toggle-container py-2 px-3 py-md-3 mb-2",
                style={
                    "background": "#f8f9fa", 
                    "borderRadius": "6px",
                    "border": "1px solid #e9ecef",
                    "transition": "all 0.2s ease",
                    "minHeight": "42px", # Better touch target size
                    "display": "flex",
                    "flexDirection": "column",
                    "justifyContent": "center"
                }
            ), 
            width=12
        )
    ], className="mb-3 toggle-row")


def create_error_message(message, severity="danger"):
    """
    Create an enhanced error message component with better styling and mobile responsiveness.
    
    Args:
        message: Error message to display
        severity: Alert severity (danger, warning, info)
        
    Returns:
        dbc.Alert: Error message component
    """
    icon_map = {
        "danger": "exclamation-triangle",
        "warning": "exclamation-circle",
        "info": "info-circle",
        "success": "check-circle"
    }
    
    icon = icon_map.get(severity, "exclamation-triangle")
    
    return dbc.Alert(
        [
            html.Div([
                html.Div([
                    # Smaller icon on mobile
                    html.I(className=f"fas fa-{icon} me-2 me-md-3", style={"fontSize": "1.25rem"}),
                    html.Div([
                        # Smaller heading on mobile
                        html.H5(f"{severity.capitalize()} Alert", className="alert-heading mb-1 fs-6"),
                        # Smaller text on mobile
                        html.P(message, className="mb-0 small")
                    ], className="flex-grow-1")
                ], className="d-flex align-items-center")
            ])
        ],
        color=severity,
        dismissable=True,
        className="mb-3 mb-md-4 shadow-sm",
        style={
            "borderRadius": "6px",
            "border": "none",
            "padding": "0.75rem 1rem"
        }
    )


def create_info_card(title, value, color="primary", icon=None, subtitle=None):
    """
    Create an information card component with enhanced styling.
    
    Args:
        title: Card title
        value: Card value
        color: Card color
        icon: Card icon
        subtitle: Card subtitle
        
    Returns:
        dbc.Card: Information card component
    """
    # Map color names to CSS variables
    color_map = {
        "primary": "#3498db",
        "secondary": "#95a5a6",
        "success": "#2ecc71",
        "danger": "#e74c3c",
        "warning": "#f39c12",
        "info": "#1abc9c",
    }
    
    bg_color = color_map.get(color, "#3498db")
    
    card_header = html.Div([
        html.I(className=f"fas fa-{icon} mr-2", style={"opacity": "0.8"}) if icon else "",
        html.Span(title, style={"fontWeight": "600"})
    ], className="d-flex align-items-center")
    
    card_style = {
        "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
        "border": "none",
        "transition": "all 0.3s ease",
        "borderRadius": "6px",
        "overflow": "hidden"
    }
    
    header_style = {
        "backgroundColor": bg_color,
        "color": "white",
        "padding": "0.75rem 1.25rem",
        "borderBottom": "none"
    }
    
    return dbc.Card([
        dbc.CardHeader(card_header, style=header_style),
        dbc.CardBody(
            # Only include subtitle P element if subtitle is provided
            [html.H3(value, className="card-title", style={"fontWeight": "700", "fontSize": "1.75rem", "marginBottom": "0.5rem"})] + 
            ([html.P(subtitle, className="card-text", style={"opacity": "0.8"})] if subtitle else []),
            className="bg-white")
    ], style=card_style, className="mb-4 metric-card hover-shadow")


def create_tab_layout():
    """
    Create the base tab layout for the dashboard using Plotly's default UI settings.
    
    Returns:
        dict: Tab layout structure with default styling
    """
    # Create the tab layout with icons using default Plotly styling
    return {
        'forecast': dbc.Tab(
            children=[], 
            label=[html.I(className="fas fa-chart-line mr-2"), "Sales Forecast"], 
            tab_id="tab-forecast"
        ),
        'inventory': dbc.Tab(
            children=[], 
            label=[html.I(className="fas fa-boxes mr-2"), "Inventory Management"], 
            tab_id="tab-inventory"
        ),
        'pricing': dbc.Tab(
            children=[], 
            label=[html.I(className="fas fa-tags mr-2"), "Price Optimization"], 
            tab_id="tab-pricing"
        ),
        'integrated': dbc.Tab(
            label=[html.I(className="fas fa-project-diagram mr-2"), "Integrated View"], 
            tab_id="tab-integrated"
        ),
        'item_stats': dbc.Tab(
            label=[html.I(className="fas fa-chart-bar mr-2"), "Item Statistics"], 
            tab_id="tab-item-statistics"
        ),
        'profit': dbc.Tab(
            label=[html.I(className="fas fa-dollar-sign mr-2"), "Profit Analysis"], 
            tab_id="tab-profit"
        ),
        'revenue': dbc.Tab(
            label=[html.I(className="fas fa-hand-holding-usd mr-2"), "Revenue Analysis"], 
            tab_id="tab-revenue"
        ),
        'loss': dbc.Tab(
            label=[html.I(className="fas fa-shield-alt mr-2"), "Loss Prevention"], 
            tab_id="tab-loss"
        ),
        'summary': dbc.Tab(
            label=[html.I(className="fas fa-clipboard-list mr-2"), "Performance Summary"], 
            tab_id="tab-summary"
        ),
        'settings': dbc.Tab(
            label=[html.I(className="fas fa-cog mr-2"), "Settings"], 
            tab_id="tab-settings"
        )
    }


def create_tab_content():
    """
    Create content containers for each tab.
    
    Returns:
        dict: Content containers for each tab
    """
    return {
        'forecast': html.Div(id="forecast-content", className="pt-3"),
        'inventory': html.Div(id="inventory-content", className="pt-3"),
        'pricing': html.Div(id="pricing-content", className="pt-3"),
        'integrated': html.Div(id="integrated-content", className="pt-3"),
        'item_stats': html.Div(id="item-statistics-content", className="pt-3"),
        'profit': html.Div(id="profit-content", className="pt-3"),
        'revenue': html.Div(id="revenue-content", className="pt-3"),
        'loss': html.Div(id="loss-content", className="pt-3"),
        'summary': html.Div(id="summary-content", className="pt-3"),
        'settings': html.Div(id="settings-content", className="pt-3")
    }


def clear_data_cache():
    """
    Clear the data cache to force reloading of data files.
    """
    global _DATA_CACHE, _DATA_CACHE_TIMESTAMPS
    _DATA_CACHE.clear()
    _DATA_CACHE_TIMESTAMPS.clear()
    logger.info("Data cache cleared")


def get_image_asset(image_path, default_image=None):
    """
    Load and encode an image file for use in Dash.
    
    Args:
        image_path: Path to the image file
        default_image: Path to default image to use if main image fails to load
        
    Returns:
        str: Base64 encoded image or None if loading fails
    """
    try:
        # Try to read and encode the image
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Determine MIME type based on file extension
        file_ext = pathlib.Path(image_path).suffix.lower()
        mime_type = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml'
        }.get(file_ext, 'image/png')
        
        # Encode the image
        encoded = base64.b64encode(image_data).decode('ascii')
        return f"data:{mime_type};base64,{encoded}"
    
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        
        # Try to load the default image if provided
        if default_image:
            try:
                with open(default_image, 'rb') as f:
                    default_data = f.read()
                encoded = base64.b64encode(default_data).decode('ascii')
                logger.info(f"Using default image {default_image} instead")
                return f"data:image/png;base64,{encoded}"
            except Exception as e2:
                logger.error(f"Error loading default image {default_image}: {e2}")
        
        # Return None if all attempts fail
        return None


def load_dashboard_data(reload=False, fallback_root=None):
    """
    Load all the data needed for the dashboard.
    
    Args:
        reload: Force reload of data even if cached
        fallback_root: Root directory for fallback data files
        
    Returns:
        dict: Dictionary of DataFrames
    """
    # Clear cache if reload is requested
    if reload:
        clear_data_cache()
    
    # Initialize data dictionary
    data_dict = {}
    
    # Define fallback paths if provided
    fallback_combined = None
    fallback_forecasts = None
    fallback_inventories = None
    fallback_prices = None
    
    try:
        if fallback_root:
            fallback_root = pathlib.Path(fallback_root)
            fallback_combined = fallback_root / "combined_pizza_data.csv"
            fallback_forecasts = fallback_root / "forecasts"
            fallback_inventories = fallback_root / "inventory"
            fallback_prices = fallback_root / "pricing"
            
            logger.info(f"Using fallback root: {fallback_root}")
            
            # Check if the directories exist, create them if they don't
            for dir_path in [fallback_forecasts, fallback_inventories, fallback_prices]:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                    logger.info(f"Created fallback directory: {dir_path}")
    except Exception as path_error:
        logger.error(f"Error setting up fallback paths: {path_error}")
    
    # Define required columns for each data type
    combined_cols = ['Store_Id', 'Item', 'Date', 'Product', 'Sales', 'Price', 'Stock_Level']
    forecast_cols = ['Store_Id', 'Item', 'Date', 'Predicted_Demand']
    inventory_cols = ['Store_Id', 'Item', 'Current_Stock', 'Recommended_Order']
    pricing_cols = ['Store_Id', 'Item', 'Product', 'Elasticity', 'Avg_Price']
    
    # Track loading errors for diagnostics
    loading_errors = {}
    
    # Load core data files with thorough error handling
    try:
        # Load combined data
        logger.info("Loading combined data file...")
        combined_data = load_data_with_fallback(
            COMBINED_DATA_FILE, 
            fallback_path=fallback_combined,
            required_columns=combined_cols, 
            cache_key='combined_data'
        )
        if combined_data is None:
            loading_errors['combined_data'] = "Failed to load combined data"
    except Exception as e:
        logger.error(f"Critical error loading combined data: {str(e)}", exc_info=True)
        combined_data = None
        loading_errors['combined_data'] = f"Exception: {str(e)}"
    
    # Load forecast data with robust error handling
    try:
        # Load PyTorch forecasts
        logger.info("Loading PyTorch forecasts...")
        pytorch_forecasts = load_data_with_fallback(
            PYTORCH_FORECASTS_FILE, 
            fallback_path=fallback_forecasts / "pytorch_forecasts.csv" if fallback_forecasts else None,
            required_columns=None,  # Don't enforce columns yet
            cache_key='pytorch_forecasts'
        )
        if pytorch_forecasts is None:
            loading_errors['pytorch_forecasts'] = "Failed to load PyTorch forecasts"
    except Exception as e:
        logger.error(f"Error loading PyTorch forecasts: {str(e)}", exc_info=True)
        pytorch_forecasts = None
        loading_errors['pytorch_forecasts'] = f"Exception: {str(e)}"
    
    try:
        # Load RF forecasts
        logger.info("Loading Random Forest forecasts...")
        rf_forecasts = load_data_with_fallback(
            RF_FORECASTS_FILE, 
            fallback_path=fallback_forecasts / "rf_forecasts.csv" if fallback_forecasts else None,
            required_columns=None,  # Don't enforce columns yet
            cache_key='rf_forecasts'
        )
        if rf_forecasts is None:
            loading_errors['rf_forecasts'] = "Failed to load RF forecasts"
    except Exception as e:
        logger.error(f"Error loading RF forecasts: {str(e)}", exc_info=True)
        rf_forecasts = None
        loading_errors['rf_forecasts'] = f"Exception: {str(e)}"
    
    # Create forecast data if needed
    try:
        # Create default forecasts if both are missing
        if pytorch_forecasts is None and rf_forecasts is None:
            logger.warning("Both forecast files missing, creating synthetic forecast data")
            forecast_cols_extended = forecast_cols + ['Lower_Bound', 'Upper_Bound']
            try:
                pytorch_forecasts = create_synthetic_data(forecast_cols_extended, 'pytorch_forecasts')
                rf_forecasts = pytorch_forecasts.copy()
                logger.info("Successfully created synthetic forecast data")
            except Exception as synth_error:
                logger.error(f"Failed to create synthetic forecast data: {str(synth_error)}")
                # Create minimal data
                data = {
                    'Store_Id': ['1'], 
                    'Item': ['1'], 
                    'Date': [pd.Timestamp.now()],
                    'Predicted_Demand': [10],
                    'Lower_Bound': [8], 
                    'Upper_Bound': [12]
                }
                pytorch_forecasts = pd.DataFrame(data)
                rf_forecasts = pytorch_forecasts.copy()
        elif pytorch_forecasts is None and rf_forecasts is not None:
            logger.warning("PyTorch forecasts missing, using Random Forest forecasts instead")
            pytorch_forecasts = rf_forecasts.copy()
        elif rf_forecasts is None and pytorch_forecasts is not None:
            logger.warning("Random Forest forecasts missing, using PyTorch forecasts instead")
            rf_forecasts = pytorch_forecasts.copy()
    except Exception as forecast_error:
        logger.error(f"Error handling forecast data: {str(forecast_error)}", exc_info=True)
        # Create emergency minimal data
        data = {
            'Store_Id': ['1'], 
            'Item': ['1'], 
            'Date': [pd.Timestamp.now()],
            'Predicted_Demand': [10],
            'Lower_Bound': [8], 
            'Upper_Bound': [12]
        }
        pytorch_forecasts = pd.DataFrame(data)
        rf_forecasts = pytorch_forecasts.copy()
    
    # Map and normalize column names
    try:
        # Process PyTorch forecasts
        if pytorch_forecasts is not None:
            # Ensure Date column is datetime
            if 'Date' in pytorch_forecasts.columns and pytorch_forecasts['Date'].dtype != 'datetime64[ns]':
                pytorch_forecasts['Date'] = pd.to_datetime(pytorch_forecasts['Date'], errors='coerce')
                
            # Check for Forecast column and map to Predicted_Demand if needed
            if 'Forecast' in pytorch_forecasts.columns and 'Predicted_Demand' not in pytorch_forecasts.columns:
                pytorch_forecasts['Predicted_Demand'] = pytorch_forecasts['Forecast']
                
            # Check for other possible column names
            alternate_names = {
                'pred': 'Predicted_Demand',
                'prediction': 'Predicted_Demand',
                'forecast': 'Predicted_Demand',
                'lower': 'Lower_Bound',
                'lower_bound': 'Lower_Bound',
                'upper': 'Upper_Bound',
                'upper_bound': 'Upper_Bound',
                'store': 'Store_Id',
                'store_id': 'Store_Id',
                'product_id': 'Item',
                'item_id': 'Item'
            }
            
            # Check and rename columns using alternate names
            for alt_name, standard_name in alternate_names.items():
                if alt_name in pytorch_forecasts.columns and standard_name not in pytorch_forecasts.columns:
                    pytorch_forecasts[standard_name] = pytorch_forecasts[alt_name]
                    
            # Generate confidence intervals if missing
            for col in ['Lower_Bound', 'Upper_Bound']:
                if col not in pytorch_forecasts.columns:
                    demand_col = 'Predicted_Demand' if 'Predicted_Demand' in pytorch_forecasts.columns else 'Forecast'
                    if demand_col in pytorch_forecasts.columns:
                        demand = pytorch_forecasts[demand_col]
                        std_dev = demand * 0.1
                        if col == 'Lower_Bound':
                            pytorch_forecasts[col] = np.maximum(0, demand - 1.96 * std_dev)
                        else:  # Upper_Bound
                            pytorch_forecasts[col] = demand + 1.96 * std_dev
        
        # Process RF forecasts
        if rf_forecasts is not None:
            # Ensure Date column is datetime
            if 'Date' in rf_forecasts.columns and rf_forecasts['Date'].dtype != 'datetime64[ns]':
                rf_forecasts['Date'] = pd.to_datetime(rf_forecasts['Date'], errors='coerce')
                
            # Check for Forecast column and map to Predicted_Demand if needed
            if 'Forecast' in rf_forecasts.columns and 'Predicted_Demand' not in rf_forecasts.columns:
                rf_forecasts['Predicted_Demand'] = rf_forecasts['Forecast']
                
            # Check for other possible column names (same mapping as for PyTorch)
            for alt_name, standard_name in alternate_names.items():
                if alt_name in rf_forecasts.columns and standard_name not in rf_forecasts.columns:
                    rf_forecasts[standard_name] = rf_forecasts[alt_name]
                    
            # Generate confidence intervals if missing
            for col in ['Lower_Bound', 'Upper_Bound']:
                if col not in rf_forecasts.columns:
                    demand_col = 'Predicted_Demand' if 'Predicted_Demand' in rf_forecasts.columns else 'Forecast'
                    if demand_col in rf_forecasts.columns:
                        demand = rf_forecasts[demand_col]
                        std_dev = demand * 0.1
                        if col == 'Lower_Bound':
                            rf_forecasts[col] = np.maximum(0, demand - 1.96 * std_dev)
                        else:  # Upper_Bound
                            rf_forecasts[col] = demand + 1.96 * std_dev
    except Exception as mapping_error:
        logger.error(f"Error mapping forecast column names: {str(mapping_error)}", exc_info=True)
    
    # Default forecasts dataset
    forecasts = pytorch_forecasts if pytorch_forecasts is not None else rf_forecasts
    data_dict['forecasts'] = forecasts  # Make sure 'forecasts' key is set
    data_dict['pytorch_forecasts'] = pytorch_forecasts
    data_dict['rf_forecasts'] = rf_forecasts
    data_dict['combined_data'] = combined_data
    
    # Load remaining data files - wrap each in try/except for robustness
    try:
        logger.info("Loading inventory recommendations...")
        inventory_recs = load_data_with_fallback(
            INVENTORY_RECOMMENDATIONS_FILE, 
            fallback_path=fallback_inventories / "inventory_recommendations.csv" if fallback_inventories else 
                       pathlib.Path.cwd() / "inventory_recommendations.csv",
            required_columns=inventory_cols, 
            cache_key='inventory_recommendations'
        )
        data_dict['inventory_recs'] = inventory_recs
    except Exception as e:
        logger.error(f"Error loading inventory recommendations: {str(e)}", exc_info=True)
        data_dict['inventory_recs'] = None
        loading_errors['inventory_recs'] = f"Exception: {str(e)}"
    
    try:
        logger.info("Loading optimized orders...")
        optimized_orders = load_data_with_fallback(
            OPTIMIZED_ORDERS_FILE, 
            fallback_path=fallback_inventories / "optimized_orders.csv" if fallback_inventories else None,
            cache_key='optimized_orders'
        )
        data_dict['optimized_orders'] = optimized_orders
    except Exception as e:
        logger.error(f"Error loading optimized orders: {str(e)}", exc_info=True)
        data_dict['optimized_orders'] = None
        loading_errors['optimized_orders'] = f"Exception: {str(e)}"
    
    try:
        logger.info("Loading inventory projection...")
        inventory_projection = load_data_with_fallback(
            INVENTORY_PROJECTION_FILE, 
            fallback_path=fallback_inventories / "inventory_projection.csv" if fallback_inventories else 
                        pathlib.Path.cwd() / "inventory_projection.csv",
            cache_key='inventory_projection'
        )
        data_dict['inventory_projection'] = inventory_projection
    except Exception as e:
        logger.error(f"Error loading inventory projection: {str(e)}", exc_info=True)
        data_dict['inventory_projection'] = None
        loading_errors['inventory_projection'] = f"Exception: {str(e)}"
    
    try:
        logger.info("Loading price elasticities...")
        price_elasticities = load_data_with_fallback(
            PRICE_ELASTICITIES_FILE, 
            fallback_path=fallback_prices / "price_elasticities.csv" if fallback_prices else None,
            required_columns=None,  # Don't enforce column names yet
            cache_key='price_elasticities'
        )
        
        # Map column names for price elasticities
        if price_elasticities is not None:
            if 'Current_Price' in price_elasticities.columns and 'Avg_Price' not in price_elasticities.columns:
                price_elasticities['Avg_Price'] = price_elasticities['Current_Price']
            elif 'Price' in price_elasticities.columns and 'Avg_Price' not in price_elasticities.columns:
                price_elasticities['Avg_Price'] = price_elasticities['Price']
                
            # Make sure we have the required pricing columns
            for required_col in pricing_cols:
                if required_col not in price_elasticities.columns:
                    # Try common alternate names
                    if required_col == 'Elasticity' and 'Price_Elasticity' in price_elasticities.columns:
                        price_elasticities['Elasticity'] = price_elasticities['Price_Elasticity']
                    elif required_col == 'Avg_Price' and 'Mean_Price' in price_elasticities.columns:
                        price_elasticities['Avg_Price'] = price_elasticities['Mean_Price']
                    elif required_col == 'Product' and 'Item_Description' in price_elasticities.columns:
                        price_elasticities['Product'] = price_elasticities['Item_Description']
        
        data_dict['price_elasticities'] = price_elasticities
    except Exception as e:
        logger.error(f"Error loading price elasticities: {str(e)}", exc_info=True)
        data_dict['price_elasticities'] = None
        loading_errors['price_elasticities'] = f"Exception: {str(e)}"
    
    try:
        logger.info("Loading price recommendations...")
        price_recommendations = load_data_with_fallback(
            PRICE_RECOMMENDATIONS_FILE, 
            fallback_path=fallback_prices / "price_recommendations.csv" if fallback_prices else None,
            cache_key='price_recommendations'
        )
        data_dict['price_recommendations'] = price_recommendations
    except Exception as e:
        logger.error(f"Error loading price recommendations: {str(e)}", exc_info=True)
        data_dict['price_recommendations'] = None
        loading_errors['price_recommendations'] = f"Exception: {str(e)}"
    
    try:
        logger.info("Loading profit impact...")
        profit_impact = load_data_with_fallback(
            PROFIT_IMPACT_FILE, 
            fallback_path=fallback_prices / "profit_impact.csv" if fallback_prices else None,
            cache_key='profit_impact'
        )
        data_dict['profit_impact'] = profit_impact
    except Exception as e:
        logger.error(f"Error loading profit impact: {str(e)}", exc_info=True)
        data_dict['profit_impact'] = None
        loading_errors['profit_impact'] = f"Exception: {str(e)}"
    
    try:
        logger.info("Loading product mix...")
        product_mix = load_data_with_fallback(
            PRODUCT_MIX_FILE, 
            fallback_path=fallback_prices / "product_mix_optimization.csv" if fallback_prices else None,
            cache_key='product_mix'
        )
        data_dict['product_mix'] = product_mix
    except Exception as e:
        logger.error(f"Error loading product mix: {str(e)}", exc_info=True)
        data_dict['product_mix'] = None
        loading_errors['product_mix'] = f"Exception: {str(e)}"
    
    # Log a summary of loading errors
    if loading_errors:
        logger.warning(f"Data loading completed with {len(loading_errors)} errors:")
        for data_key, error in loading_errors.items():
            logger.warning(f"  - {data_key}: {error}")
    else:
        logger.info("All data loaded successfully")
    
    return data_dict


def create_app(data=None, use_weather=True, fallback_root=None):
    """
    Create the Dash application with loaded data.
    
    Args:
        data: Dictionary of DataFrames (optional, will be loaded if not provided)
        use_weather: Whether to use weather service features
        fallback_root: Root directory for fallback data files
        
    Returns:
        tuple: (app, data) Dash application and data dictionary
    """
    # Create a Dash app with enhanced responsive design and important CSS fixes
    app = dash.Dash(__name__, 
                   external_stylesheets=[
                       dbc.themes.BOOTSTRAP, 
                       'https://use.fontawesome.com/releases/v5.15.4/css/all.css',
                       'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap'
                   ],
                   suppress_callback_exceptions=True,
                   meta_tags=[
                       # Responsive viewport settings for mobile devices
                       {"name": "viewport", "content": "width=device-width, initial-scale=1, shrink-to-fit=no"},
                       # Force CSS reloading to avoid cache issues
                       {"http-equiv": "Cache-Control", "content": "no-cache, no-store, must-revalidate"},
                       {"http-equiv": "Pragma", "content": "no-cache"},
                       {"http-equiv": "Expires", "content": "0"}
                   ],
                   assets_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"))
    server = app.server  # Required for Gunicorn deployment
    
    # Load data if not provided
    if data is None:
        try:
            data = load_dashboard_data(fallback_root=fallback_root)
        except Exception as e:
            logger.critical(f"Critical error loading dashboard data: {e}", exc_info=True)
            # Create minimal data to allow dashboard to load
            data = {
                'combined_data': create_synthetic_data(['Store_Id', 'Item', 'Date', 'Product', 'Sales', 'Price', 'Stock_Level'], 'combined_data'),
                'forecasts': create_synthetic_data(['Store_Id', 'Item', 'Date', 'Predicted_Demand', 'Lower_Bound', 'Upper_Bound'], 'forecasts'),
                'pytorch_forecasts': create_synthetic_data(['Store_Id', 'Item', 'Date', 'Predicted_Demand', 'Lower_Bound', 'Upper_Bound'], 'pytorch_forecasts'),
                'rf_forecasts': create_synthetic_data(['Store_Id', 'Item', 'Date', 'Predicted_Demand', 'Lower_Bound', 'Upper_Bound'], 'rf_forecasts')
            }
    
    # Initialize weather service with error handling
    app.weather_service = None
    if use_weather and WeatherService is not None:
        try:
            app.weather_service = WeatherService()
        except Exception as e:
            logger.error(f"Failed to initialize weather service: {e}")
    
    # Initialize stock adjustment state
    app.manual_stock_adjustments = {}
    app.manual_stock_adjustments_with_dates = {}
    
    # Add reload data callback
    @app.callback(
        Output("debug-output", "children"),
        Input("reload-data-button", "n_clicks"),
        prevent_initial_call=True
    )
    def reload_dashboard_data(n_clicks):
        if n_clicks:
            try:
                nonlocal data
                data = load_dashboard_data(reload=True, fallback_root=fallback_root)
                return html.Div([
                    html.P("Data reloaded successfully", className="text-success"),
                    html.P(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                ])
            except Exception as e:
                return html.Div([
                    html.P("Error reloading data", className="text-danger"),
                    html.P(str(e))
                ])
    
    return app, data