"""
Data loading and processing module for the Pizza Predictive Ordering System.
This module handles loading and integrating data from sales, purchases, and stock CSV files.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import logging
import pathlib
from typing import Union, Dict, Optional, Tuple, List

# Add parent directory to path to allow imports from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import (
    SALES_FILE, PURCHASES_FILE, STOCK_FILE, 
    COMBINED_DATA_FILE, MIN_STOCK_WEEKS, TARGET_STOCK_WEEKS
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_loader')

def validate_dataframe(df: pd.DataFrame, required_columns: List[str], name: str) -> Tuple[bool, str]:
    """
    Validate that a DataFrame has all required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
        name: Name of the dataset for logging
        
    Returns:
        Tuple[bool, str]: Success status and error message if failed
    """
    if df is None:
        return False, f"{name} dataset is None"
        
    if df.empty:
        return False, f"{name} dataset is empty"
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns in {name} dataset: {missing_columns}"
    
    # Check for invalid data types in key columns
    for col in [c for c in required_columns if c in df.columns]:
        if col in ['Store_Id', 'item', 'Item']:
            try:
                # Ensure these columns can be converted to string
                df[col] = df[col].astype(str)
            except Exception as e:
                return False, f"Invalid data in column '{col}' of {name} dataset: {str(e)}"
        elif col in ['Date', 'Proc_date']:
            # Check if date columns can be parsed
            if df[col].dtype != 'datetime64[ns]':
                try:
                    pd.to_datetime(df[col], errors='raise')
                except Exception as e:
                    return False, f"Invalid date format in column '{col}' of {name} dataset: {str(e)}"
        
    return True, ""


def load_pizza_datasets(sales_file=SALES_FILE, 
                        purchases_file=PURCHASES_FILE,
                        stock_file=STOCK_FILE,
                        fallback_dir=None,
                        create_synthetic_if_missing=True):
    """
    Load and process the three pizza datasets: sales, purchases, and stock.
    
    Args:
        sales_file (str or Path): Path to the sales CSV file
        purchases_file (str or Path): Path to the purchases CSV file
        stock_file (str or Path): Path to the stock CSV file
        
        fallback_dir (str or Path, optional): Directory with fallback data files
        
    Returns:
        pandas.DataFrame: Combined dataset with all relevant features
    """
    logger.info(f"Loading data from {sales_file}, {purchases_file}, and {stock_file}...")
    
    # Define required columns for each dataset
    sales_required_columns = ['Proc_date', 'item', 'Total_units', 'Total_Retail_$', 'Total_Cost_$', 'Item_Description', 'store_id', 'Size']
    purchases_required_columns = ['Proc_date', 'item', 'Total_units', 'Total_Retail_$', 'Total_Cost_$', 'Item_Description', 'store_id']
    stock_required_columns = ['item', 'Item_description', 'On_Hand', 'Week_4_Avg_Movement', 'Week_13_Avg_Movement', 'Weeks_of_Supply', 'Store_Id']

    # Initialize fallback paths if provided
    if fallback_dir:
        fallback_dir = pathlib.Path(fallback_dir)
        fallback_sales_file = fallback_dir / "FrozenPizzaSales.csv"
        fallback_purchases_file = fallback_dir / "FrozenPizzaPurchases.csv"
        fallback_stock_file = fallback_dir / "FrozenPizzaStock.csv"
    else:
        fallback_sales_file = None
        fallback_purchases_file = None
        fallback_stock_file = None

    # Load the datasets with fallbacks
    sales_df = None
    purchases_df = None
    stock_df = None
    errors = []
    
    # Try to load sales data
    try:
        if os.path.exists(sales_file):
            sales_df = pd.read_csv(sales_file)
            logger.info(f"Loaded {len(sales_df)} sales records from {sales_file}")
            
            # Validate required columns
            valid, error_msg = validate_dataframe(sales_df, sales_required_columns, "Sales")
            if not valid:
                logger.error(error_msg)
                sales_df = None
                errors.append(error_msg)
    except Exception as e:
        logger.error(f"Error loading sales file {sales_file}: {str(e)}")
        errors.append(f"Sales data error: {str(e)}")
        sales_df = None

    # Try fallback sales file if needed
    if sales_df is None and fallback_sales_file and os.path.exists(fallback_sales_file):
        try:
            sales_df = pd.read_csv(fallback_sales_file)
            logger.info(f"Loaded {len(sales_df)} sales records from fallback {fallback_sales_file}")
            
            # Validate required columns
            valid, error_msg = validate_dataframe(sales_df, sales_required_columns, "Sales (fallback)")
            if not valid:
                logger.error(error_msg)
                sales_df = None
                errors.append(error_msg)
        except Exception as e:
            logger.error(f"Error loading fallback sales file {fallback_sales_file}: {str(e)}")
            errors.append(f"Fallback sales data error: {str(e)}")
            sales_df = None

    # Try to load purchases data
    try:
        if os.path.exists(purchases_file):
            purchases_df = pd.read_csv(purchases_file)
            logger.info(f"Loaded {len(purchases_df)} purchase records from {purchases_file}")
            
            # Validate required columns
            valid, error_msg = validate_dataframe(purchases_df, purchases_required_columns, "Purchases")
            if not valid:
                logger.error(error_msg)
                purchases_df = None
                errors.append(error_msg)
    except Exception as e:
        logger.error(f"Error loading purchases file {purchases_file}: {str(e)}")
        errors.append(f"Purchases data error: {str(e)}")
        purchases_df = None

    # Try fallback purchases file if needed
    if purchases_df is None and fallback_purchases_file and os.path.exists(fallback_purchases_file):
        try:
            purchases_df = pd.read_csv(fallback_purchases_file)
            logger.info(f"Loaded {len(purchases_df)} purchase records from fallback {fallback_purchases_file}")
            
            # Validate required columns
            valid, error_msg = validate_dataframe(purchases_df, purchases_required_columns, "Purchases (fallback)")
            if not valid:
                logger.error(error_msg)
                purchases_df = None
                errors.append(error_msg)
        except Exception as e:
            logger.error(f"Error loading fallback purchases file {fallback_purchases_file}: {str(e)}")
            errors.append(f"Fallback purchases data error: {str(e)}")
            purchases_df = None

    # Try to load stock data
    try:
        if os.path.exists(stock_file):
            stock_df = pd.read_csv(stock_file)
            logger.info(f"Loaded {len(stock_df)} stock records from {stock_file}")
            
            # Validate required columns
            valid, error_msg = validate_dataframe(stock_df, stock_required_columns, "Stock")
            if not valid:
                logger.error(error_msg)
                stock_df = None
                errors.append(error_msg)
    except Exception as e:
        logger.error(f"Error loading stock file {stock_file}: {str(e)}")
        errors.append(f"Stock data error: {str(e)}")
        stock_df = None

    # Try fallback stock file if needed
    if stock_df is None and fallback_stock_file and os.path.exists(fallback_stock_file):
        try:
            stock_df = pd.read_csv(fallback_stock_file)
            logger.info(f"Loaded {len(stock_df)} stock records from fallback {fallback_stock_file}")
            
            # Validate required columns
            valid, error_msg = validate_dataframe(stock_df, stock_required_columns, "Stock (fallback)")
            if not valid:
                logger.error(error_msg)
                stock_df = None
                errors.append(error_msg)
        except Exception as e:
            logger.error(f"Error loading fallback stock file {fallback_stock_file}: {str(e)}")
            errors.append(f"Fallback stock data error: {str(e)}")
            stock_df = None

    # Check if we have valid data to proceed
    # Check if we have valid data to proceed or need to create synthetic data
    if sales_df is None:
        if create_synthetic_if_missing:
            logger.warning("Failed to load sales data from primary and fallback sources. Creating synthetic data.")
            from create_sample_data import create_sample_data
            
            try:
                # Create synthetic dataset
                sales_df = create_sample_data(num_stores=3, num_products=5, num_days=120)
                logger.info(f"Created synthetic sales dataset with {len(sales_df)} records")
                
                # Use the synthetic data for purchases and stock as well
                if purchases_df is None:
                    purchases_df = sales_df.copy()
                    
                if stock_df is None:
                    stock_df = sales_df[["Store_Id", "Item", "Product", "Stock_Level", "Weeks_Of_Stock", 
                                        "Recent_Daily_Sales", "Avg_Weekly_Sales_4W"]].drop_duplicates().copy()
                    stock_df = stock_df.rename(columns={
                        "Product": "Item_description",
                        "Stock_Level": "On_Hand", 
                        "Recent_Daily_Sales": "Week_13_Avg_Movement",
                        "Avg_Weekly_Sales_4W": "Week_4_Avg_Movement",
                        "Weeks_Of_Stock": "Weeks_of_Supply"
                    })
                    
            except Exception as e:
                logger.error(f"Failed to create synthetic data: {str(e)}")
                raise ValueError(f"Failed to load or create sales data: {'; '.join(errors)}, {str(e)}")
        else:
            logger.error("Failed to load sales data from primary and fallback sources. Cannot proceed.")
            raise ValueError(f"Failed to load sales data: {'; '.join(errors)}")
        
    if purchases_df is None:
        logger.warning("Failed to load purchases data. Continuing with sales data only.")
        # Create empty purchases DataFrame with required columns
        purchases_df = pd.DataFrame(columns=purchases_required_columns)
        
    if stock_df is None:
        logger.warning("Failed to load stock data. Using default stock values.")
        # Create empty stock DataFrame with required columns
        stock_df = pd.DataFrame(columns=stock_required_columns)
    
    # Process datasets
    try:
        
        # Process sales data
        sales_df['Date'] = pd.to_datetime(sales_df['Proc_date'], format='%m/%d/%Y', errors='coerce')
        sales_df.dropna(subset=['Date'], inplace=True)  # Drop rows with invalid dates
        sales_df['item'] = sales_df['item'].astype(str)
        sales_df.rename(columns={
            'Total_units': 'Units_Sold',
            'Total_Retail_$': 'Retail_Revenue',
            'Total_Cost_$': 'Cost',
            'Item_Description': 'Product_Name',
            'store_id': 'Store_Id'
        }, inplace=True)
        
        # Process purchases data (shipments to store)
        purchases_df['Date'] = pd.to_datetime(purchases_df['Proc_date'], format='%m/%d/%Y', errors='coerce')
        purchases_df.dropna(subset=['Date'], inplace=True)
        purchases_df['item'] = purchases_df['item'].astype(str)
        purchases_df.rename(columns={
            'Total_units': 'Units_Purchased',
            'Total_Retail_$': 'Purchase_Retail_Value',
            'Total_Cost_$': 'Purchase_Cost',
            'Item_Description': 'Product_Name',
            'store_id': 'Store_Id'
        }, inplace=True)
        
        # Process stock data
        stock_df['item'] = stock_df['item'].astype(str)
        stock_df.rename(columns={
            'Item_description': 'Product_Name',
            'On_Hand': 'Current_Stock',
            'Week_4_Avg_Movement': 'Avg_Weekly_Sales_4W',
            'Week_13_Avg_Movement': 'Avg_Weekly_Sales_13W',
            'Weeks_of_Supply': 'Stock_Coverage_Weeks'
        }, inplace=True)
        
        # Combine the datasets
        # First, create a daily sales summary
        sales_summary = sales_df.groupby(['Store_Id', 'item', 'Date']).agg({
            'Units_Sold': 'sum',
            'Retail_Revenue': 'sum',
            'Cost': 'sum',
            'Product_Name': 'first',
            'Size': 'first'
        }).reset_index()
        
        # Calculate daily unit price with safeguards for division by zero
        sales_summary['Unit_Price'] = np.where(
            sales_summary['Units_Sold'] > 0,
            sales_summary['Retail_Revenue'] / sales_summary['Units_Sold'],
            0
        )
        
        # Create a purchases summary
        purchases_summary = purchases_df.groupby(['Store_Id', 'item', 'Date']).agg({
            'Units_Purchased': 'sum',
            'Purchase_Cost': 'sum',
            'Purchase_Retail_Value': 'sum'
        }).reset_index()
        
        # Get all unique dates in the sales data
        min_date = sales_summary['Date'].min()
        max_date = sales_summary['Date'].max()
        date_range = pd.date_range(start=min_date, end=max_date)
        
        # Create a complete date range for each product
        all_products = sales_df[['Store_Id', 'item', 'Product_Name', 'Size']].drop_duplicates()
        
        # Create an empty dataframe to hold complete time series
        all_dates = []
        for _, row in all_products.iterrows():
            for date in date_range:
                all_dates.append({
                    'Store_Id': row['Store_Id'],
                    'item': row['item'],
                    'Date': date,
                    'Product_Name': row['Product_Name'],
                    'Size': row['Size']
                })
        
        # Create full date range dataframe
        full_dates_df = pd.DataFrame(all_dates)
        
        # Merge sales data
        merged_df = pd.merge(
            full_dates_df,
            sales_summary,
            on=['Store_Id', 'item', 'Date', 'Product_Name', 'Size'],
            how='left'
        )
        
        # Fill missing sales with 0
        merged_df['Units_Sold'].fillna(0, inplace=True)
        merged_df['Retail_Revenue'].fillna(0, inplace=True)
        merged_df['Cost'].fillna(0, inplace=True)
        
        # Forward fill unit prices
        merged_df['Unit_Price'] = merged_df.groupby(['Store_Id', 'item'])['Unit_Price'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill').fillna(0)
        )
        
        # Merge purchases data
        merged_df = pd.merge(
            merged_df,
            purchases_summary,
            on=['Store_Id', 'item', 'Date'],
            how='left'
        )
        
        # Fill missing purchases with 0
        merged_df['Units_Purchased'].fillna(0, inplace=True)
        merged_df['Purchase_Cost'].fillna(0, inplace=True)
        merged_df['Purchase_Retail_Value'].fillna(0, inplace=True)
        
        # Add stock tracking
        merged_df = merged_df.sort_values(['Store_Id', 'item', 'Date'])
        
        # Initialize stock with the values from stock_df
        initial_stock = {}
        for _, row in stock_df.iterrows():
            key = (row['Store_Id'], row['item'])
            initial_stock[key] = row['Current_Stock']
        
        # Calculate daily stock levels
        merged_df['Stock_Level'] = 0
        
        for (store, item), group in merged_df.groupby(['Store_Id', 'item']):
            # Get initial stock level for this product
            stock_level = initial_stock.get((store, item), 0)
            
            # Update stock for each day
            for idx, row in group.iterrows():
                # Add purchases
                stock_level += row['Units_Purchased']
                
                # Subtract sales
                stock_level = max(0, stock_level - row['Units_Sold'])
                
                # Record stock level
                merged_df.loc[idx, 'Stock_Level'] = stock_level
        
        # Add weekly stock metrics from stock_df
        stock_metrics = stock_df[['Store_Id', 'item', 'Avg_Weekly_Sales_4W', 'Avg_Weekly_Sales_13W', 'Stock_Coverage_Weeks']].copy()
        
        # Ensure the data types match before merging
        stock_metrics.loc[:, 'Store_Id'] = stock_metrics['Store_Id'].astype(str)
        stock_metrics.loc[:, 'item'] = stock_metrics['item'].astype(str)
        merged_df['Store_Id'] = merged_df['Store_Id'].astype(str)
        merged_df['item'] = merged_df['item'].astype(str)
        
        merged_df = pd.merge(
            merged_df,
            stock_metrics,
            on=['Store_Id', 'item'],
            how='left'
        )
        
        # Calculate profit and ensure no negative values in core metrics
        merged_df['Units_Sold'] = merged_df['Units_Sold'].apply(lambda x: max(0, x))
        merged_df['Unit_Price'] = merged_df['Unit_Price'].apply(lambda x: max(0, x))
        merged_df['Profit'] = merged_df['Retail_Revenue'] - merged_df['Cost']
        
        # Add date features
        merged_df['Day_Of_Week'] = merged_df['Date'].dt.dayofweek
        merged_df['Month'] = merged_df['Date'].dt.month
        merged_df['Year'] = merged_df['Date'].dt.year
        merged_df['Day'] = merged_df['Date'].dt.day
        
        # Add holiday information
        holidays = {
            '01-01': 'New Year',
            '02-14': 'Valentine', 
            '07-04': 'Independence Day',
            '10-31': 'Halloween',
            '11-25': 'Thanksgiving', # Approximate
            '12-25': 'Christmas',
            '04-15': 'Easter' # Approximate
        }
        
        merged_df['month_day'] = merged_df['Date'].dt.strftime('%m-%d')
        merged_df['Is_Holiday'] = merged_df['month_day'].apply(lambda x: 1 if x in holidays else 0)
        merged_df['Holiday_Name'] = merged_df['month_day'].apply(lambda x: holidays.get(x))
        merged_df.drop('month_day', axis=1, inplace=True)
        
        # Generate simulated weather data
        np.random.seed(42)
        weather_options = ['Normal', 'Heavy Rain', 'Snow', 'Storm']
        weather_probabilities = [0.85, 0.06, 0.06, 0.03]
        
        # Group by Date to assign the same weather to all products on the same day
        unique_dates = merged_df['Date'].unique()
        date_weather = {}
        
        for date in unique_dates:
            date_weather[date] = np.random.choice(weather_options, p=weather_probabilities)
        
        merged_df['Weather'] = merged_df['Date'].map(date_weather)
        
        # Add promotion flag - assume a promotion if price is significantly lower than average
        merged_df['avg_price'] = merged_df.groupby(['Store_Id', 'item'])['Unit_Price'].transform('mean')
        merged_df['Promotion'] = (merged_df['Unit_Price'] < merged_df['avg_price'] * 0.9).astype(int)
        merged_df.drop('avg_price', axis=1, inplace=True)
        
        # Add weeks of stock metric based on current stock and recent sales
        merged_df['Recent_Daily_Sales'] = merged_df.groupby(['Store_Id', 'item'])['Units_Sold'].transform(
            lambda x: x.rolling(window=28, min_periods=1).mean()
        )
        merged_df['Weeks_Of_Stock'] = merged_df['Stock_Level'] / (merged_df['Recent_Daily_Sales'] * 7)
        merged_df['Weeks_Of_Stock'].replace([np.inf, -np.inf], 4, inplace=True)  # Cap at 4 weeks
        merged_df['Weeks_Of_Stock'].fillna(4, inplace=True)  # Fill NaNs with 4 weeks
        
        # Add supply status flag (under 1 week, 1-3 weeks, over 3 weeks)
        conditions = [
            (merged_df['Weeks_Of_Stock'] < MIN_STOCK_WEEKS),
            (merged_df['Weeks_Of_Stock'] >= MIN_STOCK_WEEKS) & (merged_df['Weeks_Of_Stock'] <= TARGET_STOCK_WEEKS),
            (merged_df['Weeks_Of_Stock'] > TARGET_STOCK_WEEKS)
        ]
        values = ['Low', 'Adequate', 'Excess']
        merged_df['Stock_Status'] = np.select(conditions, values, default='Adequate')
        
        # Calculate optimal order quantity based on sales and stock policy
        # Use settings from config
        merged_df['Daily_Sales_Forecast'] = merged_df['Recent_Daily_Sales']
        merged_df['Target_Stock_Weeks'] = TARGET_STOCK_WEEKS
        merged_df['Min_Stock_Weeks'] = MIN_STOCK_WEEKS
        merged_df['Order_Point'] = merged_df['Daily_Sales_Forecast'] * 7 * merged_df['Min_Stock_Weeks']
        
        # Calculate optimal order quantity
        merged_df['Order_Quantity'] = np.where(
            merged_df['Stock_Level'] < merged_df['Order_Point'],
            (merged_df['Daily_Sales_Forecast'] * 7 * merged_df['Target_Stock_Weeks']) - merged_df['Stock_Level'],
            0
        )
        merged_df['Order_Quantity'] = merged_df['Order_Quantity'].apply(lambda x: max(0, round(x)))
        
        # Add columns for standard RF model compatibility
        merged_df.rename(columns={
            'Units_Sold': 'Sales',
            'Unit_Price': 'Price',
            'item': 'Item',
            'Product_Name': 'Product'
        }, inplace=True)
        
        # Add lag features for time series modeling
        logger.info("Creating lag features for time series modeling...")
        for product in merged_df['Item'].unique():
            for store in merged_df['Store_Id'].unique():
                product_data = merged_df[(merged_df['Item'] == product) & 
                                         (merged_df['Store_Id'] == store)].sort_values('Date')
                
                if len(product_data) > 0:
                    # Create lags for Sales
                    for lag in [1, 7, 14, 28]:
                        lag_col = f'Sales_Lag_{lag}'
                        lagged_values = product_data['Sales'].shift(lag)
                        merged_df.loc[(merged_df['Item'] == product) & 
                                     (merged_df['Store_Id'] == store), lag_col] = lagged_values.values
                    
                    # Create rolling averages
                    for window in [7, 14, 28]:
                        avg_col = f'Sales_Avg_{window}'
                        rolling_avg = product_data['Sales'].rolling(window=window, min_periods=1).mean()
                        merged_df.loc[(merged_df['Item'] == product) & 
                                     (merged_df['Store_Id'] == store), avg_col] = rolling_avg.values
        
        # Fill NaN values in lag features
        lag_cols = [col for col in merged_df.columns if col.startswith('Sales_Lag_') or col.startswith('Sales_Avg_')]
        for col in lag_cols:
            merged_df[col] = merged_df.groupby(['Store_Id', 'Item'])[col].transform(
                lambda x: x.fillna(x.mean()) if not x.isna().all() else 0
            )
        
        logger.info(f"Successfully processed pizza datasets into a combined dataset with {len(merged_df)} records")
        return merged_df
    
    except Exception as e:
        logger.error(f"Error loading pizza datasets: {str(e)}", exc_info=True)
        raise

def validate_combined_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate the combined dataset for consistency and required fields.
    
    Args:
        df: Combined DataFrame to validate
        
    Returns:
        Tuple[bool, str]: Success status and error message if failed
    """
    if df is None:
        return False, "Combined dataset is None"
        
    if df.empty:
        return False, "Combined dataset is empty"
    
    # Check for required columns
    required_columns = ['Store_Id', 'Item', 'Date', 'Product', 'Sales', 'Price']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns in combined dataset: {missing_columns}"
    
    # Check for critical data issues
    if df['Date'].isna().any():
        logger.error("Dataset contains missing dates")
        # Attempt to fix by dropping rows with missing dates
        original_len = len(df)
        df.dropna(subset=['Date'], inplace=True)
        logger.warning(f"Dropped {original_len - len(df)} rows with missing dates")
        if len(df) == 0:
            return False, "All rows had missing dates, resulting in empty dataset"
    
    if df['Store_Id'].isna().any():
        logger.error("Dataset contains missing store IDs")
        # Try to fix by filling with most common store ID
        most_common = df['Store_Id'].mode()[0]
        df['Store_Id'].fillna(most_common, inplace=True)
        logger.warning(f"Filled missing store IDs with most common value: {most_common}")
    
    if df['Item'].isna().any():
        logger.error("Dataset contains missing item IDs")
        # Try to fix by using Product name where possible
        for idx in df[df['Item'].isna()].index:
            if not pd.isna(df.loc[idx, 'Product']):
                # Create a consistent item ID from product name
                product = df.loc[idx, 'Product']
                # Find if this product has an item ID elsewhere
                matching_items = df[(df['Product'] == product) & (~df['Item'].isna())]['Item']
                if not matching_items.empty:
                    df.loc[idx, 'Item'] = matching_items.iloc[0]
                else:
                    # Create a new item ID
                    df.loc[idx, 'Item'] = f"SYNTH_{hash(product) % 10000}"
        
        # Check if we still have NaN items
        if df['Item'].isna().any():
            # Assign synthetic IDs to remaining NaNs
            for idx in df[df['Item'].isna()].index:
                df.loc[idx, 'Item'] = f"SYNTH_{idx}"
        
        logger.warning(f"Fixed missing item IDs with synthetic values")
    
    # Ensure data types are correct
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Store_Id'] = df['Store_Id'].astype(str)
        df['Item'] = df['Item'].astype(str)
    except Exception as e:
        logger.error(f"Error converting data types: {str(e)}")
        return False, f"Data type conversion error: {str(e)}"
    
    # Basic data quality checks and corrections
    if df['Sales'].isna().any():
        logger.warning("Dataset contains missing sales values")
        # Fill with either historical average or 0
        df['Sales'] = df.groupby(['Store_Id', 'Item'])['Sales'].transform(
            lambda x: x.fillna(x.mean() if not x.mean() != x.mean() else 0)
        )
    
    if (df['Sales'] < 0).any():
        logger.warning("Dataset contains negative sales values")
        logger.info("Fixing negative sales values by replacing with zero")
        df['Sales'] = df['Sales'].apply(lambda x: max(0, x))
    
    if df['Price'].isna().any():
        logger.warning("Dataset contains missing price values")
        # Fill with either product average or reasonable default
        df['Price'] = df.groupby(['Store_Id', 'Item'])['Price'].transform(
            lambda x: x.fillna(x.mean() if not x.mean() != x.mean() else 9.99)
        )
    
    if (df['Price'] < 0).any():
        logger.warning("Dataset contains negative price values")
        logger.info("Fixing negative price values by replacing with product average")
        # Replace negative prices with the product's average price
        for item in df['Item'].unique():
            item_mask = df['Item'] == item
            avg_price = df.loc[item_mask & (df['Price'] > 0), 'Price'].mean()
            if pd.isna(avg_price) or avg_price == 0:
                avg_price = 9.99  # Fallback if no positive prices available
            df.loc[item_mask & (df['Price'] < 0), 'Price'] = avg_price
    
    # Check for date range
    today = datetime.now()
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    if max_date > today + timedelta(days=30):
        logger.warning(f"Dataset contains future dates beyond 30 days: max={max_date}")
    
    if min_date > today or max_date < datetime(2020, 1, 1):
        logger.warning(f"Dataset date range appears unusual: {min_date} to {max_date}")
    
    return True, ""


def save_combined_dataset(df, output_file=COMBINED_DATA_FILE, validate=True):
    """
    Save the combined dataset to a CSV file
    
    Args:
        df (pandas.DataFrame): The combined dataset to save
        output_file (str or Path): Path where to save the CSV file
        
        validate: Whether to validate the dataset before saving
        
    Returns:
        str: Path to the saved file
    """
    try:
        # Create a copy to avoid modifying the original dataframe
        save_df = df.copy()
        
        # Ensure parent directory exists
        output_path = pathlib.Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate dataset if requested
        if validate:
            valid, error_msg = validate_combined_data(save_df)
            if not valid:
                logger.error(f"Data validation failed: {error_msg}")
                # Try to fix common issues before saving
                logger.warning("Attempting to fix common data issues")
                
                # Ensure core columns exist
                required_columns = ['Store_Id', 'Item', 'Date', 'Product', 'Sales', 'Price']
                for col in required_columns:
                    if col not in save_df.columns:
                        if col == 'Sales' and 'Units_Sold' in save_df.columns:
                            save_df['Sales'] = save_df['Units_Sold']
                        elif col == 'Price' and 'Unit_Price' in save_df.columns:
                            save_df['Price'] = save_df['Unit_Price']
                        elif col == 'Product' and 'Product_Name' in save_df.columns:
                            save_df['Product'] = save_df['Product_Name']
                        else:
                            # Create a default column
                            logger.warning(f"Creating default values for missing column: {col}")
                            if col == 'Date':
                                save_df['Date'] = datetime.now()
                            elif col == 'Store_Id':
                                save_df['Store_Id'] = '1'
                            elif col == 'Item':
                                save_df['Item'] = save_df.index.astype(str)
                            elif col == 'Product':
                                save_df['Product'] = 'Unknown Product'
                            elif col == 'Sales':
                                save_df['Sales'] = 0
                            elif col == 'Price':
                                save_df['Price'] = 9.99
        
        # Ensure dates are in the proper format
        if 'Date' in save_df.columns and save_df['Date'].dtype != 'datetime64[ns]':
            try:
                save_df['Date'] = pd.to_datetime(save_df['Date'])
            except Exception as e:
                logger.error(f"Failed to convert Date column: {str(e)}")
        
        # Save the dataset
        save_df.to_csv(output_file, index=False)
        logger.info(f"Saved combined dataset to {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error saving dataset to {output_file}: {str(e)}", exc_info=True)
        
        # Try to save to a backup location with reduced columns if needed
        try:
            backup_file = str(output_file) + ".backup"
            
            # If the error might be complex data structures, try saving only core columns
            try:
                core_columns = [col for col in df.columns if col in [
                    'Store_Id', 'Item', 'Date', 'Product', 'Sales', 'Price',
                    'Cost', 'Stock_Level', 'Weather', 'Promotion', 'Day_Of_Week',
                    'Month', 'Year', 'Day'
                ]]
                
                if len(core_columns) >= 5:  # Ensure we have enough important columns
                    reduced_df = df[core_columns].copy()
                    reduced_df.to_csv(backup_file, index=False)
                    logger.info(f"Saved reduced backup dataset to {backup_file}")
                    return backup_file
            except Exception:
                # If reduction failed, try standard backup
                pass
                
            # Standard backup attempt    
            df.to_csv(backup_file, index=False)
            logger.info(f"Saved backup dataset to {backup_file}")
            return backup_file
        except Exception as e2:
            logger.error(f"Failed to save backup: {str(e2)}")
            raise e  # Re-raise the original exception

def process_data(fallback_dir=None, create_synthetic_if_missing=True, validate=True):
    """
    Main function to load, process and save the dataset.
    This function is called when the script is run directly.
    
    Args:
        fallback_dir (str or Path, optional): Directory with fallback data files
        create_synthetic_if_missing (bool): Create synthetic data if files are missing
        validate (bool): Whether to validate the combined dataset
    
    Returns:
        str: Path to the saved combined dataset file
    """
    try:
        # Load and process the pizza datasets
        combined_df = load_pizza_datasets(fallback_dir=fallback_dir, 
                                         create_synthetic_if_missing=create_synthetic_if_missing)
        
        if combined_df is None or combined_df.empty:
            logger.critical("Failed to load or generate dataset")
            raise ValueError("Failed to load or generate valid dataset")
        
        # Validate the combined dataset
        if validate:
            valid, error_msg = validate_combined_data(combined_df)
            if not valid:
                logger.warning(f"Generated dataset has validation issues: {error_msg}")
                
                # Attempt auto-fixing common issues
                try:
                    logger.info("Attempting to auto-fix data issues...")
                    
                    # Fix data type issues
                    for col in combined_df.columns:
                        if col in ['Store_Id', 'Item']:
                            combined_df[col] = combined_df[col].astype(str)
                        elif col == 'Date' and combined_df[col].dtype != 'datetime64[ns]':
                            combined_df[col] = pd.to_datetime(combined_df[col], errors='coerce')
                    
                    # Drop rows with critical missing values
                    critical_cols = ['Store_Id', 'Item', 'Date']
                    before_len = len(combined_df)
                    combined_df.dropna(subset=critical_cols, inplace=True)
                    after_len = len(combined_df)
                    
                    if before_len > after_len:
                        logger.warning(f"Dropped {before_len - after_len} rows with missing critical data")
                    
                    # Re-validate
                    valid, error_msg = validate_combined_data(combined_df)
                    if not valid:
                        logger.error(f"Auto-fixing failed. Dataset still has validation issues: {error_msg}")
                    else:
                        logger.info("Auto-fixing succeeded. Dataset is now valid.")
                        
                except Exception as fix_error:
                    logger.error(f"Error while trying to fix data issues: {str(fix_error)}")
        
        # Save the combined dataset
        return save_combined_dataset(combined_df, validate=validate)
    except Exception as e:
        logger.critical(f"Critical error in data processing: {str(e)}", exc_info=True)
        
        # Last resort: try to create a simple synthetic dataset if everything else fails
        if create_synthetic_if_missing:
            try:
                logger.warning("Attempting to create emergency synthetic dataset...")
                from create_sample_data import create_sample_data
                emergency_df = create_sample_data(num_stores=1, num_products=3, num_days=60)
                
                if emergency_df is not None and not emergency_df.empty:
                    logger.info(f"Created emergency synthetic dataset with {len(emergency_df)} records")
                    return save_combined_dataset(emergency_df, validate=False)
            except Exception as e2:
                logger.critical(f"Failed to create emergency synthetic dataset: {str(e2)}")
                
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process pizza sales, purchases, and stock data')
    parser.add_argument('--fallback-dir', help='Directory with fallback data files')
    args = parser.parse_args()
    
    process_data(fallback_dir=args.fallback_dir)