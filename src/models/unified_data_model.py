"""
Unified Data Model for connecting inventory, pricing, and demand forecasting.

This module provides a centralized data model that connects the different
components of the system: inventory management, price optimization, and demand
forecasting. It ensures that changes in one component propagate correctly to
the others.
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('unified_data_model')

class UnifiedDataModel:
    """
    Unified data model that connects inventory, pricing, and demand forecasting.
    
    This class serves as a central hub for data processing, ensuring that 
    changes in one component (e.g., pricing) correctly affect other components
    (e.g., demand forecasting, inventory projections).
    """
    
    def __init__(self, data_dict=None):
        """
        Initialize the unified data model with the provided data.
        
        Args:
            data_dict: Dictionary containing all available data
        """
        self.data_dict = data_dict or {}
        self._validate_data()
        self.modified = False
        # Track changes applied to the data
        self.applied_changes = {
            'price_adjustments': {},
            'inventory_adjustments': {},
            'forecast_adjustments': {}
        }
    
    def _validate_data(self):
        """
        Validate the input data and standardize column names.
        Adds fallback data generation for missing datasets.
        """
        required_datasets = ['combined_data', 'forecasts', 'price_elasticities', 'inventory_projection']
        missing_datasets = []
        
        for dataset_name in required_datasets:
            if dataset_name not in self.data_dict or self.data_dict[dataset_name] is None:
                logger.warning(f"Missing required dataset: {dataset_name}")
                missing_datasets.append(dataset_name)
        
        # Generate fallback data if needed
        if missing_datasets:
            self._generate_fallback_data(missing_datasets)
        
        # Standardize column names for all datasets
        self._standardize_column_names()
        
        # Validate data types and values in each dataset
        self._validate_data_values()
    
    def get_data(self, dataset_name):
        """
        Get data for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame or None if not available
        """
        return self.data_dict.get(dataset_name)
    
    def _generate_fallback_data(self, missing_datasets):
        """
        Generate fallback data for missing datasets.
        
        Args:
            missing_datasets: List of missing dataset names
        """
        # If combined_data is missing, we need to import from the data loader
        if 'combined_data' in missing_datasets:
            try:
                logger.info("Attempting to load data through data_loader...")
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                from src.data.data_loader import load_pizza_datasets
                
                # Try to load data with synthetic generation if needed
                self.data_dict['combined_data'] = load_pizza_datasets(create_synthetic_if_missing=True)
                logger.info(f"Successfully loaded combined_data with {len(self.data_dict['combined_data'])} records")
            except Exception as e:
                logger.error(f"Failed to load combined_data: {str(e)}")
                # Create minimal synthetic data as absolute fallback
                self._create_minimal_synthetic_data()
        
        # Generate forecasts if missing
        if 'forecasts' in missing_datasets and 'combined_data' in self.data_dict:
            self._generate_forecasts_from_combined_data()
        
        # Generate price elasticities if missing
        if 'price_elasticities' in missing_datasets and 'combined_data' in self.data_dict:
            self._generate_elasticities_from_combined_data()
            
        # Generate inventory projection if missing
        if 'inventory_projection' in missing_datasets and 'combined_data' in self.data_dict:
            self._generate_inventory_from_combined_data()
    
    def _create_minimal_synthetic_data(self):
        """
        Create minimal synthetic data as a last resort.
        """
        logger.warning("Creating minimal synthetic data as fallback")
        
        # Create a date range
        today = datetime.now().date()
        dates = pd.date_range(start=today - timedelta(days=60), end=today + timedelta(days=30))
        
        # Create a simple combined dataset
        store_ids = ['1', '2']
        items = ['1001', '1002', '1003']
        products = ['Pepperoni Pizza', 'Cheese Pizza', 'Supreme Pizza']
        
        records = []
        for store_id in store_ids:
            for idx, item in enumerate(items):
                product = products[idx % len(products)]
                base_price = 8.99 + (idx * 1.0)
                base_sales = 10 + (idx * 2)
                
                for date in dates:
                    # Add some variability
                    day_factor = 1.0 + 0.2 * np.sin(date.weekday() * np.pi / 3.5)
                    sales = max(0, int(base_sales * day_factor * np.random.normal(1, 0.2)))
                    price = round(base_price * np.random.normal(1, 0.05), 2)
                    
                    records.append({
                        'Date': date,
                        'Store_Id': store_id,
                        'Item': item,
                        'Product': product,
                        'Sales': sales,
                        'Price': price,
                        'Cost': price * 0.6,
                        'Stock_Level': max(0, int(sales * 14 * np.random.normal(1, 0.3)))
                    })
        
        self.data_dict['combined_data'] = pd.DataFrame(records)
        logger.info(f"Created minimal synthetic dataset with {len(records)} records")
    
    def _generate_forecasts_from_combined_data(self):
        """
        Generate forecast data from combined_data.
        """
        logger.info("Generating synthetic forecasts from combined data")
        combined_data = self.data_dict['combined_data']
        
        if combined_data is None or combined_data.empty:
            logger.error("Cannot generate forecasts: combined_data is missing or empty")
            return
            
        # Get today's date from the data or use current date
        today = datetime.now().date()
        if 'Date' in combined_data.columns:
            try:
                max_date = combined_data['Date'].max().date()
                if max_date < today - timedelta(days=30):
                    # If data is very old, use today
                    pass
                else:
                    today = max_date
            except Exception:
                # Continue with current date if there's an error
                pass
                
        # Generate forecast dates (14 days)
        forecast_dates = pd.date_range(start=today, periods=14)
        
        # Get unique store-item combinations
        if 'Store_Id' in combined_data.columns and 'Item' in combined_data.columns:
            store_items = combined_data[['Store_Id', 'Item', 'Product']].drop_duplicates()
        else:
            # Create dummy store-item combinations if missing
            store_items = pd.DataFrame({
                'Store_Id': ['1', '2'],
                'Item': ['1001', '1002'],
                'Product': ['Pepperoni Pizza', 'Cheese Pizza']
            })
            
        # Create forecast records
        records = []
        for _, row in store_items.iterrows():
            store_id = row['Store_Id']
            item = row['Item']
            product = row['Product']
            
            # Get average sales for this product
            avg_sales = 10  # Default
            product_data = combined_data[(combined_data['Store_Id'] == store_id) & 
                                        (combined_data['Item'] == item)]
            if not product_data.empty and 'Sales' in product_data.columns:
                avg_sales = product_data['Sales'].mean()
                if pd.isna(avg_sales) or avg_sales <= 0:
                    avg_sales = 10
            
            for date in forecast_dates:
                # Create a slightly increasing/decreasing trend with randomness
                day_idx = (date - forecast_dates[0]).days
                trend_factor = 1.0 + (day_idx / 30) * 0.05  # 5% change over 30 days
                sales_forecast = max(0, int(avg_sales * trend_factor * np.random.normal(1, 0.15)))
                
                upper_bound = int(sales_forecast * 1.3)
                lower_bound = int(sales_forecast * 0.7)
                
                records.append({
                    'Date': date,
                    'Store_Id': store_id,
                    'Item': item,
                    'Product': product,
                    'Forecast': sales_forecast,
                    'Upper_Bound': upper_bound,
                    'Lower_Bound': lower_bound
                })
        
        self.data_dict['forecasts'] = pd.DataFrame(records)
        logger.info(f"Generated {len(records)} forecast records")
    
    def _generate_elasticities_from_combined_data(self):
        """
        Generate price elasticity data from combined_data.
        """
        logger.info("Generating synthetic price elasticities from combined data")
        combined_data = self.data_dict['combined_data']
        
        if combined_data is None or combined_data.empty:
            logger.error("Cannot generate elasticities: combined_data is missing or empty")
            return
            
        # Get unique store-item combinations
        if 'Store_Id' in combined_data.columns and 'Item' in combined_data.columns:
            store_items = combined_data[['Store_Id', 'Item', 'Product']].drop_duplicates()
        else:
            # Create dummy store-item combinations if missing
            store_items = pd.DataFrame({
                'Store_Id': ['1', '2'],
                'Item': ['1001', '1002'],
                'Product': ['Pepperoni Pizza', 'Cheese Pizza']
            })
        
        records = []
        for _, row in store_items.iterrows():
            store_id = row['Store_Id']
            item = row['Item']
            product = row['Product']
            
            # Get average price and cost for this product
            avg_price = 9.99  # Default
            avg_cost = 5.99   # Default
            
            product_data = combined_data[(combined_data['Store_Id'] == store_id) & 
                                        (combined_data['Item'] == item)]
            if not product_data.empty:
                if 'Price' in product_data.columns:
                    avg_price = product_data['Price'].mean()
                    if pd.isna(avg_price) or avg_price <= 0:
                        avg_price = 9.99
                        
                if 'Cost' in product_data.columns:
                    avg_cost = product_data['Cost'].mean()
                    if pd.isna(avg_cost) or avg_cost <= 0:
                        avg_cost = avg_price * 0.6
            
            # Generate a reasonable elasticity value
            # Most grocery items are between -0.5 and -2.5
            elasticity = round(np.random.uniform(-2.0, -0.5), 2)
            
            records.append({
                'Store_Id': store_id,
                'Item': item,
                'Product': product,
                'Elasticity': elasticity,
                'Current_Price': avg_price,
                'Cost': avg_cost,
                'Margin': (avg_price - avg_cost) / avg_price
            })
        
        self.data_dict['price_elasticities'] = pd.DataFrame(records)
        logger.info(f"Generated {len(records)} price elasticity records")
    
    def _generate_inventory_from_combined_data(self):
        """
        Generate inventory projection data from combined_data.
        """
        logger.info("Generating synthetic inventory projection from combined data")
        combined_data = self.data_dict['combined_data']
        forecasts = self.data_dict.get('forecasts')
        
        if combined_data is None or combined_data.empty:
            logger.error("Cannot generate inventory: combined_data is missing or empty")
            return
            
        # Get today's date from the data or use current date
        today = datetime.now().date()
        if 'Date' in combined_data.columns:
            try:
                today = combined_data['Date'].max().date()
            except Exception:
                # Continue with current date if there's an error
                pass
                
        # Get forecast dates or generate them
        if forecasts is not None and not forecasts.empty and 'Date' in forecasts.columns:
            forecast_dates = forecasts['Date'].sort_values().unique()
        else:
            forecast_dates = pd.date_range(start=today, periods=14)
        
        # Get unique store-item combinations
        if 'Store_Id' in combined_data.columns and 'Item' in combined_data.columns:
            store_items = combined_data[['Store_Id', 'Item', 'Product']].drop_duplicates()
        else:
            # Create dummy store-item combinations if missing
            store_items = pd.DataFrame({
                'Store_Id': ['1', '2'],
                'Item': ['1001', '1002'],
                'Product': ['Pepperoni Pizza', 'Cheese Pizza']
            })
        
        records = []
        for _, row in store_items.iterrows():
            store_id = row['Store_Id']
            item = row['Item']
            product = row['Product']
            
            # Get current stock level
            current_stock = 50  # Default
            product_data = combined_data[(combined_data['Store_Id'] == store_id) & 
                                        (combined_data['Item'] == item)]
            
            if not product_data.empty and 'Stock_Level' in product_data.columns:
                # Get the latest stock level
                latest_data = product_data.sort_values('Date').iloc[-1]
                if 'Stock_Level' in latest_data and not pd.isna(latest_data['Stock_Level']):
                    current_stock = latest_data['Stock_Level']
            
            # Get daily sales forecast
            daily_sales = 7  # Default
            if forecasts is not None and not forecasts.empty:
                forecast_data = forecasts[(forecasts['Store_Id'] == store_id) & 
                                         (forecasts['Item'] == item)]
                if not forecast_data.empty and 'Forecast' in forecast_data.columns:
                    daily_sales = forecast_data['Forecast'].mean()
                    if pd.isna(daily_sales) or daily_sales <= 0:
                        daily_sales = 7
            
            # Project inventory levels
            stock_level = current_stock
            for date in forecast_dates:
                if forecasts is not None and not forecasts.empty:
                    # Look up the specific forecast for this date
                    day_forecast = forecasts[
                        (forecasts['Store_Id'] == store_id) & 
                        (forecasts['Item'] == item) & 
                        (forecasts['Date'] == date)
                    ]
                    
                    if not day_forecast.empty and 'Forecast' in day_forecast.columns:
                        daily_demand = day_forecast['Forecast'].iloc[0]
                    else:
                        daily_demand = daily_sales
                else:
                    daily_demand = daily_sales
                
                # Update stock level
                stock_level = max(0, stock_level - daily_demand)
                
                # Determine stock status
                if stock_level < daily_sales * 7:  # Less than a week of stock
                    status = "Low"
                elif stock_level < daily_sales * 14:  # Less than two weeks
                    status = "Adequate"
                else:
                    status = "Excess"
                
                records.append({
                    'Date': date,
                    'Store_Id': store_id,
                    'Item': item,
                    'Product': product,
                    'Stock_Level': stock_level,
                    'Daily_Demand': daily_demand,
                    'Stock_Status': status
                })
        
        self.data_dict['inventory_projection'] = pd.DataFrame(records)
        logger.info(f"Generated {len(records)} inventory projection records")
    
    def _standardize_column_names(self):
        """
        Standardize column names across all datasets.
        """
        # Standardize forecasts column names
        if 'forecasts' in self.data_dict and self.data_dict['forecasts'] is not None:
            df = self.data_dict['forecasts']
            # Log available columns before standardization
            logger.info(f"Forecasts columns before standardization: {df.columns.tolist()}")
            # Rename columns if needed
            if 'Forecast' not in df.columns and 'Predicted_Demand' in df.columns:
                df = df.rename(columns={'Predicted_Demand': 'Forecast'})
                self.data_dict['forecasts'] = df
                logger.info("Renamed 'Predicted_Demand' to 'Forecast' in forecasts")
            elif 'Forecast' not in df.columns and 'Predicted_Sales' in df.columns:
                df = df.rename(columns={'Predicted_Sales': 'Forecast'})
                self.data_dict['forecasts'] = df
                logger.info("Renamed 'Predicted_Sales' to 'Forecast' in forecasts")
            # Special handling for ARIMA forecasts which might have other column names
            elif 'Forecast' not in df.columns:
                # Try to find any column that might contain forecast values
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                potential_forecast_cols = [col for col in numeric_cols if col not in ['Store_Id', 'Item', 'Days_In_Future']]
                if potential_forecast_cols:
                    # Use the first potential forecast column
                    forecast_col = potential_forecast_cols[0]
                    df = df.rename(columns={forecast_col: 'Forecast'})
                    self.data_dict['forecasts'] = df
                    logger.info(f"Renamed '{forecast_col}' to 'Forecast' in forecasts as fallback")
            # Log available columns after standardization
            logger.info(f"Forecasts columns after standardization: {df.columns.tolist()}")
        
        # Also standardize pytorch_forecasts, rf_forecasts and arima_forecasts column names
        for forecast_type in ['pytorch_forecasts', 'rf_forecasts', 'arima_forecasts']:
            if forecast_type in self.data_dict and self.data_dict[forecast_type] is not None:
                df = self.data_dict[forecast_type]
                # Log available columns for this forecast type
                logger.info(f"{forecast_type} columns before standardization: {df.columns.tolist()}")
                
                # Ensure either Predicted_Demand or Forecast column exists
                if 'Predicted_Demand' not in df.columns and 'Forecast' in df.columns:
                    df = df.rename(columns={'Forecast': 'Predicted_Demand'})
                    self.data_dict[forecast_type] = df
                    logger.info(f"Renamed 'Forecast' to 'Predicted_Demand' in {forecast_type}")
                elif 'Forecast' not in df.columns and 'Predicted_Demand' in df.columns:
                    df = df.rename(columns={'Predicted_Demand': 'Forecast'})
                    self.data_dict[forecast_type] = df
                    logger.info(f"Renamed 'Predicted_Demand' to 'Forecast' in {forecast_type}")
                elif 'Predicted_Sales' in df.columns:
                    # Add both standardized column names for maximum compatibility
                    df['Forecast'] = df['Predicted_Sales']
                    df['Predicted_Demand'] = df['Predicted_Sales']
                    self.data_dict[forecast_type] = df
                    logger.info(f"Added 'Forecast' and 'Predicted_Demand' based on 'Predicted_Sales' in {forecast_type}")
                # Handle case where neither standard column exists but we have numeric columns
                elif 'Predicted_Demand' not in df.columns and 'Forecast' not in df.columns:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    potential_forecast_cols = [col for col in numeric_cols if col not in ['Store_Id', 'Item', 'Days_In_Future']]
                    if potential_forecast_cols:
                        # Use the first potential forecast column
                        forecast_col = potential_forecast_cols[0]
                        df['Forecast'] = df[forecast_col]
                        df['Predicted_Demand'] = df[forecast_col]
                        self.data_dict[forecast_type] = df
                        logger.info(f"Added 'Forecast' and 'Predicted_Demand' based on '{forecast_col}' in {forecast_type}")
                
                # Log after standardization
                logger.info(f"{forecast_type} columns after standardization: {df.columns.tolist()}")
        
        # Standardize price elasticities column names
        if 'price_elasticities' in self.data_dict and self.data_dict['price_elasticities'] is not None:
            df = self.data_dict['price_elasticities']
            # Standardize price column name
            if 'Current_Price' not in df.columns and 'Avg_Price' in df.columns:
                df = df.rename(columns={'Avg_Price': 'Current_Price'})
                self.data_dict['price_elasticities'] = df
            if 'Elasticity' not in df.columns and 'Price_Elasticity' in df.columns:
                df = df.rename(columns={'Price_Elasticity': 'Elasticity'})
                self.data_dict['price_elasticities'] = df
        
        # Standardize inventory column names
        if 'inventory_projection' in self.data_dict and self.data_dict['inventory_projection'] is not None:
            df = self.data_dict['inventory_projection']
            # Standardize stock level column name
            if 'Stock_Level' not in df.columns and 'Current_Stock' in df.columns:
                df = df.rename(columns={'Current_Stock': 'Stock_Level'})
                self.data_dict['inventory_projection'] = df
            if 'Stock_Status' not in df.columns and 'Status' in df.columns:
                df = df.rename(columns={'Status': 'Stock_Status'})
                self.data_dict['inventory_projection'] = df
    
    def _validate_data_values(self):
        """
        Validate data types and values in each dataset.
        """
        # Validate combined_data
        if 'combined_data' in self.data_dict and self.data_dict['combined_data'] is not None:
            df = self.data_dict['combined_data']
            
            # Ensure Store_Id and Item are strings
            if 'Store_Id' in df.columns:
                df['Store_Id'] = df['Store_Id'].astype(str)
            
            if 'Item' in df.columns:
                df['Item'] = df['Item'].astype(str)
            
            # Ensure Date is datetime
            if 'Date' in df.columns and df['Date'].dtype != 'datetime64[ns]':
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                except Exception as e:
                    logger.error(f"Error converting Date column in combined_data: {str(e)}")
            
            # Ensure Sales and Price are non-negative
            if 'Sales' in df.columns:
                df['Sales'] = df['Sales'].apply(lambda x: max(0, x) if not pd.isna(x) else 0)
            
            if 'Price' in df.columns:
                df['Price'] = df['Price'].apply(lambda x: max(0, x) if not pd.isna(x) else 0)
            
            self.data_dict['combined_data'] = df
        
        # Validate forecasts
        if 'forecasts' in self.data_dict and self.data_dict['forecasts'] is not None:
            df = self.data_dict['forecasts']
            
            # Ensure Store_Id and Item are strings
            if 'Store_Id' in df.columns:
                df['Store_Id'] = df['Store_Id'].astype(str)
            
            if 'Item' in df.columns:
                df['Item'] = df['Item'].astype(str)
            
            # Ensure Date is datetime
            if 'Date' in df.columns and df['Date'].dtype != 'datetime64[ns]':
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                except Exception as e:
                    logger.error(f"Error converting Date column in forecasts: {str(e)}")
            
            # Ensure Forecast is non-negative
            if 'Forecast' in df.columns:
                df['Forecast'] = df['Forecast'].apply(lambda x: max(0, x) if not pd.isna(x) else 0)
            
            # Create Upper_Bound and Lower_Bound if missing
            if 'Forecast' in df.columns and ('Upper_Bound' not in df.columns or 'Lower_Bound' not in df.columns):
                df['Upper_Bound'] = df['Forecast'] * 1.3
                df['Lower_Bound'] = df['Forecast'] * 0.7
            
            self.data_dict['forecasts'] = df
        
        # Validate price elasticities
        if 'price_elasticities' in self.data_dict and self.data_dict['price_elasticities'] is not None:
            df = self.data_dict['price_elasticities']
            
            # Ensure Store_Id and Item are strings
            if 'Store_Id' in df.columns:
                df['Store_Id'] = df['Store_Id'].astype(str)
            
            if 'Item' in df.columns:
                df['Item'] = df['Item'].astype(str)
            
            # Ensure prices are non-negative
            if 'Current_Price' in df.columns:
                df['Current_Price'] = df['Current_Price'].apply(lambda x: max(0.01, x) if not pd.isna(x) else 9.99)
            
            if 'Cost' in df.columns:
                df['Cost'] = df['Cost'].apply(lambda x: max(0.01, x) if not pd.isna(x) else 5.99)
            elif 'Current_Price' in df.columns:
                df['Cost'] = df['Current_Price'] * 0.6  # Default 60% cost
            
            # Ensure elasticity is reasonable (between -3 and 0)
            if 'Elasticity' in df.columns:
                df['Elasticity'] = df['Elasticity'].apply(
                    lambda x: min(0, max(-3, x)) if not pd.isna(x) else -1.0
                )
            
            self.data_dict['price_elasticities'] = df
        
        # Validate inventory projection
        if 'inventory_projection' in self.data_dict and self.data_dict['inventory_projection'] is not None:
            df = self.data_dict['inventory_projection']
            
            # Ensure Store_Id and Item are strings
            if 'Store_Id' in df.columns:
                df['Store_Id'] = df['Store_Id'].astype(str)
            
            if 'Item' in df.columns:
                df['Item'] = df['Item'].astype(str)
            
            # Ensure Date is datetime
            if 'Date' in df.columns and df['Date'].dtype != 'datetime64[ns]':
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                except Exception as e:
                    logger.error(f"Error converting Date column in inventory_projection: {str(e)}")
            
            # Ensure Stock_Level is non-negative
            if 'Stock_Level' in df.columns:
                df['Stock_Level'] = df['Stock_Level'].apply(lambda x: max(0, x) if not pd.isna(x) else 0)
            
            self.data_dict['inventory_projection'] = df
            
    def get_product_data(self, store_id, item_id):
        """
        Get all data for a specific product across all datasets.
        
        Args:
            store_id: Store ID
            item_id: Product/item ID
            
        Returns:
            dict: Dictionary of datasets filtered for this product
        """
        # Convert store_id and item_id to string for consistent comparison
        store_id = str(store_id)
        item_id = str(item_id)
        
        product_data = {}
        
        for dataset_name, df in self.data_dict.items():
            if df is None or len(df) == 0:
                continue
                
            if 'Store_Id' in df.columns and 'Item' in df.columns:
                try:
                    # Ensure comparison columns are strings
                    df_copy = df.copy()
                    df_copy['Store_Id'] = df_copy['Store_Id'].astype(str)
                    df_copy['Item'] = df_copy['Item'].astype(str)
                    
                    filtered_df = df_copy[(df_copy['Store_Id'] == store_id) & (df_copy['Item'] == item_id)]
                    if len(filtered_df) > 0:
                        product_data[dataset_name] = filtered_df
                    else:
                        logger.warning(f"No data found for store {store_id}, product {item_id} in {dataset_name}")
                except Exception as e:
                    logger.error(f"Error filtering {dataset_name} for product: {str(e)}")
        
        # If no data was found for any dataset, try to generate some
        if not product_data:
            logger.warning(f"No data found for store {store_id}, product {item_id} in any dataset")
            
            # Try to see if any product exists with this store
            store_exists = False
            for _, df in self.data_dict.items():
                if df is not None and 'Store_Id' in df.columns:
                    df_copy = df.copy()
                    df_copy['Store_Id'] = df_copy['Store_Id'].astype(str)
                    if store_id in df_copy['Store_Id'].values:
                        store_exists = True
                        break
            
            if store_exists:
                # Store exists but item doesn't, try synthetic item data
                logger.info(f"Generating synthetic data for store {store_id}, product {item_id}")
                self._generate_synthetic_product_data(store_id, item_id)
                
                # Try to get the data again
                return self.get_product_data(store_id, item_id)
        
        return product_data
    
    def adjust_price(self, store_id, item_id, price_adjustment_pct):
        """
        Apply a price adjustment and update all affected components.
        
        Args:
            store_id: Store ID
            item_id: Product/item ID
            price_adjustment_pct: Percentage adjustment to price
            
        Returns:
            dict: Updated data for the product
        """
        # Make sure price elasticities and forecasts exist
        price_elasticities = self.data_dict.get('price_elasticities')
        forecasts = self.data_dict.get('forecasts')
        inventory_projection = self.data_dict.get('inventory_projection')
        
        if price_elasticities is None or forecasts is None:
            logger.warning("Cannot adjust price: missing required data")
            return None
        
        # Get elasticity data for the product
        elasticity_data = price_elasticities[(price_elasticities['Store_Id'] == store_id) & 
                                           (price_elasticities['Item'] == item_id)]
        
        if len(elasticity_data) == 0:
            logger.warning(f"No elasticity data found for store {store_id}, product {item_id}")
            return None
        
        # Get current price and elasticity
        elasticity = elasticity_data['Elasticity'].iloc[0]
        current_price = elasticity_data['Current_Price'].iloc[0]
        
        # Calculate new price
        new_price = current_price * (1 + price_adjustment_pct/100)
        price_ratio = new_price / current_price
        
        # Calculate demand impact based on price elasticity
        quantity_ratio = price_ratio ** elasticity
        
        # Store the price adjustment
        key = f"{store_id}_{item_id}"
        self.applied_changes['price_adjustments'][key] = {
            'price_adjustment_pct': price_adjustment_pct,
            'old_price': current_price,
            'new_price': new_price,
            'quantity_ratio': quantity_ratio
        }
        
        # Update forecasts
        forecast_data = forecasts[(forecasts['Store_Id'] == store_id) & 
                                 (forecasts['Item'] == item_id)]
        
        if len(forecast_data) > 0:
            # Deep copy to avoid modifying the original
            adjusted_forecasts = forecasts.copy()
            
            # Apply adjustment to forecasts
            mask = (adjusted_forecasts['Store_Id'] == store_id) & (adjusted_forecasts['Item'] == item_id)
            adjusted_forecasts.loc[mask, 'Forecast'] = adjusted_forecasts.loc[mask, 'Forecast'] * quantity_ratio
            
            # Update data dictionary
            self.data_dict['adjusted_forecasts'] = adjusted_forecasts
            self.modified = True
        
        # If inventory projection exists, update it based on new forecast
        if inventory_projection is not None:
            inv_data = inventory_projection[(inventory_projection['Store_Id'] == store_id) & 
                                          (inventory_projection['Item'] == item_id)]
            
            if len(inv_data) > 0:
                # Deep copy
                adjusted_inventory = inventory_projection.copy()
                
                # Get the last known stock level
                mask = (adjusted_inventory['Store_Id'] == store_id) & (adjusted_inventory['Item'] == item_id)
                latest_record = adjusted_inventory[mask].sort_values('Date').iloc[-1]
                
                # Column that contains stock level might vary
                stock_col = 'Stock_Level' if 'Stock_Level' in adjusted_inventory.columns else 'Current_Stock'
                current_stock = latest_record[stock_col]
                
                # Re-project inventory based on adjusted forecast
                dates = adjusted_inventory.loc[mask, 'Date'].sort_values().unique()
                daily_demand = adjusted_forecasts[mask]['Forecast'].values
                
                # Recalculate projected stock
                projected_stock = [current_stock]
                for i in range(1, len(daily_demand)):
                    next_stock = max(0, projected_stock[-1] - daily_demand[i-1])
                    projected_stock.append(next_stock)
                
                # Update stock projection
                adjusted_inventory.loc[mask, stock_col] = projected_stock
                
                # Update data dictionary
                self.data_dict['adjusted_inventory'] = adjusted_inventory
        
        # Return the updated data for this product
        return self.get_product_data(store_id, item_id)
    
    def adjust_inventory(self, store_id, item_id, inventory_adjustment_pct=None, absolute_value=None):
        """
        Apply an inventory adjustment and update all affected components.
        
        Args:
            store_id: Store ID
            item_id: Product/item ID
            inventory_adjustment_pct: Percentage adjustment to inventory (optional)
            absolute_value: Absolute value to set inventory to (optional)
            
        Returns:
            dict: Updated data for the product
        """
        # Make sure inventory projection exists
        inventory_projection = self.data_dict.get('inventory_projection')
        
        if inventory_projection is None:
            logger.warning("Cannot adjust inventory: missing inventory projection data")
            return None
        
        # Get inventory data for the product
        inv_data = inventory_projection[(inventory_projection['Store_Id'] == store_id) & 
                                      (inventory_projection['Item'] == item_id)]
        
        if len(inv_data) == 0:
            logger.warning(f"No inventory data found for store {store_id}, product {item_id}")
            return None
        
        # Column that contains stock level might vary
        stock_col = 'Stock_Level' if 'Stock_Level' in inventory_projection.columns else 'Current_Stock'
        
        # Get the latest stock level
        latest_record = inv_data.sort_values('Date').iloc[-1]
        current_stock = latest_record[stock_col]
        
        # Calculate new stock level
        if absolute_value is not None:
            new_stock = absolute_value
        elif inventory_adjustment_pct is not None:
            new_stock = current_stock * (1 + inventory_adjustment_pct/100)
        else:
            logger.warning("Must provide either inventory_adjustment_pct or absolute_value")
            return None
        
        # Store the adjustment
        key = f"{store_id}_{item_id}"
        self.applied_changes['inventory_adjustments'][key] = {
            'inventory_adjustment_pct': inventory_adjustment_pct,
            'old_stock': current_stock,
            'new_stock': new_stock
        }
        
        # Deep copy
        adjusted_inventory = inventory_projection.copy()
        
        # Update the stock level for all dates
        mask = (adjusted_inventory['Store_Id'] == store_id) & (adjusted_inventory['Item'] == item_id)
        
        # Get current date - assume it's the latest date in the data
        current_date = adjusted_inventory.loc[mask, 'Date'].max()
        
        # Update the stock level only for the current date
        current_date_mask = mask & (adjusted_inventory['Date'] == current_date)
        adjusted_inventory.loc[current_date_mask, stock_col] = new_stock
        
        # Re-project future inventory
        forecasts = self.data_dict.get('adjusted_forecasts', self.data_dict.get('forecasts'))
        if forecasts is not None:
            forecast_data = forecasts[(forecasts['Store_Id'] == store_id) & 
                                     (forecasts['Item'] == item_id)]
            
            if len(forecast_data) > 0:
                # Get all future dates
                future_dates = adjusted_inventory.loc[mask & (adjusted_inventory['Date'] > current_date), 'Date'].sort_values()
                
                # Get matching forecasts
                future_forecasts = forecast_data[forecast_data['Date'].isin(future_dates)].sort_values('Date')
                
                if len(future_forecasts) > 0:
                    # Recalculate projected stock
                    projected_stock = new_stock
                    for _, row in future_forecasts.iterrows():
                        forecast_date = row['Date']
                        demand = row['Forecast']
                        projected_stock = max(0, projected_stock - demand)
                        
                        # Update the projection for this date
                        date_mask = mask & (adjusted_inventory['Date'] == forecast_date)
                        if sum(date_mask) > 0:
                            adjusted_inventory.loc[date_mask, stock_col] = projected_stock
        
        # Update data dictionary
        self.data_dict['adjusted_inventory'] = adjusted_inventory
        self.modified = True
        
        # Return the updated data for this product
        return self.get_product_data(store_id, item_id)
    
    def adjust_forecast(self, store_id, item_id, forecast_adjustment_pct):
        """
        Apply a forecast adjustment and update all affected components.
        
        Args:
            store_id: Store ID
            item_id: Product/item ID
            forecast_adjustment_pct: Percentage adjustment to forecast
            
        Returns:
            dict: Updated data for the product
        """
        # Make sure forecasts exist
        forecasts = self.data_dict.get('forecasts')
        inventory_projection = self.data_dict.get('inventory_projection')
        
        if forecasts is None:
            logger.warning("Cannot adjust forecast: missing forecast data")
            return None
        
        # Get forecast data for the product
        forecast_data = forecasts[(forecasts['Store_Id'] == store_id) & 
                                 (forecasts['Item'] == item_id)]
        
        if len(forecast_data) == 0:
            logger.warning(f"No forecast data found for store {store_id}, product {item_id}")
            return None
        
        # Store the adjustment
        key = f"{store_id}_{item_id}"
        adjustment_ratio = 1 + forecast_adjustment_pct/100
        self.applied_changes['forecast_adjustments'][key] = {
            'forecast_adjustment_pct': forecast_adjustment_pct,
            'adjustment_ratio': adjustment_ratio
        }
        
        # Deep copy
        adjusted_forecasts = forecasts.copy()
        
        # Apply adjustment to forecasts
        mask = (adjusted_forecasts['Store_Id'] == store_id) & (adjusted_forecasts['Item'] == item_id)
        adjusted_forecasts.loc[mask, 'Forecast'] = adjusted_forecasts.loc[mask, 'Forecast'] * adjustment_ratio
        
        # Update data dictionary
        self.data_dict['adjusted_forecasts'] = adjusted_forecasts
        self.modified = True
        
        # If inventory projection exists, update it based on new forecast
        if inventory_projection is not None:
            self.update_inventory_projection(store_id, item_id)
        
        # Return the updated data for this product
        return self.get_product_data(store_id, item_id)
    
    def update_inventory_projection(self, store_id, item_id):
        """
        Update inventory projection based on latest forecasts.
        
        Args:
            store_id: Store ID
            item_id: Product/item ID
            
        Returns:
            DataFrame: Updated inventory projection
        """
        # Get required data
        inventory_projection = self.data_dict.get('inventory_projection')
        forecasts = self.data_dict.get('adjusted_forecasts', self.data_dict.get('forecasts'))
        
        if inventory_projection is None or forecasts is None:
            logger.warning("Cannot update inventory projection: missing required data")
            return None
        
        # Get data for the product
        inv_data = inventory_projection[(inventory_projection['Store_Id'] == store_id) & 
                                      (inventory_projection['Item'] == item_id)]
        forecast_data = forecasts[(forecasts['Store_Id'] == store_id) & 
                                 (forecasts['Item'] == item_id)]
        
        if len(inv_data) == 0 or len(forecast_data) == 0:
            logger.warning(f"Missing data for store {store_id}, product {item_id}")
            return None
        
        # Column that contains stock level might vary
        stock_col = 'Stock_Level' if 'Stock_Level' in inventory_projection.columns else 'Current_Stock'
        
        # Get the latest stock level
        latest_record = inv_data.sort_values('Date').iloc[-1]
        current_stock = latest_record[stock_col]
        
        # Check if we have an inventory adjustment for this product
        key = f"{store_id}_{item_id}"
        if key in self.applied_changes['inventory_adjustments']:
            current_stock = self.applied_changes['inventory_adjustments'][key]['new_stock']
        
        # Deep copy
        adjusted_inventory = inventory_projection.copy()
        
        # Update the stock level
        mask = (adjusted_inventory['Store_Id'] == store_id) & (adjusted_inventory['Item'] == item_id)
        
        # Get current date - assume it's the latest date in the data
        current_date = adjusted_inventory.loc[mask, 'Date'].max()
        
        # Update the stock level for the current date
        current_date_mask = mask & (adjusted_inventory['Date'] == current_date)
        adjusted_inventory.loc[current_date_mask, stock_col] = current_stock
        
        # Sort forecast data by date
        forecast_data = forecast_data.sort_values('Date')
        
        # Re-project future inventory
        future_dates = adjusted_inventory.loc[mask & (adjusted_inventory['Date'] > current_date), 'Date'].sort_values()
        
        # Get matching forecasts for future dates
        future_forecasts = forecast_data[forecast_data['Date'].isin(future_dates)]
        
        if len(future_forecasts) > 0:
            # Recalculate projected stock
            projected_stock = current_stock
            for _, row in future_forecasts.iterrows():
                forecast_date = row['Date']
                demand = row['Forecast']
                projected_stock = max(0, projected_stock - demand)
                
                # Update the projection for this date
                date_mask = mask & (adjusted_inventory['Date'] == forecast_date)
                if sum(date_mask) > 0:
                    adjusted_inventory.loc[date_mask, stock_col] = projected_stock
        
        # Update data dictionary
        self.data_dict['adjusted_inventory'] = adjusted_inventory
        self.modified = True
        
        return adjusted_inventory
    
    def _generate_synthetic_product_data(self, store_id, item_id):
        """
        Generate synthetic data for a specific product that doesn't exist.
        
        Args:
            store_id: Store ID
            item_id: Product/item ID
        """
        # Create a reasonable product name
        product_name = f"Product {item_id}"
        
        # Try to get product name from any existing records
        for _, df in self.data_dict.items():
            if df is not None and 'Product' in df.columns and 'Item' in df.columns:
                try:
                    df_copy = df.copy()
                    df_copy['Item'] = df_copy['Item'].astype(str)
                    matching = df_copy[df_copy['Item'] == item_id]['Product'].unique()
                    if len(matching) > 0:
                        product_name = matching[0]
                        break
                except Exception:
                    continue
        
        # Get today's date
        today = datetime.now().date()
        
        # Create a date range for forecasting
        forecast_dates = pd.date_range(start=today, periods=14)
        
        # Get a reasonable base price and sales
        base_price = 9.99
        base_sales = 10
        base_cost = base_price * 0.6
        
        # For combined_data
        if 'combined_data' in self.data_dict and self.data_dict['combined_data'] is not None:
            # Create 60 days of historical data
            historical_dates = pd.date_range(end=today-timedelta(days=1), periods=60)
            records = []
            
            for date in historical_dates:
                day_factor = 1.0 + 0.2 * np.sin(date.weekday() * np.pi / 3.5)
                sales = max(0, int(base_sales * day_factor * np.random.normal(1, 0.2)))
                price = round(base_price * np.random.normal(1, 0.05), 2)
                
                records.append({
                    'Date': date,
                    'Store_Id': store_id,
                    'Item': item_id,
                    'Product': product_name,
                    'Sales': sales,
                    'Price': price,
                    'Cost': price * 0.6,
                    'Stock_Level': max(0, int(sales * 14 * np.random.normal(1, 0.3)))
                })
            
            # Append to existing data
            new_data = pd.DataFrame(records)
            self.data_dict['combined_data'] = pd.concat([self.data_dict['combined_data'], new_data])
        
        # For forecasts
        if 'forecasts' in self.data_dict and self.data_dict['forecasts'] is not None:
            records = []
            
            for date in forecast_dates:
                day_idx = (date - forecast_dates[0]).days
                trend_factor = 1.0 + (day_idx / 30) * 0.05  # 5% change over 30 days
                sales_forecast = max(0, int(base_sales * trend_factor * np.random.normal(1, 0.15)))
                
                upper_bound = int(sales_forecast * 1.3)
                lower_bound = int(sales_forecast * 0.7)
                
                records.append({
                    'Date': date,
                    'Store_Id': store_id,
                    'Item': item_id,
                    'Product': product_name,
                    'Forecast': sales_forecast,
                    'Upper_Bound': upper_bound,
                    'Lower_Bound': lower_bound
                })
            
            new_data = pd.DataFrame(records)
            self.data_dict['forecasts'] = pd.concat([self.data_dict['forecasts'], new_data])
        
        # For price elasticities
        if 'price_elasticities' in self.data_dict and self.data_dict['price_elasticities'] is not None:
            elasticity = round(np.random.uniform(-2.0, -0.5), 2)
            
            records = [{
                'Store_Id': store_id,
                'Item': item_id,
                'Product': product_name,
                'Elasticity': elasticity,
                'Current_Price': base_price,
                'Cost': base_cost,
                'Margin': (base_price - base_cost) / base_price
            }]
            
            new_data = pd.DataFrame(records)
            self.data_dict['price_elasticities'] = pd.concat([self.data_dict['price_elasticities'], new_data])
        
        # For inventory projection
        if 'inventory_projection' in self.data_dict and self.data_dict['inventory_projection'] is not None:
            records = []
            current_stock = base_sales * 10  # Starting inventory level
            
            for date in forecast_dates:
                daily_demand = max(0, int(base_sales * np.random.normal(1, 0.15)))
                
                # Update stock level
                current_stock = max(0, current_stock - daily_demand)
                
                # Determine stock status
                if current_stock < base_sales * 7:  # Less than a week
                    status = "Low"
                elif current_stock < base_sales * 14:  # Less than two weeks
                    status = "Adequate"
                else:
                    status = "Excess"
                
                records.append({
                    'Date': date,
                    'Store_Id': store_id,
                    'Item': item_id,
                    'Product': product_name,
                    'Stock_Level': current_stock,
                    'Stock_Status': status
                })
            
            new_data = pd.DataFrame(records)
            self.data_dict['inventory_projection'] = pd.concat([self.data_dict['inventory_projection'], new_data])
            
        logger.info(f"Generated synthetic data for store {store_id}, product {item_id}")
        
    def _generate_fallback_metrics(self, store_id, item_id):
        """
        Generate fallback metrics when no data is available.
        
        Args:
            store_id: Store ID
            item_id: Product/item ID
            
        Returns:
            dict: Dictionary of fallback metrics
        """
        logger.warning(f"Generating fallback metrics for store {store_id}, product {item_id}")
        
        # Create fallback price metrics
        price_metrics = {
            'elasticity': -1.0,
            'current_price': 9.99,
            'new_price': 9.99,
            'price_change_pct': 0,
            'cost': 5.99,
            'current_margin': 0.4,
            'new_margin': 0.4,
            'quantity_ratio': 1.0
        }
        
        # Create fallback inventory metrics
        inventory_metrics = {
            'current_stock': 70,
            'avg_daily_sales': 10,
            'coverage_days': 7,
            'coverage_weeks': 1,
            'status': "Adequate",
            'risk_level': "Medium",
            'stockout_risk': "Medium",
            'stockout_day': 7,
            'stockout_msg': "Potential stockout in 7 days",
            'safety_stock': 70,
            'target_stock': 140,
            'reorder_needed': False,
            'reorder_amount': 0,
            'reorder_msg': "No reorder needed"
        }
        
        # Create fallback forecast metrics
        forecast_metrics = {
            'total_forecast': 140,
            'avg_daily_forecast': 10,
            'forecast_horizon_days': 14,
            'recent_avg_sales': 10,
            'forecast_vs_history_pct': 0,
            'forecast_adjustment_pct': 0
        }
        
        # Create fallback integrated metrics
        integrated_metrics = {
            'price_change_impact': {
                'original_profit': 560,
                'new_profit': 560,
                'profit_diff': 0,
                'profit_diff_pct': 0,
                'adjusted_forecast': 140,
                'original_forecast': 140,
                'forecast_diff': 0,
                'forecast_diff_pct': 0
            },
            'inventory_health': {
                'adjusted_coverage_days': 7,
                'adjusted_coverage_weeks': 1,
                'balance_status': "Adequate",
                'action_needed': "Maintain current inventory level"
            },
            'business_impact_score': 50,
            'recommendation': "Neutral Impact"
        }
        
        # Combine all fallback metrics
        metrics = {
            'price': price_metrics,
            'inventory': inventory_metrics,
            'forecast': forecast_metrics,
            'integrated': integrated_metrics
        }
        
        return metrics
        
    def calculate_metrics(self, store_id, item_id):
        """
        Calculate key metrics for the product based on the current state.
        
        Args:
            store_id: Store ID
            item_id: Product/item ID
            
        Returns:
            dict: Dictionary of key metrics
        """
        # Get required data
        product_data = self.get_product_data(store_id, item_id)
        
        if not product_data:
            logger.warning(f"No data found for store {store_id}, product {item_id}")
            
            # Return fallback metrics as a last resort
            # This ensures visualizations at least have something to display
            return self._generate_fallback_metrics(store_id, item_id)
        
        # Use adjusted forecasts if available
        forecasts = self.data_dict.get('adjusted_forecasts', self.data_dict.get('forecasts'))
        forecast_data = forecasts[(forecasts['Store_Id'] == store_id) & 
                                 (forecasts['Item'] == item_id)]
        
        # Use adjusted inventory if available
        inventory = self.data_dict.get('adjusted_inventory', self.data_dict.get('inventory_projection'))
        inv_data = inventory[(inventory['Store_Id'] == store_id) & 
                            (inventory['Item'] == item_id)]
        
        # Get price elasticities
        elasticity_data = self.data_dict.get('price_elasticities')
        if elasticity_data is not None:
            elasticity_data = elasticity_data[(elasticity_data['Store_Id'] == store_id) & 
                                             (elasticity_data['Item'] == item_id)]
        
        # Get pricing metrics
        price_metrics = self._calculate_price_metrics(store_id, item_id, elasticity_data)
        
        # Get inventory metrics
        inventory_metrics = self._calculate_inventory_metrics(store_id, item_id, inv_data, forecast_data)
        
        # Get forecast metrics
        forecast_metrics = self._calculate_forecast_metrics(store_id, item_id, forecast_data)
        
        # Get integrated metrics
        integrated_metrics = self._calculate_integrated_metrics(
            store_id, item_id, price_metrics, inventory_metrics, forecast_metrics)
        
        # Combine all metrics
        metrics = {
            'price': price_metrics,
            'inventory': inventory_metrics,
            'forecast': forecast_metrics,
            'integrated': integrated_metrics
        }
        
        return metrics
    
    def _calculate_price_metrics(self, store_id, item_id, elasticity_data):
        """Calculate pricing metrics for the product."""
        if elasticity_data is None or len(elasticity_data) == 0:
            return {}
        
        # Get elasticity and price data
        elasticity = elasticity_data['Elasticity'].iloc[0]
        current_price = elasticity_data['Current_Price'].iloc[0]
        cost = elasticity_data['Cost'].iloc[0] if 'Cost' in elasticity_data.columns else current_price * 0.6
        
        # Check if we have an adjustment
        key = f"{store_id}_{item_id}"
        if key in self.applied_changes['price_adjustments']:
            new_price = self.applied_changes['price_adjustments'][key]['new_price']
            price_change_pct = self.applied_changes['price_adjustments'][key]['price_adjustment_pct']
            quantity_ratio = self.applied_changes['price_adjustments'][key]['quantity_ratio']
        else:
            new_price = current_price
            price_change_pct = 0
            quantity_ratio = 1.0
        
        # Calculate margins
        current_margin = (current_price - cost) / current_price
        new_margin = (new_price - cost) / new_price
        
        # Metrics dictionary
        metrics = {
            'elasticity': elasticity,
            'current_price': current_price,
            'new_price': new_price,
            'price_change_pct': price_change_pct,
            'cost': cost,
            'current_margin': current_margin,
            'new_margin': new_margin,
            'quantity_ratio': quantity_ratio
        }
        
        return metrics
    
    def _calculate_inventory_metrics(self, store_id, item_id, inv_data, forecast_data):
        """Calculate inventory metrics for the product."""
        if inv_data is None or len(inv_data) == 0:
            return {}
        
        # Get inventory settings
        # Hardcoded defaults, should ideally come from config
        MIN_STOCK_WEEKS = 1
        TARGET_STOCK_WEEKS = 2
        MAX_STOCK_WEEKS = 3
        
        try:
            from config.settings import (
                MIN_STOCK_WEEKS, TARGET_STOCK_WEEKS, MAX_STOCK_WEEKS
            )
        except ImportError:
            logger.warning("Using default stock weeks settings")
        
        # Get the latest stock level
        inv_data = inv_data.sort_values('Date')
        stock_col = 'Stock_Level' if 'Stock_Level' in inv_data.columns else 'Current_Stock'
        current_stock = inv_data[stock_col].iloc[-1]
        
        # Check if we have an adjustment
        key = f"{store_id}_{item_id}"
        if key in self.applied_changes['inventory_adjustments']:
            current_stock = self.applied_changes['inventory_adjustments'][key]['new_stock']
        
        # Calculate average daily sales from forecast
        avg_daily_sales = 1.0
        if forecast_data is not None and len(forecast_data) > 0:
            avg_daily_sales = forecast_data['Forecast'].mean()
        
        # Calculate coverage
        coverage_days = current_stock / avg_daily_sales if avg_daily_sales > 0 else float('inf')
        coverage_weeks = coverage_days / 7
        
        # Determine status
        if coverage_weeks < MIN_STOCK_WEEKS:
            status = "Low"
            risk_level = "High"
        elif coverage_weeks <= TARGET_STOCK_WEEKS:
            status = "Adequate"
            risk_level = "Low"
        elif coverage_weeks <= MAX_STOCK_WEEKS:
            status = "Good"
            risk_level = "Low"
        else:
            status = "Excess"
            risk_level = "Medium"
        
        # Calculate stockout risk
        stockout_day = None
        stockout_risk = "Low"
        stockout_msg = "No stockout projected"
        
        if forecast_data is not None and len(forecast_data) > 0 and avg_daily_sales > 0:
            if coverage_days < len(forecast_data):
                stockout_day = int(coverage_days)
                stockout_risk = "High"
                stockout_msg = f"Projected stockout in {stockout_day} days"
        
        # Calculate reorder need
        safety_stock = avg_daily_sales * 7 * MIN_STOCK_WEEKS
        target_stock = avg_daily_sales * 7 * TARGET_STOCK_WEEKS
        
        if current_stock < safety_stock:
            reorder_needed = True
            reorder_amount = target_stock - current_stock
            reorder_msg = f"Reorder {int(reorder_amount)} units"
        else:
            reorder_needed = False
            reorder_amount = 0
            reorder_msg = "No reorder needed"
        
        # Metrics dictionary
        metrics = {
            'current_stock': current_stock,
            'avg_daily_sales': avg_daily_sales,
            'coverage_days': coverage_days,
            'coverage_weeks': coverage_weeks,
            'status': status,
            'risk_level': risk_level,
            'stockout_risk': stockout_risk,
            'stockout_day': stockout_day,
            'stockout_msg': stockout_msg,
            'safety_stock': safety_stock,
            'target_stock': target_stock,
            'reorder_needed': reorder_needed,
            'reorder_amount': reorder_amount,
            'reorder_msg': reorder_msg
        }
        
        return metrics
    
    def _calculate_forecast_metrics(self, store_id, item_id, forecast_data):
        """Calculate forecast metrics for the product."""
        if forecast_data is None or len(forecast_data) == 0:
            return {}
        
        # Check if we have a forecast adjustment
        key = f"{store_id}_{item_id}"
        forecast_adjustment_pct = 0
        
        if key in self.applied_changes['forecast_adjustments']:
            forecast_adjustment_pct = self.applied_changes['forecast_adjustments'][key]['forecast_adjustment_pct']
        
        # Calculate total forecast
        total_forecast = forecast_data['Forecast'].sum()
        avg_daily_forecast = forecast_data['Forecast'].mean()
        forecast_dates = forecast_data['Date'].sort_values()
        forecast_horizon_days = (forecast_dates.max() - forecast_dates.min()).days + 1
        
        # Check if we have historical data for comparison
        combined_data = self.data_dict.get('combined_data')
        historical_data = None
        
        if combined_data is not None:
            historical_data = combined_data[(combined_data['Store_Id'] == store_id) & 
                                           (combined_data['Item'] == item_id)]
        
        # Compare with recent history if available
        recent_avg_sales = None
        forecast_vs_history_pct = None
        
        if historical_data is not None and len(historical_data) > 0:
            # Calculate average sales for the most recent period matching forecast horizon
            recent_data = historical_data.sort_values('Date').tail(forecast_horizon_days)
            if len(recent_data) > 0:
                recent_avg_sales = recent_data['Sales'].mean()
                if recent_avg_sales > 0:
                    forecast_vs_history_pct = (avg_daily_forecast - recent_avg_sales) / recent_avg_sales * 100
        
        # Metrics dictionary
        metrics = {
            'total_forecast': total_forecast,
            'avg_daily_forecast': avg_daily_forecast,
            'forecast_horizon_days': forecast_horizon_days,
            'recent_avg_sales': recent_avg_sales,
            'forecast_vs_history_pct': forecast_vs_history_pct,
            'forecast_adjustment_pct': forecast_adjustment_pct
        }
        
        return metrics
    
    def _calculate_integrated_metrics(self, store_id, item_id, price_metrics, 
                                     inventory_metrics, forecast_metrics):
        """Calculate integrated metrics combining all components."""
        if not price_metrics or not inventory_metrics or not forecast_metrics:
            return {}
        
        # Calculate profit impact from price change
        price_change_impact = {}
        if 'quantity_ratio' in price_metrics and 'elasticity' in price_metrics:
            current_price = price_metrics.get('current_price')
            new_price = price_metrics.get('new_price')
            cost = price_metrics.get('cost')
            quantity_ratio = price_metrics.get('quantity_ratio')
            
            if all(v is not None for v in [current_price, new_price, cost, quantity_ratio]):
                total_forecast = forecast_metrics.get('total_forecast', 0)
                
                # Original profit
                original_profit = (current_price - cost) * total_forecast
                
                # New profit
                adjusted_forecast = total_forecast * quantity_ratio
                new_profit = (new_price - cost) * adjusted_forecast
                
                # Profit difference
                profit_diff = new_profit - original_profit
                profit_diff_pct = (profit_diff / original_profit * 100) if original_profit > 0 else 0
                
                price_change_impact = {
                    'original_profit': original_profit,
                    'new_profit': new_profit,
                    'profit_diff': profit_diff,
                    'profit_diff_pct': profit_diff_pct,
                    'adjusted_forecast': adjusted_forecast,
                    'original_forecast': total_forecast,
                    'forecast_diff': adjusted_forecast - total_forecast,
                    'forecast_diff_pct': ((adjusted_forecast - total_forecast) / total_forecast * 100) 
                                       if total_forecast > 0 else 0
                }
        
        # Calculate inventory health based on price-adjusted forecast
        inventory_health = {}
        if 'current_stock' in inventory_metrics and 'quantity_ratio' in price_metrics:
            current_stock = inventory_metrics.get('current_stock')
            avg_daily_sales = inventory_metrics.get('avg_daily_sales', 0) * price_metrics.get('quantity_ratio', 1.0)
            
            if avg_daily_sales > 0:
                adjusted_coverage_days = current_stock / avg_daily_sales
                adjusted_coverage_weeks = adjusted_coverage_days / 7
                
                # Determine if inventory is balanced for adjusted forecast
                MIN_STOCK_WEEKS = 1
                TARGET_STOCK_WEEKS = 2
                MAX_STOCK_WEEKS = 3
                
                try:
                    from config.settings import (
                        MIN_STOCK_WEEKS, TARGET_STOCK_WEEKS, MAX_STOCK_WEEKS
                    )
                except ImportError:
                    pass
                
                if adjusted_coverage_weeks < MIN_STOCK_WEEKS:
                    balance_status = "Understocked"
                    action_needed = "Order more inventory"
                elif adjusted_coverage_weeks > MAX_STOCK_WEEKS:
                    balance_status = "Overstocked"
                    action_needed = "Consider promotion or price decrease"
                else:
                    balance_status = "Balanced"
                    action_needed = "Maintain current inventory level"
                
                inventory_health = {
                    'adjusted_coverage_days': adjusted_coverage_days,
                    'adjusted_coverage_weeks': adjusted_coverage_weeks,
                    'balance_status': balance_status,
                    'action_needed': action_needed
                }
        
        # Calculate overall business impact score (1-100)
        # Higher score means more positive impact
        business_impact_score = 50  # Neutral starting point
        
        # Adjust score based on price impact
        if price_change_impact:
            profit_diff_pct = price_change_impact.get('profit_diff_pct', 0)
            business_impact_score += min(25, max(-25, profit_diff_pct))
        
        # Adjust score based on inventory health
        if inventory_health:
            if inventory_health.get('balance_status') == "Balanced":
                business_impact_score += 15
            elif inventory_health.get('balance_status') == "Understocked":
                business_impact_score -= 10
            elif inventory_health.get('balance_status') == "Overstocked":
                business_impact_score -= 5
        
        # Determine recommendation based on score
        if business_impact_score >= 75:
            recommendation = "Highly Recommended"
        elif business_impact_score >= 60:
            recommendation = "Recommended"
        elif business_impact_score >= 40:
            recommendation = "Neutral Impact"
        elif business_impact_score >= 25:
            recommendation = "Not Recommended"
        else:
            recommendation = "Strongly Not Recommended"
        
        # Integrated metrics
        metrics = {
            'price_change_impact': price_change_impact,
            'inventory_health': inventory_health,
            'business_impact_score': business_impact_score,
            'recommendation': recommendation
        }
        
        return metrics