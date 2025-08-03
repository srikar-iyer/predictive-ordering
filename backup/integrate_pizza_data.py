import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_pizza_datasets(sales_file='FrozenPizzaSales.csv', 
                        purchases_file='FrozenPizzaPurchases.csv',
                        stock_file='FrozenPizzaStock.csv'):
    """
    Load and process the three pizza datasets: sales, purchases, and stock.
    Returns a combined dataset with all relevant features.
    """
    print(f"Loading data from {sales_file}, {purchases_file}, and {stock_file}...")
    
    # Load the datasets
    try:
        sales_df = pd.read_csv(sales_file)
        purchases_df = pd.read_csv(purchases_file)
        stock_df = pd.read_csv(stock_file)
        
        print(f"Loaded {len(sales_df)} sales records")
        print(f"Loaded {len(purchases_df)} purchase records")
        print(f"Loaded {len(stock_df)} stock records")
        
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
        
        # Calculate daily unit price
        sales_summary['Unit_Price'] = sales_summary['Retail_Revenue'] / sales_summary['Units_Sold']
        sales_summary['Unit_Price'].fillna(0, inplace=True)  # Handle division by zero
        
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
        stock_metrics = stock_df[['Store_Id', 'item', 'Avg_Weekly_Sales_4W', 'Avg_Weekly_Sales_13W', 'Stock_Coverage_Weeks']]
        merged_df = pd.merge(
            merged_df,
            stock_metrics,
            on=['Store_Id', 'item'],
            how='left'
        )
        
        # Calculate profit
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
            (merged_df['Weeks_Of_Stock'] < 1),
            (merged_df['Weeks_Of_Stock'] >= 1) & (merged_df['Weeks_Of_Stock'] <= 3),
            (merged_df['Weeks_Of_Stock'] > 3)
        ]
        values = ['Low', 'Adequate', 'Excess']
        merged_df['Stock_Status'] = np.select(conditions, values, default='Adequate')
        
        # Calculate optimal order quantity based on sales and stock policy
        # Assume target of keeping 2 weeks of stock on hand
        merged_df['Daily_Sales_Forecast'] = merged_df['Recent_Daily_Sales']
        merged_df['Target_Stock_Weeks'] = 2  # 2 weeks of stock
        merged_df['Min_Stock_Weeks'] = 1  # 1 week minimum safety stock
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
        
        print(f"Successfully processed pizza datasets into a combined dataset with {len(merged_df)} records")
        return merged_df
    
    except Exception as e:
        print(f"Error loading pizza datasets: {str(e)}")
        raise

def save_combined_dataset(df, output_file='combined_pizza_data.csv'):
    """Save the combined dataset to a CSV file"""
    df.to_csv(output_file, index=False)
    print(f"Saved combined dataset to {output_file}")
    return output_file

if __name__ == "__main__":
    # Load and process the pizza datasets
    combined_df = load_pizza_datasets()
    
    # Save the combined dataset
    save_combined_dataset(combined_df)