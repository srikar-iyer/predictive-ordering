import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import gradio as gr

# Set random seed for reproducibility
np.random.seed(42)

# Load data from CSV file
def load_data_from_csv(file_path='frozen_pizza_only.csv', sample_size=0.05, random_state=42):
    """Load historical sales and inventory data from CSV file
    
    CSV structure:
    - Store_Id: Store identifier
    - Item: Product identifier
    - proc_date: Date of sale
    - Category_Name: Product category
    - Unit_Retail: Price per unit
    - Total_units: Units sold
    """
    
    print(f"Loading data from {file_path}...")
    
    # Define holidays for the dataset period
    holidays = {
        '01-01': 'New Year',
        '02-14': 'Valentine', 
        '07-04': 'Independence Day',
        '10-31': 'Halloween',
        '11-25': 'Thanksgiving', # Approximate
        '12-25': 'Christmas',
        '04-15': 'Easter' # Approximate
    }
    
    try:
        # Read the CSV file with sampling for performance
        df = pd.read_csv(file_path)
        
        # Take a sample of the data for faster processing
        if sample_size < 1.0:
            df = df.sample(frac=sample_size, random_state=random_state)
        
        # Parse date column
        df['Date'] = pd.to_datetime(df['proc_date'], format='%m/%d/%Y', errors='coerce')
        df.dropna(subset=['Date'], inplace=True)  # Drop rows with invalid dates
        
        # Convert item identifier to string
        df['Item'] = df['Item'].astype(str)
        
        # Use Item ID as the Product instead of Category_Name
        df['Product'] = df['Item']
        
        # Rename Unit_Retail to Price and Total_units to Sales
        df.rename(columns={'Unit_Retail': 'Price', 'Total_units': 'Sales'}, inplace=True)
        
        # Add day of week, month, and year
        df['Day_Of_Week'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        
        # Add holiday information
        df['month_day'] = df['Date'].dt.strftime('%m-%d')
        df['Is_Holiday'] = df['month_day'].apply(lambda x: 1 if x in holidays else 0)
        df['Holiday_Name'] = df['month_day'].apply(lambda x: holidays.get(x))
        df.drop('month_day', axis=1, inplace=True)
        
        # Generate simulated weather data since it's not in the original dataset
        # Use a consistent seed for reproducibility
        np.random.seed(42)
        weather_options = ['Normal', 'Heavy Rain', 'Snow', 'Storm']
        weather_probabilities = [0.85, 0.06, 0.06, 0.03]  # 85% Normal, 15% adverse weather
        
        # Group by Date to assign the same weather to all products on the same day
        unique_dates = df['Date'].unique()
        date_weather = {}
        
        for date in unique_dates:
            date_weather[date] = np.random.choice(weather_options, p=weather_probabilities)
        
        df['Weather'] = df['Date'].map(date_weather)
        
        # Add Lead_Time - simulated since it's not in the original dataset
        df['Lead_Time'] = np.random.normal(7, 2, size=len(df)).astype(int)
        df['Lead_Time'] = df['Lead_Time'].apply(lambda x: max(1, x))  # Minimum 1 day
        
        # Add promotion data - simulated since it's not in the original dataset
        # Assume prices that are significantly lower than average for a product are promotions
        df['avg_price'] = df.groupby(['Item'])['Price'].transform('mean')
        df['Promotion'] = (df['Price'] < df['avg_price'] * 0.9).astype(int)  # 10% below average price
        df.drop('avg_price', axis=1, inplace=True)
        
        # Simulated stock level data
        # Group by product and date to calculate stock levels based on previous sales
        df = df.sort_values(['Item', 'Date'])
        
        # Initialize stock level for each product
        df['Stock_Level'] = 0
        df['Potential_Sales'] = df['Sales']
        df['Lost_Sales'] = 0
        
        # Calculate stock levels
        for item in df['Item'].unique():
            item_data = df[df['Item'] == item].copy()
            
            # Initialize with a reasonable stock level
            initial_stock = item_data['Sales'].mean() * 10
            current_stock = initial_stock
            
            for idx, row in item_data.iterrows():
                # Calculate how much stock was delivered (approximate as mean sales * random factor)
                stock_delivered = np.random.uniform(0.9, 1.3) * row['Sales']
                
                # Update stock level
                current_stock = max(0, current_stock) + stock_delivered
                
                # Record stock level
                df.loc[idx, 'Stock_Level'] = int(current_stock)
                
                # Calculate potential sales and lost sales
                potential_sales = row['Sales']
                if current_stock < potential_sales:
                    lost_sales = potential_sales - current_stock
                    actual_sales = current_stock
                    df.loc[idx, 'Lost_Sales'] = lost_sales
                    df.loc[idx, 'Sales'] = actual_sales
                    df.loc[idx, 'Potential_Sales'] = potential_sales
                else:
                    df.loc[idx, 'Lost_Sales'] = 0
                    df.loc[idx, 'Potential_Sales'] = potential_sales
                
                # Update stock after sales
                current_stock -= df.loc[idx, 'Sales']
        
        # Drop unnecessary columns
        columns_to_drop = ['Store_Id', 'proc_date', 'Category_Name']
        for col in columns_to_drop:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
        
        print(f"Loaded {len(df)} records for {len(df['Product'].unique())} pizza item IDs")
        return df
    
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        # Fallback to generate sample data
        return generate_sample_data()

# Generate sample data (used as fallback if CSV loading fails)
def generate_sample_data():
    """Generate sample historical sales and inventory data for multiple pizza item IDs"""
    
    print("Falling back to sample data generation...")
    
    # Define pizza item IDs
    products = ['942700005', '942700010', '942700015', '942700020', '942700025', '942700030', '942700035']
    
    # Generate dates for the past 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Initialize dataframe
    data = []
    
    # Define holidays and events
    holidays = {
        '01-01': 'New Year',
        '02-14': 'Valentine', 
        '07-04': 'Independence Day',
        '10-31': 'Halloween',
        '11-25': 'Thanksgiving', # Approximate
        '12-25': 'Christmas'
    }
    
    # Generate data for each product
    for product in products:
        # Base demand parameters
        if product == 'Potatoes':
            base_demand = 500
            seasonal_amplitude = 150
            weekly_pattern = [1.0, 0.9, 0.8, 0.8, 1.1, 1.3, 1.4]  # Higher on weekends
            holiday_factor = 1.7
            price_range = (0.8, 1.5)
            
        elif product == 'Vegetables':
            base_demand = 800
            seasonal_amplitude = 200
            weekly_pattern = [1.1, 1.0, 0.9, 0.9, 1.0, 1.2, 1.3]
            holiday_factor = 1.5
            price_range = (1.0, 2.0)
            
        elif product == 'Baby Products':
            base_demand = 300
            seasonal_amplitude = 50
            weekly_pattern = [0.9, 1.0, 1.0, 1.0, 1.1, 1.2, 1.0]
            holiday_factor = 1.1
            price_range = (5.0, 20.0)
            
        elif product == 'Fruit':
            base_demand = 600
            seasonal_amplitude = 250
            weekly_pattern = [1.0, 0.9, 0.8, 0.9, 1.2, 1.4, 1.3]
            holiday_factor = 1.4
            price_range = (1.5, 4.0)
            
        elif product == 'Milk':
            base_demand = 450
            seasonal_amplitude = 50
            weekly_pattern = [1.1, 1.0, 0.9, 0.9, 1.0, 1.2, 1.3]
            holiday_factor = 1.3
            price_range = (2.5, 4.5)
            
        elif product == 'DVDs':
            base_demand = 100
            seasonal_amplitude = 30
            weekly_pattern = [0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 1.4]
            holiday_factor = 2.5
            price_range = (10.0, 25.0)
            
        else:  # Meat
            base_demand = 350
            seasonal_amplitude = 100
            weekly_pattern = [0.8, 0.7, 0.9, 1.0, 1.2, 1.5, 1.4]
            holiday_factor = 2.0
            price_range = (5.0, 15.0)
        
        # Generate daily sales for this product
        for i, date in enumerate(date_range):
            # Basic seasonality (annual)
            day_of_year = date.dayofyear
            seasonality = np.sin(day_of_year / 365 * 2 * np.pi) * seasonal_amplitude
            
            # Weekly pattern
            day_of_week = date.weekday()
            weekly_factor = weekly_pattern[day_of_week]
            
            # Holiday effect
            holiday_effect = 1.0
            month_day = f"{date.month:02d}-{date.day:02d}"
            if month_day in holidays:
                holiday_effect = holiday_factor
                
                # Days leading up to holiday also have increased sales
                days_to_holiday = 0
            elif month_day == '04-15':  # Easter approximation
                holiday_effect = holiday_factor
            
            # Weather effect (simulate some random weather events)
            if np.random.random() < 0.03:  # 3% chance of adverse weather
                weather = np.random.choice(['Heavy Rain', 'Snow', 'Storm'])
                weather_effect = 0.7  # Reduce sales during bad weather
            else:
                weather = 'Normal'
                weather_effect = 1.0
                
                # Seasonal weather patterns
                if date.month in [12, 1, 2]:  # Winter
                    if product == 'Fruit':
                        weather_effect *= 0.9
                    elif product == 'Meat':
                        weather_effect *= 1.1
                elif date.month in [6, 7, 8]:  # Summer
                    if product == 'Fruit':
                        weather_effect *= 1.2
                    elif product == 'Ice Cream':
                        weather_effect *= 1.5
            
            # Promotional effect
            if np.random.random() < 0.1:  # 10% chance of promotion
                promotion = True
                promotion_effect = np.random.uniform(1.2, 1.6)
            else:
                promotion = False
                promotion_effect = 1.0
            
            # Price calculation with some randomness
            base_price = np.random.uniform(price_range[0], price_range[1])
            if promotion:
                price = base_price * 0.8  # 20% discount
            else:
                price = base_price
            
            # Calculate final demand
            demand = (base_demand + seasonality) * weekly_factor * holiday_effect * weather_effect * promotion_effect
            
            # Add noise
            noise = np.random.normal(0, demand * 0.1)  # 10% noise
            final_sales = max(0, int(demand + noise))
            
            # Stock level calculation - simulate some stock issues
            stock_delivered = int(final_sales * np.random.uniform(0.9, 1.3))  # Order roughly what we need
            
            # Lead time in days (how far in advance orders were placed)
            lead_time = int(np.random.normal(7, 2))
            lead_time = max(1, lead_time)  # Minimum 1 day
            
            # Current stock
            if i > 0:
                # Simplistic inventory tracking
                prev_stock = prev_stock_delivered - prev_sales
                current_stock = max(0, prev_stock) + stock_delivered
            else:
                # Initial stock
                current_stock = stock_delivered
            
            # Potential lost sales due to stockouts
            stockout = max(0, final_sales - current_stock)
            actual_sales = min(final_sales, current_stock)
            
            # Store previous values for next iteration
            prev_stock_delivered = stock_delivered
            prev_sales = actual_sales
            
            # Record the data
            data.append({
                'Date': date,
                'Product': product,
                'Sales': actual_sales,
                'Potential_Sales': final_sales,  # What could have sold if stock was available
                'Lost_Sales': stockout,
                'Stock_Level': current_stock,
                'Price': price,
                'Promotion': promotion,
                'Weather': weather,
                'Lead_Time': lead_time,
                'Day_Of_Week': date.dayofweek,
                'Month': date.month,
                'Year': date.year,
                'Is_Holiday': 1 if month_day in holidays or month_day == '04-15' else 0,
                'Holiday_Name': holidays.get(month_day, 'Easter' if month_day == '04-15' else None)
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def prepare_features(df):
    """Prepare features for the forecasting model"""
    
    # Create copy to avoid modifying the original
    df_features = df.copy()
    
    # Convert Date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df_features['Date']):
        df_features['Date'] = pd.to_datetime(df_features['Date'])
    
    # One-hot encode product
    product_dummies = pd.get_dummies(df_features['Product'], prefix='Product')
    df_features = pd.concat([df_features, product_dummies], axis=1)
    
    # One-hot encode weather
    weather_dummies = pd.get_dummies(df_features['Weather'], prefix='Weather')
    df_features = pd.concat([df_features, weather_dummies], axis=1)
    
    # Cyclical encoding for day of week, month
    df_features['Day_Sin'] = np.sin(df_features['Day_Of_Week'] * (2 * np.pi / 7))
    df_features['Day_Cos'] = np.cos(df_features['Day_Of_Week'] * (2 * np.pi / 7))
    df_features['Month_Sin'] = np.sin(df_features['Month'] * (2 * np.pi / 12))
    df_features['Month_Cos'] = np.cos(df_features['Month'] * (2 * np.pi / 12))
    
    # Calculate days until closest holiday
    df_features['Days_To_Holiday'] = 365  # Default large value
    
    # Add lag features (sales from previous days)
    for product in df['Product'].unique():
        product_data = df[df['Product'] == product].sort_values('Date')
        
        # Create 7, 14, 30 day lags
        for lag in [1, 7, 14, 30]:
            lag_col = f'Sales_Lag_{lag}'
            product_data[lag_col] = product_data['Sales'].shift(lag)
            
        # Calculate rolling averages
        for window in [7, 14, 30]:
            avg_col = f'Sales_Avg_{window}'
            product_data[avg_col] = product_data['Sales'].rolling(window=window).mean()
            
        # Update in the main dataframe
        for col in product_data.columns:
            if col.startswith('Sales_Lag_') or col.startswith('Sales_Avg_'):
                df_features.loc[df_features['Product'] == product, col] = product_data[col].values
    
    # Fill NaN values for lag features
    for col in df_features.columns:
        if col.startswith('Sales_Lag_') or col.startswith('Sales_Avg_'):
            df_features[col] = df_features[col].fillna(df_features.groupby('Product')['Sales'].transform('mean'))
    
    # Features to use in the model
    feature_cols = [
        'Price', 'Promotion', 'Day_Sin', 'Day_Cos', 'Month_Sin', 'Month_Cos',
        'Is_Holiday', 'Lead_Time'
    ]
    
    # Add product and weather dummy columns
    feature_cols.extend([col for col in df_features.columns if col.startswith('Product_')])
    feature_cols.extend([col for col in df_features.columns if col.startswith('Weather_')])
    
    # Add lag features
    feature_cols.extend([col for col in df_features.columns if col.startswith('Sales_Lag_')])
    feature_cols.extend([col for col in df_features.columns if col.startswith('Sales_Avg_')])
    
    return df_features, feature_cols

def train_forecast_model(df):
    """Train a forecasting model using the provided data"""
    
    # Prepare features
    df_features, feature_cols = prepare_features(df)
    
    # Define target
    target = 'Sales'
    
    # Split into features and target
    X = df_features[feature_cols].fillna(0)  # Fill any remaining NaNs
    y = df_features[target]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train a Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, df_features, feature_cols, feature_importance, (X_test, y_test, y_pred_test)

def run_tests(model, df, feature_cols):
    """Run tests on the forecasting model with different scenarios"""
    
    results = []
    
    # Get a prepared dataset
    df_features, _ = prepare_features(df)
    
    # Test scenarios for each product
    for product in df['Product'].unique():
        
        # Filter data for this product
        product_data = df_features[df_features['Product'] == product].copy()
        
        # Get the most recent 30 days of data
        recent_data = product_data.sort_values('Date', ascending=False).head(30)
        
        # Calculate average values for this product
        avg_price = recent_data['Price'].mean()
        avg_lead_time = recent_data['Lead_Time'].mean()
        
        # Create test cases
        test_cases = [
            {
                'name': 'Baseline',
                'price': avg_price,
                'promotion': False,
                'holiday': False,
                'weather': 'Normal',
            },
            {
                'name': 'Price Reduction',
                'price': avg_price * 0.8,  # 20% price reduction
                'promotion': True,
                'holiday': False,
                'weather': 'Normal',
            },
            {
                'name': 'Holiday',
                'price': avg_price,
                'promotion': False,
                'holiday': True,
                'weather': 'Normal',
            },
            {
                'name': 'Bad Weather',
                'price': avg_price,
                'promotion': False,
                'holiday': False,
                'weather': 'Storm',
            },
            {
                'name': 'Holiday + Promotion',
                'price': avg_price * 0.8,
                'promotion': True,
                'holiday': True,
                'weather': 'Normal',
            }
        ]
        
        # Create a sample row as template (use the most recent data as a base)
        sample_row = recent_data.iloc[0].copy()
        
        # Test each scenario
        for case in test_cases:
            # Create a copy of the sample row
            test_row = sample_row.copy()
            
            # Modify the test row based on the scenario
            test_row['Price'] = case['price']
            test_row['Promotion'] = 1 if case['promotion'] else 0
            test_row['Is_Holiday'] = 1 if case['holiday'] else 0
            
            # Update weather columns
            for col in feature_cols:
                if col.startswith('Weather_'):
                    weather_type = col.replace('Weather_', '')
                    test_row[col] = 1 if weather_type == case['weather'] else 0
            
            # Extract only the features needed for prediction
            test_features = pd.DataFrame([test_row[feature_cols]])
            
            # Make prediction
            predicted_sales = model.predict(test_features)[0]
            
            # Store results
            results.append({
                'Product': product,
                'Scenario': case['name'],
                'Price': case['price'],
                'Promotion': case['promotion'],
                'Holiday': case['holiday'],
                'Weather': case['weather'],
                'Predicted_Sales': int(predicted_sales),
                'Recommended_Order': max(int(predicted_sales * 1.1), int(predicted_sales) + 1)  # Add 10% safety stock or at least 1 unit
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def generate_demand_forecast(df, model, feature_cols, days_to_forecast=30):
    """Generate demand forecast for each product using the Random Forest model"""
    
    # Prepare features
    df_features, _ = prepare_features(df)
    
    # Get the latest date in the dataset
    latest_date = df['Date'].max()
    
    # Create a DataFrame for future dates
    future_dates = pd.date_range(start=latest_date + timedelta(days=1),
                              end=latest_date + timedelta(days=days_to_forecast))
    
    # Create a DataFrame to store forecasts
    forecast_data = []
    
    # Get all unique products
    products = df['Product'].unique()
    
    # Loop through each product and forecast
    for product in products:
        # Get the latest row for this product to use as a template
        product_data = df_features[df_features['Product'] == product].sort_values('Date', ascending=False)
        
        # Skip if no data for this product
        if len(product_data) == 0:
            continue
            
        # Use the most recent row as a template
        template_row = product_data.iloc[0].copy()
        
        # Get recent sales patterns for this product
        recent_sales = product_data.sort_values('Date')
        if len(recent_sales) > 30:
            recent_sales = recent_sales.tail(30)
        
        # Create forecast for each future date
        for future_date in future_dates:
            forecast_row = template_row.copy()
            
            # Update date-related features
            forecast_row['Date'] = future_date
            forecast_row['Day_Of_Week'] = future_date.dayofweek
            forecast_row['Month'] = future_date.month
            forecast_row['Year'] = future_date.year
            
            # Update cyclical encodings
            forecast_row['Day_Sin'] = np.sin(forecast_row['Day_Of_Week'] * (2 * np.pi / 7))
            forecast_row['Day_Cos'] = np.cos(forecast_row['Day_Of_Week'] * (2 * np.pi / 7))
            forecast_row['Month_Sin'] = np.sin(forecast_row['Month'] * (2 * np.pi / 12))
            forecast_row['Month_Cos'] = np.cos(forecast_row['Month'] * (2 * np.pi / 12))
            
            # Holiday check
            month_day = f"{future_date.month:02d}-{future_date.day:02d}"
            holidays = {
                '01-01': 'New Year',
                '02-14': 'Valentine', 
                '07-04': 'Independence Day',
                '10-31': 'Halloween',
                '11-25': 'Thanksgiving', # Approximate
                '12-25': 'Christmas',
                '04-15': 'Easter' # Approximate
            }
            forecast_row['Is_Holiday'] = 1 if month_day in holidays else 0
            
            # Weather - use normal weather for forecasting
            for col in feature_cols:
                if col.startswith('Weather_'):
                    weather_type = col.replace('Weather_', '')
                    forecast_row[col] = 1 if weather_type == 'Normal' else 0
            
            # Assume no promotion in forecast
            forecast_row['Promotion'] = 0
            
            # Make prediction
            X_forecast = pd.DataFrame([forecast_row[feature_cols]])
            predicted_sales = model.predict(X_forecast)[0]
            
            # Add to forecast data
            forecast_data.append({
                'Date': future_date,
                'Product': product,
                'Predicted_Demand': max(0, int(predicted_sales))
            })
    
    return pd.DataFrame(forecast_data)

def generate_charts(df, test_results, feature_importance, test_data, model=None, feature_cols=None):
    """Generate visualization charts for the forecasting results"""
    
    charts = {}
    
    # 1. Historical Sales by Product
    plt.figure(figsize=(12, 6))
    product_sales = df.groupby(['Date', 'Product'])['Sales'].sum().unstack()
    product_sales.plot(figsize=(12, 6))
    plt.title('Historical Daily Sales by Product')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend(title='Product', loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure to static/images directory
    plt.savefig('static/images/historical_sales.png')
    charts['historical_sales'] = 'static/images/historical_sales.png'
    
    # 2. Seasonal Patterns
    plt.figure(figsize=(12, 6))
    monthly_sales = df.groupby(['Month', 'Product'])['Sales'].mean().unstack()
    monthly_sales.plot(kind='bar', figsize=(12, 6))
    plt.title('Average Monthly Sales by Product')
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Product', loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure to static/images directory
    plt.savefig('static/images/seasonal_patterns.png')
    charts['seasonal_patterns'] = 'static/images/seasonal_patterns.png'
    
    # 3. Feature Importance
    plt.figure(figsize=(12, 6))
    top_features = feature_importance.head(15)  # Top 15 features
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 15 Features Affecting Sales Predictions')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Save figure to static/images directory
    plt.savefig('static/images/feature_importance.png')
    charts['feature_importance'] = 'static/images/feature_importance.png'
    
    # 4. Prediction vs Actual (Test Set)
    X_test, y_test, y_pred_test = test_data
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Predicted vs Actual Sales')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure to static/images directory
    plt.savefig('static/images/prediction_accuracy.png')
    charts['prediction_accuracy'] = 'static/images/prediction_accuracy.png'
    
    # 5. Scenario Comparison by Product
    plt.figure(figsize=(14, 8))
    unique_products = test_results['Product'].unique()
    num_products = len(unique_products)
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_products)))
    rows = grid_size
    cols = grid_size
    
    for i, product in enumerate(unique_products):
        if i < rows * cols:
            plt.subplot(rows, cols, i+1)
            product_results = test_results[test_results['Product'] == product]
            sns.barplot(x='Scenario', y='Predicted_Sales', data=product_results)
            plt.title(f'{product}')
            plt.xticks(rotation=45, fontsize=8)
            if i % cols == 0:
                plt.ylabel('Predicted Sales')
            plt.tight_layout()
    
    plt.suptitle('Predicted Sales by Scenario for Each Product', y=1.05)
    plt.tight_layout()
    
    # Save figure to static/images directory
    plt.savefig('static/images/scenario_comparison.png')
    charts['scenario_comparison'] = 'static/images/scenario_comparison.png'
    
    # 6. Promotional Impact
    plt.figure(figsize=(12, 6))
    promo_impact = test_results.groupby(['Product', 'Promotion'])['Predicted_Sales'].mean().unstack()
    promo_impact.plot(kind='bar', figsize=(12, 6))
    plt.title('Impact of Promotions on Predicted Sales by Product')
    plt.xlabel('Product')
    plt.ylabel('Average Predicted Sales')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.legend(['No Promotion', 'Promotion'])
    plt.tight_layout()
    
    # Save figure to static/images directory
    plt.savefig('static/images/promotion_impact.png')
    charts['promotion_impact'] = 'static/images/promotion_impact.png'
    
    # 7. Demand Forecast per Item (if model is provided)
    if model is not None and feature_cols is not None:
        # Generate demand forecast
        forecast_df = generate_demand_forecast(df, model, feature_cols, days_to_forecast=30)
        
        # Plot forecast for each product
        plt.figure(figsize=(14, 8))
        
        # Get top products by sales volume for better readability
        top_products = df.groupby('Product')['Sales'].sum().nlargest(9).index.tolist()
        
        for i, product in enumerate(top_products):
            if i < 9:  # Limit to 9 products for readability
                plt.subplot(3, 3, i+1)
                
                # Plot product forecast
                product_forecast = forecast_df[forecast_df['Product'] == product]
                plt.plot(product_forecast['Date'], product_forecast['Predicted_Demand'], 'b-')
                plt.title(f'{product}')
                plt.xticks(rotation=45, fontsize=8)
                plt.grid(True, alpha=0.3)
                if i % 3 == 0:
                    plt.ylabel('Predicted Demand (units)')
                plt.tight_layout()
        
        plt.suptitle('30-Day Demand Forecast by Product (Random Forest Model)', y=1.05)
        plt.tight_layout()
        
        # Save figure to static/images directory
        plt.savefig('static/images/demand_forecast.png')
        charts['demand_forecast'] = 'static/images/demand_forecast.png'
    
    return charts

def create_gradio_ui(model, df, feature_cols):
    """Create a Gradio UI for the retail forecasting model with enhanced visuals and Random Forest insights"""
    
    # Get unique products
    products = sorted(df['Product'].unique())
    
    # Get prepared dataset
    df_features, _ = prepare_features(df)
    
    # Generate a 30-day demand forecast for all products once (to avoid recalculating)
    forecast_df = generate_demand_forecast(df, model, feature_cols, days_to_forecast=30)
    
    # Prepare top feature importance data
    _, _, _, feature_importance, _ = train_forecast_model(df)
    top_features = feature_importance.head(10)
    
    # Create the prediction function
    def predict_quantity(product, price_adjustment, promotion, holiday, weather, lead_time, safety_stock_percent):
        # Get a row for this product as a template
        product_data = df_features[df_features['Product'] == product].sort_values('Date', ascending=False)
        if len(product_data) == 0:
            return "Product not found in the dataset", None, None, None
        
        # Use the most recent row as a template
        sample_row = product_data.iloc[0].copy()
        
        # Calculate adjusted price based on percentage adjustment
        original_price = sample_row['Price']
        adjusted_price = original_price * (1 + price_adjustment / 100)
        
        # Update values based on inputs
        sample_row['Price'] = adjusted_price
        sample_row['Promotion'] = 1 if promotion else 0
        sample_row['Is_Holiday'] = 1 if holiday else 0
        sample_row['Lead_Time'] = lead_time
        
        # Update weather columns
        for col in feature_cols:
            if col.startswith('Weather_'):
                weather_type = col.replace('Weather_', '')
                sample_row[col] = 1 if weather_type == weather else 0
        
        # Extract only the features needed for prediction
        test_features = pd.DataFrame([sample_row[feature_cols]])
        
        # Make prediction
        predicted_sales = int(model.predict(test_features)[0])
        recommended_order = max(int(predicted_sales * (1 + safety_stock_percent / 100)), predicted_sales + 1)  # Add safety stock or at least 1 unit
        
        # Create a chart showing factors influencing the decision
        plt.figure(figsize=(10, 6))
        
        # Create a bar chart showing the prediction and factors
        factors = {
            'Base Prediction': predicted_sales,
            'With Safety Stock': recommended_order
        }
        
        # Add comparison scenarios
        comparison_scenarios = []
        
        # No promotion scenario
        if promotion:
            no_promo_row = sample_row.copy()
            no_promo_row['Promotion'] = 0
            no_promo_features = pd.DataFrame([no_promo_row[feature_cols]])
            factors['Without Promotion'] = int(model.predict(no_promo_features)[0])
            comparison_scenarios.append('Without Promotion')
        
        # No holiday scenario
        if holiday:
            no_holiday_row = sample_row.copy()
            no_holiday_row['Is_Holiday'] = 0
            no_holiday_features = pd.DataFrame([no_holiday_row[feature_cols]])
            factors['Without Holiday'] = int(model.predict(no_holiday_features)[0])
            comparison_scenarios.append('Without Holiday')
        
        # Normal weather scenario
        if weather != 'Normal':
            normal_weather_row = sample_row.copy()
            for col in feature_cols:
                if col.startswith('Weather_'):
                    weather_type = col.replace('Weather_', '')
                    normal_weather_row[col] = 1 if weather_type == 'Normal' else 0
            normal_weather_features = pd.DataFrame([normal_weather_row[feature_cols]])
            factors['With Normal Weather'] = int(model.predict(normal_weather_features)[0])
            comparison_scenarios.append('With Normal Weather')
        
        # Regular price scenario
        if abs(price_adjustment) > 1:
            reg_price_row = sample_row.copy()
            reg_price_row['Price'] = original_price
            reg_price_features = pd.DataFrame([reg_price_row[feature_cols]])
            factors['With Regular Price'] = int(model.predict(reg_price_features)[0])
            comparison_scenarios.append('With Regular Price')
        
        # Plot the factors - decision chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(factors.keys(), factors.values(), color=['blue', 'green'] + ['gray'] * len(comparison_scenarios))
        
        # Highlight the recommended order
        bars[1].set_color('green')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}',
                    ha='center', va='bottom', rotation=0)
        
        plt.title(f'Predicted Sales for {product} with Decision Factors')
        plt.ylabel('Quantity')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add condition information as text
        condition_text = f"Price: {'↓' if price_adjustment < 0 else '↑' if price_adjustment > 0 else '='} ({adjusted_price:.2f})\n"
        condition_text += f"Promotion: {'Yes' if promotion else 'No'}\n"
        condition_text += f"Holiday: {'Yes' if holiday else 'No'}\n"
        condition_text += f"Weather: {weather}\n"
        condition_text += f"Lead Time: {lead_time} days\n"
        condition_text += f"Safety Stock: {safety_stock_percent}%"
        
        plt.figtext(0.15, 0.01, condition_text, ha='left')
        
        # Add recommendation as text
        recommendation = f"Recommended Order: {recommended_order} units\n"
        recommendation += f"(Includes {safety_stock_percent}% safety stock above predicted sales of {predicted_sales})"
        
        plt.figtext(0.65, 0.01, recommendation, ha='right', fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        
        # Save the figure to static/images directory
        plt.savefig('static/images/prediction_chart.png')
        
        # Plot the forecast for this product
        plt.figure(figsize=(12, 6))
        product_forecast = forecast_df[forecast_df['Product'] == product]
        
        if len(product_forecast) > 0:
            plt.plot(product_forecast['Date'], product_forecast['Predicted_Demand'], 'b-', marker='o')
            plt.title(f'30-Day Demand Forecast for {product}')
            plt.xlabel('Date')
            plt.ylabel('Predicted Demand (Units)')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Add the current prediction
            plt.axhline(y=predicted_sales, color='r', linestyle='--', label=f'Current Prediction: {predicted_sales}')
            plt.legend()
            
            # Save the forecast chart to static/images directory
            plt.savefig('static/images/forecast_chart.png')
        else:
            # If no forecast data, create a placeholder
            plt.text(0.5, 0.5, f"No forecast data available for {product}", 
                     ha='center', va='center', fontsize=14)
            plt.axis('off')
            plt.savefig('static/images/forecast_chart.png')
        
        # Create a feature importance plot for this specific prediction
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title('Top 10 Features Affecting Sales Predictions')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('static/images/feature_importance_chart.png')
        
        # Return explanation and charts
        explanation = (
            f"# Prediction Results for {product}\n\n"
            f"## Recommended Order: {recommended_order} units\n\n"
            f"### Factors:\n"
            f"- Predicted Sales: {predicted_sales} units\n"
            f"- Safety Stock: {recommended_order - predicted_sales} units ({safety_stock_percent}%)\n\n"
            f"### Conditions:\n"
            f"- Price: ${adjusted_price:.2f}"
            f"{' (' + str(price_adjustment) + '%)' if price_adjustment != 0 else ''}\n"
            f"- Promotion: {'Yes' if promotion else 'No'}\n"
            f"- Holiday: {'Yes' if holiday else 'No'}\n"
            f"- Weather: {weather}\n"
            f"- Lead Time: {lead_time} days\n\n"
            f"### Reasoning:\n"
            f"The Random Forest model analyzed historical sales patterns for {product} and determined "
            f"a base predicted demand of {predicted_sales} units. With a {safety_stock_percent}% safety stock, "
            f"the recommended order quantity is {recommended_order} units.\n\n"
            f"This prediction takes into account factors like price elasticity, promotional effects, "
            f"seasonality, and external conditions like weather and holidays. The model was trained on "
            f"historical sales data and optimized to minimize prediction error.\n\n"
            f"The forecast chart shows the expected demand over the next 30 days, which can help with "
            f"inventory planning and identifying upcoming trends."
        )
        
        return explanation, 'static/images/prediction_chart.png', 'static/images/forecast_chart.png', 'static/images/feature_importance_chart.png'
    
    # Create the interface with tabs
    with gr.Blocks(title="Retail Product Order Quantity Prediction") as demo:
        gr.Markdown("# Retail Product Order Quantity Prediction\nPredict optimal order quantities for retail products based on various factors using Random Forest model.")
        
        with gr.Row():
            with gr.Column(scale=1):
                product = gr.Dropdown(products, value=products[0], label="Select Product")
                price_adjustment = gr.Slider(-20, 20, value=0, step=1, label="Price Adjustment (%)")
                promotion = gr.Checkbox(label="Promotional Period")
                holiday = gr.Checkbox(label="Holiday Period")
                weather = gr.Dropdown(["Normal", "Heavy Rain", "Snow", "Storm"], value="Normal", label="Weather Condition")
                lead_time = gr.Slider(1, 30, value=7, step=1, label="Lead Time (days)")
                safety_stock = gr.Slider(0, 30, value=10, step=1, label="Safety Stock (%)")
                submit_btn = gr.Button("Calculate Recommended Order")
            
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Recommendation"):
                        output_text = gr.Markdown(label="Prediction Results")
                        decision_chart = gr.Image(label="Decision Factors")
                    
                    with gr.TabItem("Demand Forecast"):
                        forecast_chart = gr.Image(label="30-Day Demand Forecast")
                    
                    with gr.TabItem("Feature Importance"):
                        importance_chart = gr.Image(label="Feature Importance")
        
        submit_btn.click(
            predict_quantity,
            inputs=[product, price_adjustment, promotion, holiday, weather, lead_time, safety_stock],
            outputs=[output_text, decision_chart, forecast_chart, importance_chart]
        )
    
    return demo

def main():
    """Main function to run the entire forecasting workflow"""
    
    print("Loading retail data from CSV...")
    df = load_data_from_csv()
    
    print("\nTraining forecasting model...")
    model, df_features, feature_cols, feature_importance, test_data = train_forecast_model(df)
    
    print("\nRunning test scenarios...")
    test_results = run_tests(model, df, feature_cols)
    
    print("\nGenerating visualization charts...")
    charts = generate_charts(df, test_results, feature_importance, test_data, model, feature_cols)
    
    print("\nGenerating demand forecast...")
    forecast_df = generate_demand_forecast(df, model, feature_cols, days_to_forecast=30)
    
    print("\nSaving test results...")
    # Write test results to file
    with open('testresult.txt', 'w') as f:
        f.write("# Retail Product Order Quantity Prediction Test Results\n\n")
        
        f.write("## Model Performance Metrics\n")
        X_test, y_test, y_pred_test = test_data
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        f.write(f"- Root Mean Squared Error (RMSE): {test_rmse:.2f}\n")
        f.write(f"- Mean Absolute Error (MAE): {test_mae:.2f}\n")
        f.write(f"- R² Score: {test_r2:.4f}\n\n")
        
        f.write("## Top Factors Influencing Predictions\n")
        for i, (feature, importance) in enumerate(zip(feature_importance['Feature'].head(10), 
                                                    feature_importance['Importance'].head(10))):
            f.write(f"{i+1}. {feature}: {importance:.4f}\n")
        f.write("\n")
        
        f.write("## Test Scenarios Results\n")
        for product in test_results['Product'].unique():
            f.write(f"\n### {product}\n")
            product_results = test_results[test_results['Product'] == product]
            for _, row in product_results.iterrows():
                f.write(f"- {row['Scenario']}:\n")
                f.write(f"  - Price: ${row['Price']:.2f}\n")
                f.write(f"  - Promotion: {'Yes' if row['Promotion'] else 'No'}\n")
                f.write(f"  - Holiday: {'Yes' if row['Holiday'] else 'No'}\n")
                f.write(f"  - Weather: {row['Weather']}\n")
                f.write(f"  - Predicted Sales: {row['Predicted_Sales']} units\n")
                f.write(f"  - Recommended Order: {row['Recommended_Order']} units\n")
        
        f.write("\n## 30-Day Demand Forecast (Top 5 Products)\n")
        for product in forecast_df['Product'].unique()[:5]:
            f.write(f"\n### {product}\n")
            product_forecast = forecast_df[forecast_df['Product'] == product]
            f.write(f"| Date | Predicted Demand |\n|------|-----------------|\n")
            for _, row in product_forecast.iterrows():
                f.write(f"| {row['Date'].strftime('%Y-%m-%d')} | {row['Predicted_Demand']} |\n")
    
    print("\nLaunching Gradio UI...")
    demo = create_gradio_ui(model, df, feature_cols)
    demo.launch()

if __name__ == "__main__":
    main()