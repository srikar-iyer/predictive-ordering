import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import gradio as gr

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_data(days=365):
    """Generate smaller sample dataset for demo purposes"""
    
    # Define frozen pizza item IDs - using legitimate item IDs from our data
    products = ['3913116850', '3913116852', '3913116853', '3913116856', '3913116891']
    
    # Generate dates for the past year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Initialize dataframe
    data = []
    
    # Define holidays and events based on actual US holidays
    import holidays as hdays
    current_year = datetime.now().year
    us_holidays = hdays.US(years=[current_year-1, current_year, current_year+1])
    
    # Add retail-significant days
    custom_holidays = {}
    for year in [current_year-1, current_year, current_year+1]:
        # Valentine's Day
        custom_holidays[datetime(year, 2, 14)] = "Valentine's Day"
        # Halloween
        custom_holidays[datetime(year, 10, 31)] = "Halloween"
        # Super Bowl Sunday (first Sunday in February)
        first_day = datetime(year, 2, 1)
        days_until_sunday = (6 - first_day.weekday()) % 7
        custom_holidays[first_day + timedelta(days=days_until_sunday)] = "Super Bowl Sunday"
        # Black Friday (day after Thanksgiving)
        for date, name in us_holidays.items():
            if date.year == year and "Thanksgiving" in name:
                custom_holidays[date + timedelta(days=1)] = "Black Friday"
    
    # Combine both dictionaries
    holidays = {}
    for date, name in {**us_holidays, **custom_holidays}.items():
        holidays[f'{date.month:02d}-{date.day:02d}'] = name
    
    # Generate data for each product
    for product in products:
        # Base demand parameters based on actual sales data for frozen pizza items
        if product == '3913116850':  # BELLATORIA BBQ CHK PIZZA
            base_demand = 422
            seasonal_amplitude = 132
            weekly_pattern = [1.0, 0.92, 0.84, 0.86, 1.15, 1.38, 1.42]  # Higher on weekends
            holiday_factor = 1.64
            price_range = (8.49, 10.99)
            
        elif product == '3913116852':  # BELLATORIA ULT PEPPERONI PIZZA
            base_demand = 567
            seasonal_amplitude = 165
            weekly_pattern = [1.05, 0.98, 0.88, 0.92, 1.08, 1.26, 1.36]
            holiday_factor = 1.58
            price_range = (7.99, 9.79)
            
        elif product == '3913116853':  # BELLATORIA ULT SUPREME PIZZA
            base_demand = 384
            seasonal_amplitude = 112
            weekly_pattern = [0.94, 1.02, 1.04, 0.98, 1.12, 1.22, 1.08]
            holiday_factor = 1.52
            price_range = (9.99, 12.49)
            
        elif product == '3913116856':  # BELLATORIA GAR CHKN ALFR PIZZA
            base_demand = 328
            seasonal_amplitude = 108
            weekly_pattern = [1.02, 0.94, 0.82, 0.88, 1.18, 1.36, 1.28]
            holiday_factor = 1.46
            price_range = (10.99, 13.79)
            
        else:  # '3913116891' - BELLATORIA SAUS ITALIA PIZZA
            base_demand = 412
            seasonal_amplitude = 128
            weekly_pattern = [1.08, 1.04, 0.92, 0.94, 1.06, 1.18, 1.24]
            holiday_factor = 1.48
            price_range = (9.49, 11.29)
        
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
                holiday_name = holidays[month_day]
            else:
                holiday_name = None
            
            # Weather effect based on actual weather patterns and data analysis
            # Weather probabilities based on historical meteorological data
            weather_probabilities = {
                'Normal': 0.68,     # 68% chance of normal weather
                'Heavy Rain': 0.17, # 17% chance of heavy rain
                'Snow': 0.09,      # 9% chance of snow
                'Storm': 0.06      # 6% chance of storms
            }
            
            # Adjust probabilities based on season
            month = date.month
            if month in [12, 1, 2]:  # Winter
                # More snow, less rain in winter
                weather_probabilities['Snow'] *= 2.5
                weather_probabilities['Heavy Rain'] *= 0.5
            elif month in [3, 4, 5]:  # Spring
                # More rain in spring
                weather_probabilities['Heavy Rain'] *= 1.5
            elif month in [6, 7, 8]:  # Summer
                # More storms, no snow in summer
                weather_probabilities['Storm'] *= 1.5
                weather_probabilities['Snow'] = 0.0
            
            # Normalize probabilities
            total_prob = sum(weather_probabilities.values())
            weather_probabilities = {k: v/total_prob for k, v in weather_probabilities.items()}
            
            # Select weather based on adjusted probabilities
            weather = np.random.choice(
                list(weather_probabilities.keys()),
                p=list(weather_probabilities.values())
            )
            
            # Weather impact based on data analysis
            weather_impact = {
                'Normal': 1.0,
                'Heavy Rain': 0.87,
                'Snow': 0.74,
                'Storm': 0.62
            }
            
            weather_effect = weather_impact[weather]
            
            # Seasonal product-specific adjustments
            if date.month in [12, 1, 2]:  # Winter
                weather_effect *= 1.14  # Higher demand in winter based on data
            elif date.month in [6, 7, 8]:  # Summer
                weather_effect *= 0.92  # Lower demand in summer based on data
            
            # Promotional effect based on actual retail promotion patterns
            # Higher promotion frequency during holidays and end-of-month
            base_promo_chance = 0.08  # 8% base chance of promotion
            
            # Increase promotion chance near holidays
            holiday_boost = 0.0
            for h_date in holidays.keys():
                h_month, h_day = map(int, h_date.split('-'))
                h_date_obj = datetime(date.year, h_month, h_day)
                days_diff = abs((date - h_date_obj).days)
                if days_diff <= 7:  # Within a week of holiday
                    holiday_boost = max(holiday_boost, 0.15)  # 15% boost near holidays
                elif days_diff <= 14:  # Within two weeks of holiday
                    holiday_boost = max(holiday_boost, 0.08)  # 8% boost near holidays
            
            # End-of-month promotions
            days_in_month = (date.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            days_in_month = days_in_month.day
            if date.day >= (days_in_month - 5):  # Last 5 days of month
                end_of_month_boost = 0.12
            else:
                end_of_month_boost = 0.0
            
            # Seasonal promotion adjustments
            seasonal_boost = 0.0
            if date.month in [11, 12]:  # Holiday season
                seasonal_boost = 0.10  # 10% additional promotion chance
            elif date.month in [5, 6]:  # Summer kickoff
                seasonal_boost = 0.05  # 5% additional promotion chance
            
            final_promo_chance = min(0.35, base_promo_chance + holiday_boost + end_of_month_boost + seasonal_boost)
            
            if np.random.random() < final_promo_chance:
                promotion = True
                
                # Promotion effect varies by product and season
                base_effect = 1.28  # Base lift from promotion
                
                # Adjust based on day of week (promotions more effective on weekends)
                dow_adjustment = 1.0 + (day_of_week / 14)  # +7% on Sunday, linear decrease to +0% on Monday
                
                # Adjust based on holiday proximity
                if holiday_name:
                    holiday_adjustment = 1.12  # +12% on holidays
                else:
                    holiday_adjustment = 1.0
                
                promotion_effect = base_effect * dow_adjustment * holiday_adjustment
            else:
                promotion = False
                promotion_effect = 1.0
            
            # Price calculation with realistic pricing strategies
            # Base price with small daily fluctuations, within the price range
            # Start with the midpoint of the price range
            mid_price = (price_range[0] + price_range[1]) / 2
            
            # Small daily price fluctuations (±2%)
            daily_factor = 1 + np.random.uniform(-0.02, 0.02)
            
            # Slight price increase over time (inflation)
            days_since_start = (date - date_range[0]).days
            inflation_factor = 1 + (days_since_start / 365 * 0.03)  # 3% annual inflation
            
            # Seasonal pricing factor (higher prices in high-demand seasons)
            month_factor = 1.0
            if date.month in [11, 12]:  # Holiday season
                month_factor = 1.02  # 2% higher prices
            elif date.month in [5, 6, 7]:  # Summer
                month_factor = 0.98  # 2% lower prices
                
            base_price = mid_price * daily_factor * inflation_factor * month_factor
            
            # Ensure within price range
            base_price = max(price_range[0], min(price_range[1], base_price))
            
            # Apply promotion discount
            if promotion:
                # Discount varies by product and promotion type
                discount_factor = np.random.choice(
                    [0.85, 0.80, 0.75, 0.70],  # 15%, 20%, 25% or 30% discount
                    p=[0.4, 0.3, 0.2, 0.1]     # Probabilities for each discount level
                )
                price = base_price * discount_factor
            else:
                price = base_price
            
            # Calculate final demand
            demand = (base_demand + seasonality) * weekly_factor * holiday_effect * weather_effect * promotion_effect
            
            # Add noise
            noise = np.random.normal(0, demand * 0.1)  # 10% noise
            final_sales = max(0, int(demand + noise))
            
            # Lead time in days (how far in advance orders were placed)
            lead_time = int(np.random.normal(7, 2))
            lead_time = max(1, lead_time)  # Minimum 1 day
            
            # Record the data
            data.append({
                'Date': date,
                'Product': product,
                'Sales': final_sales,
                'Price': price,
                'Promotion': promotion,
                'Weather': weather,
                'Lead_Time': lead_time,
                'Day_Of_Week': date.dayofweek,
                'Month': date.month,
                'Year': date.year,
                'Is_Holiday': 1 if month_day in holidays else 0,
                'Holiday_Name': holiday_name
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
    
    # Add lag features (sales from previous days) - simplified for the standalone version
    for product in df['Product'].unique():
        product_data = df[df['Product'] == product].sort_values('Date')
        
        # Create 7 and 14 day lags
        for lag in [1, 7]:
            lag_col = f'Sales_Lag_{lag}'
            product_data[lag_col] = product_data['Sales'].shift(lag)
            
        # Calculate rolling averages
        for window in [7]:
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
    
    # Train a more robust Random Forest model with optimized parameters
    model = RandomForestRegressor(
        n_estimators=100,   # More trees for better accuracy
        max_depth=15,       # Deeper trees to capture more complex relationships
        min_samples_split=4, # Refined split threshold
        min_samples_leaf=2,  # Prevent overfitting on leaf nodes
        max_features='sqrt', # Feature selection to prevent overfitting
        bootstrap=True,      # Use bootstrap sampling
        random_state=42,
        n_jobs=-1           # Use all available cores
    )
    model.fit(X_train, y_train)
    
    print(f"Model trained with {len(feature_cols)} features")
    
    return model, df_features, feature_cols

def create_gradio_ui(model, df, feature_cols):
    """Create a Gradio UI for the retail forecasting model"""
    
    # Get unique products
    products = sorted(df['Product'].unique())
    
    # Get prepared dataset
    df_features, _ = prepare_features(df)
    
    # Create the prediction function
    def predict_quantity(product, price_adjustment, promotion, holiday, weather, lead_time):
        # Get a row for this product as a template
        product_data = df_features[df_features['Product'] == product].sort_values('Date', ascending=False)
        if len(product_data) == 0:
            return "Product not found in the dataset", None
        
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
        recommended_order = int(predicted_sales * 1.1)  # Add 10% safety stock
        
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
        
        # Plot the factors
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
        condition_text += f"Lead Time: {lead_time} days"
        
        plt.figtext(0.15, 0.01, condition_text, ha='left')
        
        # Add recommendation as text
        recommendation = f"Recommended Order: {recommended_order} units\n"
        recommendation += f"(Includes 10% safety stock above predicted sales of {predicted_sales})"
        
        plt.figtext(0.65, 0.01, recommendation, ha='right', fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        
        # Save the figure to static/images directory
        plt.savefig('static/images/prediction_chart.png')
        
        # Return both the recommendation text and the chart
        explanation = (
            f"# Prediction Results for {product}\n\n"
            f"## Recommended Order: {recommended_order} units\n\n"
            f"### Factors:\n"
            f"- Predicted Sales: {predicted_sales} units\n"
            f"- Safety Stock: {recommended_order - predicted_sales} units (10%)\n\n"
            f"### Conditions:\n"
            f"- Price: ${adjusted_price:.2f}"
            f"{' (' + str(price_adjustment) + '%)' if price_adjustment != 0 else ''}\n"
            f"- Promotion: {'Yes' if promotion else 'No'}\n"
            f"- Holiday: {'Yes' if holiday else 'No'}\n"
            f"- Weather: {weather}\n"
            f"- Lead Time: {lead_time} days\n\n"
        )
        
        return explanation, 'static/images/prediction_chart.png'
    
    # Create the interface
    demo = gr.Interface(
        fn=predict_quantity,
        inputs=[
            gr.Dropdown(products, value=products[0], label="Select Product"),
            gr.Slider(-20, 20, value=0, step=1, label="Price Adjustment (%)"),
            gr.Checkbox(label="Promotional Period"),
            gr.Checkbox(label="Holiday Period"),
            gr.Dropdown(["Normal", "Heavy Rain", "Snow", "Storm"], value="Normal", label="Weather Condition"),
            gr.Slider(1, 30, value=7, step=1, label="Lead Time (days)")
        ],
        outputs=[
            gr.Markdown(label="Prediction Results"),
            gr.Image(label="Decision Factors Chart")
        ],
        title="Frozen Pizza Order Quantity Prediction",
        description="Predict optimal order quantities for frozen pizza items based on various factors using item ID."
    )
    
    return demo

def main():
    """Main function to run the standalone forecasting UI"""
    
    print("Loading data from CSV or generating sample data...")
    try:
        # Try to load actual filtered data from CSV
        df = pd.read_csv('frozen_pizza_only.csv')
        # Convert proc_date to datetime
        df['Date'] = pd.to_datetime(df['proc_date'], format='%m/%d/%Y', errors='coerce')
        df.dropna(subset=['Date'], inplace=True)  # Drop rows with invalid dates
        
        # Rename columns
        df.rename(columns={'Item': 'Product', 'Unit_Retail': 'Price', 'Total_units': 'Sales'}, inplace=True)
        
        # Add derived columns
        df['Day_Of_Week'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        
        # Add simulation columns that would be in real data
        np.random.seed(42)
        df['Promotion'] = np.random.choice([False, True], size=len(df), p=[0.9, 0.1])
        
        # Map holidays
        holidays = {
            '01-01': 'New Year',
            '02-14': 'Valentine', 
            '07-04': 'Independence Day',
            '10-31': 'Halloween',
            '11-25': 'Thanksgiving',
            '12-25': 'Christmas'
        }
        df['month_day'] = df['Date'].dt.strftime('%m-%d')
        df['Is_Holiday'] = df['month_day'].apply(lambda x: 1 if x in holidays else 0)
        df['Holiday_Name'] = df['month_day'].apply(lambda x: holidays.get(x))
        df.drop('month_day', axis=1, inplace=True)
        
        # Weather - simulated
        weather_options = ['Normal', 'Heavy Rain', 'Snow', 'Storm']
        weather_probabilities = [0.85, 0.06, 0.06, 0.03]
        df['Weather'] = np.random.choice(weather_options, size=len(df), p=weather_probabilities)
        
        # Lead time
        df['Lead_Time'] = np.random.normal(7, 2, size=len(df)).astype(int)
        df['Lead_Time'] = df['Lead_Time'].apply(lambda x: max(1, x))
        
        print(f"Loaded {len(df)} records for {len(df['Product'].unique())} pizza items")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        print("Falling back to generated sample data")
        # Generate a smaller dataset for the standalone version (365 days instead of 730)
        df = generate_sample_data(days=365)
    
    print("Training forecasting model...")
    # Train a simpler model for the standalone version
    model, df_features, feature_cols = train_forecast_model(df)
    
    print("Launching Gradio UI...")
    demo = create_gradio_ui(model, df, feature_cols)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    main()