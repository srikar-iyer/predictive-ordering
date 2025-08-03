#!/usr/bin/env python3
"""
Script to create a sample dataset for testing the RF model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_sample_data(num_stores=3, num_products=5, num_days=120):
    """Create a sample dataset for training the RF model
    
    Args:
        num_stores: Number of stores to simulate
        num_products: Number of products per store
        num_days: Number of days of historical data
    
    Returns:
        DataFrame with simulated data
    """
    print(f"Creating sample dataset with {num_stores} stores, {num_products} products, {num_days} days")
    
    # Create base date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=num_days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create store IDs and product IDs
    store_ids = list(range(1, num_stores + 1))
    product_ids = list(range(1, num_products + 1))
    
    # Product names
    product_names = [
        "Cheese Pizza",
        "Pepperoni Pizza",
        "Supreme Pizza",
        "Veggie Pizza",
        "Hawaiian Pizza",
        "Meat Lovers Pizza",
        "BBQ Chicken Pizza",
        "Buffalo Pizza",
        "Margherita Pizza",
        "Four Cheese Pizza"
    ]
    
    # Size options
    sizes = ["Small", "Medium", "Large", "X-Large"]
    
    # Weather options
    weather_types = ["Sunny", "Rainy", "Cloudy", "Snowy", "Stormy", "Normal"]
    
    # Create records for each combination of store, product, and date
    records = []
    
    for store_id in store_ids:
        for product_id in product_ids:
            # Assign product name and size
            product_name = product_names[(product_id - 1) % len(product_names)]
            size = sizes[(product_id - 1) % len(sizes)]
            
            # Base parameters for this product
            base_price = 8.99 + (product_id * 1.5) + (store_id * 0.5)
            base_cost = base_price * 0.4  # 40% cost
            base_demand = 10 + (product_id * 3) + (store_id * 2)
            
            # Seasonality parameters
            weekly_amplitude = 0.3  # 30% variation during the week
            yearly_amplitude = 0.2  # 20% variation during the year
            
            # Create records for each date
            for date in dates:
                # Add weekly seasonality
                day_of_week = date.weekday()  # 0=Monday, 6=Sunday
                weekly_factor = 1.0 + weekly_amplitude * np.sin(day_of_week * np.pi / 3.5)
                
                # Add yearly seasonality
                day_of_year = date.timetuple().tm_yday
                yearly_factor = 1.0 + yearly_amplitude * np.sin(day_of_year * 2 * np.pi / 365)
                
                # Add random noise
                noise = np.random.normal(1.0, 0.1)
                
                # Calculate sales with seasonality and noise
                sales = max(0, base_demand * weekly_factor * yearly_factor * noise)
                
                # Add price variation and promotions
                price = base_price
                on_promotion = False
                
                # Random promotions (10% chance)
                if np.random.random() < 0.1:
                    price = base_price * 0.8  # 20% off
                    on_promotion = True
                
                # Weather effect
                weather = np.random.choice(weather_types, p=[0.2, 0.1, 0.1, 0.05, 0.05, 0.5])
                
                # Adjust sales based on weather
                weather_factor = 1.0
                if weather == "Rainy":
                    weather_factor = 1.2  # People order more when it's raining
                elif weather == "Snowy":
                    weather_factor = 1.3  # People order even more when it's snowing
                elif weather == "Sunny":
                    weather_factor = 0.9  # People order less when it's nice out
                
                sales *= weather_factor
                
                # Calculate stock metrics
                initial_stock = base_demand * 7  # Start with a week's worth of stock
                stock_movement = sales
                
                # Randomize restock days
                if day_of_week == 1:  # Restock on Tuesdays
                    stock_added = base_demand * 7  # Restock a week's worth
                else:
                    stock_added = 0
                
                stock_level = max(0, initial_stock - stock_movement + stock_added)
                weeks_of_stock = stock_level / (base_demand * 7) if base_demand > 0 else 4
                
                # Determine stock status
                if weeks_of_stock < 1:
                    stock_status = "Low"
                elif weeks_of_stock <= 3:
                    stock_status = "Adequate"
                else:
                    stock_status = "Excess"
                
                # Create record
                record = {
                    'Date': date,
                    'Store_Id': store_id,
                    'Item': product_id,
                    'Product': product_name,
                    'Size': size,
                    'Price': round(price, 2),
                    'Cost': round(base_cost, 2),
                    'Sales': round(sales),
                    'Weather': weather,
                    'Day_Of_Week': day_of_week,
                    'Month': date.month,
                    'Year': date.year,
                    'Day': date.day,
                    'Is_Holiday': 1 if date.month == 1 and date.day == 1 else 0,  # New Year's Day
                    'Promotion': 1 if on_promotion else 0,
                    'Stock_Level': round(stock_level),
                    'Weeks_Of_Stock': round(weeks_of_stock, 2),
                    'Stock_Status': stock_status,
                    'Recent_Daily_Sales': round(sales),
                    'Avg_Weekly_Sales_4W': round(base_demand * 7),
                }
                
                records.append(record)
    
    # Create dataframe
    df = pd.DataFrame(records)
    
    # Add rolling metrics
    for store_id in store_ids:
        for product_id in product_ids:
            product_data = df[(df['Store_Id'] == store_id) & (df['Item'] == product_id)]
            df.loc[(df['Store_Id'] == store_id) & (df['Item'] == product_id), 'Avg_Weekly_Sales_4W'] = \
                product_data['Sales'].rolling(window=28, min_periods=1).mean() * 7
    
    # Make sure there are no NaN values
    df = df.fillna(0)
    
    # Save to CSV
    output_path = 'combined_pizza_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Sample dataset created with {len(df)} records")
    print(f"Saved to {output_path}")
    
    return df

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create sample data for RF model testing')
    parser.add_argument('--stores', type=int, default=3,
                       help='Number of stores (default: 3)')
    parser.add_argument('--products', type=int, default=5,
                       help='Number of products per store (default: 5)')
    parser.add_argument('--days', type=int, default=120,
                       help='Number of days of historical data (default: 120)')
    
    args = parser.parse_args()
    
    create_sample_data(num_stores=args.stores, num_products=args.products, num_days=args.days)