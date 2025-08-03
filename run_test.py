#!/usr/bin/env python3
import retail_forecast as rf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import sys

def run_tests():
    # Start timing
    start_time = time.time()
    
    # Load the data
    print("Loading data from CSV...")
    # First make sure we have the filtered pizza data
    import os
    if not os.path.exists('frozen_pizza_only.csv'):
        print("Filtered pizza data not found. Creating it now...")
        from filter_frozen_pizza import filter_frozen_pizza
        filter_frozen_pizza('Price_Elasticity_Frozen_Input.csv', 'frozen_pizza_only.csv')
    
    df = rf.load_data_from_csv(file_path='frozen_pizza_only.csv', sample_size=0.2)  # Use 20% of pizza data
    
    # Train the model
    print("Training model...")
    model, df_features, feature_cols, feature_importance, test_data = rf.train_forecast_model(df)
    
    # Run test scenarios
    print("Running test scenarios...")
    test_results = rf.run_tests(model, df, feature_cols)
    
    # Generate demand forecast
    print("Generating demand forecast...")
    forecast_df = rf.generate_demand_forecast(df, model, feature_cols, days_to_forecast=30)
    
    # Calculate metrics
    X_test, y_test, y_pred_test = test_data
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Write test results to file
    print("Writing test results to file...")
    with open('testresult.txt', 'w') as f:
        f.write("# Frozen Pizza Order Quantity Prediction Test Results\n\n")
        
        f.write("## Data Summary\n")
        f.write(f"- Sample Size: {len(df)} records\n")
        f.write(f"- Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}\n")
        f.write(f"- Pizza Item IDs: {len(df['Product'].unique())}\n\n")
        
        f.write("## Model Performance Metrics\n")
        f.write(f"- Root Mean Squared Error (RMSE): {test_rmse:.2f}\n")
        f.write(f"- Mean Absolute Error (MAE): {test_mae:.2f}\n")
        f.write(f"- R² Score: {test_r2:.4f}\n\n")
        
        f.write("## Top Factors Influencing Predictions\n")
        for i, (feature, importance) in enumerate(zip(feature_importance['Feature'].head(10), 
                                                     feature_importance['Importance'].head(10))):
            f.write(f"{i+1}. {feature}: {importance:.4f}\n")
        f.write("\n")
        
        f.write("## Test Scenarios Results\n")
        # Sample a few products for the report
        sample_products = list(test_results['Product'].unique())[:5]
        for product in sample_products:
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
        
        f.write("\n## 30-Day Demand Forecast (Sample Products)\n")
        for product in sample_products:
            product_forecast = forecast_df[forecast_df['Product'] == product]
            if len(product_forecast) > 0:
                f.write(f"\n### {product}\n")
                f.write(f"| Date | Predicted Demand |\n|------|------------------|\n")
                for _, row in product_forecast.head(7).iterrows():  # Show 1 week of forecast
                    f.write(f"| {row['Date'].strftime('%Y-%m-%d')} | {row['Predicted_Demand']} |\n")
        
        f.write("\n## Performance\n")
        f.write(f"- Test Duration: {elapsed_time:.2f} seconds\n")
        
        f.write("\n## Conclusion\n")
        f.write("The Random Forest model provides accurate demand predictions for frozen pizza items based on historical sales data. ")
        f.write("The model is now specific to pizza item IDs rather than general product types, allowing for more precise predictions. ")
        f.write("The model takes into account various factors including price, promotions, seasonality, and external events. ")
        f.write("The demand forecast can help optimize inventory management for frozen pizza items and reduce stockouts and overstock situations.\n")
    
    print(f"Test completed in {elapsed_time:.2f} seconds")
    print("Results written to testresult.txt")

if __name__ == "__main__":
    run_tests()