#!/usr/bin/env python3
"""
Test script to evaluate the Random Forest model's performance with and without retraining
This script helps demonstrate the effectiveness of the toggle option
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import os
import argparse

# Import functions from rf_model_update
from rf_model_update import (
    prepare_features, 
    train_forecast_model, 
    load_model,
    save_model,
    generate_demand_forecast
)

def test_model_performance(use_existing=False, test_days=14):
    """Test the model performance with and without retraining
    
    Args:
        use_existing (bool): Whether to use the existing model instead of retraining
        test_days (int): Number of days to use for testing (these will be held out)
    
    Returns:
        dict: Dictionary of performance metrics
    """
    # Load the data
    print(f"\n{'='*80}")
    print(f"TESTING RANDOM FOREST MODEL PERFORMANCE")
    print(f"Mode: {'Using existing model' if use_existing else 'Training new model'}")
    print(f"{'='*80}")
    
    # Start timing
    start_time = time.time()
    
    # Load the integrated dataset
    df = pd.read_csv('combined_pizza_data.csv')
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Split into train and test - use the last test_days for testing
    unique_dates = df['Date'].unique()
    unique_dates.sort()
    
    if len(unique_dates) <= test_days:
        print(f"WARNING: Not enough data for testing. Using half the data for testing.")
        split_idx = len(unique_dates) // 2
    else:
        split_idx = len(unique_dates) - test_days
        
    train_dates = unique_dates[:split_idx]
    test_dates = unique_dates[split_idx:]
    
    train_df = df[df['Date'].isin(train_dates)].copy()
    test_df = df[df['Date'].isin(test_dates)].copy()
    
    print(f"Train data: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
    print(f"Test data: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")
    
    # Train or load model based on the flag
    if use_existing and os.path.exists('models/rf_model.pkl'):
        print("Loading existing model...")
        model, feature_cols = load_model()
        # Prepare features for consistency
        df_features, _ = prepare_features(train_df)
    else:
        print("Training new model...")
        model, df_features, feature_cols, feature_importance, _ = train_forecast_model(train_df)
        if not use_existing:  # Only save if we're not using an existing model
            save_model(model, feature_cols)
    
    # Generate forecasts for the test period
    print("Generating forecasts for test period...")
    forecast_df = generate_demand_forecast(train_df, model, feature_cols, days_to_forecast=len(test_dates))
    
    # Merge forecast with actual test data for comparison
    test_actuals = test_df[['Date', 'Store_Id', 'Item', 'Sales']]
    test_actuals = test_actuals.rename(columns={'Sales': 'Actual_Sales'})
    
    # Filter forecast to only include dates in the test period
    forecast_df = forecast_df[forecast_df['Date'].isin(test_dates)]
    
    # Merge forecasts with actuals
    comparison_df = pd.merge(
        forecast_df, 
        test_actuals,
        on=['Date', 'Store_Id', 'Item'],
        how='inner'
    )
    
    # Calculate accuracy metrics
    if len(comparison_df) > 0:
        mae = mean_absolute_error(comparison_df['Actual_Sales'], comparison_df['Predicted_Demand'])
        rmse = np.sqrt(mean_squared_error(comparison_df['Actual_Sales'], comparison_df['Predicted_Demand']))
        r2 = r2_score(comparison_df['Actual_Sales'], comparison_df['Predicted_Demand'])
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Handle division by zero by replacing zeros with a small value
        actual_nonzero = np.maximum(comparison_df['Actual_Sales'], 0.1)
        mape = np.mean(np.abs((comparison_df['Actual_Sales'] - comparison_df['Predicted_Demand']) / actual_nonzero)) * 100
        
        # Calculate additional metrics
        bias = np.mean(comparison_df['Predicted_Demand'] - comparison_df['Actual_Sales'])
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'Bias': bias,
            'Execution_Time': execution_time,
            'Records_Compared': len(comparison_df)
        }
        
        # Print metrics
        print(f"\n{'='*80}")
        print(f"MODEL PERFORMANCE METRICS")
        print(f"{'='*80}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"R² Score: {r2:.4f}")
        print(f"Bias: {bias:.2f}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Records Compared: {len(comparison_df)}")
        
        # Create a simple visualization of actual vs predicted
        if len(comparison_df) > 0:
            plt.figure(figsize=(12, 6))
            
            # Get top 5 store-product combinations by volume
            top_combos = comparison_df.groupby(['Store_Id', 'Item', 'Product'])['Actual_Sales'].sum().nlargest(5)
            
            for i, ((store, item, product), _) in enumerate(top_combos.items()):
                combo_data = comparison_df[(comparison_df['Store_Id'] == store) & (comparison_df['Item'] == item)]
                
                if len(combo_data) > 0:
                    plt.subplot(2, 3, i+1)
                    plt.plot(combo_data['Date'], combo_data['Actual_Sales'], 'b-', label='Actual')
                    plt.plot(combo_data['Date'], combo_data['Predicted_Demand'], 'r--', label='Predicted')
                    plt.title(f"{product} (Store {store})")
                    plt.xticks(rotation=45)
                    plt.legend()
                    plt.tight_layout()
            
            # Create directory if it doesn't exist
            os.makedirs('static/images', exist_ok=True)
            plt.savefig(f'static/images/rf_model_test_{"existing" if use_existing else "new"}.png')
            plt.close()
            
            # Also create a scatter plot of actual vs predicted
            plt.figure(figsize=(8, 8))
            plt.scatter(comparison_df['Actual_Sales'], comparison_df['Predicted_Demand'], alpha=0.5)
            plt.plot([0, comparison_df['Actual_Sales'].max()], [0, comparison_df['Actual_Sales'].max()], 'r--')
            plt.xlabel('Actual Sales')
            plt.ylabel('Predicted Demand')
            plt.title('Actual vs Predicted')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'static/images/rf_model_scatter_{"existing" if use_existing else "new"}.png')
            plt.close()
        
        return metrics
    else:
        print("ERROR: No matching records found between forecasts and actuals")
        return {
            'MAE': float('nan'),
            'RMSE': float('nan'),
            'MAPE': float('nan'),
            'R2': float('nan'),
            'Bias': float('nan'),
            'Execution_Time': time.time() - start_time,
            'Records_Compared': 0
        }

def compare_models():
    """Run tests for both modes (training and using existing) and compare results"""
    print("\nTESTING NEW MODEL TRAINING...")
    new_metrics = test_model_performance(use_existing=False)
    
    print("\nTESTING EXISTING MODEL...")
    existing_metrics = test_model_performance(use_existing=True)
    
    # Compare and output the results
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'Metric':<20} {'New Model':<15} {'Existing Model':<15} {'Difference':<15} {'% Change':<10}")
    print(f"{'-'*65}")
    
    for metric in ['MAE', 'RMSE', 'MAPE', 'R2', 'Bias', 'Execution_Time']:
        new_val = new_metrics[metric]
        existing_val = existing_metrics[metric]
        diff = existing_val - new_val
        pct_change = (diff / new_val * 100) if new_val != 0 else float('inf')
        
        # For R2, higher is better, for others lower is better
        if metric == 'R2':
            pct_change = -pct_change
            
        # Format the output with color indicators
        if metric != 'Bias':  # For bias, the sign matters more than the magnitude
            direction = "↓" if pct_change > 0 and metric != 'R2' or pct_change < 0 and metric == 'R2' else "↑"
        else:
            # For bias, closer to zero is better
            direction = "↓" if abs(existing_val) < abs(new_val) else "↑"
            
        print(f"{metric:<20} {new_val:<15.4f} {existing_val:<15.4f} {diff:<15.4f} {pct_change:<10.2f}% {direction}")
    
    # Print records compared
    print(f"\nRecords compared: New={new_metrics['Records_Compared']}, Existing={existing_metrics['Records_Compared']}")
    
    # Generate recommendation
    if existing_metrics['RMSE'] <= new_metrics['RMSE'] * 1.05:  # Within 5% of new model performance
        recommendation = "RECOMMENDATION: Using the existing model is suitable for faster execution with minimal performance impact."
    else:
        recommendation = "RECOMMENDATION: Training a new model is recommended for better accuracy despite longer execution time."
    
    print(f"\n{recommendation}")
    
    return new_metrics, existing_metrics
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Random Forest Model Performance')
    parser.add_argument('--use-existing', action='store_true',
                       help='Use existing saved model instead of retraining')
    parser.add_argument('--compare', action='store_true',
                       help='Run both modes and compare results')
    parser.add_argument('--test-days', type=int, default=14,
                       help='Number of days to use for testing (default: 14)')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models()
    else:
        test_model_performance(use_existing=args.use_existing, test_days=args.test_days)