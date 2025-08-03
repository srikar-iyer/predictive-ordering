#!/usr/bin/env python3
"""
Test script to validate the Random Forest model improvements
- Compares NaN handling robustness
- Tests time series specific enhancements
- Evaluates prediction accuracy with different settings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import time
import sys
import argparse
import logging
from datetime import datetime, timedelta

# Add parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the rf_model module
from models.rf_model import (
    train_random_forest_model,
    create_forecasts,
    prepare_features,
    load_model,
    save_model
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_rf_improvements')

def create_test_dataset_with_nans(data_file, nan_rate=0.1):
    """
    Create a test dataset with artificially introduced NaN values
    
    Args:
        data_file (str): Path to the original data file
        nan_rate (float): Rate of NaN values to introduce (0-1)
        
    Returns:
        DataFrame: Data with NaNs
    """
    logger.info(f"Creating test dataset with {nan_rate:.1%} NaN values")
    
    # Load the data
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Make a copy to avoid modifying the original
    df_with_nans = df.copy()
    
    # Identify numeric columns that are features (not ID or date columns)
    exclude_cols = ['Date', 'Store_Id', 'Item', 'Product', 'Size', 'Weather']
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Randomly introduce NaN values
    for col in feature_cols:
        # Create random mask for NaN placement
        nan_mask = np.random.random(size=len(df)) < nan_rate
        df_with_nans.loc[nan_mask, col] = np.nan
        
        # Count NaNs introduced
        nan_count = nan_mask.sum()
        logger.info(f"Introduced {nan_count} NaN values in column {col}")
        
    return df_with_nans

def test_nan_handling(data_file, nan_rate=0.1, test_days=14):
    """
    Test the robustness of the NaN handling in the random forest model
    
    Args:
        data_file (str): Path to the data file
        nan_rate (float): Rate of NaN values to introduce
        test_days (int): Number of days to use for testing
    
    Returns:
        dict: Performance metrics with and without NaNs
    """
    logger.info("=" * 80)
    logger.info("TESTING NaN HANDLING ROBUSTNESS")
    logger.info("=" * 80)
    
    # Load the clean data
    df_clean = pd.read_csv(data_file)
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    
    # Create dataset with artificial NaNs
    df_with_nans = create_test_dataset_with_nans(data_file, nan_rate)
    
    # Split data for both datasets
    unique_dates = df_clean['Date'].unique()
    unique_dates.sort()
    
    if len(unique_dates) <= test_days:
        logger.warning("Not enough data for testing. Using half for testing.")
        split_idx = len(unique_dates) // 2
    else:
        split_idx = len(unique_dates) - test_days
        
    train_dates = unique_dates[:split_idx]
    test_dates = unique_dates[split_idx:]
    
    # Create train/test splits
    train_clean = df_clean[df_clean['Date'].isin(train_dates)].copy()
    test_df = df_clean[df_clean['Date'].isin(test_dates)].copy()
    
    train_with_nans = df_with_nans[df_with_nans['Date'].isin(train_dates)].copy()
    
    logger.info(f"Train data: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
    logger.info(f"Test data: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")
    
    # Train models on both datasets
    logger.info("Training model on clean data...")
    start_time_clean = time.time()
    model_clean, features_clean = train_random_forest_model(train_clean)
    training_time_clean = time.time() - start_time_clean
    
    logger.info("Training model on data with NaNs...")
    start_time_nans = time.time()
    model_nans, features_nans = train_random_forest_model(train_with_nans)
    training_time_nans = time.time() - start_time_nans
    
    # Generate forecasts
    logger.info("Generating forecasts with both models...")
    forecast_clean = create_forecasts(model_clean, features_clean, train_clean, days_to_forecast=len(test_dates))
    forecast_nans = create_forecasts(model_nans, features_nans, train_with_nans, days_to_forecast=len(test_dates))
    
    # Prepare test data for comparison
    test_actuals = test_df[['Date', 'Store_Id', 'Item', 'Sales']].rename(columns={'Sales': 'Actual_Sales'})
    
    # Filter forecasts to only include dates in the test period
    forecast_clean = forecast_clean[forecast_clean['Date'].isin(test_dates)]
    forecast_nans = forecast_nans[forecast_nans['Date'].isin(test_dates)]
    
    # Merge forecasts with actuals
    comparison_clean = pd.merge(
        forecast_clean,
        test_actuals,
        on=['Date', 'Store_Id', 'Item'],
        how='inner'
    )
    
    comparison_nans = pd.merge(
        forecast_nans,
        test_actuals,
        on=['Date', 'Store_Id', 'Item'],
        how='inner'
    )
    
    # Calculate metrics
    results = {}
    
    if len(comparison_clean) > 0:
        results['clean'] = {
            'MAE': mean_absolute_error(comparison_clean['Actual_Sales'], comparison_clean['Forecast']),
            'RMSE': np.sqrt(mean_squared_error(comparison_clean['Actual_Sales'], comparison_clean['Forecast'])),
            'R2': r2_score(comparison_clean['Actual_Sales'], comparison_clean['Forecast']),
            'Training_Time': training_time_clean,
            'Records': len(comparison_clean)
        }
    else:
        logger.error("No matching records found for clean model comparison")
    
    if len(comparison_nans) > 0:
        results['nans'] = {
            'MAE': mean_absolute_error(comparison_nans['Actual_Sales'], comparison_nans['Forecast']),
            'RMSE': np.sqrt(mean_squared_error(comparison_nans['Actual_Sales'], comparison_nans['Forecast'])),
            'R2': r2_score(comparison_nans['Actual_Sales'], comparison_nans['Forecast']),
            'Training_Time': training_time_nans,
            'Records': len(comparison_nans)
        }
    else:
        logger.error("No matching records found for NaN model comparison")
    
    # Print comparison results
    if 'clean' in results and 'nans' in results:
        logger.info("=" * 80)
        logger.info("NaN HANDLING PERFORMANCE COMPARISON")
        logger.info("=" * 80)
        logger.info(f"{'Metric':<15} {'Clean Data':<15} {'NaN Data':<15} {'Difference':<15} {'% Change':<10}")
        logger.info("-" * 70)
        
        for metric in ['MAE', 'RMSE', 'R2', 'Training_Time']:
            clean_val = results['clean'][metric]
            nans_val = results['nans'][metric]
            diff = nans_val - clean_val
            pct_change = (diff / clean_val * 100) if clean_val != 0 else float('inf')
            
            # For R2, higher is better, for others lower is better
            if metric == 'R2':
                pct_change = -pct_change
                
            # Format the output
            logger.info(f"{metric:<15} {clean_val:<15.4f} {nans_val:<15.4f} {diff:<15.4f} {pct_change:<10.2f}%")
        
        logger.info(f"Records compared: Clean={results['clean']['Records']}, NaNs={results['nans']['Records']}")
        
        # Evaluate the NaN handling effectiveness
        nan_effect = (results['nans']['RMSE'] - results['clean']['RMSE']) / results['clean']['RMSE'] * 100
        
        if nan_effect <= 5:
            logger.info("CONCLUSION: Excellent NaN handling! Performance degradation is less than 5%")
        elif nan_effect <= 10:
            logger.info("CONCLUSION: Good NaN handling. Performance degradation is between 5-10%")
        elif nan_effect <= 20:
            logger.info("CONCLUSION: Acceptable NaN handling. Performance degradation is between 10-20%")
        else:
            logger.info("CONCLUSION: NaN handling needs improvement. Performance degradation exceeds 20%")
    
    return results

def test_time_series_improvements(data_file, test_days=14):
    """
    Test the effectiveness of time series specific improvements
    
    Args:
        data_file (str): Path to the data file
        test_days (int): Number of days to use for testing
    
    Returns:
        dict: Performance metrics with and without time series enhancements
    """
    logger.info("=" * 80)
    logger.info("TESTING TIME SERIES IMPROVEMENTS")
    logger.info("=" * 80)
    
    # Load the data
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Split data
    unique_dates = df['Date'].unique()
    unique_dates.sort()
    
    if len(unique_dates) <= test_days:
        logger.warning("Not enough data for testing. Using half for testing.")
        split_idx = len(unique_dates) // 2
    else:
        split_idx = len(unique_dates) - test_days
        
    train_dates = unique_dates[:split_idx]
    test_dates = unique_dates[split_idx:]
    
    train_df = df[df['Date'].isin(train_dates)].copy()
    test_df = df[df['Date'].isin(test_dates)].copy()
    
    logger.info(f"Train data: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
    logger.info(f"Test data: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")
    
    # Train standard model
    logger.info("Training model without bootstrap confidence intervals...")
    model, features = train_random_forest_model(train_df)
    
    # Generate forecasts with and without bootstrap uncertainty
    logger.info("Generating forecasts with standard confidence intervals...")
    forecast_standard = create_forecasts(model, features, train_df, days_to_forecast=len(test_dates), 
                                         use_bootstrap=False)
    
    logger.info("Generating forecasts with bootstrap confidence intervals...")
    forecast_bootstrap = create_forecasts(model, features, train_df, days_to_forecast=len(test_dates), 
                                         use_bootstrap=True)
    
    # Prepare test data for comparison
    test_actuals = test_df[['Date', 'Store_Id', 'Item', 'Sales']].rename(columns={'Sales': 'Actual_Sales'})
    
    # Filter forecasts to only include dates in the test period
    forecast_standard = forecast_standard[forecast_standard['Date'].isin(test_dates)]
    forecast_bootstrap = forecast_bootstrap[forecast_bootstrap['Date'].isin(test_dates)]
    
    # Merge forecasts with actuals
    comparison_standard = pd.merge(
        forecast_standard,
        test_actuals,
        on=['Date', 'Store_Id', 'Item'],
        how='inner'
    )
    
    comparison_bootstrap = pd.merge(
        forecast_bootstrap,
        test_actuals,
        on=['Date', 'Store_Id', 'Item'],
        how='inner'
    )
    
    # Calculate confidence interval coverage
    def calculate_coverage(comparison_df):
        total = len(comparison_df)
        within_bounds = ((comparison_df['Actual_Sales'] >= comparison_df['Lower_Bound']) & 
                         (comparison_df['Actual_Sales'] <= comparison_df['Upper_Bound'])).sum()
        return (within_bounds / total) * 100 if total > 0 else 0
    
    standard_coverage = calculate_coverage(comparison_standard)
    bootstrap_coverage = calculate_coverage(comparison_bootstrap)
    
    # Calculate average CI width as percentage of forecast
    def calculate_avg_ci_width(comparison_df):
        width = (comparison_df['Upper_Bound'] - comparison_df['Lower_Bound'])
        return (width / comparison_df['Forecast']).mean() * 100
    
    standard_width = calculate_avg_ci_width(comparison_standard)
    bootstrap_width = calculate_avg_ci_width(comparison_bootstrap)
    
    # Calculate metrics
    results = {
        'standard': {
            'MAE': mean_absolute_error(comparison_standard['Actual_Sales'], comparison_standard['Forecast']),
            'RMSE': np.sqrt(mean_squared_error(comparison_standard['Actual_Sales'], comparison_standard['Forecast'])),
            'R2': r2_score(comparison_standard['Actual_Sales'], comparison_standard['Forecast']),
            'CI_Coverage': standard_coverage,
            'CI_Width': standard_width,
            'Records': len(comparison_standard)
        },
        'bootstrap': {
            'MAE': mean_absolute_error(comparison_bootstrap['Actual_Sales'], comparison_bootstrap['Forecast']),
            'RMSE': np.sqrt(mean_squared_error(comparison_bootstrap['Actual_Sales'], comparison_bootstrap['Forecast'])),
            'R2': r2_score(comparison_bootstrap['Actual_Sales'], comparison_bootstrap['Forecast']),
            'CI_Coverage': bootstrap_coverage,
            'CI_Width': bootstrap_width,
            'Records': len(comparison_bootstrap)
        }
    }
    
    # Print comparison results
    logger.info("=" * 80)
    logger.info("TIME SERIES IMPROVEMENTS COMPARISON")
    logger.info("=" * 80)
    logger.info(f"{'Metric':<15} {'Standard':<15} {'Bootstrap':<15} {'Difference':<15}")
    logger.info("-" * 60)
    
    for metric in ['MAE', 'RMSE', 'R2', 'CI_Coverage', 'CI_Width']:
        std_val = results['standard'][metric]
        boot_val = results['bootstrap'][metric]
        diff = boot_val - std_val
        
        logger.info(f"{metric:<15} {std_val:<15.4f} {boot_val:<15.4f} {diff:<15.4f}")
    
    logger.info(f"Records compared: Standard={results['standard']['Records']}, Bootstrap={results['bootstrap']['Records']}")
    
    # Create visualization of confidence intervals
    plt.figure(figsize=(12, 6))
    
    # Get a sample product-store combo
    store_items = comparison_standard[['Store_Id', 'Item']].drop_duplicates().head(1).values[0]
    store_id, item = store_items
    
    # Filter data for this combo
    std_sample = comparison_standard[(comparison_standard['Store_Id'] == store_id) & 
                                     (comparison_standard['Item'] == item)]
    boot_sample = comparison_bootstrap[(comparison_bootstrap['Store_Id'] == store_id) & 
                                      (comparison_bootstrap['Item'] == item)]
    
    if len(std_sample) > 0 and len(boot_sample) > 0:
        # Plot standard CI
        plt.subplot(1, 2, 1)
        plt.plot(std_sample['Date'], std_sample['Actual_Sales'], 'b-', label='Actual')
        plt.plot(std_sample['Date'], std_sample['Forecast'], 'r--', label='Forecast')
        plt.fill_between(std_sample['Date'], 
                         std_sample['Lower_Bound'], 
                         std_sample['Upper_Bound'], 
                         color='r', alpha=0.2)
        plt.title(f"Standard CI (Coverage: {standard_coverage:.1f}%)")
        plt.xticks(rotation=45)
        plt.legend()
        
        # Plot bootstrap CI
        plt.subplot(1, 2, 2)
        plt.plot(boot_sample['Date'], boot_sample['Actual_Sales'], 'b-', label='Actual')
        plt.plot(boot_sample['Date'], boot_sample['Forecast'], 'r--', label='Forecast')
        plt.fill_between(boot_sample['Date'], 
                         boot_sample['Lower_Bound'], 
                         boot_sample['Upper_Bound'], 
                         color='r', alpha=0.2)
        plt.title(f"Bootstrap CI (Coverage: {bootstrap_coverage:.1f}%)")
        plt.xticks(rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('static/images/confidence_interval_comparison.png')
        plt.close()
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test RF Model Improvements')
    parser.add_argument('--data-file', type=str, default='combined_pizza_data.csv',
                        help='Path to the data file')
    parser.add_argument('--test-days', type=int, default=14,
                        help='Number of days to use for testing')
    parser.add_argument('--nan-rate', type=float, default=0.1,
                        help='Rate of NaN values to introduce (0-1)')
    parser.add_argument('--test-nans', action='store_true',
                        help='Test NaN handling')
    parser.add_argument('--test-ts', action='store_true',
                        help='Test time series improvements')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs('static/images', exist_ok=True)
    
    if args.test_nans:
        nan_results = test_nan_handling(args.data_file, args.nan_rate, args.test_days)
    
    if args.test_ts:
        ts_results = test_time_series_improvements(args.data_file, args.test_days)
    
    # If no specific test was selected, run both
    if not (args.test_nans or args.test_ts):
        logger.info("Running all tests...")
        nan_results = test_nan_handling(args.data_file, args.nan_rate, args.test_days)
        ts_results = test_time_series_improvements(args.data_file, args.test_days)
    
    logger.info("Testing completed successfully")

if __name__ == "__main__":
    main()