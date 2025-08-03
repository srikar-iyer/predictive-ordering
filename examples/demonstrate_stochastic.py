#!/usr/bin/env python3
"""
Script to demonstrate the stochastic behavior of the Random Forest model
This shows how the toggle option allows using existing models with stochastic behavior
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import os

# Import functions from rf_model_update
# Import the module directly so that we have access to all classes
import rf_model_update
from rf_model_update import (
    prepare_features,
    load_model,
    generate_demand_forecast,
    EnsembleModel  # Explicitly import EnsembleModel class
)

def demonstrate_stochastic_behavior(num_runs=5):
    """Generate multiple forecast runs to demonstrate stochastic behavior
    
    Args:
        num_runs (int): Number of forecast runs to generate
    """
    print("\nDEMONSTRATING STOCHASTIC BEHAVIOR OF RF MODEL")
    print("=" * 60)
    
    # Load the data
    df = pd.read_csv('combined_pizza_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Load the model
    try:
        model, feature_cols = load_model()
        print("Successfully loaded existing model")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run rf_model_update.py first to train and save a model")
        return
    
    # Prepare features for consistency
    df_features, _ = prepare_features(df)
    
    # Select a specific store and product for demonstration
    store_id = df['Store_Id'].iloc[0]  # Just use first store
    product_id = df['Item'].iloc[0]    # Just use first product
    
    # Filter to just that product
    product_name = df[(df['Store_Id'] == store_id) & (df['Item'] == product_id)]['Product'].iloc[0]
    print(f"Generating stochastic forecasts for {product_name} (Store {store_id}, Item {product_id})")
    
    # Generate multiple forecasts with stochastic behavior enabled
    forecast_dataframes = []
    for i in range(num_runs):
        print(f"Generating forecast run {i+1} of {num_runs}...")
        forecast_df = generate_demand_forecast(
            df,
            model,
            feature_cols,
            days_to_forecast=30,
            add_stochastic_variation=True
        )
        
        # Filter to our chosen product
        product_forecast = forecast_df[
            (forecast_df['Store_Id'] == store_id) & 
            (forecast_df['Item'] == product_id)
        ].copy()
        
        # Save for comparison
        product_forecast['Run'] = i + 1
        forecast_dataframes.append(product_forecast)
    
    # Also generate one deterministic forecast for comparison
    print("Generating deterministic forecast for comparison...")
    deterministic_df = generate_demand_forecast(
        df,
        model,
        feature_cols,
        days_to_forecast=30,
        add_stochastic_variation=False
    )
    
    # Filter to our chosen product
    deterministic_forecast = deterministic_df[
        (deterministic_df['Store_Id'] == store_id) & 
        (deterministic_df['Item'] == product_id)
    ].copy()
    deterministic_forecast['Run'] = 'Deterministic'
    
    # Combine all forecasts
    all_forecasts = pd.concat(forecast_dataframes + [deterministic_forecast])
    
    # Calculate statistics on the stochastic runs
    stochastic_only = all_forecasts[all_forecasts['Run'] != 'Deterministic']
    stats = stochastic_only.groupby('Date')['Predicted_Demand'].agg(['mean', 'std', 'min', 'max']).reset_index()
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot the stochastic runs
    for run in range(1, num_runs + 1):
        run_data = all_forecasts[all_forecasts['Run'] == run]
        plt.plot(run_data['Date'], run_data['Predicted_Demand'], 
                alpha=0.5, linewidth=1, label=f"Run {run}" if run == 1 else None)
    
    # Plot the deterministic run with a thicker line
    det_data = all_forecasts[all_forecasts['Run'] == 'Deterministic']
    plt.plot(det_data['Date'], det_data['Predicted_Demand'], 
            'k--', linewidth=2, label='Deterministic')
    
    # Plot the mean of stochastic runs
    plt.plot(stats['Date'], stats['mean'], 
            'b-', linewidth=2, label='Mean of Stochastic Runs')
    
    # Add shaded area for min-max range
    plt.fill_between(stats['Date'], stats['min'], stats['max'], 
                    color='blue', alpha=0.2, label='Min-Max Range')
    
    # Formatting
    plt.title(f"Stochastic vs Deterministic Forecasts\n{product_name} (Store {store_id})", fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Predicted Demand', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('static/images', exist_ok=True)
    plt.savefig('static/images/stochastic_demonstration.png')
    plt.close()
    
    print("\nStochastic behavior demonstration completed!")
    print(f"Visualization saved to: static/images/stochastic_demonstration.png")
    
    # Calculate coefficient of variation to measure stochasticity
    stats['cv'] = stats['std'] / stats['mean']
    avg_cv = stats['cv'].mean()
    
    print("\nSTOCHASTIC BEHAVIOR METRICS:")
    print(f"Average Coefficient of Variation: {avg_cv:.4f} ({avg_cv*100:.1f}%)")
    print(f"Min Coefficient of Variation: {stats['cv'].min():.4f} ({stats['cv'].min()*100:.1f}%)")
    print(f"Max Coefficient of Variation: {stats['cv'].max():.4f} ({stats['cv'].max()*100:.1f}%)")
    
    # Calculate how the stochasticity changes over time
    stats['days_ahead'] = range(1, len(stats) + 1)
    correlation = stats['cv'].corr(stats['days_ahead'])
    
    print(f"\nCorrelation between stochasticity and forecast horizon: {correlation:.4f}")
    if correlation > 0.5:
        print("Stochasticity increases with forecast horizon, as expected")
    elif correlation < -0.5:
        print("Stochasticity decreases with forecast horizon - this is unusual")
    else:
        print("Stochasticity is relatively stable across the forecast horizon")
    
    return all_forecasts, stats

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Demonstrate RF Model Stochastic Behavior')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of stochastic runs to generate (default: 5)')
    
    args = parser.parse_args()
    demonstrate_stochastic_behavior(num_runs=args.runs)