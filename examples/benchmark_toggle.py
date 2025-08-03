#!/usr/bin/env python3
"""
Benchmark script to measure the performance difference between
training a new model and using an existing model
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rf_model_update import EnsembleModel, prepare_features, train_forecast_model, load_model, generate_demand_forecast

def benchmark_toggle():
    print("\nBENCHMARKING RF MODEL TOGGLE FUNCTIONALITY")
    print("=" * 60)
    
    # Load the dataset
    try:
        df = pd.read_csv('combined_pizza_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Clean dataset to avoid numeric issues
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Use a sample for faster benchmarking
        if len(df) > 5000:
            df = df.sample(5000, random_state=42)
        
        print(f"Loaded dataset with {len(df)} rows")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Check if a model exists
    if not os.path.exists('models/rf_model.pkl'):
        print("No existing model found. Training one first...")
        model, df_features, feature_cols, _, _ = train_forecast_model(df)
        from rf_model_update import save_model
        save_model(model, feature_cols)
        print("Model saved for benchmarking")
    
    # Measure performance with new training
    print("\nBenchmarking with new model training...")
    train_start = time.time()
    model, df_features, feature_cols, _, _ = train_forecast_model(df)
    train_time = time.time() - train_start
    
    print(f"New model training took {train_time:.2f} seconds")
    
    # Measure performance with existing model
    print("\nBenchmarking with existing model...")
    load_start = time.time()
    model, feature_cols = load_model()
    load_time = time.time() - load_start
    
    print(f"Loading existing model took {load_time:.2f} seconds")
    
    # Compare times
    speedup = train_time / load_time if load_time > 0 else float('inf')
    print(f"\nUsing existing model is {speedup:.1f}x faster than retraining")
    
    # Benchmark forecast generation
    print("\nBenchmarking forecast generation...")
    
    # With loaded model
    load_forecast_start = time.time()
    forecast_df = generate_demand_forecast(df, model, feature_cols, days_to_forecast=30)
    load_forecast_time = time.time() - load_forecast_start
    
    print(f"Generating forecasts with loaded model: {load_forecast_time:.2f} seconds")
    
    # Create visualization
    labels = ['Training New Model', 'Using Existing Model']
    times = [train_time, load_time]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, times, color=['red', 'green'])
    plt.title('Performance Comparison: New Model vs Existing Model')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y', alpha=0.3)
    
    # Add time values on top of bars
    for i, v in enumerate(times):
        plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')
    
    # Add speedup text
    plt.text(0.5, max(times) * 0.5, f"{speedup:.1f}x Faster", 
             ha='center', fontsize=14, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('static/images', exist_ok=True)
    plt.savefig('static/images/toggle_benchmark.png')
    print(f"Benchmark visualization saved to static/images/toggle_benchmark.png")
    
    return {
        'train_time': train_time,
        'load_time': load_time,
        'speedup': speedup,
        'forecast_time': load_forecast_time
    }

if __name__ == '__main__':
    benchmark_toggle()