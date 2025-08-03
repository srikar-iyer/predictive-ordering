"""
Demonstration script for the unified data model.

This script shows how to use the UnifiedDataModel to connect inventory,
pricing, and demand forecasting components.
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the unified data model
from src.models.unified_data_model import UnifiedDataModel

# Import data loading functionality
from ui.core import load_dashboard_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('demonstrate_unified_model')

def demonstrate_unified_model():
    """
    Demonstrate how to use the unified data model to connect components.
    """
    print("Loading data...")
    data_dict = load_dashboard_data()
    
    # Check if required data is available
    required_data = ['combined_data', 'forecasts', 'price_elasticities', 'inventory_projection']
    missing_data = [key for key in required_data if key not in data_dict or data_dict[key] is None]
    
    if missing_data:
        print(f"Missing required data: {', '.join(missing_data)}")
        print("Please ensure all required data files are available.")
        return
    
    # Create the unified data model
    print("Creating unified data model...")
    model = UnifiedDataModel(data_dict)
    
    # Get a sample store and product
    print("Getting sample store and product...")
    combined_data = data_dict['combined_data']
    store_id = combined_data['Store_Id'].iloc[0]
    item_id = combined_data['Item'].iloc[0]
    
    print(f"Using Store ID: {store_id}, Item ID: {item_id}")
    
    # Get the initial metrics for the product
    print("\nInitial metrics:")
    initial_metrics = model.calculate_metrics(store_id, item_id)
    print_key_metrics(initial_metrics)
    
    # Demonstrate price adjustment
    print("\nAdjusting price by +5%...")
    model.adjust_price(store_id, item_id, 5)
    
    # Get updated metrics
    price_adj_metrics = model.calculate_metrics(store_id, item_id)
    print("\nMetrics after price adjustment:")
    print_key_metrics(price_adj_metrics)
    
    # Demonstrate inventory adjustment
    print("\nAdjusting inventory by +20%...")
    model.adjust_inventory(store_id, item_id, 20)
    
    # Get updated metrics
    inv_adj_metrics = model.calculate_metrics(store_id, item_id)
    print("\nMetrics after inventory adjustment:")
    print_key_metrics(inv_adj_metrics)
    
    # Demonstrate forecast adjustment
    print("\nAdjusting forecast by +10%...")
    model.adjust_forecast(store_id, item_id, 10)
    
    # Get updated metrics
    forecast_adj_metrics = model.calculate_metrics(store_id, item_id)
    print("\nMetrics after forecast adjustment:")
    print_key_metrics(forecast_adj_metrics)
    
    # Show what changes were applied
    print("\nApplied changes:")
    for change_type, changes in model.applied_changes.items():
        print(f"\n{change_type}:")
        for key, value in changes.items():
            print(f"  {key}: {value}")
    
    print("\nDemonstration complete!")

def print_key_metrics(metrics):
    """
    Print key metrics from the metrics dictionary in a readable format.
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics()
    """
    if not metrics:
        print("No metrics available")
        return
    
    # Print price metrics
    if 'price' in metrics:
        price_metrics = metrics['price']
        print(f"  Price: ${price_metrics.get('current_price', 0):.2f} -> ${price_metrics.get('new_price', 0):.2f}")
        print(f"  Elasticity: {price_metrics.get('elasticity', 0):.2f}")
        print(f"  Margin: {price_metrics.get('current_margin', 0)*100:.1f}% -> {price_metrics.get('new_margin', 0)*100:.1f}%")
    
    # Print inventory metrics
    if 'inventory' in metrics:
        inv_metrics = metrics['inventory']
        print(f"  Current Stock: {inv_metrics.get('current_stock', 0):.0f} units")
        print(f"  Coverage: {inv_metrics.get('coverage_weeks', 0):.1f} weeks")
        print(f"  Status: {inv_metrics.get('status', 'Unknown')}")
        print(f"  Stockout Risk: {inv_metrics.get('stockout_risk', 'Unknown')}")
    
    # Print forecast metrics
    if 'forecast' in metrics:
        forecast_metrics = metrics['forecast']
        print(f"  Avg Daily Forecast: {forecast_metrics.get('avg_daily_forecast', 0):.1f} units")
        print(f"  Total Forecast: {forecast_metrics.get('total_forecast', 0):.0f} units")
    
    # Print integrated metrics
    if 'integrated' in metrics:
        integrated_metrics = metrics['integrated']
        
        if 'price_change_impact' in integrated_metrics:
            price_impact = integrated_metrics['price_change_impact']
            print(f"  Profit Impact: {price_impact.get('profit_diff_pct', 0):+.1f}% (${price_impact.get('profit_diff', 0):+.2f})")
            print(f"  Demand Impact: {price_impact.get('forecast_diff_pct', 0):+.1f}%")
        
        if 'business_impact_score' in integrated_metrics:
            print(f"  Business Impact Score: {integrated_metrics['business_impact_score']:.0f}/100")
            print(f"  Recommendation: {integrated_metrics.get('recommendation', 'None')}")

if __name__ == "__main__":
    demonstrate_unified_model()