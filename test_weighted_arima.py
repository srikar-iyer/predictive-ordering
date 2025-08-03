#!/usr/bin/env python3
"""
Test script for Weighted Averaged ARIMA model implementation
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our weighted ARIMA modules
from src.models.weighted_arima_model import run_weighted_arima_forecasting
from src.models.integrated_forecasting_with_weighted_arima import IntegratedForecasterWithWeightedARIMA

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_weighted_arima')

def test_standalone_weighted_arima(data_path, days=30, use_existing=False):
    """
    Test the standalone weighted ARIMA model
    
    Args:
        data_path: Path to data file
        days: Number of days to forecast
        use_existing: Whether to use existing models
    """
    logger.info("Testing standalone weighted ARIMA model")
    
    # Run forecasting
    forecasts = run_weighted_arima_forecasting(
        data_file=data_path,
        days_to_forecast=days,
        use_existing=use_existing
    )
    
    # Check results
    if forecasts is not None and len(forecasts) > 0:
        logger.info(f"Generated {len(forecasts)} forecast rows")
        
        # Show summary statistics
        forecast_sum = forecasts.groupby('Date')['Forecast'].sum()
        logger.info(f"Total forecast: {forecast_sum.sum():.2f} units")
        logger.info(f"Average daily forecast: {forecast_sum.mean():.2f} units")
        
        # Check confidence intervals
        has_ci = 'Lower_Bound' in forecasts.columns and 'Upper_Bound' in forecasts.columns
        if has_ci:
            ci_width_pct = (forecasts['Upper_Bound'] - forecasts['Lower_Bound']) / forecasts['Forecast']
            logger.info(f"Average CI width: {ci_width_pct.mean():.2%} of forecast")
        
        logger.info("Standalone test completed successfully")
        return True
    else:
        logger.error("Failed to generate forecasts")
        return False

def test_integrated_forecasting(data_path, days=30, use_existing=False, create_visuals=True):
    """
    Test the integrated forecasting with weighted ARIMA model
    
    Args:
        data_path: Path to data file
        days: Number of days to forecast
        use_existing: Whether to use existing models
        create_visuals: Whether to create visualizations
    """
    logger.info("Testing integrated forecasting with weighted ARIMA")
    
    # Create integrated forecaster
    forecaster = IntegratedForecasterWithWeightedARIMA(data_path=data_path)
    
    # Run integrated optimization
    forecasts, price_recommendations, inventory_recommendations = forecaster.run_integrated_optimization(
        days_to_forecast=days,
        use_existing=use_existing,
        create_visuals=create_visuals
    )
    
    # Check results
    success = True
    if forecasts is not None and len(forecasts) > 0:
        logger.info(f"Generated {len(forecasts)} forecast rows")
    else:
        logger.error("Failed to generate forecasts")
        success = False
    
    if price_recommendations is not None and len(price_recommendations) > 0:
        logger.info(f"Generated {len(price_recommendations)} price recommendations")
    else:
        logger.error("Failed to generate price recommendations")
        success = False
    
    if inventory_recommendations is not None and len(inventory_recommendations) > 0:
        logger.info(f"Generated {len(inventory_recommendations)} inventory recommendations")
    else:
        logger.error("Failed to generate inventory recommendations")
        success = False
    
    if success:
        logger.info("Integrated forecasting test completed successfully")
    
    return success

def main():
    """
    Main function
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test weighted ARIMA model')
    parser.add_argument('--data', type=str, default='combined_pizza_data.csv', help='Path to data file')
    parser.add_argument('--days', type=int, default=30, help='Number of days to forecast')
    parser.add_argument('--use-existing', action='store_true', help='Use existing models')
    parser.add_argument('--no-visuals', dest='create_visuals', action='store_false', help='Disable visualizations')
    parser.add_argument('--integrated', action='store_true', help='Test integrated forecasting')
    parser.set_defaults(create_visuals=True)
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        return False
    
    # Run tests
    if args.integrated:
        success = test_integrated_forecasting(
            args.data, 
            days=args.days, 
            use_existing=args.use_existing, 
            create_visuals=args.create_visuals
        )
    else:
        success = test_standalone_weighted_arima(
            args.data, 
            days=args.days, 
            use_existing=args.use_existing
        )
    
    return success

if __name__ == "__main__":
    success = main()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)