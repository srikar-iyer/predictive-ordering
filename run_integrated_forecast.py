#!/usr/bin/env python3
"""
Command-line interface for running the integrated forecasting system.
This script connects ARIMA forecasting with inventory and pricing optimization.
"""
import os
import sys
import argparse
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from src.models.integrated_forecasting import IntegratedForecaster
from config.settings import (
    COMBINED_DATA_FILE, ARIMA_FORECASTS_FILE, INVENTORY_RECOMMENDATIONS_FILE,
    PRICE_RECOMMENDATIONS_FILE
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('run_integrated_forecast')

def main():
    """
    Main function to run the integrated forecasting system
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Integrated forecasting and optimization')
    parser.add_argument('--days', type=int, default=30, help='Number of days to forecast')
    parser.add_argument('--use-existing', action='store_true', help='Use existing models if available')
    parser.add_argument('--data', type=str, default=COMBINED_DATA_FILE, help='Path to input data')
    parser.add_argument('--forecast-only', action='store_true', help='Run only the forecasting component')
    parser.add_argument('--optimization-only', action='store_true', help='Run only price optimization')
    parser.add_argument('--inventory-only', action='store_true', help='Run only inventory optimization')
    parser.add_argument('--no-visuals', dest='create_visuals', action='store_false', help='Disable visualization creation')
    parser.set_defaults(create_visuals=True)
    
    args = parser.parse_args()
    
    try:
        # Create forecaster
        forecaster = IntegratedForecaster(data_path=args.data)
        
        # Load data
        df = forecaster.load_data()
        
        if args.forecast_only:
            # Run only forecasting
            logger.info("Running ARIMA forecasting only")
            forecasts = forecaster.generate_forecasts(days_to_forecast=args.days, use_existing=args.use_existing)
            logger.info(f"Generated forecasts for {len(forecasts)} store-item combinations over {args.days} days")
            logger.info(f"Forecasts saved to {ARIMA_FORECASTS_FILE}")
            
        elif args.optimization_only:
            # Run price optimization only
            logger.info("Running price optimization only")
            forecaster.generate_forecasts(days_to_forecast=args.days, use_existing=args.use_existing)
            elasticities, recommendations, impact = forecaster.optimize_prices()
            logger.info(f"Generated {len(recommendations)} price recommendations")
            logger.info(f"Price recommendations saved to {PRICE_RECOMMENDATIONS_FILE}")
            
        elif args.inventory_only:
            # Run inventory optimization only
            logger.info("Running inventory optimization only")
            forecaster.generate_forecasts(days_to_forecast=args.days, use_existing=args.use_existing)
            inventory_recommendations = forecaster.optimize_inventory()
            logger.info(f"Generated {len(inventory_recommendations)} inventory recommendations")
            logger.info(f"Inventory recommendations saved to {INVENTORY_RECOMMENDATIONS_FILE}")
            
        else:
            # Run full integrated system
            logger.info("Running full integrated forecasting and optimization")
            forecasts, price_recommendations, inventory_recommendations = forecaster.run_integrated_optimization(
                days_to_forecast=args.days,
                use_existing=args.use_existing,
                create_visuals=args.create_visuals
            )
            
            logger.info(f"Generated forecasts for {len(forecasts)} store-item combinations over {args.days} days")
            logger.info(f"Generated {len(price_recommendations)} price recommendations")
            logger.info(f"Generated {len(inventory_recommendations)} inventory recommendations")
            logger.info("Integrated system execution completed successfully")
        
    except Exception as e:
        logger.error(f"Error running integrated system: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()