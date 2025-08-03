"""
Run script for item-based statistics with extended forecasting.
This script provides a command-line interface for running the item statistics analysis
and extended forecasting capabilities.
"""
import argparse
import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.item_statistics import (
    run_item_statistics_analysis, ItemStatisticsAnalyzer
)
from config.settings import (
    COMBINED_DATA_FILE, MODELS_DIR, STATIC_DIR,
    WEIGHTED_ARIMA_FORECASTS_FILE, ARIMA_FORECASTS_FILE,
    ITEM_STATISTICS_FILE, EXTENDED_FORECASTS_FILE
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"item_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger('run_item_statistics')

def main():
    """
    Main function to run the item statistics analysis from command line
    """
    parser = argparse.ArgumentParser(description='Item Statistics Analysis with Extended Forecasting')
    
    # Basic parameters
    parser.add_argument('--data', type=str, default=COMBINED_DATA_FILE, 
                        help='Path to combined data file')
    parser.add_argument('--forecast', type=str, 
                        help='Path to forecast file (uses weighted ARIMA or ARIMA forecasts if not specified)')
    parser.add_argument('--days-to-extend', type=int, default=60, 
                        help='Number of days to extend forecasts')
    
    # Output parameters
    parser.add_argument('--stats-output', type=str, 
                        help='Path for saving item statistics output (default: models/item_statistics.csv)')
    parser.add_argument('--forecast-output', type=str, 
                        help='Path for saving extended forecast output (default: models/extended_forecasts.csv)')
    parser.add_argument('--viz-dir', type=str, 
                        help='Directory for saving visualizations (default: static/images/)')
    
    # Control flags
    parser.add_argument('--no-viz', action='store_true', 
                        help='Skip creating visualizations')
    parser.add_argument('--no-plotly', action='store_true', 
                        help='Skip creating interactive Plotly visualizations')
    parser.add_argument('--stats-only', action='store_true', 
                        help='Only calculate statistics, skip forecast extension')
    parser.add_argument('--forecast-only', action='store_true', 
                        help='Only extend forecasts, skip detailed statistics')
    parser.add_argument('--top-items', type=int, default=5,
                        help='Number of top items to visualize individually')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting item statistics and extended forecasting process")
        
        # Initialize analyzer
        analyzer = ItemStatisticsAnalyzer(args.data)
        
        # Step 1: Load data
        logger.info(f"Loading data from {args.data}")
        analyzer.load_data()
        
        # Step 2: Load forecasts
        forecast_file = args.forecast
        if forecast_file is None:
            # Try weighted ARIMA forecasts first, then regular ARIMA
            if os.path.exists(WEIGHTED_ARIMA_FORECASTS_FILE):
                forecast_file = WEIGHTED_ARIMA_FORECASTS_FILE
            elif os.path.exists(ARIMA_FORECASTS_FILE):
                forecast_file = ARIMA_FORECASTS_FILE
        
        if not args.stats_only:
            logger.info(f"Loading forecasts from {forecast_file}")
            forecasts = analyzer.load_forecasts(forecast_file)
            
            if forecasts is None:
                logger.error("No forecast data available. Cannot continue with forecast extension.")
                if not args.forecast_only:
                    logger.info("Continuing with statistics calculation only.")
                else:
                    return 1
            
        # Step 3: Calculate item statistics
        if not args.forecast_only:
            logger.info("Calculating item statistics")
            item_stats = analyzer.calculate_item_statistics()
            
            # Save item statistics
            stats_output = args.stats_output
            if item_stats is not None:
                logger.info(f"Saving item statistics to {stats_output}")
                analyzer.save_item_statistics(stats_output)
                
                # Create item statistics visualizations
                if not args.no_viz:
                    # Standard matplotlib visualizations
                    viz_dir = args.viz_dir
                    if viz_dir:
                        stats_viz_dir = os.path.join(viz_dir, 'item_statistics')
                    else:
                        stats_viz_dir = os.path.join(STATIC_DIR, 'images', 'item_statistics')
                    
                    logger.info(f"Creating standard item statistics visualizations in {stats_viz_dir}")
                    analyzer.create_item_statistics_visualizations(stats_viz_dir)
                    
                    # Interactive Plotly visualizations
                    if not args.no_plotly:
                        if viz_dir:
                            plotly_dir = os.path.join(viz_dir, 'plotly_visualizations')
                        else:
                            plotly_dir = os.path.join(STATIC_DIR, 'plotly_visualizations')
                        
                        os.makedirs(plotly_dir, exist_ok=True)
                        logger.info(f"Creating interactive Plotly visualizations in {plotly_dir}")
                        analyzer.create_plotly_visualizations()
        
        # Step 4: Extend forecasts
        if not args.stats_only and forecasts is not None:
            logger.info(f"Extending forecasts by {args.days_to_extend} days")
            extended_forecasts = analyzer.extend_forecast(args.days_to_extend)
            
            # Save extended forecasts
            forecast_output = args.forecast_output
            if extended_forecasts is not None:
                logger.info(f"Saving extended forecasts to {forecast_output}")
                analyzer.save_extended_forecasts(forecast_output)
                
                # Create extended forecast visualizations
                if not args.no_viz:
                    # Standard matplotlib visualizations
                    viz_dir = args.viz_dir
                    if viz_dir:
                        forecast_viz_dir = os.path.join(viz_dir, 'extended_forecasts')
                    else:
                        forecast_viz_dir = os.path.join(STATIC_DIR, 'images', 'extended_forecasts')
                    
                    logger.info(f"Creating standard extended forecast visualizations in {forecast_viz_dir}")
                    analyzer.create_extended_forecast_visualizations(forecast_viz_dir)
                    
                    # Note: Plotly visualizations for forecasts are created in the create_plotly_visualizations method
                    # which was called earlier if not args.no_plotly
        
        logger.info("Item statistics and extended forecasting process completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error in item statistics process: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())