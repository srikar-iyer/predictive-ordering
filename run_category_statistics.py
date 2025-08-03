"""
Run script for category-based statistics and aggregated analysis.
This script provides a command-line interface for running the category statistics analysis.
"""
import argparse
import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.category_statistics import (
    run_category_statistics_analysis, CategoryStatisticsAnalyzer
)
from config.settings import (
    COMBINED_DATA_FILE, MODELS_DIR, STATIC_DIR,
    ITEM_STATISTICS_FILE, CATEGORY_STATISTICS_FILE
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"category_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger('run_category_statistics')

def main():
    """
    Main function to run the category statistics analysis from command line
    """
    parser = argparse.ArgumentParser(description='Category Statistics Analysis')
    
    # Basic parameters
    parser.add_argument('--data', type=str, default=COMBINED_DATA_FILE, 
                        help='Path to combined data file')
    parser.add_argument('--item-stats', type=str, default=ITEM_STATISTICS_FILE, 
                        help='Path to item statistics file')
    
    # Output parameters
    parser.add_argument('--stats-output', type=str, default=CATEGORY_STATISTICS_FILE,
                        help='Path for saving category statistics output')
    parser.add_argument('--viz-dir', type=str, 
                        help='Directory for saving visualizations (default: static/images/)')
    
    # Control flags
    parser.add_argument('--no-viz', action='store_true', 
                        help='Skip creating visualizations')
    parser.add_argument('--no-plotly', action='store_true', 
                        help='Skip creating interactive Plotly visualizations')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting category statistics analysis process")
        
        # Initialize analyzer
        analyzer = CategoryStatisticsAnalyzer(args.data, args.item_stats)
        
        # Step 1: Load data
        logger.info(f"Loading data from {args.data}")
        analyzer.load_data()
        
        # Step 2: Load item statistics
        logger.info(f"Loading item statistics from {args.item_stats}")
        analyzer.load_item_statistics()
        
        # Step 3: Calculate category statistics
        logger.info("Calculating category statistics")
        category_stats = analyzer.calculate_category_statistics()
        
        # Step 4: Save category statistics
        if category_stats is not None:
            logger.info(f"Saving category statistics to {args.stats_output}")
            analyzer.save_category_statistics(args.stats_output)
            
            # Create category statistics visualizations
            if not args.no_viz:
                # Standard matplotlib visualizations
                viz_dir = args.viz_dir
                if viz_dir:
                    stats_viz_dir = os.path.join(viz_dir, 'category_statistics')
                else:
                    stats_viz_dir = os.path.join(STATIC_DIR, 'images', 'category_statistics')
                
                logger.info(f"Creating standard category statistics visualizations in {stats_viz_dir}")
                analyzer.create_category_statistics_visualizations(stats_viz_dir)
                
                # Create category forecast aggregation
                logger.info("Creating category forecast aggregation")
                analyzer.create_category_forecast_aggregation()
                
                # Save category forecasts
                logger.info("Saving category forecasts")
                analyzer.save_category_forecasts()
                
                # Interactive Plotly visualizations
                if not args.no_plotly:
                    if viz_dir:
                        plotly_dir = os.path.join(viz_dir, 'plotly_visualizations')
                    else:
                        plotly_dir = os.path.join(STATIC_DIR, 'plotly_visualizations')
                    
                    os.makedirs(plotly_dir, exist_ok=True)
                    logger.info(f"Creating interactive Plotly visualizations in {plotly_dir}")
                    analyzer.create_plotly_visualizations()
        
        logger.info("Category statistics analysis process completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error in category statistics process: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())