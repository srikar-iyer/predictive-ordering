"""
Test script for item statistics and extended forecasting.
"""
import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.item_statistics import ItemStatisticsAnalyzer, run_item_statistics_analysis
from config.settings import COMBINED_DATA_FILE, MODELS_DIR, STATIC_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_item_statistics')

def test_item_statistics(use_plotly=True):
    """
    Test the item statistics module.
    
    Args:
        use_plotly: Whether to test Plotly visualizations
    """
    logger.info("Testing item statistics module...")
    
    try:
        # Create analyzer
        analyzer = ItemStatisticsAnalyzer()
        
        # Load data
        df = analyzer.load_data()
        if df is None or len(df) == 0:
            logger.error("Failed to load data")
            return False
            
        logger.info(f"Loaded {len(df)} records from {analyzer.data_path}")
        
        # Calculate item statistics for a subset of data to make testing faster
        store_items = df[['Store_Id', 'Item']].drop_duplicates().head(5)
        test_data = pd.DataFrame()
        
        for _, row in store_items.iterrows():
            store_id, item = row['Store_Id'], row['Item']
            item_data = df[(df['Store_Id'] == store_id) & (df['Item'] == item)]
            test_data = pd.concat([test_data, item_data])
        
        # Create temporary analyzer with test data
        test_analyzer = ItemStatisticsAnalyzer()
        test_analyzer.df = test_data
        
        # Calculate item statistics
        stats = test_analyzer.calculate_item_statistics()
        if stats is None or len(stats) == 0:
            logger.error("Failed to calculate item statistics")
            return False
            
        logger.info(f"Calculated statistics for {len(stats)} store-item combinations")
        
        # Create standard visualizations
        viz_dir = os.path.join(STATIC_DIR, 'test_images')
        os.makedirs(viz_dir, exist_ok=True)
        
        plots = test_analyzer.create_item_statistics_visualizations(viz_dir)
        logger.info(f"Created {len(plots)} visualization plots")
        
        # Test Plotly visualizations if requested
        if use_plotly:
            plotly_dir = os.path.join(STATIC_DIR, 'test_plotly_visualizations')
            os.makedirs(plotly_dir, exist_ok=True)
            
            test_analyzer.plotly_dir = plotly_dir
            logger.info("Testing Plotly visualizations...")
            plotly_plots = test_analyzer.create_plotly_visualizations()
            logger.info(f"Created {len(plotly_plots)} Plotly visualization plots")
        
        # Load forecasts if available
        forecasts = test_analyzer.load_forecasts()
        if forecasts is not None:
            # Extend forecasts
            extended = test_analyzer.extend_forecast(days_to_extend=30)
            if extended is not None:
                logger.info(f"Extended forecasts by 30 days")
                
                # Create extended forecast visualizations
                forecast_plots = test_analyzer.create_extended_forecast_visualizations(viz_dir)
                logger.info(f"Created {len(forecast_plots)} forecast visualization plots")
        
        logger.info("Item statistics test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error testing item statistics: {e}", exc_info=True)
        return False

def test_run_analysis(use_plotly=True):
    """
    Test the full analysis run.
    
    Args:
        use_plotly: Whether to test Plotly visualizations
    """
    logger.info("Testing full item statistics analysis run...")
    
    try:
        # Run analysis with a small number of days to extend for quicker testing
        stats, forecasts = run_item_statistics_analysis(
            days_to_extend=15, 
            use_plotly=use_plotly
        )
        
        if stats is None:
            logger.error("Failed to calculate item statistics")
            return False
            
        if forecasts is None:
            logger.warning("No forecasts generated (this may be expected if no forecast data was available)")
        
        # Check if Plotly visualizations were created
        if use_plotly:
            plotly_dir = os.path.join(STATIC_DIR, 'plotly_visualizations')
            if os.path.exists(plotly_dir):
                plotly_files = [f for f in os.listdir(plotly_dir) if f.endswith('.html')]
                logger.info(f"Found {len(plotly_files)} Plotly visualization files")
        
        logger.info("Full analysis run completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error running full analysis: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Item Statistics Module')
    parser.add_argument('--no-plotly', action='store_true', help='Skip Plotly visualization tests')
    parser.add_argument('--full', action='store_true', help='Run full analysis test')
    args = parser.parse_args()
    
    # Run tests
    test_item_statistics(use_plotly=not args.no_plotly)
    
    # Test the full analysis run if requested
    if args.full:
        test_run_analysis(use_plotly=not args.no_plotly)