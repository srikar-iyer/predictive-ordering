"""
Wrapper module for Weighted Averaged ARIMA models to maintain compatibility with the original codebase.
"""
import os
import argparse
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.weighted_arima_model import run_weighted_arima_forecasting

def main():
    """
    Parse arguments and run Weighted ARIMA forecasting
    """
    parser = argparse.ArgumentParser(description='Weighted ARIMA time series forecasting')
    parser.add_argument('--days', type=int, default=30, help='Number of days to forecast')
    parser.add_argument('--use-existing', action='store_true', help='Use existing models if available')
    parser.add_argument('--no-parallel', dest='parallel', action='store_false', help='Disable parallel processing')
    args = parser.parse_args()
    
    run_weighted_arima_forecasting(
        days_to_forecast=args.days,
        use_existing=args.use_existing,
        parallel=args.parallel
    )

if __name__ == "__main__":
    main()