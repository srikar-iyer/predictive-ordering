"""
Wrapper module for ARIMA time series models to maintain compatibility with the original codebase.
"""
import os
import argparse
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.arima_model import run_arima_forecasting

def main():
    """
    Parse arguments and run ARIMA forecasting
    """
    parser = argparse.ArgumentParser(description='ARIMA time series forecasting')
    parser.add_argument('--days', type=int, default=30, help='Number of days to forecast')
    parser.add_argument('--use-existing', action='store_true', help='Use existing models if available')
    args = parser.parse_args()
    
    run_arima_forecasting(
        days_to_forecast=args.days,
        use_existing=args.use_existing
    )

if __name__ == "__main__":
    main()