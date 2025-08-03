"""
Wrapper module for the Random Forest model to maintain compatibility with the original codebase.
"""
import os
import argparse
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.rf_model import run_rf_forecasting

def main():
    """
    Parse arguments and run RF model processing
    """
    parser = argparse.ArgumentParser(description='Random Forest model training and forecasting')
    parser.add_argument('--use-existing', action='store_true', help='Use existing model instead of retraining')
    args = parser.parse_args()
    
    run_rf_forecasting(use_existing=args.use_existing)

if __name__ == "__main__":
    main()