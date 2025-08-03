"""
Wrapper module for PyTorch time series models to maintain compatibility with the original codebase.
"""
import os
import argparse
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.time_series import run_time_series_forecasting

def main():
    """
    Parse arguments and run time series forecasting
    """
    parser = argparse.ArgumentParser(description='Time series model training and forecasting')
    parser.add_argument('--days', type=int, default=30, help='Number of days to forecast')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with limited processing')
    args = parser.parse_args()
    
    # In debug mode, only run a subset of the data to test the fixes
    if args.debug:
        print("Running in debug mode with limited data processing")
        # Add a try-except block to catch and report any errors
        try:
            # Mock a simple forecast to test core functionality
            from src.models.time_series import LSTMModel
            import torch
            import numpy as np
            
            # Create a simple model instance
            model = LSTMModel(input_dim=17, hidden_dim=64, num_layers=2, output_dim=1)
            
            # Create some fake data
            X = np.random.rand(3, 28, 17).astype(np.float32)  # 3 sequences, 28 timesteps, 17 features
            X_tensor = torch.tensor(X)
            
            # Try to get predictions
            model.eval()
            with torch.no_grad():
                pred = model(X_tensor)
            
            print("Model can process inputs successfully!")
            print(f"Input shape: {X.shape}, Output shape: {pred.shape}")
            print("Prediction sample:", pred[0].item())
            return
        except Exception as e:
            print(f"Error in debug mode: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Regular mode - run the full forecasting process
    run_time_series_forecasting(days_to_forecast=args.days)

if __name__ == "__main__":
    main()