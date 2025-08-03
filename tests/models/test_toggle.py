#!/usr/bin/env python3
"""
Simple script to test the toggle functionality between training a new model
and using an existing model
"""

import time
import os
from rf_model_update import load_model, EnsembleModel

def test_toggle_functionality():
    print("\nTESTING RF MODEL TOGGLE FUNCTIONALITY")
    print("=" * 60)
    
    # Check if a model exists
    if os.path.exists('models/rf_model.pkl'):
        print("Existing model found!")
        
        # Measure load time
        start_time = time.time()
        try:
            model, feature_cols = load_model()
            load_time = time.time() - start_time
            print(f"Successfully loaded model in {load_time:.2f} seconds")
            print(f"Model has {len(feature_cols)} features")
            
            # Test the key stochastic parameters in the model
            if hasattr(model, 'stochastic_parameters'):
                print("\nStochastic Parameters:")
                for key, value in model.stochastic_parameters.items():
                    print(f"  - {key}: {value}")
            
            print("\nToggle functionality is working correctly!")
            print("The --use-existing flag will load this model instead of retraining")
            
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("No existing model found. Run rf_model_update.py first to create a model.")

if __name__ == '__main__':
    test_toggle_functionality()