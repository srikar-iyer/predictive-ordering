#!/usr/bin/env python3
"""
Test script for the enhanced data loading mechanism
"""
import sys
import os
import pathlib
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.core import load_dashboard_data, clear_data_cache

def main():
    """Test the data loader"""
    print(f"Testing data loader at {datetime.now()}")
    
    # Clear cache and reload data
    clear_data_cache()
    data = load_dashboard_data(reload=True)
    
    # Print data information
    print("\nSuccessfully loaded data:")
    for k, v in data.items():
        if v is not None:
            print(f"{k}: {type(v).__name__} - {len(v)} rows")
        else:
            print(f"{k}: None")
    
    # Test fallback directory
    print("\nTesting with fallback directory")
    fallback_dir = pathlib.Path(__file__).parent / "data"
    
    clear_data_cache()
    data = load_dashboard_data(reload=True, fallback_root=fallback_dir)
    
    # Print data information
    print("\nData with fallback:")
    for k, v in data.items():
        if v is not None:
            print(f"{k}: {type(v).__name__} - {len(v)} rows")
        else:
            print(f"{k}: None")

if __name__ == "__main__":
    main()