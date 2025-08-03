#!/usr/bin/env python3
"""
Wrapper for legacy plotly_dashboard.py.
This script ensures backward compatibility by importing from the refactored modules when available.
"""
import os
import importlib.util
import sys

def main():
    """
    Main function that decides whether to use refactored modules or legacy code.
    """
    # Check if refactored dashboard module exists
    if os.path.exists(os.path.join("ui", "dashboard.py")):
        print("Using refactored dashboard module")
        try:
            # Add parent directory to path for imports
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            
            # Import the run_dashboard function from the refactored module
            from ui.dashboard import run_dashboard
            
            # Run the refactored dashboard
            return run_dashboard()
            
        except Exception as e:
            print(f"Error using refactored dashboard: {e}")
            print("Falling back to legacy dashboard...")
    
    # Fall back to legacy dashboard
    print("Using legacy dashboard module")
    import plotly_dashboard
    return plotly_dashboard.app.run(debug=True, host='0.0.0.0', port=8050)

if __name__ == "__main__":
    main()