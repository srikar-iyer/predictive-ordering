#!/usr/bin/env python3
"""
Test script for the inventory visualization plots.
This script verifies that the modified plotting functions work correctly.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# Create mock data for testing
def create_mock_data():
    """Create a simple mock dataset for testing the plotting functions"""
    print("Creating mock data for testing...")
    
    # Date range for testing
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Create a DataFrame with required columns
    data = {
        'Store_Id': [104.0] * len(dates),
        'Item': [3913116850.0] * len(dates),
        'Date': dates,
        'Product': ['TEST PIZZA'] * len(dates),
        'Price': [6.99] * len(dates),
        'Sales': np.random.normal(loc=10, scale=3, size=len(dates)),
        'Stock_Level': np.random.normal(loc=50, scale=10, size=len(dates)),
        'Recent_Daily_Sales': np.random.normal(loc=10, scale=3, size=len(dates)),
        'Avg_Weekly_Sales_4W': np.random.normal(loc=70, scale=10, size=len(dates)),
        'Avg_Weekly_Sales_13W': np.random.normal(loc=65, scale=15, size=len(dates)),
        'Stock_Coverage_Weeks': np.random.normal(loc=2, scale=0.5, size=len(dates)),
        'Cost': [4.50] * len(dates),
    }
    
    df = pd.DataFrame(data)
    
    # Ensure all stock levels and sales are positive
    df['Stock_Level'] = df['Stock_Level'].apply(lambda x: max(0, x))
    df['Sales'] = df['Sales'].apply(lambda x: max(0, x))
    df['Recent_Daily_Sales'] = df['Recent_Daily_Sales'].apply(lambda x: max(0, x))
    df['Avg_Weekly_Sales_4W'] = df['Avg_Weekly_Sales_4W'].apply(lambda x: max(0, x))
    df['Avg_Weekly_Sales_13W'] = df['Avg_Weekly_Sales_13W'].apply(lambda x: max(0, x))
    df['Stock_Coverage_Weeks'] = df['Stock_Coverage_Weeks'].apply(lambda x: max(0, x))
    
    print(f"Created mock dataset with {len(df)} rows")
    return df

def test_inventory_chart():
    """Test the update_inventory_chart function"""
    from plotly_dashboard_inventory import update_inventory_chart
    
    # Create a simple mock class for the weather service
    class MockWeatherService:
        def get_weather_forecast(self, location):
            return [{'date': '2023-01-01', 'temperature': 72, 'weather_category': 'Sunny'}]
            
        def get_weather_adjusted_demand(self, base_demand, forecast, product_type=None):
            # Return the original demand with small adjustments
            adjustments = np.random.normal(loc=1.0, scale=0.1, size=len(base_demand))
            return base_demand * adjustments, adjustments
    
    # Create a simple mock class for the app
    class MockApp:
        def __init__(self):
            self.manual_stock_adjustments = {}
    
    # Create mock data
    mock_data = create_mock_data()
    
    # Create mock objects
    mock_weather_service = MockWeatherService()
    mock_app = MockApp()
    
    # Monkeypatch the imports
    import sys
    import types
    
    mod = types.ModuleType('plotly_dashboard')
    mod.combined_data = mock_data
    mod.weather_service = mock_weather_service
    mod.app = mock_app
    sys.modules['plotly_dashboard'] = mod
    
    print("Testing update_inventory_chart function...")
    fig = update_inventory_chart(1, None, 104.0, 3913116850.0, False, None, None)
    
    # Save the figure to HTML for inspection
    if not os.path.exists('test_output'):
        os.makedirs('test_output')
        
    fig.write_html('test_output/inventory_chart_test.html')
    print("Saved inventory chart test output to test_output/inventory_chart_test.html")
    
    # Test with weather adjustments
    print("Testing update_inventory_chart function with weather adjustments...")
    fig_weather = update_inventory_chart(1, None, 104.0, 3913116850.0, True, "10001", None)
    fig_weather.write_html('test_output/inventory_chart_weather_test.html')
    print("Saved inventory chart with weather test output to test_output/inventory_chart_weather_test.html")
    
    return True

def test_stock_velocity_chart():
    """Test the update_stock_velocity_chart function"""
    from plotly_dashboard_inventory import update_stock_velocity_chart
    import plotly.graph_objects as go
    
    # Create mock data
    mock_data = create_mock_data()
    
    print("Testing update_stock_velocity_chart function...")
    fig = update_stock_velocity_chart(1, 104.0, 3913116850.0, mock_data, go, np)
    
    # Save the figure to HTML for inspection
    if not os.path.exists('test_output'):
        os.makedirs('test_output')
        
    fig.write_html('test_output/stock_velocity_chart_test.html')
    print("Saved stock velocity chart test output to test_output/stock_velocity_chart_test.html")
    
    return True

def test_stock_penalty_chart():
    """Test the update_stock_penalty_chart function"""
    from plotly_dashboard_inventory import update_stock_penalty_chart
    import plotly.graph_objects as go
    
    # Create mock data
    mock_data = create_mock_data()
    
    print("Testing update_stock_penalty_chart function...")
    fig = update_stock_penalty_chart(1, 104.0, 3913116850.0, mock_data, go, np)
    
    # Save the figure to HTML for inspection
    if not os.path.exists('test_output'):
        os.makedirs('test_output')
        
    fig.write_html('test_output/stock_penalty_chart_test.html')
    print("Saved stock penalty chart test output to test_output/stock_penalty_chart_test.html")
    
    return True

def test_error_handling():
    """Test the error handling in the plotting functions"""
    from plotly_dashboard_inventory import update_inventory_chart
    
    # Create a simple mock class for the app
    class MockApp:
        def __init__(self):
            self.manual_stock_adjustments = {}
    
    # Monkeypatch the imports with None data to test error handling
    import sys
    import types
    
    mod = types.ModuleType('plotly_dashboard')
    mod.combined_data = None
    mod.weather_service = None
    mod.app = MockApp()
    sys.modules['plotly_dashboard'] = mod
    
    print("Testing error handling with missing data...")
    fig = update_inventory_chart(1, None, 104.0, 3913116850.0, False, None, None)
    
    # Save the figure to HTML for inspection
    if not os.path.exists('test_output'):
        os.makedirs('test_output')
        
    fig.write_html('test_output/error_handling_test.html')
    print("Saved error handling test output to test_output/error_handling_test.html")
    
    return True

def run_all_tests():
    """Run all test functions"""
    print("Starting inventory visualization tests...")
    
    tests = [
        test_inventory_chart,
        test_stock_velocity_chart,
        test_stock_penalty_chart,
        test_error_handling
    ]
    
    results = []
    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            result = test()
            results.append(result)
            print(f"{test.__name__}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"{test.__name__}: ERROR - {str(e)}")
            results.append(False)
    
    success_count = sum(1 for r in results if r)
    print(f"\nTest Results: {success_count}/{len(tests)} tests passed")
    
    return all(results)

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)