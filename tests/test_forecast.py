import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path so we can import the main module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the main code (assuming it's in the parent directory)
try:
    import retail_forecast as rf
except ImportError:
    # Try to import from current directory
    try:
        sys.path.append(os.path.abspath(os.path.dirname(__file__)))
        import retail_forecast as rf
    except ImportError:
        print("ERROR: Could not import retail_forecast module")
        sys.exit(1)

class TestRetailForecast(unittest.TestCase):
    """Test cases for the retail forecasting system"""
    
    def setUp(self):
        """Setup test data"""
        np.random.seed(42)  # For reproducibility
        # Try to load CSV data first
        try:
            self.test_data = rf.load_data_from_csv()
            
            # Limit to 30 days for faster testing
            end_date = self.test_data['Date'].max()
            start_date = end_date - timedelta(days=30)
            self.test_data = self.test_data[(self.test_data['Date'] >= start_date) & 
                                            (self.test_data['Date'] <= end_date)]
        except Exception as e:
            print(f"Warning: Failed to load CSV data, falling back to sample data: {e}")
            # Fall back to the sample data if CSV loading fails
            self.test_data = rf.generate_sample_data()
            # Limit to 30 days for faster testing
            end_date = self.test_data['Date'].max()
            start_date = end_date - timedelta(days=30)
            self.test_data = self.test_data[(self.test_data['Date'] >= start_date) & 
                                            (self.test_data['Date'] <= end_date)]
    
    def test_data_loading(self):
        """Test that data is loaded correctly"""
        # Check data has the expected columns
        expected_columns = [
            'Date', 'Product', 'Sales', 'Potential_Sales', 'Lost_Sales',
            'Stock_Level', 'Price', 'Promotion', 'Weather', 'Lead_Time',
            'Day_Of_Week', 'Month', 'Year', 'Is_Holiday', 'Holiday_Name'
        ]
        
        for col in expected_columns:
            self.assertIn(col, self.test_data.columns, f"Missing column: {col}")
        
        # Check that we have at least one product
        self.assertGreater(len(self.test_data['Product'].unique()), 0, "No products found in the data")
            
        # Sales should be non-negative
        self.assertTrue((self.test_data['Sales'] >= 0).all(), "Found negative sales values")
        
        # Stock levels should be non-negative
        self.assertTrue((self.test_data['Stock_Level'] >= 0).all(), "Found negative stock levels")
    
    def test_feature_preparation(self):
        """Test that feature engineering works correctly"""
        # Prepare features
        df_features, feature_cols = rf.prepare_features(self.test_data)
        
        # Check that one-hot encoding worked for products
        for product in self.test_data['Product'].unique():
            product_col = f'Product_{product}'
            self.assertIn(product_col, feature_cols, f"Missing one-hot encoded column: {product_col}")
        
        # Check that lag features were created
        for lag in [1, 7, 14, 30]:
            lag_col = f'Sales_Lag_{lag}'
            self.assertIn(lag_col, feature_cols, f"Missing lag feature: {lag_col}")
            
        # Check that rolling averages were created
        for window in [7, 14, 30]:
            avg_col = f'Sales_Avg_{window}'
            self.assertIn(avg_col, feature_cols, f"Missing rolling avg feature: {avg_col}")
            
        # Check that cyclical encoding was done
        self.assertIn('Day_Sin', feature_cols, "Missing cyclic feature: Day_Sin")
        self.assertIn('Day_Cos', feature_cols, "Missing cyclic feature: Day_Cos")
        self.assertIn('Month_Sin', feature_cols, "Missing cyclic feature: Month_Sin")
        self.assertIn('Month_Cos', feature_cols, "Missing cyclic feature: Month_Cos")
    
    def test_model_training(self):
        """Test that the model trains successfully"""
        # Train the model (using a small subset for speed)
        model, df_features, feature_cols, feature_importance, test_data = rf.train_forecast_model(self.test_data)
        
        # Check the model is a RandomForest
        self.assertEqual(model.__class__.__name__, "RandomForestRegressor")
        
        # Check feature importance is calculated
        self.assertEqual(len(feature_importance), len(feature_cols))
        self.assertGreater(feature_importance['Importance'].sum(), 0)
        
        # Make a test prediction
        X_test, y_test, y_pred = test_data
        self.assertEqual(len(y_pred), len(y_test))
    
    def test_scenario_generation(self):
        """Test that test scenarios can be generated"""
        # Train the model (using a small subset for speed)
        model, df_features, feature_cols, feature_importance, test_data = rf.train_forecast_model(self.test_data)
        
        # Generate test scenarios
        test_results = rf.run_tests(model, self.test_data, feature_cols)
        
        # Check that we have scenarios for each product
        for product in self.test_data['Product'].unique():
            product_scenarios = test_results[test_results['Product'] == product]
            self.assertGreater(len(product_scenarios), 0, f"No scenarios for {product}")
            
        # Check that we have the expected scenarios
        expected_scenarios = ['Baseline', 'Price Reduction', 'Holiday', 'Bad Weather', 'Holiday + Promotion']
        for scenario in expected_scenarios:
            self.assertIn(scenario, test_results['Scenario'].unique(), f"Missing scenario: {scenario}")
            
        # Validate recommended order logic
        # Check that recommended orders are always greater than predicted sales (safety stock)
        self.assertTrue(all(test_results['Recommended_Order'] > test_results['Predicted_Sales']))

    def test_demand_forecast(self):
        """Test that demand forecast works correctly"""
        # Train a model on a small subset for speed
        model, df_features, feature_cols, feature_importance, test_data = rf.train_forecast_model(self.test_data)
        
        # Generate a forecast for 10 days
        forecast_df = rf.generate_demand_forecast(self.test_data, model, feature_cols, days_to_forecast=10)
        
        # Check that the forecast has the expected structure
        expected_columns = ['Date', 'Product', 'Predicted_Demand']
        for col in expected_columns:
            self.assertIn(col, forecast_df.columns, f"Missing column in forecast: {col}")
            
        # Check that all products in the forecast are in the original data
        original_products = set(self.test_data['Product'].unique())
        forecast_products = set(forecast_df['Product'].unique())
        self.assertTrue(forecast_products.issubset(original_products), 
                        "Forecast contains products not in the original data")
        
        # Check that the forecast covers the expected date range
        latest_original_date = self.test_data['Date'].max()
        forecast_start_date = forecast_df['Date'].min()
        forecast_end_date = forecast_df['Date'].max()
        
        expected_start_date = latest_original_date + timedelta(days=1)
        expected_end_date = latest_original_date + timedelta(days=10)
        
        # Allow a small tolerance for date comparison due to timestamp differences
        self.assertLessEqual(abs((forecast_start_date - expected_start_date).days), 1, 
                           "Forecast starts too far from expected start date")
        self.assertLessEqual(abs((expected_end_date - forecast_end_date).days), 1, 
                           "Forecast ends too far from expected end date")
        
        # Check that all predicted demand values are non-negative
        self.assertTrue((forecast_df['Predicted_Demand'] >= 0).all(), 
                        "Found negative predicted demand values")

if __name__ == '__main__':
    unittest.main()
