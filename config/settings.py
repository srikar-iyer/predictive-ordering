"""
Configuration settings for the Pizza Predictive Ordering System.

This module defines global settings and constants used throughout the system.
"""
import os

# Base directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data file paths
SALES_FILE = os.path.join(ROOT_DIR, "data", "raw", "FrozenPizzaSales.csv")
PURCHASES_FILE = os.path.join(ROOT_DIR, "data", "raw", "FrozenPizzaPurchases.csv")
STOCK_FILE = os.path.join(ROOT_DIR, "data", "raw", "FrozenPizzaStock.csv")
COMBINED_DATA_FILE = os.path.join(ROOT_DIR, "data", "processed", "combined_pizza_data.csv")
PYTORCH_FORECASTS_FILE = os.path.join(ROOT_DIR, "data", "processed", "pytorch_forecasts.csv")
RF_FORECASTS_FILE = os.path.join(ROOT_DIR, "data", "processed", "rf_forecasts.csv")
ARIMA_FORECASTS_FILE = os.path.join(ROOT_DIR, "data", "processed", "arima_forecasts.csv")
WEIGHTED_ARIMA_FORECASTS_FILE = os.path.join(ROOT_DIR, "data", "processed", "weighted_arima_forecasts.csv")
PRICE_ELASTICITIES_FILE = os.path.join(ROOT_DIR, "data", "processed", "price_elasticities.csv")
PRICE_RECOMMENDATIONS_FILE = os.path.join(ROOT_DIR, "data", "processed", "price_recommendations.csv")
INVENTORY_PROJECTION_FILE = os.path.join(ROOT_DIR, "data", "processed", "inventory", "inventory_projection.csv")
INVENTORY_RECOMMENDATIONS_FILE = os.path.join(ROOT_DIR, "data", "processed", "inventory", "inventory_recommendations.csv")
PROFIT_IMPACT_FILE = os.path.join(ROOT_DIR, "data", "processed", "profit_impact.csv")
PRODUCT_MIX_OPTIMIZATION_FILE = os.path.join(ROOT_DIR, "data", "processed", "product_mix_optimization.csv")
ITEM_STATISTICS_FILE = os.path.join(ROOT_DIR, "data", "processed", "item_statistics.csv")
EXTENDED_FORECASTS_FILE = os.path.join(ROOT_DIR, "data", "processed", "extended_forecasts.csv")
CATEGORY_STATISTICS_FILE = os.path.join(ROOT_DIR, "data", "processed", "category_statistics.csv")

# Inventory management settings
MIN_STOCK_WEEKS = 1.0      # Minimum weeks of inventory (safety stock)
TARGET_STOCK_WEEKS = 2.0   # Target weeks of inventory
MAX_STOCK_WEEKS = 3.0      # Maximum weeks of inventory before considered excess

# Price optimization settings
MAX_PRICE_INCREASE = 15    # Maximum price increase percentage allowed
MAX_PRICE_DECREASE = 10    # Maximum price decrease percentage allowed
MIN_MARGIN = 20            # Minimum margin percentage required
ELASTICITY_CONSTRAINT = 5  # Maximum price change allowed for elastic products (percentage)

# Forecasting settings
DEFAULT_FORECAST_DAYS = 14     # Default number of days to forecast
CONFIDENCE_INTERVAL_WIDTH = 0.9  # Width of confidence interval (0.9 = 90%)

# UI settings
DEFAULT_THEME = "light"    # Default UI theme (light or dark)
SHOW_ITEM_NUMBERS = True   # Whether to show item numbers in product names

# Directory settings
STATIC_DIR = os.path.join(ROOT_DIR, "static")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# Integration settings
ENABLE_UNIFIED_MODEL = True  # Whether to use the unified data model for integration
ENABLE_WEATHER_SERVICE = True  # Whether to enable weather service integration

# Dashboard display settings
DEFAULT_TAB = "integrated"  # Default tab to show on dashboard load
ENABLE_KPI_CARDS = True     # Whether to show KPI cards on dashboard