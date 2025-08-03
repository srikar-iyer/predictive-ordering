# Pizza Predictive Ordering System

An advanced machine learning-based system for optimizing pizza inventory and pricing decisions. This specialized system leverages multiple models including Random Forest and PyTorch time series models to predict demand, optimize inventory levels, and maximize profits for pizza retailers.

## Overview

The Pizza Predictive Ordering System helps retailers optimize their frozen pizza inventory and pricing strategies by providing data-driven recommendations. The system uses a combination of machine learning approaches:

1. **Random Forest models** for comprehensive demand forecasting with multiple factors
2. **PyTorch LSTM/Transformer models** for accurate time series forecasting
3. **Inventory management logic** with 1-3 weeks of stock policy
4. **Price elasticity analysis** and profit maximization algorithms
5. **Interactive Plotly/Dash dashboard** for visualization and decision support

## New Directory Structure

The codebase has been refactored into a more modular and maintainable structure:

```
predictive_ordering/
├── config/                 # Configuration settings
│   ├── __init__.py
│   └── settings.py         # Centralized configuration 
├── data/                   # Input/output data files
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── data/               # Data processing
│   │   ├── __init__.py
│   │   └── data_loader.py  # Data integration
│   ├── models/             # Forecasting models
│   │   ├── __init__.py
│   │   ├── rf_model.py     # Random Forest model
│   │   └── time_series.py  # PyTorch time series models
│   ├── optimization/       # Optimization logic
│   │   ├── __init__.py
│   │   └── profit_optimizer.py # Price optimization
│   └── services/           # External services
│       ├── __init__.py
│       └── weather_service.py  # Weather data integration
├── ui/                     # User interface components
│   ├── __init__.py
│   └── core.py             # Core UI components
├── static/                 # Static files and images
└── tests/                  # Test modules
```

## Features

### Core Capabilities
- **Multi-Model Demand Forecasting**: Combine Random Forest and PyTorch time series models for robust predictions
- **Smart Inventory Management**: Apply 1-3 weeks stock logic with automatic order recommendations
- **Price Optimization**: Calculate price elasticity and recommend profit-maximizing prices
- **Product Mix Analysis**: Optimize across multiple products considering cross-effects
- **Interactive Visualization**: Comprehensive Plotly-based dashboard
- **Complete Data Pipeline**: Integrated workflow from raw data to recommendations
- **Model Toggle Option**: Use existing models without retraining for faster execution

### Key Components
1. **Data Integration**: Combines sales, purchase and stock data into a unified dataset
2. **Time Series Modeling**: LSTM and Transformer neural networks for sequential prediction
3. **Ensemble Learning**: Random Forest models for feature-rich prediction
4. **Inventory Logic**: Smart stock level management with safety stock calculation
5. **Price Elasticity**: Log-log regression models for elasticity calculation
6. **Profit Optimization**: Mathematical optimization for price setting
7. **Interactive Dashboard**: Web-based visualization and exploration

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Docker (for containerized deployment)

### Dependencies
```
pandas
numpy
matplotlib
seaborn
scikit-learn
torch
plotly
dash
dash-bootstrap-components
scipy
gradio
openmeteo-requests
requests-cache
retry-requests
holidays
```

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/pizza-predictive-ordering.git
   cd pizza-predictive-ordering
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place data files in the project directory:
   - `FrozenPizzaSales.csv`
   - `FrozenPizzaPurchases.csv`
   - `FrozenPizzaStock.csv`

## Usage

### Running the Full Pipeline

```bash
python main.py
```

This runs the complete pipeline:
1. Data integration
2. Random Forest model training
3. PyTorch time series model training
4. Inventory management
5. Profit optimization
6. Dashboard launch

For faster execution, you have two options:

```bash
# Skip the time-intensive PyTorch model training
python main.py --skip-time-series

# Use existing Random Forest model
python main.py --use-existing-rf

# Combine both options for maximum speed
python main.py --skip-time-series --use-existing-rf
```

### Using Individual Components

Each component can also be used independently:

```python
# Data processing
from src.data.data_loader import load_pizza_datasets
combined_data = load_pizza_datasets()

# Random Forest forecasting
from src.models.rf_model import run_rf_forecasting
rf_forecasts = run_rf_forecasting(data_file='combined_pizza_data.csv')

# PyTorch time series forecasting
from src.models.time_series import run_time_series_forecasting
ts_forecasts = run_time_series_forecasting(data_file='combined_pizza_data.csv')

# Profit optimization
from src.optimization.profit_optimizer import run_profit_optimization
optimizer = run_profit_optimization()

# Weather service
from src.services.weather_service import get_weather_service
weather = get_weather_service().get_current_weather("10001")
```

### Accessing the Dashboard

Once the pipeline runs, access the dashboard at:
```
http://localhost:8050
```

## Docker Deployment

The application can be run in Docker containers for easy deployment without worrying about dependencies.

### Local Deployment with Docker Compose (Development)

1. Build and start the application locally:
   ```bash
   docker-compose up
   ```

   This will:
   - Build the Docker image from the Dockerfile
   - Start the application container
   - Make the dashboard available at http://localhost:8050
   - Create a persistent volume for output data

2. To stop the application:
   ```bash
   docker-compose down
   ```

## Customization

### Configuration Settings

Edit settings in `config/settings.py` to customize behavior:

```python
# Adjust inventory policy
MIN_STOCK_WEEKS = 1
TARGET_STOCK_WEEKS = 2
MAX_STOCK_WEEKS = 3

# Adjust price optimization constraints
MAX_PRICE_INCREASE = 20  # Maximum 20% price increase
MAX_PRICE_DECREASE = 15  # Maximum 15% price decrease
MIN_MARGIN = 25          # Minimum 25% margin
```

## Backward Compatibility

This refactored version maintains compatibility with the original file structure. All component scripts can be run directly from the root directory as before.