# Weighted Averaged ARIMA Model

This document provides an overview of the Weighted Averaged ARIMA model implementation that replaces the Random Forest models in the predictive ordering system.

## Overview

The Weighted Averaged ARIMA (AutoRegressive Integrated Moving Average) model provides improved time series forecasting by combining multiple ARIMA models trained on different time windows. Each time window model captures different temporal patterns, and the weighted ensemble approach provides more robust and accurate forecasts.

## Key Features

1. **Weighted Time Window Approach**
   - Daily window (7 days): Captures most recent trends and rapid changes
   - Weekly window (28 days): Captures weekly patterns and medium-term trends
   - Monthly window (90 days): Captures monthly seasonality and medium-term patterns
   - Quarterly window (180 days): Captures quarterly business cycles
   - Yearly window (365 days): Captures long-term trends and yearly seasonality

2. **Automatic Seasonality Detection**
   - Detects and models multiple seasonality patterns
   - Adaptively selects appropriate ARIMA parameters
   - Combines seasonal and non-seasonal models as needed

3. **Confidence Interval Calculation**
   - Provides robust uncertainty quantification
   - Accounts for model disagreement in ensemble
   - Delivers more accurate stock level projections

4. **Integration with Inventory Optimization**
   - Weighted confidence measures for inventory decision-making
   - Extended forecasting capabilities
   - Improved handling of seasonal patterns

## Technical Implementation

### Model Structure

The weighted ARIMA implementation uses a combination of models:

```
TIME_WINDOWS = {
    'daily': {'days': 7, 'weight': 0.30},      # Recent daily patterns (1 week)
    'weekly': {'days': 28, 'weight': 0.25},    # Recent weekly patterns (4 weeks)
    'monthly': {'days': 90, 'weight': 0.20},   # Monthly patterns (3 months)
    'quarterly': {'days': 180, 'weight': 0.15}, # Quarterly patterns (6 months)
    'yearly': {'days': 365, 'weight': 0.10}    # Yearly patterns
}
```

### Forecast Combination

Forecasts from individual models are combined using a weighted average:

```
weighted_forecast = Σ (weight_i * forecast_i)
```

Where:
- weight_i is the importance weight of model i
- forecast_i is the forecast from model i

### Confidence Interval Calculation

Confidence intervals incorporate uncertainty from:
1. Individual model uncertainties (weighted variance)
2. Model disagreement (ensemble variance)
3. Forecast horizon (increasing uncertainty over time)

## Usage

### Basic Usage

```python
from src.models.weighted_arima_model import run_weighted_arima_forecasting

# Run forecasting
forecasts = run_weighted_arima_forecasting(
    data_file='combined_pizza_data.csv',
    days_to_forecast=30,
    use_existing=False,  # Train new models
    parallel=True        # Use parallel processing
)
```

### Integrated Forecasting

```python
from src.models.integrated_forecasting_with_weighted_arima import IntegratedForecasterWithWeightedARIMA

# Create integrated forecaster
forecaster = IntegratedForecasterWithWeightedARIMA(data_path='combined_pizza_data.csv')

# Run complete optimization process
forecasts, price_recs, inventory_recs = forecaster.run_integrated_optimization(
    days_to_forecast=30,
    use_existing=True,    # Use existing models if available
    create_visuals=True   # Generate visualization charts
)
```

### Command Line Usage

```bash
# Run standalone weighted ARIMA
python -m src.models.weighted_arima_model_wrapper --days 30 --use-existing

# Run integrated forecasting with weighted ARIMA
python -m src.models.integrated_forecasting_with_weighted_arima --days 30 --use-existing
```

## Benefits over Random Forest Models

1. **Time Series Focus**
   - Explicitly models time dependence and autocorrelation
   - Better handling of seasonal patterns
   - More accurate for extended forecast horizons

2. **Uncertainty Quantification**
   - Provides principled confidence intervals
   - Quantifies forecast uncertainty over time
   - Enables risk-aware inventory decisions

3. **Ensemble Robustness**
   - Combines strengths of multiple time windows
   - Less sensitive to anomalies in specific periods
   - Balances recency bias with historical patterns

4. **Adaptive Training**
   - Automatically detects and models seasonality
   - Applies appropriate differencing for stationarity
   - Uses optimal ARIMA parameters for each window

## Visualizations

The implementation provides enhanced visualization capabilities:

1. **Forecast plots with confidence intervals**
2. **Uncertainty analysis over time**
3. **Confidence comparison by product**
4. **Integrated dashboard with inventory and pricing**

## Dependencies

- statsmodels
- pmdarima
- joblib
- pandas
- numpy
- matplotlib
- seaborn

## Future Improvements

- Add automated anomaly detection
- Implement hierarchical forecasting
- Explore regime switching for handling structural breaks
- Add automatic model selection between RF, ARIMA, and PyTorch models