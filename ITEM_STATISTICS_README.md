# Item-Based Statistics with Extended Forecasting

This module provides comprehensive item-level statistics and extends forecasts beyond the current time range to support long-term planning and analysis. It integrates with the existing forecasting models (especially ARIMA) to provide deeper insights into product performance.

## Features

1. **Comprehensive Item-Level Statistics**
   - Sales patterns and trends
   - Seasonality detection and analysis
   - Price sensitivity and elasticity
   - Weather impact analysis
   - Day-of-week and monthly sales patterns
   - Inventory and stock coverage analysis
   - Profit and margin calculations

2. **Extended Forecasting**
   - Extends existing forecasts up to 60+ days into the future
   - Uses multiple forecasting methods based on item characteristics:
     - Trend and seasonality extrapolation
     - Day-of-week pattern-based forecasting
     - Linear trend extrapolation
   - Automatically increases uncertainty over time
   - Maintains confidence intervals with growing width

3. **Advanced Visualization**
   - Item statistics visualizations
   - Extended forecast charts
   - Comparative performance views
   - Sales pattern analysis
   - Weather and seasonality impact visualizations

4. **Dashboard Integration**
   - New "Item Statistics" tab in the dashboard
   - Interactive item selection and analysis
   - Multiple view options (basic statistics, detailed analysis, extended forecast)
   - Item comparison tools

## Files

### Core Functionality
- `/src/models/item_statistics.py` - Main module with statistics and forecasting logic
- `/run_item_statistics.py` - Command-line tool for running analysis

### UI Components
- `/ui/item_statistics.py` - Dashboard UI component for the item statistics tab

### Testing
- `/test_item_statistics.py` - Test script for the module

### Configuration
- Updated `config/settings.py` with new file paths and settings

## Usage

### Command Line

Run the item statistics analysis with default settings:
```bash
python run_item_statistics.py
```

Extended options:
```bash
python run_item_statistics.py --data /path/to/data.csv --days-to-extend 60 --no-viz
```

### Dashboard

1. Launch the dashboard application:
```bash
python -m ui.run_dashboard
```

2. Navigate to the "Item Statistics" tab
3. Select a store and product
4. Choose between basic statistics, detailed analysis, or extended forecast views

## Output Files

- `item_statistics.csv` - Comprehensive item statistics
- `extended_forecasts.csv` - Extended forecast data

## Integration with Existing Models

The module integrates with:

- ARIMA forecasting models
- Weighted ARIMA models
- Time series forecasting data
- Inventory recommendations
- Price optimization

It builds upon the existing forecast data to provide extended projections and more detailed item-level analysis.

## Technical Implementation

The implementation focuses on:

1. **Statistical Robustness** - Using proven statistical methods for trend detection, seasonality analysis, and extended forecasting.

2. **Visualization Quality** - Creating clear, informative visualizations with appropriate context and confidence intervals.

3. **Usability** - Providing an intuitive interface for exploring item-level data and forecasts.

4. **Performance** - Efficiently processing data and generating forecasts, even for large datasets.

5. **Integration** - Seamlessly working with existing models and data structures.

## Future Enhancements

- Cross-item correlation analysis
- Product category aggregation and analysis
- Interactive forecast adjustment
- Export functionality for reports
- Anomaly detection and flagging