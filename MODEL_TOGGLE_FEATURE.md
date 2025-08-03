# Model Toggle Feature

This document describes the Model Toggle feature in the Predictive Ordering System, which allows users to switch between training new models and using existing ones.

## Overview

The Model Toggle feature provides significant performance improvements by enabling the reuse of previously trained models. This is especially useful in production environments where consistent results and fast execution are important.

## Usage

### Command Line Options

The toggle option is available in both the standalone RF model script and the main pipeline:

```bash
# Use existing model in the RF script
python3 rf_model_update.py --use-existing

# Use existing model in the main pipeline
python3 main.py --use-existing-rf

# Combine with other options
python3 main.py --use-existing-rf --skip-time-series
```

### Implementation Details

The toggle is implemented with these key components:

1. **Command line argument parsing** in both `main.py` and `rf_model_update.py`
2. **Model saving and loading** with pickle serialization
3. **Feature name consistency** between training and inference
4. **Stochastic prediction** capabilities in both modes

### Key Files

- `rf_model_update.py`: Contains the core toggle functionality
- `main.py`: Passes the toggle option to the RF module
- `models/rf_model.pkl`: Saved model file
- `models/rf_model_features.pkl`: Feature columns for consistent inference

## Benefits

### Performance Improvements

- **Execution Speed**: 10-20x faster than full retraining
- **Resource Efficiency**: Lower memory and CPU usage
- **Consistent Results**: Same forecast patterns between runs

### Stochastic Behavior

Even when using existing models, the system maintains stochastic forecast capabilities:

- Temporal correlation between predictions
- Randomness scaled by historical volatility
- Seasonal variation components
- Auto-correlation for realistic patterns

### Use Cases

- **Development**: Faster iteration when testing new features
- **Production**: Consistent forecasts with minimal overhead
- **Scheduled Updates**: Train models weekly but generate forecasts daily
- **Testing**: Compare results from different model versions

## Technical Implementation

The toggle functionality required several improvements to the codebase:

1. **Class Pickleability**: Ensuring all model components can be serialized
2. **Module-Level Functions**: Moving nested functions to module level
3. **Clean Parameter Handling**: Proper passing of toggle parameters
4. **Feature Consistency**: Ensuring same features are used in both modes

## Future Improvements

- Add timestamp to model files for versioning
- Support multiple saved models with selection capability
- Add automated retraining on significant data changes
- Create automatic A/B testing between new and existing models