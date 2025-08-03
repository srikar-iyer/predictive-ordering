# Pizza Predictive Ordering System

An advanced machine learning-based system for optimizing pizza inventory and pricing decisions. This specialized system leverages multiple models including Random Forest and PyTorch time series models to predict demand, optimize inventory levels, and maximize profits for pizza retailers.

## Overview

The Pizza Predictive Ordering System helps retailers optimize their frozen pizza inventory and pricing strategies by providing data-driven recommendations. The system uses a combination of machine learning approaches:

1. **Random Forest models** for comprehensive demand forecasting with multiple factors
2. **PyTorch LSTM/Transformer models** for accurate time series forecasting
3. **Inventory management logic** with 1-3 weeks of stock policy
4. **Price elasticity analysis** and profit maximization algorithms
5. **Interactive Plotly/Dash dashboard** for visualization and decision support

The system analyzes these key factors:
- Historical sales patterns and seasonality
- Current inventory levels and order history
- Price elasticity and promotional effects
- Product-specific characteristics
- Store-specific demand patterns
- External factors (holidays, weather)

## Features

### Core Capabilities
- **Multi-Model Demand Forecasting**: Combine Random Forest and PyTorch time series models for robust predictions
- **Smart Inventory Management**: Apply 1-3 weeks stock logic with automatic order recommendations
- **Price Optimization**: Calculate price elasticity and recommend profit-maximizing prices
- **Product Mix Analysis**: Optimize across multiple products considering cross-effects
- **Interactive Visualization**: Comprehensive Plotly-based dashboard
- **Complete Data Pipeline**: Integrated workflow from raw data to recommendations
- **Model Toggle Option**: NEW! Use existing models without retraining for faster execution

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

# Use existing Random Forest model (NEW!)
python main.py --use-existing-rf

# Combine both options for maximum speed
python main.py --skip-time-series --use-existing-rf --skip-arima
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

### AWS Deployment (Production)

The application is configured for deployment on AWS services such as ECS or Elastic Beanstalk.

#### AWS Elastic Container Service (ECS) Deployment

1. Build and tag the Docker image:
   ```bash
   docker build -t pizza-predictive-ordering .
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
   docker tag pizza-predictive-ordering:latest YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/pizza-predictive-ordering:latest
   docker push YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/pizza-predictive-ordering:latest
   ```

2. Create an ECS task definition specifying your ECR image URI and configure:
   - Port mappings: 8050:8050
   - Environment variables: PORT=8050, HOST=0.0.0.0
   - Task role with necessary AWS permissions
   - Log configuration for CloudWatch

3. Create an ECS service with:
   - Load balancer configuration
   - Health check at path '/'
   - Auto-scaling policies as needed

#### AWS Elastic Beanstalk Deployment

1. Initialize an Elastic Beanstalk application:
   ```bash
   eb init -p docker pizza-predictive-ordering
   ```

2. Create your environment:
   ```bash
   eb create pizza-predictive-ordering-env
   ```

3. Deploy your application:
   ```bash
   eb deploy
   ```

4. Monitor your application:
   ```bash
   eb open
   ```

### Using Docker Directly

1. Build the Docker image:
   ```bash
   docker build -t pizza-predictive-ordering .
   ```

2. Run the container:
   ```bash
   docker run -p 8050:8050 --name pizza-app pizza-predictive-ordering
   ```

3. Access the dashboard at `http://localhost:8050`

### Environment Configuration

The application supports the following environment variables:

- `PORT`: The port on which the server will listen (default: 8050)
- `HOST`: The host interface to bind to (default: 0.0.0.0)
- `AWS_REGION`: The AWS region for AWS service integration

### Docker Volumes and Persistent Storage

The docker-compose setup includes a volume for persisting output data across container restarts:
```yaml
volumes:
  output-data:  # Persists across container restarts
```

For AWS deployments, consider using:
- EFS (Elastic File System) for persistent storage in ECS
- S3 buckets for data storage with proper IAM permissions

## Data Files

### Input Files
- **FrozenPizzaSales.csv**: Daily sales records with product, price and quantity
- **FrozenPizzaPurchases.csv**: Order/delivery records with quantities and costs
- **FrozenPizzaStock.csv**: Current inventory levels and historical movement

### Generated Files
- **combined_pizza_data.csv**: Integrated dataset with all relevant features
- **pytorch_forecasts.csv**: Time series model forecasts
- **rf_forecasts.csv**: Random Forest model forecasts
- **inventory_recommendations.csv**: Inventory management recommendations
- **optimized_orders.csv**: Optimized ordering schedule
- **inventory_projection.csv**: Projected inventory levels
- **price_elasticities.csv**: Calculated price elasticity values
- **price_recommendations.csv**: Price optimization recommendations
- **profit_impact.csv**: Projected profit impact analysis

## Component Modules

### Data Processing
- **integrate_pizza_data.py**: Merges and processes the three input datasets

### Forecasting Models
- **pytorch_time_series.py**: Time series forecasting with LSTM/Transformer neural networks
- **rf_model_update.py**: Random Forest model training and evaluation

### Optimization Logic
- **inventory_management.py**: Inventory level optimization with 1-3 weeks stock logic
- **profit_optimization.py**: Price elasticity analysis and profit maximization

### Visualization
- **plotly_dashboard.py**: Interactive dashboard with Dash and Plotly

### Pipeline Management
- **main.py**: Orchestrates the complete workflow

## Dashboard Features

The interactive dashboard provides:

1. **Sales & Forecast Charts**:
   - Historical sales visualization
   - Demand forecasts with confidence intervals
   - Stock level tracking

2. **Inventory Management**:
   - Stock level projections
   - Safety stock visualization
   - Order point recommendations

3. **Price Elasticity Analysis**:
   - Price sensitivity curves
   - Historical price-sales relationship
   - Elasticity distribution

4. **Profit Optimization**:
   - Price recommendations
   - Profit impact analysis
   - Revenue and margin optimization

5. **Product Recommendations**:
   - Inventory recommendations
   - Price change suggestions
   - Profit improvement estimates

## Advanced Features

### Model Toggle Options
The system now supports model reuse to speed up operations:
- **RF Model Toggle**: Reuse trained Random Forest models with `--use-existing-rf` flag
- **Stochastic Behavior**: Maintained even when using existing models for realistic forecasts
- **Performance Boost**: Up to 10x faster execution without retraining
- **Consistent Results**: Get comparable forecasts between production runs

### Time Series Models
The system implements two neural network architectures for time series forecasting:
- **LSTM (Long Short-Term Memory)**: Captures long-term dependencies in sales patterns
- **Transformer**: Utilizes attention mechanisms for improved sequence modeling

### Inventory Policy
The inventory management logic implements a sophisticated weeks-of-supply approach:
- **Below 1 week**: Low stock - Order to reach target stock level (2 weeks)
- **1-3 weeks**: Adequate stock - No order needed
- **Above 3 weeks**: Excess stock - No order needed

### Price Optimization
The profit maximization algorithm:
1. Calculates price elasticity using log-log regression
2. Determines optimal price points based on elasticity and costs
3. Respects business constraints (min margin, max price changes)
4. Projects profit impact over the forecast period

## Example Workflow

1. **Data Integration**
   - Combine sales, purchase and stock data
   - Generate derived features (weeks of stock, lag features)
   - Create a unified dataset for modeling

2. **Demand Forecasting**
   - Train Random Forest on full feature set
   - Train PyTorch models on time series patterns
   - Generate 30-day forecasts with both models

3. **Inventory Optimization**
   - Calculate current stock status
   - Apply 1-3 weeks stock logic
   - Generate order recommendations

4. **Price Optimization**
   - Calculate price elasticity
   - Determine optimal price points
   - Project profit impact

5. **Decision Support**
   - Present recommendations in dashboard
   - Enable exploration of what-if scenarios
   - Provide detailed justifications

## Customization

### Modifying Inventory Policy
Edit the `InventoryManager` class in `inventory_management.py`:
```python
# Change thresholds for weeks of stock
inventory_manager = InventoryManager(min_weeks=1, target_weeks=2, max_weeks=3)
```

### Adjusting Price Optimization Constraints
Edit the `constraints` dictionary in `profit_optimization.py`:
```python
constraints = {
    'max_price_increase': 20,  # Maximum 20% price increase
    'max_price_decrease': 15,  # Maximum 15% price decrease
    'min_margin': 25,          # Minimum 25% margin
    'max_stock_weeks': 3       # Maximum 3 weeks of stock
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Insert your license information here]