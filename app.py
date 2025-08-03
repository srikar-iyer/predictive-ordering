import os
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import plotly
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Create Flask app instance first
app = Flask(__name__)

# Import visualization modules
from src.models.plotly_visualizations import (
    create_elasticity_distribution_plot,
    create_price_sensitivity_curve,
    create_profit_impact_waterfall
)
from src.models.integrated_visualizations import (
    create_integrated_chart,
    create_impact_heatmap,
    create_kpi_indicators
)

# Add routes for recommendation endpoints
@app.route('/price-recommendations')
def price_recommendations_view():
    """Render the price recommendations visualization page."""
    return render_template('price_recommendations.html')

@app.route('/inventory-recommendations')
def inventory_recommendations_view():
    """Render the inventory recommendations visualization page."""
    return render_template('inventory_recommendations.html')

# Data loader function to fetch appropriate data from CSV files
def get_data(data_type):
    try:
        # Get root directory - this helps ensure we have the right paths regardless of where the app is run from
        root_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(root_dir, 'data', 'processed')
        
        # Define paths to data files
        data_paths = {
            'forecast': os.path.join(data_dir, 'rf_forecasts.csv'),
            'rf_forecasts': os.path.join(data_dir, 'rf_forecasts.csv'),
            'inventory': os.path.join(data_dir, 'inventory', 'inventory_projection.csv'),
            'elasticity': os.path.join(data_dir, 'price_elasticities.csv'),
            'profit_impact': os.path.join(data_dir, 'profit_impact.csv'),
            'arima_forecasts': os.path.join(data_dir, 'arima_forecasts.csv'),
            'pytorch_forecasts': os.path.join(data_dir, 'pytorch_forecasts.csv'),
            'price_recommendations': os.path.join(data_dir, 'price_recommendations.csv'),
            'rf_recommendations': os.path.join(data_dir, 'rf_recommendations.csv'),
            'combined_data': os.path.join(data_dir, 'combined_pizza_data.csv')
        }
        
        print(f"Looking for {data_type} at {data_paths.get(data_type, 'unknown path')}")
        
        # Check if requested data file exists
        if data_type in data_paths and os.path.exists(data_paths[data_type]):
            print(f"Found data file for {data_type}: {data_paths[data_type]}")
            data = pd.read_csv(data_paths[data_type])
            print(f"Loaded {len(data)} rows for {data_type}")
            return data
        else:
            if data_type in data_paths:
                print(f"Data file not found: {data_paths[data_type]}")
            else:
                print(f"No path defined for data type: {data_type}")
            return None
    except Exception as e:
        print(f"Error loading {data_type} data: {str(e)}")
        return None

# Generate sample data as fallback when real data is not available
def generate_sample_data():
    # Create sample forecast data using pizza product data
    dates = pd.date_range(start='2025-07-02', end='2025-08-31', freq='D')
    np.random.seed(42)
    
    # Sample forecast data for BELLATORIA BBQ CHK PIZZA
    forecast_data = pd.DataFrame({
        'Date': dates,
        'Forecast': np.random.normal(15, 3, len(dates)),  # More realistic units (not percentages)
        'Upper_Bound': np.random.normal(18, 3, len(dates)),
        'Lower_Bound': np.random.normal(12, 3, len(dates)),
        'Std_Dev': np.random.uniform(2, 3, len(dates))
    })
    
    # Add store and product information to match actual pizza data format
    forecast_data['Store_Id'] = 104.0
    forecast_data['Item'] = 3913116850.0
    forecast_data['Product'] = 'BELLATORIA BBQ CHK PIZZA'
    forecast_data['Size'] = '15.51 OZ'
    
    # Sample elasticity data using real pizza products
    elasticity_data = pd.DataFrame({
        'Store_Id': [104.0] * 20,
        'Item': [3913116850.0, 3913116852.0, 3913116853.0, 3913116856.0, 3913116891.0, 
                3913116957.0, 4219700201.0, 4219700203.0, 4219700204.0, 4280010700.0, 
                4280010800.0, 4280011300.0, 4280011400.0, 4280011520.0, 4280011600.0, 
                7218063429.0, 7218063473.0, 7465379187.0, 7218063247.0, 7218063244.0],
        'Product': ['BELLATORIA BBQ CHK PIZZA', 'BELLATORIA ULT PEPPERONI PIZZA', 'BELLATORIA ULT SUPREME PIZZA',
                  'BELLATORIA GAR CHKN ALFR PIZZA', 'BELLATORIA SAUS ITALIA PIZZA', 'BELLATORIA HAWAIIAN BBQ PIZZA',
                  'BREW PUB PEPPERONI PIZZA', 'BREW PUB SAUS PEP PIZZA', 'BREW PUB CHEESE PIZZA', 'TOTINO SUPREME PIZZA',
                  'TOTINO 3 MEAT PIZZA', 'TOTINO CHEESE PIZZA', 'TOTINO PEPPERONI PIZZA', 'TOTINO 3 CHEESE PIZZA',
                  'TOTINO COMBINATION PIZZA', 'RED BARON 4 CHEESE PIZZA', 'RED BARON PEPPERONI PIZZA',
                  'JACKS PEPPERONI PIZZA', 'RED BARON SUPREME PIZZA', 'RED BARON FH PEPPERONI'],
        'Elasticity': [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.9, -1.5, -1.5, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        'Current_Price': [7.99, 7.99, 7.99, 7.99, 7.99, 7.99, 8.99, 8.99, 8.99, 2.19, 2.19, 2.19, 2.19, 2.19, 2.19, 4.99, 4.99, 4.99, 4.99, 4.99],
        'Cost': [4.75, 4.75, 4.75, 4.75, 4.75, 4.75, 4.95, 4.95, 4.95, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 2.50, 2.50, 2.50, 2.50, 2.50],
        'Margin_Pct': [40.55, 40.55, 40.55, 40.55, 40.55, 40.55, 44.94, 44.94, 44.94, 49.77, 49.77, 49.77, 49.77, 49.77, 49.77, 49.90, 49.90, 49.90, 49.90, 49.90],
        'R_Squared': [0.39, 0.21, 0.27, 0.20, 0.22, 0.12, 0.31, 0.34, 0.37, 0.18, 0.19, 0.22, 0.25, 0.20, 0.21, 0.19, 0.25, 0.23, 0.25, 0.28],
        'Is_Significant': [True, False, False, False, False, False, True, True, True, False, False, False, False, False, False, False, False, False, False, False],
        'Status': ['Valid elasticity', 'Poor fit', 'Poor fit', 'Poor fit', 'Poor fit', 'Poor fit', 'Valid elasticity', 'Valid elasticity', 'Valid elasticity', 
                'Poor fit', 'Poor fit', 'Poor fit', 'Poor fit', 'Poor fit', 'Poor fit', 'Poor fit', 'Poor fit', 'Poor fit', 'Poor fit', 'Poor fit']
    })
    
    # Sample profit impact data based on real pizza products
    profit_impact_data = pd.DataFrame({
        'Store_Id': [104.0] * 10,
        'Item': [4280011400.0, 7218063473.0, 7465379187.0, 4280010800.0, 7218063429.0, 4280011600.0, 4280011520.0, 4280010700.0, 4280011300.0, 3913116850.0],
        'Product': ['TOTINO PEPPERONI PIZZA', 'RED BARON PEPPERONI PIZZA', 'JACKS PEPPERONI PIZZA', 'TOTINO 3 MEAT PIZZA', 'RED BARON 4 CHEESE PIZZA',
                  'TOTINO COMBINATION PIZZA', 'TOTINO 3 CHEESE PIZZA', 'TOTINO SUPREME PIZZA', 'TOTINO CHEESE PIZZA', 'BELLATORIA BBQ CHK PIZZA'],
        'Current_Price': [2.19, 4.99, 4.99, 2.19, 4.99, 2.19, 2.19, 2.19, 2.19, 7.99],
        'Optimal_Price': [2.63, 5.99, 5.99, 2.63, 5.99, 2.63, 2.63, 2.63, 2.63, 9.59],
        'Price_Change_Pct': [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        'Elasticity': [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0],
        'Total_Current_Profit': [143.81, 168.82, 299.18, 83.89, 89.41, 49.34, 49.26, 44.30, 43.36, 35.00],
        'Total_New_Profit': [307.30, 309.84, 425.78, 179.26, 164.09, 105.44, 105.26, 94.67, 92.66, 42.00],
        'Total_Profit_Difference': [163.49, 141.02, 126.60, 95.37, 74.68, 56.10, 56.00, 50.37, 49.30, 7.00],
        'Profit_Change_Pct': [113.69, 83.53, 42.32, 113.69, 83.53, 113.69, 113.69, 113.69, 113.69, 20.00],
        'Forecast_Days': [30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
        'Daily_Profit_Impact': [5.45, 4.70, 4.22, 3.18, 2.49, 1.87, 1.87, 1.68, 1.64, 0.23]
    })
    
    # Sample price recommendations based on real pizza products
    price_recommendations_data = pd.DataFrame({
        'Store_Id': [104.0] * 10,
        'Item': [3913116850.0, 3913116856.0, 3913116891.0, 4219700201.0, 4219700203.0, 4219700204.0, 4219700205.0, 4219700206.0, 4219700207.0, 4280010700.0],
        'Product': ['BELLATORIA BBQ CHK PIZZA', 'BELLATORIA GAR CHKN ALFR PIZZA', 'BELLATORIA SAUS ITALIA PIZZA', 
                  'BREW PUB PEPPERONI PIZZA', 'BREW PUB SAUS PEP PIZZA', 'BREW PUB CHEESE PIZZA', 'BREW PUB 4 MEAT PIZZA', 
                  'BREW PUB BACON CHSBRGR PIZZA', 'BREW PUB SUPREME PIZZA', 'TOTINO SUPREME PIZZA'],
        'Elasticity': [-2.0, -1.0, -1.0, -1.88, -1.48, -1.51, -2.23, -2.21, -2.57, -1.0],
        'Current_Price': [7.99, 7.99, 7.99, 8.99, 8.99, 8.99, 8.99, 8.99, 8.99, 2.19],
        'Optimal_Price': [8.79, 9.59, 9.59, 9.89, 7.64, 8.09, 8.09, 8.09, 8.09, 2.63],
        'Price_Change_Pct': [10.0, 20.0, 20.0, 10.0, -15.0, -10.0, -10.0, -10.0, -10.0, 20.0],
        'Expected_Sales_Change_Pct': [-17.38, -16.67, -16.67, -16.37, 27.28, 17.30, 26.52, 26.20, 31.15, -16.67],
        'Expected_Profit_Change_Pct': [2.99, 24.43, 24.43, 2.24, 8.19, 5.57, 13.87, 13.58, 18.03, 113.69],
        'Cost': [4.75, 4.75, 4.75, 4.95, 0.0, 0.0, 0.0, 0.0, 0.0, 1.91],
        'Current_Margin_Pct': [40.55, 40.55, 40.55, 44.94, 100.0, 100.0, 100.0, 100.0, 100.0, 12.79],
        'New_Margin_Pct': [45.96, 50.46, 50.46, 49.94, 100.0, 100.0, 100.0, 100.0, 100.0, 27.32],
        'Constraint_Applied': ['elasticity_upper_bound', 'standard_upper_bound', 'standard_upper_bound', 'elasticity_upper_bound', 
                             'standard_lower_bound', 'elasticity_lower_bound', 'elasticity_lower_bound', 'elasticity_lower_bound', 
                             'elasticity_lower_bound', 'margin_upper_bound'],
        'Recommendation': ['Increase price by 10.0%', 'Increase price by 20.0%', 'Increase price by 20.0%', 'Increase price by 10.0%', 
                          'Decrease price by 15.0%', 'Decrease price by 10.0%', 'Decrease price by 10.0%', 'Decrease price by 10.0%', 
                          'Decrease price by 10.0%', 'Increase price by 20.0%']
    })
    
    # Sample inventory recommendations data using pizza products
    inventory_recommendations_data = pd.DataFrame({
        'Store_Id': [104.0] * 15,
        'Item': [3913116850.0, 3913116852.0, 3913116853.0, 3913116856.0, 3913116891.0, 
               4219700201.0, 4219700203.0, 4219700204.0, 4280010700.0, 4280010800.0,
               4280011300.0, 4280011400.0, 4280011520.0, 4280011600.0, 7218063429.0],
        'Current_Stock': [195, 233, 273, 302, 337, 216, 255, 292, 334, 373, 243, 274, 327, 355, 394],
        'Avg_Daily_Demand': [14.97, 17.0, 24.0, 31.1, 39.27, 23.1, 32.47, 28.87, 31.73, 36.43, 25.7, 39.73, 33.47, 30.7, 28.87],
        'Weeks_Of_Stock': [1.86, 1.96, 1.63, 1.39, 1.23, 1.34, 1.12, 1.45, 1.50, 1.46, 1.35, 0.99, 1.40, 1.65, 1.95],
        'Target_Stock_Weeks': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        'Min_Stock_Weeks': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'Target_Stock_Units': [209.5, 238.0, 336.0, 435.4, 549.7, 323.4, 454.5, 404.1, 444.3, 510.1, 359.8, 556.3, 468.5, 429.8, 404.1],
        'Recommended_Order_Quantity': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 282, 0, 0, 0],
        'Product': ['BELLATORIA BBQ CHK PIZZA', 'BELLATORIA ULT PEPPERONI PIZZA', 'BELLATORIA ULT SUPREME PIZZA',
                  'BELLATORIA GAR CHKN ALFR PIZZA', 'BELLATORIA SAUS ITALIA PIZZA', 'BREW PUB PEPPERONI PIZZA', 
                  'BREW PUB SAUS PEP PIZZA', 'BREW PUB CHEESE PIZZA', 'TOTINO SUPREME PIZZA', 'TOTINO 3 MEAT PIZZA', 
                  'TOTINO CHEESE PIZZA', 'TOTINO PEPPERONI PIZZA', 'TOTINO 3 CHEESE PIZZA', 'TOTINO COMBINATION PIZZA',
                  'RED BARON 4 CHEESE PIZZA'],
        'Size': ['15.51 OZ', '15.51 OZ', '16.03 OZ', '16.03 OZ', '16.35 OZ', '23 OZ', '23 OZ', '21.95 OZ', '9.8 OZ', '9.8 OZ', 
               '9.8 OZ', '9.8 OZ', '10.2 OZ', '10.2 OZ', '20.6 OZ'],
        'Stock_Status': ['Adequate', 'Adequate', 'Adequate', 'Adequate', 'Adequate', 'Adequate', 'Adequate', 'Adequate', 'Adequate', 'Adequate', 
                        'Adequate', 'Low', 'Adequate', 'Adequate', 'Adequate'],
        'Unit_Profit': [2.50, 4.80, 3.93, 3.39, 1.62, 1.62, 1.23, 4.46, 3.40, 3.83, 1.08, 4.88, 4.33, 1.85, 1.73],
        'Profit_Impact': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1376.06, 0.0, 0.0, 0.0]
    })
    
    return {
        'forecast': forecast_data,
        'elasticity': elasticity_data,
        'profit_impact': profit_impact_data,
        'price_recommendations': price_recommendations_data,
        'rf_recommendations': inventory_recommendations_data
    }

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Render the main dashboard page."""
    return render_template('dashboard.html')

@app.route('/forecast')
def forecast():
    """Render the forecast visualization page."""
    return render_template('forecast.html')

@app.route('/forecast/rf')
def rf_forecast():
    """Render the Random Forest forecast visualization page."""
    return render_template('forecast_rf.html')

@app.route('/forecast/arima')
def arima_forecast():
    """Render the ARIMA forecast visualization page."""
    return render_template('forecast_arima.html')

@app.route('/forecast/pytorch')
def pytorch_forecast():
    """Render the PyTorch forecast visualization page."""
    return render_template('forecast_pytorch.html')

@app.route('/pricing')
def pricing():
    """Render the pricing visualization page."""
    return render_template('pricing.html')

@app.route('/inventory')
def inventory():
    """Render the inventory visualization page."""
    return render_template('inventory.html')

@app.route('/api/inventory-forecast-data')
def inventory_forecast_data():
    """API endpoint to get inventory projection with forecast data."""
    try:
        # Get parameters from request
        store_id = request.args.get('store_id', '104.0')
        product_id = request.args.get('product_id', '3913116850.0')
        forecast_model = request.args.get('forecast_model', 'rf')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Determine which forecast data to load based on model
        forecast_type = f"{forecast_model}_forecasts" if forecast_model in ['rf', 'arima', 'pytorch'] else 'forecast'
        
        # Load inventory and forecast data
        inventory_df = get_data('inventory')
        forecast_df = get_data(forecast_type)
        
        # Filter inventory data if available
        if inventory_df is not None:
            inventory_df = inventory_df[(inventory_df['Store_Id'] == float(store_id)) & 
                                       (inventory_df['Item'] == float(product_id))]
        
        # Filter forecast data if available
        if forecast_df is not None:
            forecast_df = forecast_df[(forecast_df['Store_Id'] == float(store_id)) & 
                                     (forecast_df['Item'] == float(product_id))]
        
        # Check if we have sufficient data
        if inventory_df is None or len(inventory_df) == 0 or forecast_df is None or len(forecast_df) == 0:
            # Use sample data if files don't exist or no data for selected store/product
            data = generate_sample_data()
            dates = data['forecast']['Date']
            
            # Create figure
            fig = go.Figure()
            
            # Sample stock levels
            np.random.seed(42)  # For reproducibility
            initial_stock = 100
            stock_levels = [initial_stock]
            for i in range(1, len(dates)):
                # Daily sales from forecast with some randomness
                sales = data['forecast']['Forecast'].iloc[i] * (0.8 + 0.4 * np.random.random())
                # Random restock every 7 days
                restock = np.random.choice([0, 50, 100]) if i % 7 == 0 else 0
                # New stock level
                new_stock = max(0, stock_levels[-1] - sales + restock)
                stock_levels.append(new_stock)
            
            # Add stock level line
            fig.add_trace(go.Scatter(
                x=dates,
                y=stock_levels,
                mode='lines+markers',
                name='Current Stock',
                line=dict(color='blue', width=2)
            ))
            
            # Add forecast line
            fig.add_trace(go.Scatter(
                x=dates,
                y=data['forecast']['Forecast'],
                mode='lines',
                name=f'{forecast_model.upper()} Forecast',
                line=dict(color='red', width=2),
                yaxis='y2'
            ))
            
            # Add reorder point line
            reorder_point = 30  # Example reorder point
            fig.add_trace(go.Scatter(
                x=dates,
                y=[reorder_point] * len(dates),
                mode='lines',
                name='Reorder Point',
                line=dict(color='orange', dash='dash', width=1.5)
            ))
            
            # Add target stock line
            target_stock = 80  # Example target stock
            fig.add_trace(go.Scatter(
                x=dates,
                y=[target_stock] * len(dates),
                mode='lines',
                name='Target Stock',
                line=dict(color='green', dash='dash', width=1.5)
            ))
            
            # Update layout
            fig.update_layout(
                title='Inventory Projection with Demand Forecast (Sample Data)',
                xaxis_title='Date',
                yaxis_title='Stock Level (Units)',
                yaxis2=dict(
                    title='Forecast Demand (Units)',
                    overlaying='y',
                    side='right',
                    showgrid=False
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=-0.1,
                    xanchor='center',
                    x=0.5,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='rgba(0,0,0,0.1)',
                    borderwidth=1,
                    font=dict(size=12)
                )
            )
        else:
            # Convert dates if needed
            if not pd.api.types.is_datetime64_dtype(inventory_df['Date']):
                inventory_df['Date'] = pd.to_datetime(inventory_df['Date'])
            
            if not pd.api.types.is_datetime64_dtype(forecast_df['Date']):
                forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
            
            # Apply date filters if provided
            if start_date:
                start_date = pd.to_datetime(start_date)
                inventory_df = inventory_df[inventory_df['Date'] >= start_date]
                forecast_df = forecast_df[forecast_df['Date'] >= start_date]
            
            if end_date:
                end_date = pd.to_datetime(end_date)
                inventory_df = inventory_df[inventory_df['Date'] <= end_date]
                forecast_df = forecast_df[forecast_df['Date'] <= end_date]
            
            # Sort by date
            inventory_df = inventory_df.sort_values('Date')
            forecast_df = forecast_df.sort_values('Date')
            
            # Get product name
            product_name = "Unknown"
            if 'Product' in inventory_df.columns and len(inventory_df) > 0:
                product_name = inventory_df['Product'].iloc[0]
            elif 'Product' in forecast_df.columns and len(forecast_df) > 0:
                product_name = forecast_df['Product'].iloc[0]
            
            # Create figure
            fig = go.Figure()
            
            # Add stock level line
            fig.add_trace(go.Scatter(
                x=inventory_df['Date'],
                y=inventory_df['Stock_Level'],
                mode='lines+markers',
                name='Current Stock',
                line=dict(color='blue', width=2)
            ))
            
            # Add forecast line
            forecast_column = 'Forecast'
            if forecast_column in forecast_df.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df[forecast_column],
                    mode='lines',
                    name=f'{forecast_model.upper()} Forecast',
                    line=dict(color='red', width=2),
                    yaxis='y2'
                ))
            
            # Add reorder point and target stock if available
            if 'Order_Point' in inventory_df.columns:
                fig.add_trace(go.Scatter(
                    x=inventory_df['Date'],
                    y=inventory_df['Order_Point'],
                    mode='lines',
                    name='Reorder Point',
                    line=dict(color='orange', dash='dash', width=1.5)
                ))
            
            # Add target stock based on min_stock_weeks * avg_weekly_demand
            if 'Avg_Weekly_Demand' in inventory_df.columns and 'Min_Stock_Weeks' in inventory_df.columns:
                target_stock = inventory_df['Avg_Weekly_Demand'] * inventory_df['Target_Stock_Weeks']
                fig.add_trace(go.Scatter(
                    x=inventory_df['Date'],
                    y=target_stock,
                    mode='lines',
                    name='Target Stock',
                    line=dict(color='green', dash='dash', width=1.5)
                ))
            
            # Update layout
            fig.update_layout(
                title=f'Inventory Projection with {forecast_model.upper()} Forecast for {product_name} (Store {store_id})',
                xaxis_title='Date',
                yaxis_title='Stock Level (Units)',
                yaxis2=dict(
                    title='Forecast Demand (Units)',
                    overlaying='y',
                    side='right',
                    showgrid=False
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=-0.1,
                    xanchor='center',
                    x=0.5,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='rgba(0,0,0,0.1)',
                    borderwidth=1,
                    font=dict(size=12)
                )
            )
        
        return jsonify(plotly.io.to_json(fig))
    except Exception as e:
        # Generate detailed error with traceback
        import traceback
        error_stack = traceback.format_exc()
        print(f"Error in inventory projection: {str(e)}\n{error_stack}")
        
        # Return detailed error message
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error creating inventory projection chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.6, showarrow=False
        )
        
        # Add file and line information
        tb_lines = error_stack.split('\n')
        file_line_info = [line for line in tb_lines if 'File "' in line]
        if file_line_info:
            error_fig.add_annotation(
                text=file_line_info[-1].strip(),
                xref="paper", yref="paper",
                x=0.5, y=0.4, showarrow=False,
                font=dict(size=10)
            )
            
        return jsonify(plotly.io.to_json(error_fig))

@app.route('/integrated')
def integrated():
    """Render the integrated visualization page."""
    return render_template('integrated.html')

@app.route('/price-elasticity')
def price_elasticity():
    """Render the price elasticity analysis page."""
    return render_template('price_elasticity.html')

@app.route('/revenue-impact')
def revenue_impact():
    """Render the revenue impact analysis page."""
    return render_template('revenue_impact.html')

@app.route('/model-comparison')
def model_comparison_page():
    """Render the forecast model comparison page."""
    return render_template('model_comparison.html')

@app.route('/api/forecast-data')
def forecast_data():
    """API endpoint to get forecast data with historical sales with enhanced time series controls."""
    try:
        # Get parameters from request with improved defaults
        store_id = request.args.get('store_id', '104.0')
        product_id = request.args.get('product_id', '3913116850.0')
        forecast_model = request.args.get('forecast_model', 'rf')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        timespan = int(request.args.get('timespan', '14'))  # Default to 2 weeks (14 days)
        show_confidence = request.args.get('show_confidence', 'true').lower() == 'true'
        chart_height = int(request.args.get('chart_height', '600'))
        
        # Set default date range to show data from 2023 to 07/08/2025 if not provided
        if not start_date or not end_date:
            if not start_date:
                start_date = '2023-01-01'  # Start from beginning of 2023
            if not end_date:
                end_date = '2025-07-08'    # Up to July 8, 2025
        
        # Determine which forecast data to load based on model
        forecast_type = f"{forecast_model}_forecasts" if forecast_model in ['rf', 'arima', 'pytorch'] else 'forecast'
        
        # Load forecast data
        forecast_df = get_data(forecast_type)
        
        # Load historical sales data
        combined_data = get_data('combined_data')
        
        if forecast_df is not None:
            # Handle forecast column naming based on model type
            forecast_col = 'Forecast'
            if forecast_model == 'pytorch':
                if 'Predicted_Demand' in forecast_df.columns:
                    forecast_col = 'Predicted_Demand'
                elif 'Predicted_Sales' in forecast_df.columns:
                    forecast_col = 'Predicted_Sales'
            elif forecast_model == 'arima':
                if 'Forecast' in forecast_df.columns:
                    forecast_col = 'Forecast'
                elif 'Predicted_Demand' in forecast_df.columns:
                    forecast_col = 'Predicted_Demand'
                elif 'Predicted_Sales' in forecast_df.columns:
                    forecast_col = 'Predicted_Sales'
                else:
                    # Try to find any column that might contain forecast values
                    numeric_cols = forecast_df.select_dtypes(include=[np.number]).columns.tolist()
                    potential_forecast_cols = [col for col in numeric_cols 
                                              if col not in ['Store_Id', 'Item'] 
                                              and not col.startswith('Lower_') 
                                              and not col.startswith('Upper_')]
                    
                    if potential_forecast_cols:
                        forecast_col = potential_forecast_cols[0]
                        print(f"Using '{forecast_col}' as fallback forecast column for ARIMA")
            
            # Filter by store and product
            forecast_df = forecast_df[(forecast_df['Store_Id'] == float(store_id)) & 
                                      (forecast_df['Item'] == float(product_id))]
            
            # Convert dates if needed
            if not pd.api.types.is_datetime64_dtype(forecast_df['Date']):
                forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
            
            # Apply date filters if provided
            if start_date:
                start_date = pd.to_datetime(start_date)
                forecast_df = forecast_df[forecast_df['Date'] >= start_date]
                
            if end_date:
                end_date = pd.to_datetime(end_date)
                forecast_df = forecast_df[forecast_df['Date'] <= end_date]
            
            # Get historical data if available
            historical_df = None
            if combined_data is not None:
                # Filter by store and product
                historical_df = combined_data[(combined_data['Store_Id'] == float(store_id)) & 
                                              (combined_data['Item'] == float(product_id))]
                
                # Convert dates if needed
                if not pd.api.types.is_datetime64_dtype(historical_df['Date']):
                    historical_df['Date'] = pd.to_datetime(historical_df['Date'])
                
                # Sort by date
                historical_df = historical_df.sort_values('Date')
                
                # Find overlap date between forecast and history to avoid double counting
                if not forecast_df.empty and not historical_df.empty:
                    min_forecast_date = forecast_df['Date'].min()
                    # Use only historical data before forecast starts
                    historical_df = historical_df[historical_df['Date'] < min_forecast_date]
                
                # Apply timespan filter - only use most recent historical data
                if len(historical_df) > 0:
                    max_hist_date = historical_df['Date'].max()
                    # Get data within the timespan
                    cutoff_date = max_hist_date - pd.Timedelta(days=timespan)
                    historical_df = historical_df[historical_df['Date'] >= cutoff_date]
            
            if not forecast_df.empty:
                # Sort by date
                forecast_df = forecast_df.sort_values('Date')
                
                # Create figure
                fig = go.Figure()
                
                # Add historical sales data if available
                if historical_df is not None and len(historical_df) > 0:
                    # Determine sales column name
                    sales_col = 'Sales' 
                    if 'Units_Sold' in historical_df.columns and 'Sales' not in historical_df.columns:
                        sales_col = 'Units_Sold'
                        
                    fig.add_trace(go.Scatter(
                        x=historical_df['Date'],
                        y=historical_df[sales_col],
                        mode='lines+markers',
                        name='Historical Sales',
                        line=dict(color='darkblue', width=2),
                        marker=dict(size=4, color='darkblue'),
                        hovertemplate='<b>Historical Sales</b><br>Date: %{x}<br>Units: %{y:.0f}<extra></extra>'
                    ))
                
                # Add historical sales data from CSV (if available)
                historical_data = get_data('combined_data')
                if historical_data is not None and len(historical_data) > 0:
                    historical_df = historical_data[(historical_data['Store_Id'] == float(store_id)) & 
                                                    (historical_data['Item'] == float(product_id))]
                    
                    # Convert dates if needed and ensure sorted
                    if len(historical_df) > 0:
                        if 'Date' in historical_df.columns and not pd.api.types.is_datetime64_dtype(historical_df['Date']):
                            historical_df['Date'] = pd.to_datetime(historical_df['Date'])
                        historical_df = historical_df.sort_values('Date')
                        
                        # Determine which column has the sales data
                        sales_col = 'Sales'
                        if 'Sales' not in historical_df.columns and 'Units_Sold' in historical_df.columns:
                            sales_col = 'Units_Sold'
                            
                        # Add historical data trace
                        fig.add_trace(go.Scatter(
                            x=historical_df['Date'],
                            y=historical_df[sales_col],
                            mode='markers',
                            name='Historical Sales',
                            marker=dict(size=5, color='blue'),
                            hovertemplate='<b>Historical Sales</b><br>Date: %{x}<br>Units: %{y:.0f}<extra></extra>'
                        ))
                
                # Add forecast line with enhanced styling
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df[forecast_col],
                    mode='lines+markers',
                    name=f'{forecast_model.upper()} Forecast',
                    line=dict(color='orange', width=3, dash='solid'),
                    marker=dict(size=6, color='orange', symbol='diamond'),
                    hovertemplate=f'<b>{forecast_model.upper()} Forecast</b><br>Date: %{{x}}<br>Predicted Units: %{{y:.1f}}<extra></extra>'
                ))
                
                # Add confidence interval if available
                upper_bound_col = 'Upper_Bound'
                lower_bound_col = 'Lower_Bound'
                
                # Check for alternative column names (for PyTorch model)
                if 'Upper_CI' in forecast_df.columns and 'Upper_Bound' not in forecast_df.columns:
                    upper_bound_col = 'Upper_CI'
                if 'Lower_CI' in forecast_df.columns and 'Lower_Bound' not in forecast_df.columns:
                    lower_bound_col = 'Lower_CI'
                    
                # Generate confidence intervals if not present
                if (upper_bound_col not in forecast_df.columns or lower_bound_col not in forecast_df.columns) and forecast_col in forecast_df.columns:
                    # Create 95% confidence intervals
                    std_dev = forecast_df[forecast_col] * 0.1  # Assuming 10% standard deviation
                    forecast_df['Upper_Bound'] = forecast_df[forecast_col] + 1.96 * std_dev  # 95% CI upper bound
                    forecast_df['Lower_Bound'] = forecast_df[forecast_col].apply(lambda x: max(0, x - 1.96 * std_dev))  # 95% CI lower bound
                    upper_bound_col = 'Upper_Bound'
                    lower_bound_col = 'Lower_Bound'
                    print(f"Generated 95% confidence intervals based on {forecast_col}")
                
                # Add confidence intervals only if requested
                if show_confidence and upper_bound_col in forecast_df.columns and lower_bound_col in forecast_df.columns:
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df[upper_bound_col],
                        mode='lines',
                        name='95% Confidence Interval',
                        line=dict(width=0),
                        showlegend=False,
                        hovertemplate='Upper Bound: %{y:.1f}<extra></extra>'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df[lower_bound_col],
                        mode='lines',
                        name='Lower Bound',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255, 165, 0, 0.2)',  # Orange with transparency
                        showlegend=True,
                        hovertemplate='Lower Bound: %{y:.1f}<extra></extra>'
                    ))
                
                # Get product name (with empty DataFrame check)
                product_name = forecast_df['Product'].iloc[0] if 'Product' in forecast_df.columns and not forecast_df.empty else f'Product {product_id}'
                
                # Calculate dynamic y-axis range to include all data
                y_values = []
                if historical_df is not None and len(historical_df) > 0:
                    if sales_col in historical_df.columns:
                        y_values.extend(historical_df[sales_col].dropna().tolist())
                
                if forecast_col in forecast_df.columns:
                    y_values.extend(forecast_df[forecast_col].dropna().tolist())
                if upper_bound_col in forecast_df.columns:
                    y_values.extend(forecast_df[upper_bound_col].dropna().tolist())
                
                # Set y-axis range with 10% padding
                y_min = max(0, min(y_values) * 0.9) if y_values else 0
                y_max = max(y_values) * 1.1 if y_values else 100
                
                # Enhanced layout with better time series controls
                fig.update_layout(
                    title=dict(
                        text=f'Sales History + {forecast_model.upper()} Forecast for {product_name}',
                        font=dict(size=16)
                    ),
                    xaxis_title='Date',
                    yaxis_title='Units Sold',
                    legend_title='Data Series',
                    height=chart_height,
                    yaxis=dict(
                        range=[y_min, y_max],
                        title_font=dict(size=14),
                        tickfont=dict(size=12),
                        gridwidth=1,
                        gridcolor='rgba(128,128,128,0.2)'
                    ),
                    xaxis=dict(
                        rangeslider=dict(visible=True),
                        type='date',
                        tickformat='%b %d, %Y',  # Improved date format (Aug 01, 2025)
                        title_font=dict(size=14),
                        tickfont=dict(size=12),
                        gridwidth=1,
                        gridcolor='rgba(128,128,128,0.2)',
                        rangeselector=dict(
                            buttons=list([
                                dict(count=7, label='7d', step='day', stepmode='backward'),
                                dict(count=14, label='2w', step='day', stepmode='backward'),
                                dict(count=30, label='1m', step='day', stepmode='backward'),
                                dict(count=90, label='3m', step='day', stepmode='backward'),
                                dict(count=365, label='1y', step='day', stepmode='backward'),
                                dict(step='all', label='All')
                            ]),
                            font=dict(size=12),
                            bgcolor='rgba(150,150,150,0.1)'
                        )
                    ),
                    hovermode='x unified',
                    plot_bgcolor='rgba(250,250,250,1)',  # Slightly off-white for better contrast
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                # Add grid for better readability
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                
                return jsonify(plotly.io.to_json(fig))
        
        # Fallback to sample data if no real data
        data = generate_sample_data()
        forecast_data = data['forecast']
        
        # Create a line chart for forecast data with simulated history
        fig = go.Figure()
        
        # Generate some historical data
        today = datetime.strptime('2025-08-01', '%Y-%m-%d')
        historical_dates = pd.date_range(end=today - timedelta(days=1), periods=60)
        
        # Simulate historical sales with some seasonality
        np.random.seed(42)  # For reproducibility
        base_value = 25  # Base value for sales
        sales_noise = 0.3  # Noise level
        
        # Create sales with weekly seasonality
        historical_sales = []
        for date in historical_dates:
            # Weekly pattern: higher on weekends
            day_factor = 1.0 + 0.5 * (1 if date.dayofweek >= 5 else 0) 
            # Monthly pattern
            month_factor = 1.0 + 0.2 * np.sin(2 * np.pi * date.day / 30)
            # Add trend
            trend_factor = 1.0 + 0.001 * (date - historical_dates[0]).days
            # Combine factors
            value = base_value * day_factor * month_factor * trend_factor
            # Add noise
            value *= np.random.normal(1, sales_noise)
            historical_sales.append(max(0, value))
        
        # Add historical sales to chart
        fig.add_trace(go.Scatter(
            x=historical_dates,
            y=historical_sales,
            mode='lines',
            name='Historical Sales',
            line=dict(color='darkblue', width=2)
        ))
        
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=forecast_data['Date'],
            y=forecast_data['Forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='orange')
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_data['Date'],
            y=forecast_data['Upper_Bound'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_data['Date'],
            y=forecast_data['Lower_Bound'],
            mode='lines',
            name='Lower Bound',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 165, 0, 0.2)',  # Orange with transparency
            showlegend=False
        ))
        
        product_name = forecast_data['Product'].iloc[0] if 'Product' in forecast_data.columns else f'Product {product_id}'
        
        # Calculate dynamic y-axis range
        all_y_values = historical_sales + forecast_data['Forecast'].tolist() + forecast_data['Upper_Bound'].tolist()
        y_min = max(0, min(all_y_values) * 0.9)
        y_max = max(all_y_values) * 1.1
        
        fig.update_layout(
            title=f'Sales History + Forecast for {product_name} (Sample Data)',
            xaxis_title='Date',
            yaxis_title='Units',
            legend_title='Legend',
            height=600,
            yaxis=dict(range=[y_min, y_max])
        )
        
        return jsonify(plotly.io.to_json(fig))
    except Exception as e:
        # Generate detailed error with traceback
        import traceback
        error_stack = traceback.format_exc()
        print(f"Error in forecast chart: {str(e)}\n{error_stack}")
        
        # Return detailed error message
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error creating forecast chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.6, showarrow=False
        )
        
        # Add traceback information
        tb_lines = error_stack.split('\n')
        file_line_info = [line for line in tb_lines if 'File "' in line]
        if file_line_info:
            error_fig.add_annotation(
                text=file_line_info[-1].strip(),
                xref="paper", yref="paper",
                x=0.5, y=0.4, showarrow=False,
                font=dict(size=10)
            )
            
        return jsonify(plotly.io.to_json(error_fig))

@app.route('/api/model-comparison-data')
def model_comparison_data():
    """API endpoint to compare multiple forecast models side by side."""
    try:
        # Get parameters
        store_id = request.args.get('store_id', '104.0')
        product_id = request.args.get('product_id', '3913116850.0')
        models = request.args.get('models', 'rf,arima,pytorch').split(',')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Load historical data
        combined_data = get_data('combined_data')
        historical_df = None
        if combined_data is not None:
            historical_df = combined_data[
                (combined_data['Store_ID'].astype(str) == store_id) &
                (combined_data['Product_ID'].astype(str) == product_id)
            ].copy()
            if 'Date' in historical_df.columns:
                historical_df['Date'] = pd.to_datetime(historical_df['Date'])
                historical_df = historical_df.sort_values('Date')
        
        # Create comparison chart
        fig = go.Figure()
        
        # Add historical data
        if historical_df is not None and len(historical_df) > 0:
            sales_col = 'Units_Sold' if 'Units_Sold' in historical_df.columns else 'Sales'
            fig.add_trace(go.Scatter(
                x=historical_df['Date'],
                y=historical_df[sales_col],
                mode='lines+markers',
                name='Historical Sales',
                line=dict(color='darkblue', width=2),
                marker=dict(size=4)
            ))
        
        # Color palette for different models
        colors = {'rf': 'orange', 'arima': 'green', 'pytorch': 'red'}
        
        # Add forecast data for each model
        for model in models:
            if model.strip():
                forecast_type = f"{model.strip()}_forecasts"
                forecast_df = get_data(forecast_type)
                
                if forecast_df is not None:
                    # Filter for specific store/product
                    model_data = forecast_df[
                        (forecast_df['Store_ID'].astype(str) == store_id) &
                        (forecast_df['Product_ID'].astype(str) == product_id)
                    ].copy()
                    
                    if len(model_data) > 0:
                        # Determine forecast column
                        forecast_col = 'Forecast'
                        if model == 'pytorch' and 'Predicted_Demand' in model_data.columns:
                            forecast_col = 'Predicted_Demand'
                        
                        if 'Date' in model_data.columns and forecast_col in model_data.columns:
                            model_data['Date'] = pd.to_datetime(model_data['Date'])
                            model_data = model_data.sort_values('Date')
                            
                            fig.add_trace(go.Scatter(
                                x=model_data['Date'],
                                y=model_data[forecast_col],
                                mode='lines+markers',
                                name=f'{model.upper()} Forecast',
                                line=dict(color=colors.get(model, 'gray'), width=2),
                                marker=dict(size=5, symbol='diamond')
                            ))
        
        # Enhanced layout
        fig.update_layout(
            title=dict(
                text=f'Model Comparison - Store {store_id}, Product {product_id}',
                font=dict(size=16)
            ),
            xaxis_title='Date',
            yaxis_title='Units Sold',
            legend_title='Models',
            height=700,
            hovermode='x unified',
            xaxis=dict(
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label='7d', step='day', stepmode='backward'),
                        dict(count=14, label='2w', step='day', stepmode='backward'),
                        dict(count=30, label='1m', step='day', stepmode='backward'),
                        dict(count=90, label='3m', step='day', stepmode='backward'),
                        dict(step='all')
                    ])
                )
            )
        )
        
        return jsonify(plotly.io.to_json(fig))
        
    except Exception as e:
        import traceback
        print(f"Error in model comparison: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)})

@app.route('/api/export-forecast-data')
def export_forecast_data():
    """API endpoint to export time series forecast data as CSV/JSON."""
    try:
        # Get parameters
        store_id = request.args.get('store_id', '104.0')
        product_id = request.args.get('product_id', '3913116850.0')
        model = request.args.get('model', 'rf')
        format_type = request.args.get('format', 'csv').lower()
        
        # Load forecast data
        forecast_type = f"{model}_forecasts"
        forecast_df = get_data(forecast_type)
        
        # Load historical data
        combined_data = get_data('combined_data')
        
        export_data = []
        
        # Add historical data
        if combined_data is not None:
            historical = combined_data[
                (combined_data['Store_ID'].astype(str) == store_id) &
                (combined_data['Product_ID'].astype(str) == product_id)
            ].copy()
            
            for _, row in historical.iterrows():
                export_data.append({
                    'Date': row.get('Date', ''),
                    'Store_ID': store_id,
                    'Product_ID': product_id,
                    'Type': 'Historical',
                    'Value': row.get('Units_Sold', row.get('Sales', 0)),
                    'Model': 'Actual'
                })
        
        # Add forecast data
        if forecast_df is not None:
            forecast = forecast_df[
                (forecast_df['Store_ID'].astype(str) == store_id) &
                (forecast_df['Product_ID'].astype(str) == product_id)
            ].copy()
            
            forecast_col = 'Forecast'
            if model == 'pytorch' and 'Predicted_Demand' in forecast.columns:
                forecast_col = 'Predicted_Demand'
            
            for _, row in forecast.iterrows():
                export_data.append({
                    'Date': row.get('Date', ''),
                    'Store_ID': store_id,
                    'Product_ID': product_id,
                    'Type': 'Forecast',
                    'Value': row.get(forecast_col, 0),
                    'Model': model.upper(),
                    'Upper_Bound': row.get('Upper_Bound', ''),
                    'Lower_Bound': row.get('Lower_Bound', '')
                })
        
        if format_type == 'json':
            return jsonify(export_data)
        else:
            # Return CSV format
            import io
            df = pd.DataFrame(export_data)
            output = io.StringIO()
            df.to_csv(output, index=False)
            csv_data = output.getvalue()
            
            return csv_data, 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': f'attachment; filename=forecast_export_{store_id}_{product_id}_{model}.csv'
            }
            
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/elasticity-data')
def elasticity_data():
    """API endpoint to get elasticity distribution data."""
    try:
        # Get parameters
        store_id = request.args.get('store_id', '104.0')
        valid_only = request.args.get('valid_only', 'false').lower() == 'true'
        
        # Load elasticity data
        elasticity_df = get_data('elasticity')
        
        if elasticity_df is not None:
            # Filter by store
            elasticity_df = elasticity_df[elasticity_df['Store_Id'] == float(store_id)]
            
            # Filter to only valid elasticities if requested
            if valid_only:
                elasticity_df = elasticity_df[elasticity_df['Is_Significant'] == True]
        else:
            # Fallback to sample data
            data = generate_sample_data()
            elasticity_df = data['elasticity']
            
            # Filter by store
            elasticity_df = elasticity_df[elasticity_df['Store_Id'] == float(store_id)]
            
            # Filter by validity if requested
            if valid_only:
                elasticity_df = elasticity_df[elasticity_df['Is_Significant'] == True]
        
        # Create elasticity distribution plot
        fig = create_elasticity_distribution_plot(elasticity_df)
        
        return jsonify(plotly.io.to_json(fig))
    except Exception as e:
        # Return error message
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error loading elasticity data: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return jsonify(plotly.io.to_json(error_fig))

@app.route('/api/price-sensitivity')
def price_sensitivity():
    """API endpoint to get price sensitivity curve."""
    try:
        # Get parameters from request
        store_id = request.args.get('store_id', '104.0')
        product_id = request.args.get('product_id', '3913116850.0')
        # Add parameter to toggle display of product name in chart
        show_product_name = request.args.get('show_product_name', 'true').lower() == 'true'
        
        # Load elasticity data
        elasticity_df = get_data('elasticity')
        
        if elasticity_df is not None:
            # Filter by store and product
            product_data = elasticity_df[(elasticity_df['Store_Id'] == float(store_id)) & 
                                         (elasticity_df['Item'] == float(product_id))]
            
            if not product_data.empty:
                product = product_data.iloc[0]
                
                # Apply constraint if needed
                constraints = None
                if request.args.get('apply_constraints') == 'true':
                    constraints = {
                        'max_price_increase': float(request.args.get('max_increase', 10)),
                        'max_price_decrease': float(request.args.get('max_decrease', 10)),
                        'elasticity_constraint': float(request.args.get('elasticity_constraint', 5))
                    }
                
                # Get product name if needed
                product_name = None
                if show_product_name and 'Product' in product:
                    product_name = product['Product']
                
                # Create price sensitivity curve
                fig = create_price_sensitivity_curve(
                    elasticity=product['Elasticity'],
                    current_price=product['Current_Price'],
                    cost=product['Cost'],
                    constraints=constraints,
                    product_name=product_name
                )
                
                return jsonify(plotly.io.to_json(fig))
        
        # Fallback to sample data if no real data or matching product
        data = generate_sample_data()
        sample_elasticity = data['elasticity']
        
        # Get parameter to toggle display of product name in chart
        show_product_name = request.args.get('show_product_name', 'true').lower() == 'true'
        
        # Find the specific product in our sample data
        product_data = sample_elasticity[(sample_elasticity['Store_Id'] == float(store_id)) & 
                                         (sample_elasticity['Item'] == float(product_id))]
        
        if product_data.empty:
            # Use first product if no match
            product = sample_elasticity.iloc[0]
        else:
            product = product_data.iloc[0]
        
        # Apply constraint if needed
        constraints = None
        if request.args.get('apply_constraints') == 'true':
            constraints = {
                'max_price_increase': float(request.args.get('max_increase', 10)),
                'max_price_decrease': float(request.args.get('max_decrease', 10)),
                'elasticity_constraint': float(request.args.get('elasticity_constraint', 5))
            }
        
        # Get product name if needed
        product_name = None
        if show_product_name and 'Product' in product:
            product_name = product['Product']
        
        # Create price sensitivity curve
        fig = create_price_sensitivity_curve(
            elasticity=product['Elasticity'],
            current_price=product['Current_Price'],
            cost=product['Cost'],
            constraints=constraints,
            product_name=product_name
        )
        
        return jsonify(plotly.io.to_json(fig))
    except Exception as e:
        # Return error message
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error generating price sensitivity curve: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return jsonify(plotly.io.to_json(error_fig))

@app.route('/api/profit-impact')
def profit_impact():
    """API endpoint to get profit impact waterfall."""
    try:
        # Get parameters from request
        store_id = request.args.get('store_id', '104.0')
        limit = int(request.args.get('limit', '10'))
        strategy = request.args.get('strategy', 'optimal')  # For future use with different pricing strategies
        # Add parameter to toggle display of product names
        show_product_names = request.args.get('show_product_names', 'true').lower() == 'true'
        
        # Load profit impact data
        profit_df = get_data('profit_impact')
        
        if profit_df is not None and not profit_df.empty:
            # Filter by store
            profit_df = profit_df[profit_df['Store_Id'] == float(store_id)]
            
            if not profit_df.empty:
                # Check if we have all required columns
                required_columns = ['Product', 'Total_Current_Profit', 'Total_New_Profit', 
                                   'Total_Profit_Difference', 'Price_Change_Pct']
                
                if all(col in profit_df.columns for col in required_columns):
                    # If product names should be hidden, replace with generic names
                    if not show_product_names:
                        # Create a copy of the DataFrame to avoid modifying the original
                        profit_df = profit_df.copy()
                        # Replace product names with generic identifiers
                        profit_df['Product'] = ['Product ' + str(i+1) for i in range(len(profit_df))]
                    
                    # Create profit impact waterfall with actual data
                    fig = create_profit_impact_waterfall(profit_df, limit)
                    return jsonify(plotly.io.to_json(fig))
        
        # Fallback to sample data if no real data available
        data = generate_sample_data()
        profit_df = data['profit_impact']
        
        # Filter by store
        profit_df = profit_df[profit_df['Store_Id'] == float(store_id)]
        
        # Get parameter to toggle display of product names
        show_product_names = request.args.get('show_product_names', 'true').lower() == 'true'
        
        # If product names should be hidden, replace with generic names
        if not show_product_names:
            # Create a copy of the DataFrame to avoid modifying the original
            profit_df = profit_df.copy()
            # Replace product names with generic identifiers
            profit_df['Product'] = ['Product ' + str(i+1) for i in range(len(profit_df))]
        
        # Create profit impact waterfall with sample data
        fig = create_profit_impact_waterfall(profit_df, limit)
        
        return jsonify(plotly.io.to_json(fig))
    except Exception as e:
        # Return error message
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error generating profit impact chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return jsonify(plotly.io.to_json(error_fig))

@app.route('/api/rf-forecast-data')
def rf_forecast_data():
    """API endpoint to get Random Forest forecast data."""
    try:
        # Try to load the RF forecasts from file
        rf_forecasts_path = os.path.join('data', 'processed', 'rf_forecasts.csv')
        if os.path.exists(rf_forecasts_path):
            df = pd.read_csv(rf_forecasts_path)
            
            # Get store and product from query parameters or use defaults
            store_id = request.args.get('store_id', '104.0')
            product_id = request.args.get('product_id', '3913116850.0')
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            
            # Filter data
            df = df[(df['Store_Id'] == float(store_id)) & (df['Item'] == float(product_id))]
            
            # Apply date filters if provided
            if start_date:
                df = df[df['Date'] >= start_date]
            if end_date:
                df = df[df['Date'] <= end_date]
            
            # Convert dates if they aren't already
            if not pd.api.types.is_datetime64_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'])
                
            # Sort by date
            df = df.sort_values('Date')
            
            # Create figure
            fig = go.Figure()
            
            # Add forecast line
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Forecast'],
                mode='lines',
                name='RF Forecast',
                line=dict(color='green', width=2)
            ))
            
            # Add confidence interval if available
            if 'Lower_Bound' in df.columns and 'Upper_Bound' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Upper_Bound'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Lower_Bound'],
                    mode='lines',
                    name='Lower Bound',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(0, 128, 0, 0.2)',
                    showlegend=False
                ))
            
            # Try to get product name (with empty DataFrame check)
            product_name = df['Product'].iloc[0] if 'Product' in df.columns and not df.empty else f'Product {product_id}'
            
            # Update layout
            fig.update_layout(
                title=f'Random Forest Sales Forecast for {product_name} (Store {store_id})',
                xaxis_title='Date',
                yaxis_title='Forecast Units',
                legend_title='Legend',
                height=600
            )
            
            return jsonify(plotly.io.to_json(fig))
        else:
            # Use sample data if file doesn't exist
            data = generate_sample_data()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data['forecast']['Date'],
                y=data['forecast']['Forecast'] * 0.9,  # Slight difference from regular forecast
                mode='lines',
                name='RF Forecast',
                line=dict(color='green', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=data['forecast']['Date'],
                y=data['forecast']['Upper_Bound'] * 0.9,
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=data['forecast']['Date'],
                y=data['forecast']['Lower_Bound'] * 0.9,
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0, 128, 0, 0.2)',
                showlegend=False
            ))
            
            fig.update_layout(
                title='Random Forest Sales Forecast with Confidence Intervals',
                xaxis_title='Date',
                yaxis_title='Sales Units',
                legend_title='Legend',
                height=600
            )
            
            return jsonify(plotly.io.to_json(fig))
    except Exception as e:
        # Return error message
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error loading RF forecast data: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return jsonify(plotly.io.to_json(error_fig))

@app.route('/api/arima-forecast-data')
def arima_forecast_data():
    """API endpoint to get ARIMA forecast data."""
    try:
        # Try to load the ARIMA forecasts from file
        arima_forecasts_path = os.path.join('data', 'processed', 'arima_forecasts.csv')
        if os.path.exists(arima_forecasts_path):
            df = pd.read_csv(arima_forecasts_path)
            
            # Get store and product from query parameters or use defaults
            store_id = request.args.get('store_id', '104.0')
            product_id = request.args.get('product_id', '3913116850.0')
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            
            # Filter data
            df = df[(df['Store_Id'] == float(store_id)) & (df['Item'] == float(product_id))]
            
            # Check if the DataFrame is empty after filtering
            if df.empty:
                print(f"Warning: No ARIMA forecast data found for Store ID {store_id} and Product ID {product_id}. Will use fallback data.")
            
            # Apply date filters if provided (only if DataFrame is not empty)
            if not df.empty:
                if start_date:
                    df = df[df['Date'] >= start_date]
                if end_date:
                    df = df[df['Date'] <= end_date]
            
            # Convert dates if they aren't already (only if DataFrame is not empty)
            if not df.empty and not pd.api.types.is_datetime64_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'])
                
            # Sort by date (only if DataFrame is not empty)
            if not df.empty:
                df = df.sort_values('Date')
            
            # Create figure
            fig = go.Figure()
            
            # Only add data traces if DataFrame is not empty
            if not df.empty:
                # Add forecast line
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Forecast'],
                    mode='lines',
                    name='ARIMA Forecast',
                    line=dict(color='red', width=2)
                ))
            
            # Add confidence interval if available (only if DataFrame is not empty)
            if not df.empty and 'Lower_Bound' in df.columns and 'Upper_Bound' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Upper_Bound'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Lower_Bound'],
                    mode='lines',
                    name='Lower Bound',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    showlegend=False
                ))
            
            # Try to get product name (with empty DataFrame check)
            product_name = df['Product'].iloc[0] if 'Product' in df.columns and not df.empty else f'Product {product_id}'
            
            # Update layout
            fig.update_layout(
                title=f'ARIMA Sales Forecast for {product_name} (Store {store_id})',
                xaxis_title='Date',
                yaxis_title='Forecast Units',
                legend_title='Legend',
                height=600
            )
            
            return jsonify(plotly.io.to_json(fig))
        else:
            # Use sample data if file doesn't exist
            data = generate_sample_data()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data['forecast']['Date'],
                y=data['forecast']['Forecast'] * 1.1,  # Slight difference from regular forecast
                mode='lines',
                name='ARIMA Forecast',
                line=dict(color='red', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=data['forecast']['Date'],
                y=data['forecast']['Upper_Bound'] * 1.1,
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=data['forecast']['Date'],
                y=data['forecast']['Lower_Bound'] * 1.1,
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.2)',
                showlegend=False
            ))
            
            fig.update_layout(
                title='ARIMA Sales Forecast with Confidence Intervals',
                xaxis_title='Date',
                yaxis_title='Sales Units',
                legend_title='Legend',
                height=600
            )
            
            return jsonify(plotly.io.to_json(fig))
    except Exception as e:
        # Return error message
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error loading ARIMA forecast data: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return jsonify(plotly.io.to_json(error_fig))

@app.route('/api/pytorch-forecast-data')
def pytorch_forecast_data():
    """API endpoint to get PyTorch forecast data."""
    try:
        # Try to load the PyTorch forecasts from file
        pytorch_forecasts_path = os.path.join('data', 'processed', 'pytorch_forecasts.csv')
        if os.path.exists(pytorch_forecasts_path):
            df = pd.read_csv(pytorch_forecasts_path)
            
            # Get store and product from query parameters or use defaults
            store_id = request.args.get('store_id', '104.0')
            product_id = request.args.get('product_id', '3913116850.0')
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            
            # Filter data
            df = df[(df['Store_Id'] == float(store_id)) & (df['Item'] == float(product_id))]
            
            # Apply date filters if provided
            if start_date:
                df = df[df['Date'] >= start_date]
            if end_date:
                df = df[df['Date'] <= end_date]
            
            # Convert dates if they aren't already
            if not pd.api.types.is_datetime64_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'])
                
            # Sort by date
            df = df.sort_values('Date')
            
            # Create figure
            fig = go.Figure()
            
            # Add forecast line
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Forecast'],
                mode='lines',
                name='PyTorch Forecast',
                line=dict(color='blue', width=2)
            ))
            
            # Add confidence interval if available
            if 'Lower_Bound' in df.columns and 'Upper_Bound' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Upper_Bound'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Lower_Bound'],
                    mode='lines',
                    name='Lower Bound',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(0, 0, 255, 0.2)',
                    showlegend=False
                ))
            
            # Try to get product name (with empty DataFrame check)
            product_name = df['Product'].iloc[0] if 'Product' in df.columns and not df.empty else f'Product {product_id}'
            
            # Update layout
            fig.update_layout(
                title=f'PyTorch Sales Forecast for {product_name} (Store {store_id})',
                xaxis_title='Date',
                yaxis_title='Forecast Units',
                legend_title='Legend',
                height=600
            )
            
            return jsonify(plotly.io.to_json(fig))
        else:
            # Use sample data if file doesn't exist
            data = generate_sample_data()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data['forecast']['Date'],
                y=data['forecast']['Forecast'] * 1.05,  # Slight difference from regular forecast
                mode='lines',
                name='PyTorch Forecast',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=data['forecast']['Date'],
                y=data['forecast']['Upper_Bound'] * 1.05,
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=data['forecast']['Date'],
                y=data['forecast']['Lower_Bound'] * 1.05,
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0, 0, 255, 0.2)',
                showlegend=False
            ))
            
            fig.update_layout(
                title='PyTorch Sales Forecast with Confidence Intervals',
                xaxis_title='Date',
                yaxis_title='Sales Units',
                legend_title='Legend',
                height=600
            )
            
            return jsonify(plotly.io.to_json(fig))
    except Exception as e:
        # Return error message
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error loading PyTorch forecast data: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return jsonify(plotly.io.to_json(error_fig))

@app.route('/api/model-comparison-chart')
def model_comparison_chart():
    """API endpoint to get model comparison data."""
    try:
        # Try to load all forecast files
        rf_path = os.path.join('data', 'processed', 'rf_forecasts.csv')
        pytorch_path = os.path.join('data', 'processed', 'pytorch_forecasts.csv')
        arima_path = os.path.join('data', 'processed', 'arima_forecasts.csv')
        
        # Initialize dicts to store forecasts
        forecasts = {}
        
        # Get store and product from query parameters or use defaults
        store_id = request.args.get('store_id', '104.0')
        product_id = request.args.get('product_id', '3913116850.0')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Load RF forecasts if available
        if os.path.exists(rf_path):
            rf_df = pd.read_csv(rf_path)
            rf_df = rf_df[(rf_df['Store_Id'] == float(store_id)) & (rf_df['Item'] == float(product_id))]
            if not rf_df.empty:
                if 'Date' in rf_df.columns and not pd.api.types.is_datetime64_dtype(rf_df['Date']):
                    rf_df['Date'] = pd.to_datetime(rf_df['Date'])
                # Apply date filters if provided
                if start_date:
                    rf_df = rf_df[rf_df['Date'] >= start_date]
                if end_date:
                    rf_df = rf_df[rf_df['Date'] <= end_date]
                forecasts['rf'] = rf_df.sort_values('Date')
        
        # Load PyTorch forecasts if available
        if os.path.exists(pytorch_path):
            pytorch_df = pd.read_csv(pytorch_path)
            pytorch_df = pytorch_df[(pytorch_df['Store_Id'] == float(store_id)) & (pytorch_df['Item'] == float(product_id))]
            if not pytorch_df.empty:
                if 'Date' in pytorch_df.columns and not pd.api.types.is_datetime64_dtype(pytorch_df['Date']):
                    pytorch_df['Date'] = pd.to_datetime(pytorch_df['Date'])
                # Apply date filters if provided
                if start_date:
                    pytorch_df = pytorch_df[pytorch_df['Date'] >= start_date]
                if end_date:
                    pytorch_df = pytorch_df[pytorch_df['Date'] <= end_date]
                forecasts['pytorch'] = pytorch_df.sort_values('Date')
        
        # Load ARIMA forecasts if available
        if os.path.exists(arima_path):
            arima_df = pd.read_csv(arima_path)
            arima_df = arima_df[(arima_df['Store_Id'] == float(store_id)) & (arima_df['Item'] == float(product_id))]
            if not arima_df.empty:
                if 'Date' in arima_df.columns and not pd.api.types.is_datetime64_dtype(arima_df['Date']):
                    arima_df['Date'] = pd.to_datetime(arima_df['Date'])
                # Apply date filters if provided
                if start_date:
                    arima_df = arima_df[arima_df['Date'] >= start_date]
                if end_date:
                    arima_df = arima_df[arima_df['Date'] <= end_date]
                forecasts['arima'] = arima_df.sort_values('Date')
        
        if not forecasts:
            # Use sample data if no real forecasts available
            data = generate_sample_data()
            
            # Create figure
            fig = go.Figure()
            
            # Add sample forecasts with slight variations
            fig.add_trace(go.Scatter(
                x=data['forecast']['Date'],
                y=data['forecast']['Forecast'] * 0.9,
                mode='lines',
                name='Random Forest',
                line=dict(color='green', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=data['forecast']['Date'],
                y=data['forecast']['Forecast'] * 1.1,
                mode='lines',
                name='ARIMA',
                line=dict(color='red', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=data['forecast']['Date'],
                y=data['forecast']['Forecast'] * 1.05,
                mode='lines',
                name='PyTorch',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title='Forecast Model Comparison (Sample Data)',
                xaxis_title='Date',
                yaxis_title='Sales Units',
                legend_title='Models',
                height=600
            )
            
            return jsonify(plotly.io.to_json(fig))
        
        # Create figure with actual forecast data
        fig = go.Figure()
        
        # Add each forecast model to the chart with a different color
        model_colors = {'rf': 'green', 'arima': 'red', 'pytorch': 'blue'}
        model_names = {'rf': 'Random Forest', 'arima': 'ARIMA', 'pytorch': 'PyTorch'}
        
        for model, df in forecasts.items():
            if 'Forecast' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Forecast'],
                    mode='lines',
                    name=model_names.get(model, model),
                    line=dict(color=model_colors.get(model, 'gray'), width=2)
                ))
        
        # Try to get product name
        product_name = None
        for model, df in forecasts.items():
            if 'Product' in df.columns and not df.empty:
                product_name = df['Product'].iloc[0]
                break
        
        if not product_name:
            product_name = f'Product {product_id}'
        
        # Update layout
        fig.update_layout(
            title=f'Forecast Model Comparison for {product_name} (Store {store_id})',
            xaxis_title='Date',
            yaxis_title='Forecast Units',
            legend_title='Models',
            height=600
        )
        
        return jsonify(plotly.io.to_json(fig))
    except Exception as e:
        # Return error message
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error creating model comparison chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return jsonify(plotly.io.to_json(error_fig))

@app.route('/api/detailed-forecast-data')
def detailed_forecast_data():
    """API endpoint to get detailed forecast data."""
    try:
        # Get parameters from request
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        store_id = request.args.get('store_id', '104.0')
        product_id = request.args.get('product_id', '3913116850.0')
        forecast_model = request.args.get('forecast_model', 'rf')
        
        # Paths to forecast data files
        forecast_paths = {
            'rf': os.path.join('data', 'processed', 'rf_forecasts.csv'),
            'pytorch': os.path.join('data', 'processed', 'pytorch_forecasts.csv'),
            'arima': os.path.join('data', 'processed', 'arima_forecasts.csv')
        }
        
        # Check if data file exists
        if forecast_model in forecast_paths and os.path.exists(forecast_paths[forecast_model]):
            # Load and filter data
            forecast_df = pd.read_csv(forecast_paths[forecast_model])
            forecast_df = forecast_df[(forecast_df['Store_Id'] == float(store_id)) & 
                                    (forecast_df['Item'] == float(product_id))]
            
            # Convert dates if needed
            if not pd.api.types.is_datetime64_dtype(forecast_df['Date']):
                forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
            
            # Filter by date range if provided
            if start_date:
                start_date = pd.to_datetime(start_date)
                forecast_df = forecast_df[forecast_df['Date'] >= start_date]
                
            if end_date:
                end_date = pd.to_datetime(end_date)
                forecast_df = forecast_df[forecast_df['Date'] <= end_date]
                
            # Sort by date
            forecast_df = forecast_df.sort_values('Date')
            
            # Create detailed forecast figure
            fig = go.Figure()
            
            # Add main forecast line
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Forecast'],
                mode='lines',
                name=f'{forecast_model.upper()} Forecast',
                line=dict(color='blue', width=2)
            ))
            
            # Add confidence intervals if available
            if 'Upper_Bound' in forecast_df.columns and 'Lower_Bound' in forecast_df.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Upper_Bound'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Lower_Bound'],
                    mode='lines',
                    name='Lower Bound',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(0, 0, 255, 0.2)',
                    showlegend=False
                ))
            
            # Add actual sales if available
            if 'Actual' in forecast_df.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Actual'],
                    mode='lines+markers',
                    name='Actual Sales',
                    line=dict(color='green', width=2)
                ))
            
            # Get product name if available
            product_name = forecast_df['Product'].iloc[0] if 'Product' in forecast_df.columns and len(forecast_df) > 0 else f'Product {product_id}'
            
            # Update layout
            fig.update_layout(
                title=f'Detailed Sales Forecast for {product_name} (Store {store_id})',
                xaxis_title='Date',
                yaxis_title='Units',
                legend_title='Data',
                height=600
            )
            
            return jsonify(plotly.io.to_json(fig))
        else:
            # Create sample data if file doesn't exist
            data = generate_sample_data()
            
            # Create detailed forecast figure with sample data
            fig = go.Figure()
            
            # Add main forecast line
            fig.add_trace(go.Scatter(
                x=data['forecast']['Date'],
                y=data['forecast']['Forecast'],
                mode='lines',
                name='Forecast',
                line=dict(color='blue', width=2)
            ))
            
            # Add confidence intervals
            fig.add_trace(go.Scatter(
                x=data['forecast']['Date'],
                y=data['forecast']['Upper_Bound'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=data['forecast']['Date'],
                y=data['forecast']['Lower_Bound'],
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0, 0, 255, 0.2)',
                showlegend=False
            ))
            
            # Add actual sales
            fig.add_trace(go.Scatter(
                x=data['forecast']['Date'],
                y=data['forecast']['Actual'],
                mode='lines+markers',
                name='Actual Sales',
                line=dict(color='green', width=2)
            ))
            
            # Add annotations for key insights
            max_forecast = max(data['forecast']['Forecast'])
            max_date = data['forecast']['Date'][data['forecast']['Forecast'].argmax()]
            
            fig.add_annotation(
                x=max_date,
                y=max_forecast,
                text=f"Peak Demand: {max_forecast:.1f}",
                showarrow=True,
                arrowhead=1
            )
            
            # Update layout
            fig.update_layout(
                title='Detailed Sales Forecast with Insights (Sample Data)',
                xaxis_title='Date',
                yaxis_title='Units',
                legend_title='Data',
                height=600
            )
            
            return jsonify(plotly.io.to_json(fig))
            
    except Exception as e:
        # Generate detailed error with traceback
        import traceback
        error_stack = traceback.format_exc()
        print(f"Error in detailed forecast: {str(e)}\n{error_stack}")
        
        # Return detailed error message
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error creating detailed forecast: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.6, showarrow=False
        )
        
        # Add file and line information
        tb_lines = error_stack.split('\n')
        file_line_info = [line for line in tb_lines if 'File "' in line]
        if file_line_info:
            error_fig.add_annotation(
                text=file_line_info[-1].strip(),
                xref="paper", yref="paper",
                x=0.5, y=0.4, showarrow=False,
                font=dict(size=10)
            )
            
        return jsonify(plotly.io.to_json(error_fig))

@app.route('/api/product-dropdown-data')
def product_dropdown_data():
    """API endpoint to get product data for dropdowns"""
    try:
        # Get store ID from request parameters (default to '104.0')
        store_id = request.args.get('store_id', '104.0')
        
        # Load the combined pizza data
        combined_data = get_data('combined_data')
        
        if combined_data is not None:
            # Filter by store if specified
            if store_id != 'all':
                combined_data = combined_data[combined_data['Store_Id'] == float(store_id)]
            
            # Get unique products with their IDs
            products = combined_data[['Item', 'Product', 'Size']].drop_duplicates()
            
            # Sort alphabetically by product name
            products = products.sort_values('Product')
            
            # Convert to list of dictionaries
            product_list = [{
                'id': str(row['Item']),
                'name': row['Product'],
                'size': row['Size']
            } for _, row in products.iterrows()]
            
            return jsonify({'products': product_list})
        else:
            # Fallback to sample data
            sample_products = [
                {'id': '3913116850.0', 'name': 'BELLATORIA BBQ CHK PIZZA', 'size': '15.51 OZ'},
                {'id': '3913116852.0', 'name': 'BELLATORIA ULT PEPPERONI PIZZA', 'size': '15.51 OZ'},
                {'id': '4219700201.0', 'name': 'BREW PUB PEPPERONI PIZZA', 'size': '23 OZ'},
                {'id': '4280011400.0', 'name': 'TOTINO PEPPERONI PIZZA', 'size': '9.8 OZ'},
                {'id': '7218063473.0', 'name': 'RED BARON PEPPERONI PIZZA', 'size': '20.6 OZ'}
            ]
            return jsonify({'products': sample_products})
    except Exception as e:
        # Return error message
        return jsonify({
            'error': str(e),
            'products': [{'id': '3913116850.0', 'name': 'BELLATORIA BBQ CHK PIZZA', 'size': '15.51 OZ'}]
        })

@app.route('/api/store-dropdown-data')
def store_dropdown_data():
    """API endpoint to get store data for dropdowns"""
    try:
        # Load the combined pizza data
        combined_data = get_data('combined_data')
        
        if combined_data is not None:
            # Get unique store IDs
            stores = combined_data['Store_Id'].unique()
            
            # Convert to list of dictionaries
            store_list = [{
                'id': str(store_id),
                'name': f'Store {store_id}'
            } for store_id in sorted(stores)]
            
            return jsonify({'stores': store_list})
        else:
            # Fallback to sample data
            sample_stores = [{'id': '104.0', 'name': 'Store 104'}]
            return jsonify({'stores': sample_stores})
    except Exception as e:
        # Return error message
        return jsonify({
            'error': str(e),
            'stores': [{'id': '104.0', 'name': 'Store 104'}]
        })

@app.route('/api/forecast-distribution-data')
def forecast_distribution_data():
    """API endpoint to get forecast distribution data."""
    try:
        # Get parameters from request
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        store_id = request.args.get('store_id', '104.0')
        product_id = request.args.get('product_id', '3913116850.0')
        forecast_model = request.args.get('forecast_model', 'rf')
        
        # Paths to forecast data files
        forecast_paths = {
            'rf': os.path.join('data', 'processed', 'rf_forecasts.csv'),
            'pytorch': os.path.join('data', 'processed', 'pytorch_forecasts.csv'),
            'arima': os.path.join('data', 'processed', 'arima_forecasts.csv')
        }
        
        # Check if data file exists
        if forecast_model in forecast_paths and os.path.exists(forecast_paths[forecast_model]):
            # Load and filter data
            forecast_df = pd.read_csv(forecast_paths[forecast_model])
            forecast_df = forecast_df[(forecast_df['Store_Id'] == float(store_id)) & 
                                    (forecast_df['Item'] == float(product_id))]
            
            # Convert dates if needed
            if not pd.api.types.is_datetime64_dtype(forecast_df['Date']):
                forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
            
            # Filter by date range if provided
            if start_date:
                start_date = pd.to_datetime(start_date)
                forecast_df = forecast_df[forecast_df['Date'] >= start_date]
                
            if end_date:
                end_date = pd.to_datetime(end_date)
                forecast_df = forecast_df[forecast_df['Date'] <= end_date]
            
            # Identify the forecast column based on model type
            forecast_col = 'Forecast'
            if forecast_model == 'pytorch' and 'Predicted_Demand' in forecast_df.columns:
                forecast_col = 'Predicted_Demand'
            elif forecast_model == 'pytorch' and 'Predicted_Sales' in forecast_df.columns:
                forecast_col = 'Predicted_Sales'
                
            # Calculate distribution metrics
            avg_forecast = forecast_df[forecast_col].mean()
            std_dev = forecast_df[forecast_col].std()
            min_forecast = forecast_df[forecast_col].min()
            max_forecast = forecast_df[forecast_col].max()
            
            # Ensure we have a reasonable range (avoid very small std_dev)
            if std_dev < 0.1 * avg_forecast:
                std_dev = 0.1 * avg_forecast
                
            # Create a range that covers the full extent of the data
            forecast_range = np.linspace(max(0, min_forecast - std_dev), max_forecast + std_dev, 100)
            
            # Create distribution figure
            fig = go.Figure()
            
            # Add histogram of forecast values
            fig.add_trace(go.Histogram(
                x=forecast_df[forecast_col],
                nbinsx=20,
                name='Forecast Distribution',
                marker=dict(color='blue', opacity=0.7)
            ))
            
            # Add normal distribution curve if we have enough data points
            if len(forecast_df) > 5:
                # Create normal distribution based on forecast mean and std dev
                from scipy.stats import norm
                normal_y = norm.pdf(forecast_range, avg_forecast, std_dev)
                
                # Scale the normal distribution to match histogram height
                bin_width = (max_forecast - min_forecast) / 20
                if bin_width > 0:
                    scale_factor = len(forecast_df) * bin_width
                    normal_y_scaled = normal_y * scale_factor
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_range,
                        y=normal_y_scaled,
                        mode='lines',
                        name='Normal Distribution',
                        line=dict(color='red')
                    ))
            
            # Add vertical line for average
            fig.add_shape(
                type="line",
                x0=avg_forecast,
                y0=0,
                x1=avg_forecast,
                y1=1,
                yref="paper",
                line=dict(color="green", width=2, dash="dash")
            )
            
            # Add annotation for average
            fig.add_annotation(
                x=avg_forecast,
                y=0.95,
                yref="paper",
                text=f"Average: {avg_forecast:.2f}",
                showarrow=True,
                arrowhead=1
            )
            
            # Add standard deviation annotation
            fig.add_annotation(
                x=avg_forecast + std_dev,
                y=0.85,
                yref="paper",
                text=f"σ: {std_dev:.2f}",
                showarrow=True,
                arrowhead=1
            )
            
            # Get product name if available
            product_name = forecast_df['Product'].iloc[0] if 'Product' in forecast_df.columns and len(forecast_df) > 0 else f'Product {product_id}'
            
            # Update layout
            fig.update_layout(
                title=f'Forecast Distribution for {product_name} (Store {store_id})',
                xaxis_title='Forecast Units',
                yaxis_title='Frequency',
                height=500,
                xaxis=dict(
                    # Set a reasonable range based on the data
                    range=[max(0, min_forecast - std_dev), max_forecast + std_dev]
                )
            )
            
            return jsonify(plotly.io.to_json(fig))
        else:
            # Create sample data if file doesn't exist
            data = generate_sample_data()
            
            # Create sample forecast distribution
            np.random.seed(42)
            sample_forecasts = np.random.normal(100, 15, 100)
            
            # Create figure
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=sample_forecasts,
                nbinsx=20,
                name='Forecast Distribution',
                marker=dict(color='blue', opacity=0.7)
            ))
            
            # Add normal distribution curve
            avg_forecast = np.mean(sample_forecasts)
            std_dev = np.std(sample_forecasts)
            min_forecast = np.min(sample_forecasts)
            max_forecast = np.max(sample_forecasts)
            forecast_range = np.linspace(min_forecast - std_dev, max_forecast + std_dev, 100)
            
            from scipy.stats import norm
            normal_y = norm.pdf(forecast_range, avg_forecast, std_dev)
            bin_width = (max_forecast - min_forecast) / 20
            normal_y_scaled = normal_y * len(sample_forecasts) * bin_width
            
            fig.add_trace(go.Scatter(
                x=forecast_range,
                y=normal_y_scaled,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red')
            ))
            
            # Add vertical line for average
            fig.add_shape(
                type="line",
                x0=avg_forecast,
                y0=0,
                x1=avg_forecast,
                y1=1,
                yref="paper",
                line=dict(color="green", width=2, dash="dash")
            )
            
            # Add annotation for average
            fig.add_annotation(
                x=avg_forecast,
                y=0.95,
                yref="paper",
                text=f"Average: {avg_forecast:.2f}",
                showarrow=True,
                arrowhead=1
            )
            
            # Add standard deviation annotation
            fig.add_annotation(
                x=avg_forecast + std_dev,
                y=0.85,
                yref="paper",
                text=f"σ: {std_dev:.2f}",
                showarrow=True,
                arrowhead=1
            )
            
            # Update layout
            fig.update_layout(
                title='Forecast Distribution (Sample Data)',
                xaxis_title='Forecast Units',
                yaxis_title='Frequency',
                height=500,
                xaxis=dict(
                    range=[min_forecast - std_dev, max_forecast + std_dev]
                )
            )
            
            return jsonify(plotly.io.to_json(fig))
            
    except Exception as e:
        # Return error message
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error creating forecast distribution: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return jsonify(plotly.io.to_json(error_fig))

@app.route('/api/price-recommendations')
def price_recommendations():
    """API endpoint to get price recommendation data."""
    try:
        # Get parameters from request
        store_id = request.args.get('store_id', '104.0')
        limit = int(request.args.get('limit', '10'))
        sort_by = request.args.get('sort_by', 'Expected_Profit_Change_Pct')
        ascending = request.args.get('ascending', 'false').lower() == 'true'
        
        # Load price recommendations data
        recommendations_df = get_data('price_recommendations')
        
        if recommendations_df is not None:
            # Filter by store
            recommendations_df = recommendations_df[recommendations_df['Store_Id'] == float(store_id)]
            
            if not recommendations_df.empty:
                # Sort by specified column
                if sort_by in recommendations_df.columns:
                    recommendations_df = recommendations_df.sort_values(sort_by, ascending=ascending)
                
                # Limit to requested number of rows
                recommendations_df = recommendations_df.head(limit)
                
                # Create recommendation chart - bar chart showing profit impact
                fig = go.Figure()
                
                # Add bars for profit change percentage
                fig.add_trace(go.Bar(
                    y=recommendations_df['Product'],
                    x=recommendations_df['Expected_Profit_Change_Pct'],
                    orientation='h',
                    name='Expected Profit Change %',
                    marker_color=recommendations_df['Expected_Profit_Change_Pct'].apply(
                        lambda x: 'green' if x > 0 else 'red'
                    ),
                    hovertemplate=(
                        '<b>%{y}</b><br>' +
                        'Current Price: $%{customdata[0]:.2f}<br>' +
                        'Recommended Price: $%{customdata[1]:.2f}<br>' +
                        'Price Change: %{customdata[2]:.1f}%<br>' +
                        'Expected Sales Change: %{customdata[3]:.1f}%<br>' +
                        'Expected Profit Change: %{x:.1f}%<br>' +
                        'Elasticity: %{customdata[4]:.2f}' +
                        '<extra></extra>'
                    ),
                    customdata=np.column_stack((
                        recommendations_df['Current_Price'],
                        recommendations_df['Optimal_Price'],
                        recommendations_df['Price_Change_Pct'],
                        recommendations_df['Expected_Sales_Change_Pct'],
                        recommendations_df['Elasticity']
                    ))
                ))
                
                # Add annotations for recommendations
                for i, row in enumerate(recommendations_df.itertuples()):
                    fig.add_annotation(
                        y=row.Product,
                        x=row.Expected_Profit_Change_Pct + (5 if row.Expected_Profit_Change_Pct > 0 else -5),
                        text=row.Recommendation,
                        showarrow=False,
                        font=dict(size=10),
                        xanchor='left' if row.Expected_Profit_Change_Pct > 0 else 'right'
                    )
                
                # Update layout
                fig.update_layout(
                    title='Price Recommendations by Expected Profit Impact',
                    xaxis_title='Expected Profit Change (%)',
                    yaxis_title='Product',
                    yaxis={'categoryorder':'total ascending'},
                    height=max(500, 50 * len(recommendations_df) + 150)  # Dynamic height based on number of products
                )
                
                return jsonify(plotly.io.to_json(fig))
        
        # Fallback to sample data if no real data
        data = generate_sample_data()
        recommendations_df = data['price_recommendations']
        
        # Filter by store
        recommendations_df = recommendations_df[recommendations_df['Store_Id'] == float(store_id)]
        
        # Sort by specified column
        if sort_by in recommendations_df.columns:
            recommendations_df = recommendations_df.sort_values(sort_by, ascending=ascending)
        
        # Limit to requested number of rows
        recommendations_df = recommendations_df.head(limit)
        
        # Create recommendation chart - bar chart showing profit impact
        fig = go.Figure()
        
        # Add bars for profit change percentage
        fig.add_trace(go.Bar(
            y=recommendations_df['Product'],
            x=recommendations_df['Expected_Profit_Change_Pct'],
            orientation='h',
            name='Expected Profit Change %',
            marker_color=recommendations_df['Expected_Profit_Change_Pct'].apply(
                lambda x: 'green' if x > 0 else 'red'
            ),
            hovertemplate=(
                '<b>%{y}</b><br>' +
                'Current Price: $%{customdata[0]:.2f}<br>' +
                'Recommended Price: $%{customdata[1]:.2f}<br>' +
                'Price Change: %{customdata[2]:.1f}%<br>' +
                'Expected Sales Change: %{customdata[3]:.1f}%<br>' +
                'Expected Profit Change: %{x:.1f}%<br>' +
                'Elasticity: %{customdata[4]:.2f}' +
                '<extra></extra>'
            ),
            customdata=np.column_stack((
                recommendations_df['Current_Price'],
                recommendations_df['Optimal_Price'],
                recommendations_df['Price_Change_Pct'],
                recommendations_df['Expected_Sales_Change_Pct'],
                recommendations_df['Elasticity']
            ))
        ))
        
        # Add annotations for recommendations
        for i, row in enumerate(recommendations_df.itertuples()):
            fig.add_annotation(
                y=row.Product,
                x=row.Expected_Profit_Change_Pct + (5 if row.Expected_Profit_Change_Pct > 0 else -5),
                text=row.Recommendation,
                showarrow=False,
                font=dict(size=10),
                xanchor='left' if row.Expected_Profit_Change_Pct > 0 else 'right'
            )
        
        # Update layout
        fig.update_layout(
            title='Price Recommendations by Expected Profit Impact (Sample Data)',
            xaxis_title='Expected Profit Change (%)',
            yaxis_title='Product',
            yaxis={'categoryorder':'total ascending'},
            height=max(500, 50 * len(recommendations_df) + 150)  # Dynamic height based on number of products
        )
        
        return jsonify(plotly.io.to_json(fig))
            
    except Exception as e:
        # Return error message
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error creating price recommendations chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return jsonify(plotly.io.to_json(error_fig))

@app.route('/api/inventory-recommendations')
def inventory_recommendations():
    """API endpoint to get inventory recommendations data."""
    try:
        # Get parameters from request
        store_id = request.args.get('store_id', '1')
        sort_by = request.args.get('sort_by', 'Recommended_Order_Quantity')
        ascending = request.args.get('ascending', 'false').lower() == 'true'
        show_all = request.args.get('show_all', 'false').lower() == 'true'
        
        # Load inventory recommendations data
        inventory_df = get_data('rf_recommendations')
        
        if inventory_df is not None:
            # Filter by store
            inventory_df = inventory_df[inventory_df['Store_Id'] == float(store_id)]
            
            if not inventory_df.empty:
                # Filter to only show items that need ordering unless show_all is true
                if not show_all:
                    inventory_df = inventory_df[inventory_df['Recommended_Order_Quantity'] > 0]
                
                # If we don't have any recommendations to show and show_all is false, include everything
                if len(inventory_df) == 0 and not show_all:
                    inventory_df = get_data('rf_recommendations')
                    inventory_df = inventory_df[inventory_df['Store_Id'] == float(store_id)]
                
                # Sort by specified column
                if sort_by in inventory_df.columns:
                    inventory_df = inventory_df.sort_values(sort_by, ascending=ascending)
                
                # Create inventory recommendations chart
                fig = go.Figure()
                
                # Add bars for weeks of stock
                fig.add_trace(go.Bar(
                    x=inventory_df['Product'],
                    y=inventory_df['Weeks_Of_Stock'],
                    name='Current Weeks of Stock',
                    marker_color=inventory_df['Weeks_Of_Stock'].apply(
                        lambda x: 'red' if x < 1 else ('orange' if x < 1.5 else 'green')
                    ),
                    hovertemplate=(
                        '<b>%{x}</b><br>' +
                        'Current Stock: %{customdata[0]:.0f} units<br>' +
                        'Avg Daily Demand: %{customdata[1]:.1f} units<br>' +
                        'Weeks of Stock: %{y:.1f} weeks<br>' +
                        'Target Stock: %{customdata[2]:.0f} units<br>' +
                        'Recommended Order: %{customdata[3]:.0f} units' +
                        '<extra></extra>'
                    ),
                    customdata=np.column_stack((
                        inventory_df['Current_Stock'],
                        inventory_df['Avg_Daily_Demand'],
                        inventory_df['Target_Stock_Units'],
                        inventory_df['Recommended_Order_Quantity']
                    ))
                ))
                
                # Add line for target weeks of stock
                fig.add_trace(go.Scatter(
                    x=inventory_df['Product'],
                    y=inventory_df['Target_Stock_Weeks'],
                    mode='lines+markers',
                    name='Target Weeks of Stock',
                    line=dict(color='blue', width=2, dash='dash')
                ))
                
                # Add line for minimum weeks of stock
                fig.add_trace(go.Scatter(
                    x=inventory_df['Product'],
                    y=inventory_df['Min_Stock_Weeks'],
                    mode='lines',
                    name='Minimum Weeks of Stock',
                    line=dict(color='red', width=2, dash='dot')
                ))
                
                # Update layout
                fig.update_layout(
                    title=f'Inventory Status and Recommendations for Store {store_id}',
                    xaxis_title='Product',
                    yaxis_title='Weeks of Stock',
                    barmode='group',
                    height=max(500, 30 * len(inventory_df) + 200)  # Dynamic height based on number of products
                )
                
                return jsonify(plotly.io.to_json(fig))
        
        # Fallback to sample data if no real data
        data = generate_sample_data()
        inventory_df = data['rf_recommendations']
        
        # Filter by store
        inventory_df = inventory_df[inventory_df['Store_Id'] == float(store_id)]
        
        # Filter to only show items that need ordering unless show_all is true
        if not show_all:
            inventory_df = inventory_df[inventory_df['Recommended_Order_Quantity'] > 0]
        
        # If we don't have any recommendations to show and show_all is false, include everything
        if len(inventory_df) == 0 and not show_all:
            inventory_df = data['rf_recommendations']
            inventory_df = inventory_df[inventory_df['Store_Id'] == float(store_id)]
        
        # Sort by specified column
        if sort_by in inventory_df.columns:
            inventory_df = inventory_df.sort_values(sort_by, ascending=ascending)
        
        # Create inventory recommendations chart
        fig = go.Figure()
        
        # Add bars for weeks of stock
        fig.add_trace(go.Bar(
            x=inventory_df['Product'],
            y=inventory_df['Weeks_Of_Stock'],
            name='Current Weeks of Stock',
            marker_color=inventory_df['Weeks_Of_Stock'].apply(
                lambda x: 'red' if x < 1 else ('orange' if x < 1.5 else 'green')
            ),
            hovertemplate=(
                '<b>%{x}</b><br>' +
                'Current Stock: %{customdata[0]:.0f} units<br>' +
                'Avg Daily Demand: %{customdata[1]:.1f} units<br>' +
                'Weeks of Stock: %{y:.1f} weeks<br>' +
                'Target Stock: %{customdata[2]:.0f} units<br>' +
                'Recommended Order: %{customdata[3]:.0f} units' +
                '<extra></extra>'
            ),
            customdata=np.column_stack((
                inventory_df['Current_Stock'],
                inventory_df['Avg_Daily_Demand'],
                inventory_df['Target_Stock_Units'],
                inventory_df['Recommended_Order_Quantity']
            ))
        ))
        
        # Add line for target weeks of stock
        fig.add_trace(go.Scatter(
            x=inventory_df['Product'],
            y=inventory_df['Target_Stock_Weeks'],
            mode='lines+markers',
            name='Target Weeks of Stock',
            line=dict(color='blue', width=2, dash='dash')
        ))
        
        # Add line for minimum weeks of stock
        fig.add_trace(go.Scatter(
            x=inventory_df['Product'],
            y=inventory_df['Min_Stock_Weeks'],
            mode='lines',
            name='Minimum Weeks of Stock',
            line=dict(color='red', width=2, dash='dot')
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Inventory Status and Recommendations for Store {store_id} (Sample Data)',
            xaxis_title='Product',
            yaxis_title='Weeks of Stock',
            barmode='group',
            height=max(500, 30 * len(inventory_df) + 200)  # Dynamic height based on number of products
        )
        
        return jsonify(plotly.io.to_json(fig))
            
    except Exception as e:
        # Return error message
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error creating inventory recommendations chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return jsonify(plotly.io.to_json(error_fig))

@app.route('/api/historical-sales-data')
def historical_sales_data():
    """API endpoint to get historical sales data in a format suitable for plotting with Plotly.
    
    Returns a Plotly figure with historical sales data visualization.
    Parameters:
    - store_id: Store identifier
    - product_id: Product identifier
    - start_date: Filter data from this date onwards
    - end_date: Filter data up to this date
    """
    try:
        # Get parameters from request
        store_id = request.args.get('store_id', '104.0')
        product_id = request.args.get('product_id', '3913116850.0')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Load historical sales data
        historical_df = get_data('combined_data')
        
        if historical_df is not None:
            # Filter by store and product
            historical_df = historical_df[(historical_df['Store_Id'] == float(store_id)) & 
                                         (historical_df['Item'] == float(product_id))]
            
            # Convert dates if needed
            if not pd.api.types.is_datetime64_dtype(historical_df['Date']):
                historical_df['Date'] = pd.to_datetime(historical_df['Date'])
            
            # Apply date filters if provided
            if start_date:
                start_date = pd.to_datetime(start_date)
                historical_df = historical_df[historical_df['Date'] >= start_date]
                
            if end_date:
                end_date = pd.to_datetime(end_date)
                historical_df = historical_df[historical_df['Date'] <= end_date]
                
            # Sort by date
            historical_df = historical_df.sort_values('Date')
            
            # Create Plotly figure
            if not historical_df.empty:
                # Get product name if available (with empty DataFrame check)
                product_name = historical_df['Product'].iloc[0] if 'Product' in historical_df.columns and not historical_df.empty else f'Product {product_id}'
                
                # Create figure with secondary y-axis
                fig = go.Figure()
                
                # Add historical sales trace
                sales_col = 'Sales' if 'Sales' in historical_df.columns else 'Units_Sold'
                
                if sales_col in historical_df.columns:
                    fig.add_trace(go.Scatter(
                        x=historical_df['Date'],
                        y=historical_df[sales_col],
                        mode='lines+markers',
                        name='Sales Units',
                        line=dict(color='blue', width=2),
                        marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')),
                        hovertemplate='Date: %{x}<br>Sales: %{y:.1f} units<extra></extra>'
                    ))
                
                # Add revenue trace if available
                if 'Retail_Revenue' in historical_df.columns:
                    fig.add_trace(go.Scatter(
                        x=historical_df['Date'],
                        y=historical_df['Retail_Revenue'],
                        mode='lines',
                        name='Revenue',
                        line=dict(color='green', width=2),
                        yaxis='y2',
                        hovertemplate='Date: %{x}<br>Revenue: $%{y:.2f}<extra></extra>'
                    ))
                
                # Add moving average for trend visualization
                if len(historical_df) >= 7 and sales_col in historical_df.columns:
                    historical_df['MA7'] = historical_df[sales_col].rolling(window=7).mean()
                    fig.add_trace(go.Scatter(
                        x=historical_df['Date'],
                        y=historical_df['MA7'],
                        mode='lines',
                        name='7-Day Moving Avg',
                        line=dict(color='red', width=2, dash='dash'),
                        hovertemplate='Date: %{x}<br>7-Day Avg: %{y:.1f} units<extra></extra>'
                    ))
                
                # Add price data if available
                if 'Price' in historical_df.columns:
                    fig.add_trace(go.Scatter(
                        x=historical_df['Date'],
                        y=historical_df['Price'],
                        mode='lines+markers',
                        name='Price',
                        line=dict(color='purple', width=2),
                        marker=dict(size=5),
                        yaxis='y3',
                        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                    ))
                
                # Update layout with enhanced design
                fig.update_layout(
                    title=f'Historical Sales Data for {product_name} (Store {store_id})',
                    xaxis=dict(
                        title='Date',
                        title_font=dict(size=14),
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(211,211,211,0.5)',
                        rangeslider=dict(visible=True),
                        rangeselector=dict(
                            buttons=list([
                                dict(count=7, label='1w', step='day', stepmode='backward'),
                                dict(count=1, label='1m', step='month', stepmode='backward'),
                                dict(count=3, label='3m', step='month', stepmode='backward'),
                                dict(count=6, label='6m', step='month', stepmode='backward'),
                                dict(step='all')
                            ]),
                            bgcolor='rgba(150,150,150,0.2)',
                            font=dict(size=12)
                        )
                    ),
                    yaxis=dict(
                        title='Units Sold',
                        titlefont=dict(color='blue', size=14),
                        tickfont=dict(color='blue'),
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(211,211,211,0.5)',
                        zeroline=True,
                        zerolinewidth=1,
                        zerolinecolor='black'
                    ),
                    yaxis2=dict(
                        title='Revenue ($)',
                        titlefont=dict(color='green', size=14),
                        tickfont=dict(color='green'),
                        anchor='x',
                        overlaying='y',
                        side='right',
                        showgrid=False
                    ),
                    yaxis3=dict(
                        title='Price ($)',
                        titlefont=dict(color='purple', size=14),
                        tickfont=dict(color='purple'),
                        anchor='free',
                        overlaying='y',
                        side='right',
                        position=1.0,
                        showgrid=False
                    ),
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='center',
                        x=0.5,
                        bgcolor='rgba(255,255,255,0.8)'
                    ),
                    margin=dict(l=60, r=60, t=80, b=60),
                    hovermode='x unified',
                    plot_bgcolor='white',
                    height=800,
                    width=1600,
                    shapes=[
                        # Add horizontal line at zero sales
                        dict(
                            type='line',
                            xref='paper',
                            yref='y',
                            x0=0,
                            y0=0,
                            x1=1,
                            y1=0,
                            line=dict(color='black', width=1)
                        )
                    ]
                )
                
                # Add annotations for key insights if there's enough data
                if len(historical_df) > 5 and sales_col in historical_df.columns:
                    # Annotate maximum sales point
                    max_sales_idx = historical_df[sales_col].idxmax()
                    max_date = historical_df.loc[max_sales_idx, 'Date']
                    max_sales = historical_df.loc[max_sales_idx, sales_col]
                    
                    fig.add_annotation(
                        x=max_date,
                        y=max_sales,
                        text=f"Peak: {max_sales:.1f}",
                        showarrow=True,
                        arrowhead=1,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor='black',
                        ax=30,
                        ay=-40
                    )
                    
                    # Annotate minimum sales point
                    min_sales_idx = historical_df[sales_col].idxmin()
                    min_date = historical_df.loc[min_sales_idx, 'Date']
                    min_sales = historical_df.loc[min_sales_idx, sales_col]
                    
                    fig.add_annotation(
                        x=min_date,
                        y=min_sales,
                        text=f"Low: {min_sales:.1f}",
                        showarrow=True,
                        arrowhead=1,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor='black',
                        ax=-30,
                        ay=40
                    )
                
                return jsonify(plotly.io.to_json(fig))
            else:
                # No data found for the specified parameters
                fig = go.Figure()
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    text=f"No historical data found for Product {product_id} in Store {store_id}",
                    showarrow=False,
                    font=dict(size=14)
                )
                fig.update_layout(height=400)
                return jsonify(plotly.io.to_json(fig))
        
        # Generate sample data if real data is not available
        np.random.seed(42)
        today = datetime.now()
        
        # Create date range for historical data
        if start_date:
            start = pd.to_datetime(start_date)
        else:
            start = today - timedelta(days=365)  # Default to 1 year of data
            
        if end_date:
            end = pd.to_datetime(end_date)
        else:
            end = today
        
        # Generate daily dates
        dates = pd.date_range(start=start, end=end, freq='D')
        
        # Generate sales with patterns
        base_sales = 20
        sales_data = []
        
        # Add seasonal, weekly, and random patterns
        for date in dates:
            # Weekly pattern (higher on weekends)
            day_factor = 1.5 if date.dayofweek >= 5 else 1.0
            
            # Monthly pattern (higher at beginning/end of month)
            day_of_month = date.day
            month_factor = 1.2 if day_of_month <= 5 or day_of_month >= 25 else 1.0
            
            # Seasonal pattern (higher in winter, lower in summer)
            month = date.month
            if month in [11, 12, 1, 2]:  # Winter
                season_factor = 1.3
            elif month in [6, 7, 8]:  # Summer
                season_factor = 0.7
            else:  # Spring/Fall
                season_factor = 1.0
                
            # Long-term trend (slight growth over time)
            days_from_start = (date - dates[0]).days
            trend_factor = 1.0 + (days_from_start / 365) * 0.1
            
            # Price variations
            if date.month % 3 == 0 and date.day < 7:
                price = 6.99  # Promotional pricing
            else:
                price = 7.99  # Regular pricing
                
            # Combine all factors with some randomness
            daily_sales = base_sales * day_factor * month_factor * season_factor * trend_factor
            daily_sales = max(0, np.random.normal(daily_sales, daily_sales * 0.2))  # Add noise
            
            # Add special events/promotions (e.g., holidays)
            if (month == 12 and day_of_month >= 20) or (month == 11 and day_of_month >= 23 and day_of_month <= 26):
                daily_sales *= 1.5  # Holiday boost
                
            sales_data.append({
                'Date': date,
                'Sales': round(daily_sales, 1),
                'Price': price,
                'Retail_Revenue': round(daily_sales * price, 2)
            })
        
        # Create sample DataFrame
        sample_df = pd.DataFrame(sales_data)
        
        # Calculate moving average
        sample_df['MA7'] = sample_df['Sales'].rolling(window=7).mean()
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add sales trace
        fig.add_trace(go.Scatter(
            x=sample_df['Date'],
            y=sample_df['Sales'],
            mode='lines+markers',
            name='Sales Units',
            line=dict(color='blue', width=2),
            marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')),
            hovertemplate='Date: %{x}<br>Sales: %{y:.1f} units<extra></extra>'
        ))
        
        # Add revenue trace
        fig.add_trace(go.Scatter(
            x=sample_df['Date'],
            y=sample_df['Retail_Revenue'],
            mode='lines',
            name='Revenue',
            line=dict(color='green', width=2),
            yaxis='y2',
            hovertemplate='Date: %{x}<br>Revenue: $%{y:.2f}<extra></extra>'
        ))
        
        # Add moving average trace
        fig.add_trace(go.Scatter(
            x=sample_df['Date'],
            y=sample_df['MA7'],
            mode='lines',
            name='7-Day Moving Avg',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='Date: %{x}<br>7-Day Avg: %{y:.1f} units<extra></extra>'
        ))
        
        # Add price trace
        fig.add_trace(go.Scatter(
            x=sample_df['Date'],
            y=sample_df['Price'],
            mode='lines+markers',
            name='Price',
            line=dict(color='purple', width=2),
            marker=dict(size=5),
            yaxis='y3',
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Update layout with enhanced design
        fig.update_layout(
            title=f'Historical Sales Data for Product {product_id} (Store {store_id}) - Sample Data',
            xaxis=dict(
                title='Date',
                titlefont=dict(size=14),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(211,211,211,0.5)',
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label='1w', step='day', stepmode='backward'),
                        dict(count=1, label='1m', step='month', stepmode='backward'),
                        dict(count=3, label='3m', step='month', stepmode='backward'),
                        dict(count=6, label='6m', step='month', stepmode='backward'),
                        dict(step='all')
                    ]),
                    bgcolor='rgba(150,150,150,0.2)',
                    font=dict(size=12)
                )
            ),
            yaxis=dict(
                title='Units Sold',
                titlefont=dict(color='blue', size=14),
                tickfont=dict(color='blue'),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(211,211,211,0.5)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='black'
            ),
            yaxis2=dict(
                title='Revenue ($)',
                titlefont=dict(color='green', size=14),
                tickfont=dict(color='green'),
                anchor='x',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            yaxis3=dict(
                title='Price ($)',
                titlefont=dict(color='purple', size=14),
                tickfont=dict(color='purple'),
                anchor='free',
                overlaying='y',
                side='right',
                position=1.0,
                showgrid=False
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(255,255,255,0.8)'
            ),
            margin=dict(l=60, r=60, t=80, b=60),
            hovermode='x unified',
            plot_bgcolor='white',
            height=600,
            width=1000
        )
        
        # Add annotations for sample data insights
        # Identify a peak in the data
        max_sales_idx = sample_df['Sales'].idxmax()
        max_date = sample_df.loc[max_sales_idx, 'Date']
        max_sales = sample_df.loc[max_sales_idx, 'Sales']
        
        fig.add_annotation(
            x=max_date,
            y=max_sales,
            text=f"Peak: {max_sales:.1f}",
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='black',
            ax=30,
            ay=-40
        )
        
        # Identify a period where price dropped
        price_changes = sample_df[sample_df['Price'] < 7.99]
        if not price_changes.empty:
            promo_start = price_changes.iloc[0]['Date']
            fig.add_annotation(
                x=promo_start,
                y=price_changes.iloc[0]['Price'],
                text="Price Promotion",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='purple',
                ax=0,
                ay=30
            )
        
        # Add seasonal pattern annotation
        winter_data = sample_df[(sample_df['Date'].dt.month == 12) & (sample_df['Date'].dt.day > 15)]
        if not winter_data.empty:
            holiday_date = winter_data.iloc[len(winter_data)//2]['Date']
            fig.add_annotation(
                x=holiday_date,
                y=winter_data.iloc[len(winter_data)//2]['Sales'],
                text="Holiday Sales Boost",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='red',
                ax=-40,
                ay=-40
            )
        
        return jsonify(plotly.io.to_json(fig))
        
    except Exception as e:
        # Generate detailed error with traceback
        import traceback
        error_stack = traceback.format_exc()
        print(f"Error in historical sales data endpoint: {str(e)}\n{error_stack}")
        
        # Return error visualization
        error_fig = go.Figure()
        error_fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=f"Error retrieving historical sales data: {str(e)}",
            showarrow=False,
            font=dict(size=14, color='red')
        )
        error_fig.update_layout(height=400)
        return jsonify(plotly.io.to_json(error_fig))

@app.route('/api/historical-sales')
def historical_sales():
    """API endpoint to get historical sales data."""
    try:
        # Get parameters from request
        store_id = request.args.get('store_id', '104.0')
        product_id = request.args.get('product_id', '3913116850.0')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        format_type = request.args.get('format', 'json').lower()  # json or csv
        include_metadata = request.args.get('include_metadata', 'true').lower() == 'true'
        group_by = request.args.get('group_by', None)  # Optional grouping (day, week, month)
        
        # Load historical sales data
        historical_df = get_data('combined_data')
        
        if historical_df is not None:
            # Filter by store and product
            historical_df = historical_df[(historical_df['Store_Id'] == float(store_id)) & 
                                         (historical_df['Item'] == float(product_id))]
            
            # Convert dates if needed
            if not pd.api.types.is_datetime64_dtype(historical_df['Date']):
                historical_df['Date'] = pd.to_datetime(historical_df['Date'])
            
            # Apply date filters if provided
            if start_date:
                start_date = pd.to_datetime(start_date)
                historical_df = historical_df[historical_df['Date'] >= start_date]
                
            if end_date:
                end_date = pd.to_datetime(end_date)
                historical_df = historical_df[historical_df['Date'] <= end_date]
                
            # Sort by date
            historical_df = historical_df.sort_values('Date')
            
            # Apply grouping if specified
            if group_by:
                if group_by == 'day':
                    # Data is already at daily level, no grouping needed
                    pass
                elif group_by == 'week':
                    # Group by week
                    historical_df['Week'] = historical_df['Date'].dt.to_period('W').apply(lambda r: r.start_time)
                    grouped = historical_df.groupby('Week').agg({
                        'Sales': 'sum',
                        'Retail_Revenue': 'sum',
                        'Cost': 'sum',
                        'Price': 'mean',
                        'Units_Purchased': 'sum',
                        'Purchase_Cost': 'sum',
                        'Profit': 'sum'
                    }).reset_index()
                    historical_df = grouped.rename(columns={'Week': 'Date'})
                elif group_by == 'month':
                    # Group by month
                    historical_df['Month'] = historical_df['Date'].dt.to_period('M').apply(lambda r: r.start_time)
                    grouped = historical_df.groupby('Month').agg({
                        'Sales': 'sum',
                        'Retail_Revenue': 'sum',
                        'Cost': 'sum',
                        'Price': 'mean',
                        'Units_Purchased': 'sum',
                        'Purchase_Cost': 'sum',
                        'Profit': 'sum'
                    }).reset_index()
                    historical_df = grouped.rename(columns={'Month': 'Date'})
            
            # Filter columns if include_metadata is False
            if not include_metadata:
                essential_columns = ['Date', 'Sales', 'Retail_Revenue', 'Cost', 'Price', 
                                    'Units_Purchased', 'Purchase_Cost', 'Profit']
                available_columns = [col for col in essential_columns if col in historical_df.columns]
                historical_df = historical_df[available_columns]
            
            # Return data in requested format
            if format_type == 'csv':
                import io
                output = io.StringIO()
                historical_df.to_csv(output, index=False)
                csv_data = output.getvalue()
                
                return csv_data, 200, {
                    'Content-Type': 'text/csv',
                    'Content-Disposition': f'attachment; filename=historical_sales_{store_id}_{product_id}.csv'
                }
            else:  # json format
                # Convert datetime to string for JSON serialization
                if 'Date' in historical_df.columns:
                    historical_df['Date'] = historical_df['Date'].dt.strftime('%Y-%m-%d')
                
                # Create visualization if requested
                if request.args.get('visualize', 'false').lower() == 'true':
                    fig = go.Figure()
                    
                    # Add sales line
                    if 'Sales' in historical_df.columns:
                        fig.add_trace(go.Scatter(
                            x=historical_df['Date'],
                            y=historical_df['Sales'],
                            mode='lines+markers',
                            name='Sales Units',
                            line=dict(color='blue', width=2),
                            hovertemplate='<b>Date:</b> %{x}<br><b>Sales:</b> %{y:.1f} units<extra></extra>'
                        ))
                    
                    # Add revenue line on secondary y-axis if available
                    if 'Retail_Revenue' in historical_df.columns:
                        fig.add_trace(go.Scatter(
                            x=historical_df['Date'],
                            y=historical_df['Retail_Revenue'],
                            mode='lines',
                            name='Revenue',
                            line=dict(color='green', width=2, dash='dot'),
                            yaxis='y2',
                            hovertemplate='<b>Date:</b> %{x}<br><b>Revenue:</b> $%{y:.2f}<extra></extra>'
                        ))
                    
                    # Get product name if available
                    product_name = historical_df['Product'].iloc[0] if 'Product' in historical_df.columns else f'Product {product_id}'
                    
                    # Update layout
                    title_suffix = ""
                    if group_by:
                        title_suffix = f" (Grouped by {group_by})"
                    
                    fig.update_layout(
                        title=f'Historical Sales for {product_name}{title_suffix}',
                        xaxis_title='Date',
                        yaxis_title='Sales Units',
                        yaxis2=dict(
                            title='Revenue ($)',
                            overlaying='y',
                            side='right',
                            showgrid=False
                        ),
                        legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=-0.1,
                    xanchor='center',
                    x=0.5,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='rgba(0,0,0,0.1)',
                    borderwidth=1,
                    font=dict(size=12)
                ),
                        height=800,
                    width=1600,
                        hovermode='x unified'
                    )
                    
                    return jsonify({
                        'data': historical_df.to_dict('records'),
                        'visualization': plotly.io.to_json(fig),
                        'metadata': {
                            'store_id': store_id,
                            'product_id': product_id,
                            'row_count': len(historical_df),
                            'start_date': historical_df['Date'].min() if len(historical_df) > 0 else None,
                            'end_date': historical_df['Date'].max() if len(historical_df) > 0 else None
                        }
                    })
                else:
                    # Return just the data
                    return jsonify({
                        'data': historical_df.to_dict('records'),
                        'metadata': {
                            'store_id': store_id,
                            'product_id': product_id,
                            'row_count': len(historical_df),
                            'start_date': historical_df['Date'].min() if len(historical_df) > 0 else None,
                            'end_date': historical_df['Date'].max() if len(historical_df) > 0 else None
                        }
                    })
        else:
            # Create sample historical sales data if real data is not available
            today = datetime.now()
            dates = pd.date_range(end=today, periods=365)  # 1 year of data
            
            # Create sample sales data with some patterns
            np.random.seed(42)
            
            # Base sales with weekly pattern (higher on weekends)
            sales = []
            revenue = []
            price = 7.99
            
            for date in dates:
                # Weekly pattern
                day_factor = 1.5 if date.dayofweek >= 5 else 1.0
                # Monthly pattern (higher at beginning/end of month)
                day_of_month = date.day
                month_factor = 1.2 if day_of_month <= 5 or day_of_month >= 25 else 1.0
                # Seasonal pattern
                month = date.month
                season_factor = 1.3 if month in [11, 12, 1, 2] else (0.8 if month in [6, 7, 8] else 1.0)
                
                # Combine factors and add some randomness
                base_sales = 15 * day_factor * month_factor * season_factor
                daily_sales = max(0, np.random.normal(base_sales, base_sales * 0.3))
                sales.append(round(daily_sales, 1))
                
                # Calculate revenue
                daily_revenue = daily_sales * price
                revenue.append(round(daily_revenue, 2))
            
            # Create DataFrame
            sample_df = pd.DataFrame({
                'Date': dates.strftime('%Y-%m-%d'),
                'Sales': sales,
                'Retail_Revenue': revenue,
                'Price': [price] * len(dates)
            })
            
            # Apply date filters if provided
            if start_date:
                start_date_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
                sample_df = sample_df[sample_df['Date'] >= start_date_str]
                
            if end_date:
                end_date_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
                sample_df = sample_df[sample_df['Date'] <= end_date_str]
            
            # Return sample data
            if format_type == 'csv':
                import io
                output = io.StringIO()
                sample_df.to_csv(output, index=False)
                csv_data = output.getvalue()
                
                return csv_data, 200, {
                    'Content-Type': 'text/csv',
                    'Content-Disposition': f'attachment; filename=sample_historical_sales_{store_id}_{product_id}.csv'
                }
            else:  # json format
                # Create visualization if requested
                if request.args.get('visualize', 'false').lower() == 'true':
                    fig = go.Figure()
                    
                    # Add sales line
                    fig.add_trace(go.Scatter(
                        x=sample_df['Date'],
                        y=sample_df['Sales'],
                        mode='lines+markers',
                        name='Sales Units',
                        line=dict(color='blue', width=2),
                        hovertemplate='<b>Date:</b> %{x}<br><b>Sales:</b> %{y:.1f} units<extra></extra>'
                    ))
                    
                    # Add revenue line
                    fig.add_trace(go.Scatter(
                        x=sample_df['Date'],
                        y=sample_df['Retail_Revenue'],
                        mode='lines',
                        name='Revenue',
                        line=dict(color='green', width=2, dash='dot'),
                        yaxis='y2',
                        hovertemplate='<b>Date:</b> %{x}<br><b>Revenue:</b> $%{y:.2f}<extra></extra>'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f'Sample Historical Sales for Product {product_id}',
                        xaxis_title='Date',
                        yaxis_title='Sales Units',
                        yaxis2=dict(
                            title='Revenue ($)',
                            overlaying='y',
                            side='right',
                            showgrid=False
                        ),
                        legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=-0.1,
                    xanchor='center',
                    x=0.5,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='rgba(0,0,0,0.1)',
                    borderwidth=1,
                    font=dict(size=12)
                ),
                        height=800,
                    width=1600,
                        hovermode='x unified'
                    )
                    
                    return jsonify({
                        'data': sample_df.to_dict('records'),
                        'visualization': plotly.io.to_json(fig),
                        'metadata': {
                            'store_id': store_id,
                            'product_id': product_id,
                            'row_count': len(sample_df),
                            'sample_data': True,
                            'start_date': sample_df['Date'].min() if len(sample_df) > 0 else None,
                            'end_date': sample_df['Date'].max() if len(sample_df) > 0 else None
                        }
                    })
                else:
                    # Return just the data
                    return jsonify({
                        'data': sample_df.to_dict('records'),
                        'metadata': {
                            'store_id': store_id,
                            'product_id': product_id,
                            'row_count': len(sample_df),
                            'sample_data': True,
                            'start_date': sample_df['Date'].min() if len(sample_df) > 0 else None,
                            'end_date': sample_df['Date'].max() if len(sample_df) > 0 else None
                        }
                    })
    except Exception as e:
        # Generate detailed error with traceback
        import traceback
        error_stack = traceback.format_exc()
        print(f"Error retrieving historical sales data: {str(e)}\n{error_stack}")
        
        # Return detailed error message
        return jsonify({
            'error': str(e),
            'traceback': error_stack.split('\n')
        }), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the app
    app.run(debug=True, port=8050)