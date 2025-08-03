import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import os
import datetime
import base64
from io import BytesIO

# Import the weather service, inventory module, and summary dashboard
from weather_service import WeatherService
from plotly_dashboard_inventory_new import update_inventory_chart, update_stock_velocity_chart, update_stock_penalty_chart, update_inventory_summary_stats, update_stock_recommendations
import summary_dashboard


# Helper function to load data with fallback
def load_data_with_fallback(file_path, fallback_path=None):
    try:
        df = pd.read_csv(file_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        print(f"Successfully loaded {file_path} with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        if fallback_path:
            try:
                df = pd.read_csv(fallback_path)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                print(f"Using fallback {fallback_path} instead")
                return df
            except Exception as e2:
                print(f"Error loading fallback {fallback_path}: {e2}")
        return None

# Load all the data
combined_data = load_data_with_fallback('combined_pizza_data.csv')
pytorch_forecasts = load_data_with_fallback('pytorch_forecasts.csv')
rf_forecasts = load_data_with_fallback('rf_forecasts.csv')

# Create empty DataFrames with expected columns if both forecast files are missing
if pytorch_forecasts is None and rf_forecasts is None:
    print("Creating empty forecasts DataFrame as fallback")
    # Use as a fallback for both
    forecasts = pd.DataFrame(columns=['Store_Id', 'Item', 'Date', 'Predicted_Sales', 'Predicted_Demand'])
    pytorch_forecasts = forecasts.copy()
    rf_forecasts = forecasts.copy()
elif pytorch_forecasts is None:
    pytorch_forecasts = rf_forecasts.copy()
elif rf_forecasts is None:
    rf_forecasts = pytorch_forecasts.copy()

# Default forecasts dataset for backward compatibility
forecasts = pytorch_forecasts
inventory_recs = load_data_with_fallback('inventory_recommendations.csv')
optimized_orders = load_data_with_fallback('optimized_orders.csv')
inventory_projection = load_data_with_fallback('inventory_projection.csv')
price_elasticities = load_data_with_fallback('price_elasticities.csv')
price_recommendations = load_data_with_fallback('price_recommendations.csv')
profit_impact = load_data_with_fallback('profit_impact.csv')
product_mix = load_data_with_fallback('product_mix_optimization.csv')

# Create a Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Required for Gunicorn deployment

# Initialize weather service
weather_service = WeatherService()

# Initialize stock adjustment state
app.manual_stock_adjustments = {}
# Initialize stock adjustment state with dates
app.manual_stock_adjustments_with_dates = {}

# Centralized function to apply stock adjustments to dataset
def apply_stock_adjustments(data_frame, store, product, stock_adjustment=None, adjust_clicks=None, adjustment_date=None):
    """
    Apply stock adjustments to a dataset in a consistent way across all visualizations
    
    Parameters:
    data_frame (DataFrame): The DataFrame to adjust
    store (int): Store ID
    product (int): Product ID
    stock_adjustment (int, optional): New stock level value
    adjust_clicks (int, optional): Clicks on adjustment button to trigger adjustment
    adjustment_date (str, optional): Specific date for the adjustment
    
    Returns:
    DataFrame: Adjusted DataFrame
    bool: Whether an update was applied
    """
    if data_frame is None or len(data_frame) == 0 or store is None or product is None:
        return data_frame, False
    
    # Make a copy to avoid modifying the original
    adjusted_data = data_frame.copy()
    update_applied = False
    key = f"{store}_{product}"
    date_key = None
    
    # Store the adjustment in app's state if adjustment is being made now
    if adjust_clicks and stock_adjustment is not None:
        # Store regular key adjustment
        app.manual_stock_adjustments[key] = stock_adjustment
        
        # Store with date if provided
        if adjustment_date is not None:
            try:
                # Convert string date to datetime if needed
                if isinstance(adjustment_date, str):
                    adj_date = pd.to_datetime(adjustment_date).date()
                else:
                    adj_date = adjustment_date
                    
                date_key = f"{store}_{product}_{adj_date}"
                if not hasattr(app, 'manual_stock_adjustments_with_dates'):
                    app.manual_stock_adjustments_with_dates = {}
                app.manual_stock_adjustments_with_dates[date_key] = stock_adjustment
            except Exception as e:
                print(f"Error storing date-specific adjustment: {e}")
                # If date conversion fails, fall back to just using the regular key
                pass
    
    # First try to apply date-specific adjustment if date is provided
    if adjustment_date is not None:
        try:
            # Convert string date to datetime if needed
            adj_date = pd.to_datetime(adjustment_date).date() if isinstance(adjustment_date, str) else adjustment_date
            date_key = f"{store}_{product}_{adj_date}"
            
            if hasattr(app, 'manual_stock_adjustments_with_dates') and date_key in app.manual_stock_adjustments_with_dates:
                # Find the specific date in the data
                match_date = pd.Timestamp(adj_date)
                date_idx = adjusted_data[adjusted_data['Date'] == match_date].index
                
                if len(date_idx) > 0:
                    # Apply the adjustment to all stock level fields
                    stock_value = app.manual_stock_adjustments_with_dates[date_key]
                    
                    # Update all stock-related columns
                    for col in ['Stock_Level', 'On_Hand']:
                        if col in adjusted_data.columns:
                            adjusted_data.loc[date_idx, col] = stock_value
                    
                    # Recalculate derived metrics if available
                    if 'Recent_Daily_Sales' in adjusted_data.columns:
                        avg_daily_sales = adjusted_data.loc[date_idx, 'Recent_Daily_Sales'].values[0]
                        
                        # Update weeks of stock/supply if daily sales > 0
                        if avg_daily_sales > 0:
                            weeks_of_stock = stock_value / (avg_daily_sales * 7)
                            
                            # Update all weeks of supply columns
                            for col in ['Weeks_Of_Stock', 'Weeks_of_Supply', 'Stock_Coverage_Weeks']:
                                if col in adjusted_data.columns:
                                    adjusted_data.loc[date_idx, col] = weeks_of_stock
                            
                            # Update stock status if available
                            if 'Stock_Status' in adjusted_data.columns:
                                if weeks_of_stock < 1:
                                    stock_status = 'Low'
                                elif weeks_of_stock <= 3:
                                    stock_status = 'Adequate'
                                else:
                                    stock_status = 'Excess'
                                    
                                adjusted_data.loc[date_idx, 'Stock_Status'] = stock_status
                    
                    update_applied = True
        except Exception as e:
            print(f"Error applying date-specific adjustment: {e}")
            # Continue to try the regular key if date-specific fails
            pass
    
    # If no date-specific update was applied, try the regular key
    if not update_applied and key in app.manual_stock_adjustments:
        # Find the latest date in the data
        try:
            latest_date = adjusted_data['Date'].max()
            latest_idx = adjusted_data[adjusted_data['Date'] == latest_date].index
            
            if len(latest_idx) > 0:
                # Apply the adjustment to all stock level fields
                stock_value = app.manual_stock_adjustments[key]
                
                # Update all stock-related columns
                for col in ['Stock_Level', 'On_Hand']:
                    if col in adjusted_data.columns:
                        adjusted_data.loc[latest_idx, col] = stock_value
                
                # Recalculate derived metrics if available
                if 'Recent_Daily_Sales' in adjusted_data.columns:
                    avg_daily_sales = adjusted_data.loc[latest_idx, 'Recent_Daily_Sales'].values[0]
                    
                    # Update weeks of stock/supply if daily sales > 0
                    if avg_daily_sales > 0:
                        weeks_of_stock = stock_value / (avg_daily_sales * 7)
                        
                        # Update all weeks of supply columns
                        for col in ['Weeks_Of_Stock', 'Weeks_of_Supply', 'Stock_Coverage_Weeks']:
                            if col in adjusted_data.columns:
                                adjusted_data.loc[latest_idx, col] = weeks_of_stock
                        
                        # Update stock status if available
                        if 'Stock_Status' in adjusted_data.columns:
                            if weeks_of_stock < 1:
                                stock_status = 'Low'
                            elif weeks_of_stock <= 3:
                                stock_status = 'Adequate'
                            else:
                                stock_status = 'Excess'
                                
                            adjusted_data.loc[latest_idx, 'Stock_Status'] = stock_status
                
                update_applied = True
        except Exception as e:
            print(f"Error applying regular adjustment: {e}")
    
    return adjusted_data, update_applied

# Initialize item number display state
app.show_item_numbers = True

# Get unique store and product information
if combined_data is not None:
    stores = sorted(combined_data['Store_Id'].unique())
    all_products = combined_data.drop_duplicates(subset=['Item', 'Product'])[['Item', 'Product']].copy()
    all_products['Label'] = all_products['Product'] + " (" + all_products['Item'].astype(str) + ")"
    product_options = [{"label": row['Label'], "value": row['Item']} for _, row in all_products.iterrows()]
else:
    stores = []
    product_options = []

# Define the layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Pizza Predictive Ordering Dashboard", className="text-center mb-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Filters", className="text-center")),
                dbc.CardBody([
                    html.P("Store:"),
                    dcc.Dropdown(
                        id="store-dropdown",
                        options=[{"label": f"Store {s}", "value": s} for s in stores],
                        value=stores[0] if len(stores) > 0 else None,
                        clearable=False
                    ),
                    html.P("Product:", className="mt-2"),
                    dcc.Dropdown(
                        id="product-dropdown",
                        options=product_options,
                        value=product_options[0]["value"] if len(product_options) > 0 else None,
                        clearable=False
                    ),
                    dbc.Switch(
                        id="item-number-toggle",
                        label="Show Item Numbers",
                        value=False,
                        className="mt-2"
                    ),
                    html.P("Date Range:", className="mt-2"),
                    dcc.DatePickerRange(
                        id="date-range",
                        min_date_allowed=(combined_data['Date'].min() if combined_data is not None else datetime.date.today() - datetime.timedelta(days=365)),
                        max_date_allowed=(combined_data['Date'].max() + datetime.timedelta(days=90) if combined_data is not None else datetime.date.today() + datetime.timedelta(days=90)),
                        initial_visible_month=(combined_data['Date'].max() if combined_data is not None else datetime.date.today()),
                        start_date=(combined_data['Date'].max() - datetime.timedelta(days=90) if combined_data is not None else datetime.date.today() - datetime.timedelta(days=90)),
                        end_date=(combined_data['Date'].max() + datetime.timedelta(days=30) if combined_data is not None else datetime.date.today() + datetime.timedelta(days=30))
                    ),
                    
                    # Weather toggle section
                    html.Hr(),
                    html.H5("Model Options", className="mt-3"),
                    dbc.Switch(
                        id="rf-model-toggle",
                        label="Retrain Model With Updated Data",
                        value=False,
                        className="mt-2"
                    ),
                    html.Hr(),
                    html.H5("Weather Impact", className="mt-3"),
                    dbc.Switch(
                        id="weather-toggle",
                        label="Apply Weather Adjustments",
                        value=False,
                        className="mt-2"
                    ),
                    dbc.Collapse(
                        [
                            html.P("Location:", className="mt-2 mb-0"),
                            dbc.Input(
                                id="location-input",
                                type="text",
                                placeholder="Enter ZIP, city name, or coordinates",
                                value="10001",  # Default to NYC
                                className="mb-2"
                            ),
                            html.Small("Examples: '10001', 'New York, NY', or '40.7128, -74.0060'", className="text-muted d-block mb-2"),
                            html.Div(
                                [
                                    html.Button(
                                        "Get Weather", 
                                        id="get-weather-btn", 
                                        className="btn btn-primary btn-sm"
                                    ),
                                    html.Div(id="weather-data-container", className="mt-2")
                                ]
                            )
                        ],
                        id="weather-collapse"
                    ),
                    html.Div([
                        dbc.Button("Apply Filters", id="apply-filters", color="primary", className="mt-3")
                    ], className="d-grid gap-2")
                ])
            ], className="mb-4")
        ], width=3),
        
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="Sales & Forecast", tab_id="tab-forecast", children=[
                    dbc.Card([
                        dbc.CardHeader(html.H4("Sales & Demand Forecast", className="text-center")),
                        dbc.CardBody([
                            dcc.Graph(id="sales-forecast-chart", style={"height": "500px"})
                        ])
                    ])
                ]),
                dbc.Tab(label="Price Elasticity", tab_id="tab-elasticity", children=[
                    dbc.Card([
                        dbc.CardHeader(html.H4("Price Sensitivity Analysis", className="text-center")),
                        dbc.CardBody([
                            dcc.Graph(id="price-elasticity-chart", style={"height": "500px"})
                        ])
                    ])
                ]),
                dbc.Tab(label="Profit Optimization", tab_id="tab-profit", children=[
                    dbc.Card([
                        dbc.CardHeader(html.H4("Profit Impact Analysis", className="text-center")),
                        dbc.CardBody([
                            dcc.Graph(id="profit-impact-chart", style={"height": "500px"})
                        ])
                    ])
                ]),
                dbc.Tab(label="Inventory Management", tab_id="tab-inventory", children=[
                    dbc.Card([
                        dbc.CardHeader(html.H4("Inventory Management", className="text-center")),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id="inventory-chart", style={"height": "400px"})
                                ], width=12)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.H5("Update Inventory Count", className="mt-3"),
                                    html.P("Enter on-hand inventory count for a specific date:"),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Select Date:", className="mt-2 mb-1"),
                                            dcc.DatePickerSingle(
                                                id="stock-adjustment-date",
                                                min_date_allowed=(combined_data['Date'].min() if combined_data is not None else datetime.date.today() - datetime.timedelta(days=365)),
                                                max_date_allowed=(combined_data['Date'].max() + datetime.timedelta(days=90) if combined_data is not None else datetime.date.today() + datetime.timedelta(days=90)),
                                                initial_visible_month=(combined_data['Date'].max() if combined_data is not None else datetime.date.today()),
                                                date=(combined_data['Date'].max() if combined_data is not None else datetime.date.today()),
                                                display_format="YYYY-MM-DD",
                                                className="mb-3"
                                            ),
                                        ], width=12)
                                    ]),
                                    dbc.InputGroup([
                                        dbc.Input(
                                            id="stock-adjustment-input",
                                            type="number",
                                            placeholder="Enter new inventory count",
                                            min=0
                                        ),
                                        dbc.InputGroupText("units"),
                                        dbc.Button("Update Count", id="apply-stock-adjustment", color="primary")
                                    ]),
                                    html.Small("This will update the inventory count for the selected product on the selected date and recalculate needs based on demand forecast.", className="text-muted"),
                                    html.Div([
                                        dbc.Alert(
                                            "Accurate counts help ensure better inventory management!",
                                            color="info",
                                            className="mt-3 mb-0",
                                            style={"fontSize": "0.85rem"}
                                        )
                                    ])
                                ], width=6),
                                dbc.Col([
                                    html.Div(id="stock-recommendations", className="mt-3")
                                ], width=6)
                            ])
                        ])
                    ])
                ]),
                dbc.Tab(label="Inventory Performance", tab_id="tab-stock-analysis", children=[
                    dbc.Card([
                        dbc.CardHeader(html.H4("Inventory Performance Analysis", className="text-center py-3")),
                        dbc.CardBody([
                            html.Div(className="py-3"),  # Extra spacing at the top of the card body
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id="inventory-turnover-chart", style={"height": "550px"})
                                ], width=12)
                            ], className="mb-5"),  # Added significant bottom margin between graphs
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id="inventory-cost-impact-chart", style={"height": "550px"})
                                ], width=12)
                            ], className="mb-5"),  # Added significant bottom margin before summary section
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id="inventory-summary-stats", className="mt-4 mb-3")
                                ], width=12)
                            ]),
                            html.Div(className="py-3")  # Extra spacing at the bottom of the card body
                        ], className="py-2")
                    ])
                ]),
                dbc.Tab(label="Weather Analysis", tab_id="tab-weather", children=[
                    dbc.Card([
                        dbc.CardHeader(html.H4("Lead Time Weather Analysis", className="text-center")),
                        dbc.CardBody([
                            dbc.Alert(
                                [
                                    html.H4("Weather Impact", className="alert-heading"),
                                    html.P("Toggle weather adjustments and enter a ZIP code to analyze weather impact on your supply chain.")
                                ],
                                color="info",
                                id="weather-alert",
                                is_open=True,
                                dismissable=True
                            ),
                            dcc.Graph(id="lead-time-weather-chart", style={"height": "500px"}),
                            html.Div(id="lead-time-weather-details", className="mt-3")
                        ])
                    ])
                ]),
                dbc.Tab(label="Summary Dashboard", tab_id="tab-summary", children=[
                    dbc.Card([
                        dbc.CardHeader(html.H4("Store Performance Summary", className="text-center")),
                        dbc.CardBody(html.Div(id="summary-dashboard-content"))
                    ])
                ])
            ], id="main-tabs", active_tab="tab-summary")
        ], width=9)
    ]),
    
        
    dbc.Row([
        dbc.Col([
            html.H4("Additional Insights on Profit Impact", className="text-center mt-4 mb-4")
        ], width=12)
    ]),
    
    # First full-width chart
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="price-impact-summary")
        ], width=12, className="mb-5")
    ]),
    
    # Second full-width chart with added vertical spacing
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="profit-comparison-chart")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col(html.Hr(), width=12),
        dbc.Col(html.P("Pizza Predictive Ordering System - Powered by Machine Learning", className="text-center"), width=12)
    ])
], fluid=True)

# Callback to toggle weather settings
@app.callback(
    [
        Output("weather-collapse", "is_open"),
        Output("weather-alert", "is_open")
    ],
    [Input("weather-toggle", "value")]
)
def toggle_weather_settings(toggle_value):
    try:
        return toggle_value, not toggle_value
    except Exception as e:
        print(f"Error toggling weather settings: {e}")
        return False, True

# Callback to update the weather data display
@app.callback(
    Output("weather-data-container", "children"),
    [Input("get-weather-btn", "n_clicks")],
    [State("location-input", "value")],
    prevent_initial_call=True
)
def update_weather_data(n_clicks, location):
    if n_clicks is None or not location:
        return html.P("Enter a location and click 'Get Weather'")
    
    # Get current weather and forecast
    try:
        current_weather = weather_service.get_current_weather(location)
        forecast = weather_service.get_weather_forecast(location, days=7)
        
        # Create weather display
        current_weather_card = dbc.Card(
            [
                dbc.CardHeader(f"Current Weather for {location}"),
                dbc.CardBody(
                    [
                        html.H5(f"{current_weather['temperature_f']:.1f}°F - {current_weather['weather_category']}"),
                        html.P(f"Weather Impact: {current_weather['weather_category']}"),
                        html.P(f"Humidity: {current_weather['humidity']}% | Wind: {current_weather['wind_speed']} m/s")
                    ]
                )
            ],
            className="mb-3"
        )
        
        # Create forecast table
        forecast_rows = []
        for day in forecast:
            forecast_rows.append(
                html.Tr([
                    html.Td(day['date']),
                    html.Td(f"{day['temperature_f']:.1f}°F"),
                    html.Td(day['weather_category']),
                    html.Td(day['weather_category'])
                ])
            )
        
        forecast_table = dbc.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("Date"),
                        html.Th("Temp"),
                        html.Th("Condition"),
                        html.Th("Impact")
                    ])
                ),
                html.Tbody(forecast_rows)
            ],
            bordered=True,
            hover=True,
            responsive=True,
            size="sm"
        )
        
        return [
            current_weather_card,
            html.H6("7-Day Forecast"),
            forecast_table,
            html.P("Weather data will be used to adjust demand predictions.", className="text-muted mt-2")
        ]
        
    except Exception as e:
        return html.Div([
            html.P("Error retrieving weather data:", className="text-danger"),
            html.P(str(e), className="text-danger")
        ])

# Callback to update the product dropdown based on store selection or toggle state
@app.callback(
    [Output("product-dropdown", "options"),
     Output("product-dropdown", "value")],
    [Input("store-dropdown", "value"),
     Input("item-number-toggle", "value")],
    [State("product-dropdown", "value")]
)
def update_product_dropdown(selected_store, show_item_numbers, current_product):
    if combined_data is None or selected_store is None:
        return product_options, current_product
    
    # Update global setting
    app.show_item_numbers = show_item_numbers
    
    # Filter products for the selected store
    try:
        store_products = combined_data[combined_data['Store_Id'] == selected_store].drop_duplicates(subset=['Item', 'Product'])
        
        # Set label based on toggle state
        if show_item_numbers:
            store_products['Label'] = store_products['Product'] + " (" + store_products['Item'].astype(str) + ")"
        else:
            store_products['Label'] = store_products['Product']
        
        options = [{"label": row['Label'], "value": row['Item']} for _, row in store_products.iterrows()]
        
        # Keep the same product selected if it exists in the new options
        valid_product = current_product if any(opt["value"] == current_product for opt in options) else options[0]["value"] if options else None
        
        return options, valid_product
    except Exception as e:
        print(f"Error updating product dropdown: {e}")
        return product_options, current_product

# Callback to update the sales and forecast chart
@app.callback(
    Output("sales-forecast-chart", "figure"),
    [Input("apply-filters", "n_clicks"),
     Input("rf-model-toggle", "value")],
    [State("store-dropdown", "value"),
     State("product-dropdown", "value"),
     State("date-range", "start_date"),
     State("date-range", "end_date"),
     State("weather-toggle", "value"),
     State("location-input", "value")]
)
def update_sales_forecast_chart(n_clicks, show_rf_model, store, product, start_date, end_date, apply_weather, location):
    try:
        # Debug logging for toggle state
        print(f"RF Model Toggle: {show_rf_model}")
        
        if combined_data is None or store is None or product is None:
            return go.Figure()
        
        # Filter data for selected store, product, and date range
        filtered_data = combined_data[(combined_data['Store_Id'] == store) & 
                                     (combined_data['Item'] == product)]
        
        # Convert string dates to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Filter by date range
        filtered_data = filtered_data[(filtered_data['Date'] >= start_date) & 
                                     (filtered_data['Date'] <= end_date)]
        
        # Get forecast data for both models if available
        pytorch_forecast_data = None
        rf_forecast_data = None
        
        if pytorch_forecasts is not None:
            pytorch_forecast_data = pytorch_forecasts[(pytorch_forecasts['Store_Id'] == store) & 
                                                  (pytorch_forecasts['Item'] == product)]
        
        if rf_forecasts is not None:
            rf_forecast_data = rf_forecasts[(rf_forecasts['Store_Id'] == store) & 
                                        (rf_forecasts['Item'] == product)]
            # Debug logging for RF data
            print(f"RF data filter results: store={store}, product={product}, rows found={len(rf_forecast_data)}")
            if len(rf_forecast_data) == 0:
                # Check if there's any data with this store ID
                store_data = rf_forecasts[rf_forecasts['Store_Id'] == store]
                print(f"RF data for store {store}: {len(store_data)} rows")
                
                # Check what store IDs are available in the RF data
                available_stores = rf_forecasts['Store_Id'].unique()
                print(f"Available store IDs in RF data: {available_stores}")
                
                # Check what item IDs are available for this store
                if len(store_data) > 0:
                    available_items = store_data['Item'].unique()
                    print(f"Available item IDs for store {store}: {available_items}")
        
        # For backward compatibility
        forecast_data = pytorch_forecast_data
        
        # Create the figure
        fig = go.Figure()
        
        # Add historical sales
        fig.add_trace(
            go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data['Sales'],
                mode='lines+markers',
                name='Actual Sales',
                line=dict(color='blue')
            )
        )
        
        # No historical stock level display
        
        # Add forecasts if available
        if pytorch_forecast_data is not None or rf_forecast_data is not None:
            # Apply weather adjustment if toggle is on
            weather_adjusted_forecast = None
            if apply_weather and location:
                try:
                    # Get weather forecast
                    weather_forecast = weather_service.get_weather_forecast(location)
                    
                    # Get product information for product-specific adjustments
                    product_name = filtered_data['Product'].iloc[0] if len(filtered_data) > 0 else None
                    
                    # Get the base demand values
                    base_demand = forecast_data['Predicted_Demand'].values if 'Predicted_Demand' in forecast_data.columns else forecast_data['Predicted_Sales'].values
                    
                    # Adjust demand based on weather forecast
                    adjusted_demand, adjustment_factors = weather_service.get_weather_adjusted_demand(
                        base_demand, weather_forecast, product_type=product_name
                    )
                    
                    # Create a copy of the forecast data with adjusted values
                    weather_adjusted_forecast = forecast_data.copy()
                    if 'Predicted_Demand' in weather_adjusted_forecast.columns:
                        weather_adjusted_forecast['Predicted_Demand'] = adjusted_demand
                        weather_adjusted_forecast['Weather_Impact'] = adjustment_factors
                    else:
                        weather_adjusted_forecast['Predicted_Sales'] = adjusted_demand
                        weather_adjusted_forecast['Weather_Impact'] = adjustment_factors
                except Exception as e:
                    print(f"Error applying weather adjustment: {e}")
            
            # Use weather-adjusted forecast if available, otherwise use original forecast
            # Get the product name
            product_name = filtered_data['Product'].iloc[0] if len(filtered_data) > 0 else "Product"
            
            if weather_adjusted_forecast is not None:
                # Add weather-adjusted forecast
                fig.add_trace(
                    go.Scatter(
                        x=weather_adjusted_forecast['Date'],
                        y=weather_adjusted_forecast['Predicted_Demand'] if 'Predicted_Demand' in weather_adjusted_forecast.columns else weather_adjusted_forecast['Predicted_Sales'],
                        mode='lines+markers',
                        name='Weather-Adjusted Forecast',
                        line=dict(color='orange'),
                        marker=dict(symbol='triangle-up')
                    )
                )
                
                # Add original forecast as a reference (dotted line)
                fig.add_trace(
                    go.Scatter(
                        x=forecast_data['Date'],
                        y=forecast_data['Predicted_Demand'] if 'Predicted_Demand' in forecast_data.columns else forecast_data['Predicted_Sales'],
                        mode='lines',
                        name='Base Forecast',
                        line=dict(color='red', dash='dot')
                    )
                )
            else:
                # Add both forecasts if available
                # Always show PyTorch forecast
                if pytorch_forecast_data is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=pytorch_forecast_data['Date'],
                            y=pytorch_forecast_data['Predicted_Demand'] if 'Predicted_Demand' in pytorch_forecast_data.columns else pytorch_forecast_data['Predicted_Sales'],
                            mode='lines+markers',
                            name='AI based Forecast',
                            line=dict(color='red'),
                            marker=dict(symbol='triangle-up')
                        )
                    )
            
            # Show Random Forest model if toggle is on (moved outside weather conditional)
            if rf_forecast_data is not None and show_rf_model:
                fig.add_trace(
                    go.Scatter(
                        x=rf_forecast_data['Date'],
                        y=rf_forecast_data['Predicted_Demand'] if 'Predicted_Demand' in rf_forecast_data.columns else rf_forecast_data['Predicted_Sales'],
                        mode='lines+markers',
                        name='Alternate AI Based Forecast',
                        line=dict(color='purple', dash='dot'),
                        marker=dict(symbol='circle')
                    )
                )
        
        # Update layout
        fig.update_layout(
            title=f"Sales History(over Date Range) and Two Week Forecast for {filtered_data['Product'].iloc[0] if len(filtered_data) > 0 else 'Product'}",
            xaxis_title="Date",
            yaxis_title="Sales (units)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            height=500
        )
        
        # Update y-axis label
        fig.update_yaxes(title_text="Sales (units)")
        
        # Add confidence interval for PyTorch forecast (95% CI)
        if pytorch_forecast_data is not None and len(pytorch_forecast_data) > 0:
            # Estimate standard error as 20% of predicted value (can be adjusted based on model accuracy)
            std_error = pytorch_forecast_data['Predicted_Demand'].values * 0.2 if 'Predicted_Demand' in pytorch_forecast_data.columns else pytorch_forecast_data['Predicted_Sales'].values * 0.2
            
            # Calculate upper and lower bounds
            upper_bound = (pytorch_forecast_data['Predicted_Demand'] if 'Predicted_Demand' in pytorch_forecast_data.columns else pytorch_forecast_data['Predicted_Sales']) + 1.96 * std_error
            lower_bound = (pytorch_forecast_data['Predicted_Demand'] if 'Predicted_Demand' in pytorch_forecast_data.columns else pytorch_forecast_data['Predicted_Sales']) - 1.96 * std_error
            lower_bound = lower_bound.clip(lower=0)  # Ensure non-negative values
            
            # Add confidence interval
            fig.add_trace(
                go.Scatter(
                    x=pytorch_forecast_data['Date'],
                    y=upper_bound,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=pytorch_forecast_data['Date'],
                    y=lower_bound,
                    mode='lines',
                    fill='tonexty',
                    line=dict(width=0),
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    name='AI based 95% Confidence Interval'
                )
            )
            
        # Add confidence interval for RF forecast if toggle is on and data is available (95% CI)
        if rf_forecast_data is not None and len(rf_forecast_data) > 0 and show_rf_model:
            # Estimate standard error as 25% of predicted value for RF (typically less accurate)
            std_error = rf_forecast_data['Predicted_Demand'].values * 0.25 if 'Predicted_Demand' in rf_forecast_data.columns else rf_forecast_data['Predicted_Sales'].values * 0.25
            
            # Calculate upper and lower bounds
            upper_bound = (rf_forecast_data['Predicted_Demand'] if 'Predicted_Demand' in rf_forecast_data.columns else rf_forecast_data['Predicted_Sales']) + 1.96 * std_error
            lower_bound = (rf_forecast_data['Predicted_Demand'] if 'Predicted_Demand' in rf_forecast_data.columns else rf_forecast_data['Predicted_Sales']) - 1.96 * std_error
            lower_bound = lower_bound.clip(lower=0)  # Ensure non-negative values
            
            # Add confidence interval
            fig.add_trace(
                go.Scatter(
                    x=rf_forecast_data['Date'],
                    y=upper_bound,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=rf_forecast_data['Date'],
                    y=lower_bound,
                    mode='lines',
                    fill='tonexty',
                    line=dict(width=0),
                    fillcolor='rgba(128, 0, 128, 0.1)',
                    name='Alternate AI Based 95% CI'
                )
            )
        
        return fig
        
    except Exception as e:
        print(f"Error updating sales forecast chart: {e}")
        # Return an empty figure with error message
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error Loading Sales Forecast Data",
            annotations=[{
                'text': f"Error: {str(e)}",
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5
            }]
        )
        return empty_fig

# Callback to update the inventory chart
'''@app.callback(
    Output("inventory-chart", "figure"),
    [Input("apply-filters", "n_clicks"),
     Input("apply-stock-adjustment", "n_clicks")],
    [State("store-dropdown", "value"),
     State("product-dropdown", "value"),
     State("weather-toggle", "value"),
     State("location-input", "value"),
     State("stock-adjustment-input", "value")]
)'''
def update_inventory_chart(n_clicks, adjust_clicks, store, product, apply_weather, location, stock_adjustment, adjustment_date=None):
    try:
        if combined_data is None or store is None or product is None:
            return go.Figure()
        
        # Filter data for selected store and product
        filtered_data = inventory_projection[(inventory_projection['Store_Id'] == store) & 
                                          (inventory_projection['Item'] == product)]
        
        # Create the figure with subplots for dual y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add stock level
        fig.add_trace(
            go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data['Stock_Level'],
                mode='lines',
                name='Stock Level',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add predicted demand on secondary y-axis
        if apply_weather and location:
            try:
                # Get weather forecast
                weather_forecast = weather_service.get_weather_forecast(location)
                
                # Get product information for product-specific adjustments
                product_name = filtered_data['Product'].iloc[0] if len(filtered_data) > 0 else None
                
                # Get the base demand values
                base_demand = filtered_data['Predicted_Demand'].values
                
                # Adjust demand based on weather forecast
                adjusted_demand, adjustment_factors = weather_service.get_weather_adjusted_demand(
                    base_demand, weather_forecast, product_type=product_name
                )
                
                # Add both original and weather-adjusted demand
                fig.add_trace(
                    go.Scatter(
                        x=filtered_data['Date'],
                        y=base_demand,
                        mode='lines',
                        name='Base Demand',
                        line=dict(color='red', dash='dot')
                    ),
                    secondary_y=True
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=filtered_data['Date'],
                        y=adjusted_demand,
                        mode='lines',
                        name='Weather-Adjusted Demand',
                        line=dict(color='orange')
                    ),
                    secondary_y=True
                )
            except Exception as e:
                print(f"Error applying weather adjustment: {e}")
                # Fall back to original demand if there's an error
                fig.add_trace(
                    go.Scatter(
                        x=filtered_data['Date'],
                        y=filtered_data['Predicted_Demand'],
                        mode='lines',
                        name='Daily Demand',
                        line=dict(color='red', dash='dash')
                    ),
                    secondary_y=True
                )
        else:
            # Use original demand
            fig.add_trace(
                go.Scatter(
                    x=filtered_data['Date'],
                    y=filtered_data['Predicted_Demand'],
                    mode='lines',
                    name='Daily Demand',
                    line=dict(color='red', dash='dash')
                ),
                secondary_y=True
            )
        
        # Add order points if available
        if 'Orders_Received' in filtered_data.columns:
            orders = filtered_data[filtered_data['Orders_Received'] > 0]
            if len(orders) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=orders['Date'],
                        y=orders['Stock_Level'],
                        mode='markers',
                        name='Order Received',
                        marker=dict(color='green', size=10, symbol='triangle-up')
                    )
                )
        
        # Add safety stock and target stock lines if available
        if 'Stock_Status' in filtered_data.columns and len(filtered_data) > 0:
            # Estimate safety stock as 1 week of demand
            avg_daily_demand = filtered_data['Predicted_Demand'].mean()
            safety_stock = avg_daily_demand * 7  # 1 week
            target_stock = avg_daily_demand * 14  # 2 weeks
            
            fig.add_trace(
                go.Scatter(
                    x=filtered_data['Date'],
                    y=[safety_stock] * len(filtered_data),
                    mode='lines',
                    name='Safety Stock (1 week)',
                    line=dict(color='red', dash='dot')
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=filtered_data['Date'],
                    y=[target_stock] * len(filtered_data),
                    mode='lines',
                    name='Target Stock (2 weeks)',
                    line=dict(color='green', dash='dot')
                )
            )
        
        # Update layout
        product_name = filtered_data['Product'].iloc[0] if len(filtered_data) > 0 else "Product"
        
        fig.update_layout(
            title=f"Inventory Projection for {product_name}",
            xaxis_title="Date",
            yaxis_title="Stock Level (units)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            height=500
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Stock Level (units)", secondary_y=False)
        fig.update_yaxes(title_text="Daily Demand (units)", secondary_y=True)
        
        return fig
        
    except Exception as e:
        print(f"Error updating inventory chart: {e}")
        # Return an empty figure with error message
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error Loading Inventory Data",
            annotations=[{
                'text': f"Error: {str(e)}",
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5
            }]
        )
        return empty_fig

# Callback for deep learning demand prediction tab
'''@app.callback(
    Output("deep-learning-chart", "figure"),
    [Input("apply-filters", "n_clicks"),
     Input("model-type-selector", "value"),
     Input("seasonality-features", "value")],
    [State("store-dropdown", "value"),
     State("product-dropdown", "value"),
     State("date-range", "start_date"),
     State("date-range", "end_date"),
     State("weather-toggle", "value"),
     State("location-input", "value")]
)'''
def update_deep_learning_chart(n_clicks, model_type, seasonality_features, 
                              store, product, start_date, end_date, 
                              apply_weather, location):
    try:
        if combined_data is None or store is None or product is None:
            return go.Figure()
        
        # Filter data for selected store, product, and date range
        filtered_data = combined_data[(combined_data['Store_Id'] == store) & 
                                     (combined_data['Item'] == product)]
        
        # Convert string dates to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Filter by date range
        filtered_data = filtered_data[(filtered_data['Date'] >= start_date) & 
                                     (filtered_data['Date'] <= end_date)]
        
        # Get forecast data
        pytorch_forecast_data = None
        if pytorch_forecasts is not None:
            pytorch_forecast_data = pytorch_forecasts[(pytorch_forecasts['Store_Id'] == store) & 
                                                  (pytorch_forecasts['Item'] == product)]
        
        # Create the figure
        fig = go.Figure()
        
        # Add historical sales
        fig.add_trace(
            go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data['Sales'],
                mode='lines+markers',
                name='Actual Sales',
                line=dict(color='blue')
            )
        )
        
        if pytorch_forecast_data is not None and len(pytorch_forecast_data) > 0:
            # Apply weather adjustment if toggle is on
            weather_adjusted_forecast = None
            if apply_weather and location:
                try:
                    # Get weather forecast
                    weather_forecast = weather_service.get_weather_forecast(location)
                    
                    # Get product information for product-specific adjustments
                    product_name = filtered_data['Product'].iloc[0] if len(filtered_data) > 0 else None
                    
                    # Get the base demand values
                    base_demand = pytorch_forecast_data['Predicted_Demand'].values if 'Predicted_Demand' in pytorch_forecast_data.columns else pytorch_forecast_data['Predicted_Sales'].values
                    
                    # Adjust demand based on weather forecast
                    adjusted_demand, adjustment_factors = weather_service.get_weather_adjusted_demand(
                        base_demand, weather_forecast, product_type=product_name
                    )
                    
                    # Create a copy of the forecast data with adjusted values
                    weather_adjusted_forecast = pytorch_forecast_data.copy()
                    if 'Predicted_Demand' in weather_adjusted_forecast.columns:
                        weather_adjusted_forecast['Predicted_Demand'] = adjusted_demand
                        weather_adjusted_forecast['Weather_Impact'] = adjustment_factors
                    else:
                        weather_adjusted_forecast['Predicted_Sales'] = adjusted_demand
                        weather_adjusted_forecast['Weather_Impact'] = adjustment_factors
                except Exception as e:
                    print(f"Error applying weather adjustment: {e}")
            
            # Define model color based on selected type
            model_color = 'purple' if model_type == 'transformer' else 'red'
            model_name = 'Transformer' if model_type == 'transformer' else 'LSTM'
            
            # Use weather-adjusted forecast if available, otherwise use original forecast
            forecast_to_use = weather_adjusted_forecast if weather_adjusted_forecast is not None else pytorch_forecast_data
            
            # Add forecast line
            fig.add_trace(
                go.Scatter(
                    x=forecast_to_use['Date'],
                    y=forecast_to_use['Predicted_Demand'] if 'Predicted_Demand' in forecast_to_use.columns else forecast_to_use['Predicted_Sales'],
                    mode='lines+markers',
                    name=f'{model_name} Forecast',
                    line=dict(color=model_color),
                    marker=dict(symbol='triangle-up')
                )
            )
            
            # Calculate confidence interval based on model type
            # Transformers generally have narrower confidence intervals than LSTMs
            ci_width = 0.15 if model_type == 'transformer' else 0.20
            
            # Estimate standard error as percentage of predicted value
            std_error = forecast_to_use['Predicted_Demand'].values * ci_width if 'Predicted_Demand' in forecast_to_use.columns else forecast_to_use['Predicted_Sales'].values * ci_width
            
            # Calculate upper and lower bounds
            upper_bound = (forecast_to_use['Predicted_Demand'] if 'Predicted_Demand' in forecast_to_use.columns else forecast_to_use['Predicted_Sales']) + 1.96 * std_error
            lower_bound = (forecast_to_use['Predicted_Demand'] if 'Predicted_Demand' in forecast_to_use.columns else forecast_to_use['Predicted_Sales']) - 1.96 * std_error
            lower_bound = lower_bound.clip(lower=0)  # Ensure non-negative values
            
            # Add confidence interval
            fig.add_trace(
                go.Scatter(
                    x=forecast_to_use['Date'],
                    y=upper_bound,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                )
            )
            
            fillcolor = 'rgba(128, 0, 128, 0.1)' if model_type == 'transformer' else 'rgba(255, 0, 0, 0.1)'
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_to_use['Date'],
                    y=lower_bound,
                    mode='lines',
                    fill='tonexty',
                    line=dict(width=0),
                    fillcolor=fillcolor,
                    name=f'{model_name} 95% CI'
                )
            )
        
        # Get product name for title
        product_name = filtered_data['Product'].iloc[0] if len(filtered_data) > 0 else "Product"
        
        # Create list of active seasonality features for title
        active_features = ', '.join([f.capitalize() for f in seasonality_features])
        
        # Update layout with seasonality features info
        fig.update_layout(
            title=f"Deep Learning Prediction for {product_name} using {model_type.upper()} Model<br><sub>Features: {active_features} seasonality</sub>",
            xaxis_title="Date",
            yaxis_title="Sales (units)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            height=500
        )
        
        # Add annotations about seasonality
        feature_descriptions = {
            'daily': 'Day-of-week patterns',
            'weekly': '7-day window patterns',
            'monthly': 'Monthly seasonality',
            'yearly': 'Annual seasonal trends',
            'holidays': 'Holiday effects'
        }
        
        annotations = []
        
        for i, feature in enumerate(seasonality_features):
            annotations.append(
                dict(
                    x=0.01,
                    y=0.97 - (i * 0.05),
                    xref="paper",
                    yref="paper",
                    text=f"✓ {feature_descriptions.get(feature, feature)}",
                    showarrow=False,
                    font=dict(size=10),
                    bgcolor="rgba(255, 255, 255, 0.7)",
                    bordercolor="#c7c7c7",
                    borderwidth=1,
                    borderpad=4
                )
            )
        
        fig.update_layout(annotations=annotations)
        
        return fig
        
    except Exception as e:
        print(f"Error updating deep learning chart: {e}")
        # Return an empty figure with error message
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error Loading Deep Learning Prediction Data",
            annotations=[{
                'text': f"Error: {str(e)}",
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5
            }]
        )
        return empty_fig

# Callback to update the price elasticity chart
@app.callback(
    Output("price-elasticity-chart", "figure"),
    [Input("apply-filters", "n_clicks")],
    [State("store-dropdown", "value"),
     State("product-dropdown", "value")]
)
def update_price_elasticity_chart(n_clicks, store, product):
    try:
        if price_elasticities is None or combined_data is None or store is None or product is None:
            return go.Figure()
            
        # Filter data for selected store and product
        elasticity_data = price_elasticities[(price_elasticities['Store_Id'] == store) & 
                                          (price_elasticities['Item'] == product)]
        
        sales_data = combined_data[(combined_data['Store_Id'] == store) & 
                                 (combined_data['Item'] == product)]
        
        if len(elasticity_data) == 0 or len(sales_data) == 0:
            return go.Figure()
        
        # Create the figure
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=("Price vs. Sales", "Price Sensitivity Curve"),
                           specs=[[{"type": "scatter"}, {"type": "scatter"}]])
        
        # Add price vs. sales scatter plot
        fig.add_trace(
            go.Scatter(
                x=sales_data['Price'],
                y=sales_data['Sales'],
                mode='markers',
                name='Historical Data',
                marker=dict(color='blue', size=8)
            ),
            row=1, col=1
        )
        
        # Create price sensitivity curve if elasticity is available
        if len(elasticity_data) > 0:
            elasticity = elasticity_data['Elasticity'].iloc[0]
            avg_price = elasticity_data['Avg_Price'].iloc[0]
            unit_cost = elasticity_data['Unit_Cost'].iloc[0]
            
            # Generate price range for curve
            price_range = np.linspace(avg_price * 0.7, avg_price * 1.3, 100)
            
            # Calculate predicted sales at each price point using elasticity
            # Q2/Q1 = (P1/P2)^e
            base_sales = sales_data['Sales'].mean()
            predicted_sales = [base_sales * (avg_price / p) ** elasticity for p in price_range]
            
            # Calculate profit at each price point
            profit = [(p - unit_cost) * s for p, s in zip(price_range, predicted_sales)]
            
            # Find optimal profit price
            optimal_idx = np.argmax(profit)
            optimal_price = price_range[optimal_idx]
            
            # Add price sensitivity curve
            fig.add_trace(
                go.Scatter(
                    x=price_range,
                    y=predicted_sales,
                    mode='lines',
                    name='Price Sensitivity',
                    line=dict(color='red')
                ),
                row=1, col=2
            )
            
            '''# Add optimal price marker
            fig.add_trace(
                go.Scatter(
                    x=[optimal_price],
                    y=[predicted_sales[optimal_idx]],
                    mode='markers',
                    name=f'Optimal Price: ${optimal_price:.2f}',
                    marker=dict(color='green', size=12, symbol='star')
                ),
                row=1, col=2
            )'''
            
            # Add current price marker
            fig.add_trace(
                go.Scatter(
                    x=[avg_price],
                    y=[base_sales],
                    mode='markers',
                    name=f'Current Avg: ${avg_price:.2f}',
                    marker=dict(color='black', size=10, symbol='star')
                ),
                row=1, col=2
            )
            
            # Add unit cost line
            fig.add_trace(
                go.Scatter(
                    x=[unit_cost, unit_cost],
                    y=[0, max(predicted_sales) * 1.1],
                    mode='lines',
                    name=f'Unit Cost: ${unit_cost:.2f}',
                    line=dict(color='black', dash='dash')
                ),
                row=1, col=2
            )
            
            # Add elasticity value as annotation
            fig.add_annotation(
                text=f"Elasticity: {elasticity:.2f}",
                xref="x2", yref="y2",
                x=price_range[int(len(price_range) * 0.8)],
                y=predicted_sales[int(len(predicted_sales) * 0.8)],
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
        
        # Update layout
        product_name = sales_data['Product'].iloc[0]
        
        fig.update_layout(
            title=f"Price Elasticity Analysis for {product_name}",
            xaxis_title="Price ($)",
            yaxis_title="Sales (units)",
            xaxis2_title="Price ($)",
            yaxis2_title="Predicted Demand (units)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            height=500,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        print(f"Error updating price elasticity chart: {e}")
        # Return an empty figure with error message
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error Loading Price Elasticity Data",
            annotations=[{
                'text': f"Error: {str(e)}",
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5
            }]
        )
        return empty_fig

# Callback to update the profit impact chart
@app.callback(
    Output("profit-impact-chart", "figure"),
    [Input("apply-filters", "n_clicks")],
    [State("store-dropdown", "value"),
     State("product-dropdown", "value")]
)
def update_profit_impact_chart(n_clicks, store, product):
    try:
        if profit_impact is None or price_recommendations is None or store is None or product is None:
            return go.Figure()
        
        # Filter data for selected store and product
        profit_data = profit_impact[(profit_impact['Store_Id'] == store) & 
                                  (profit_impact['Item'] == product)]
        
        price_rec = price_recommendations[(price_recommendations['Store_Id'] == store) & 
                                       (price_recommendations['Item'] == product)]
        
        if len(profit_data) == 0 or len(price_rec) == 0:
            return go.Figure()
        
        # Create the figure
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=("Profit Comparison", "Price Recommendation"),
                           specs=[[{"type": "bar"}, {"type": "indicator"}]])
        
        # Add profit comparison bar chart
        baseline_profit = profit_data['Baseline_Profit'].iloc[0]
        projected_profit = profit_data['Projected_Profit'].iloc[0]
        profit_impact_val = profit_data['Profit_Impact'].iloc[0]
        profit_impact_pct = profit_data['Profit_Impact_Pct'].iloc[0]
        
        fig.add_trace(
            go.Bar(
                x=['Current', 'Optimized'],
                y=[baseline_profit, projected_profit],
                text=[f"${baseline_profit:.2f}", f"${projected_profit:.2f}"],
                textposition='auto',
                name='Profit',
                marker_color=['blue', 'green'],
                hoverinfo='y+text',
                hovertemplate='<b>%{x}</b><br>Profit: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add price recommendation gauge chart
        current_price = price_rec['Current_Price'].iloc[0]
        optimal_price = price_rec['Optimal_Price'].iloc[0]
        unit_cost = profit_data['Unit_Cost'].iloc[0]
        
        price_diff_pct = (optimal_price - current_price) / current_price * 100
        
        # Create a range from cost to max price
        price_range = [unit_cost, current_price * 1.5]
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=optimal_price,
                number={"prefix": "$", "valueformat": ".2f"},
                delta={"reference": current_price, 
                       "valueformat": ".2f", 
                       "prefix": "$",
                       "increasing": {"color": "green" if price_diff_pct > 0 else "red"}},
                gauge={
                    "axis": {"range": price_range},
                    "bar": {"color": "green" if price_diff_pct > 0 else "red"},
                    "steps": [
                        {"range": [unit_cost, current_price], "color": "lightgray"},
                        {"range": [current_price, price_range[1]], "color": "white"}
                    ],
                    "threshold": {
                        "line": {"color": "blue", "width": 4},
                        "thickness": 0.75,
                        "value": current_price
                    }
                },
                title={"text": "Recommended Price"}
            ),
            row=1, col=2
        )
        
        # Update layout
        product_name = profit_data['Product'].iloc[0]
        
        fig.update_layout(
            title=f"Profit Optimization for {product_name}",
            template="plotly_white",
            height=500
        )
        
        # Add annotations for profit impact
        fig.add_annotation(
            xref="x1", yref="paper",
            x=0.5, y=-0.15,
            text=f"Profit Impact using Sales and Purchase Data: ${profit_impact_val:.2f} ({profit_impact_pct:.1f}%)",
            showarrow=False,
            font=dict(
                size=14,
                color="green" if profit_impact_val > 0 else "red"
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error updating profit impact chart: {e}")
        # Return an empty figure with error message
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error Loading Profit Impact Data",
            annotations=[{
                'text': f"Error: {str(e)}",
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5
            }]
        )
        return empty_fig

# Removed recommendations container callback
        inv_rec = inventory_recs[(inventory_recs['Store_Id'] == store) & 
                                (inventory_recs['Item'] == product)]
        
        if len(inv_rec) > 0:
            order_qty = inv_rec['Order_Quantity'].iloc[0]
            stock_status = inv_rec['Stock_Status'].iloc[0]
            weeks_supply = inv_rec['Weeks_Of_Supply'].iloc[0]
            
            # Apply weather adjustment to order quantity if needed
            weather_adjusted_qty = order_qty
            weather_info = ""
            
            if apply_weather and location:
                try:
                    # Get product information for product-specific adjustments
                    product_name = inv_rec['Product'].iloc[0] if len(inv_rec) > 0 else None
                    
                    # Get detailed weather impact analysis
                    impact_analysis = weather_service.get_detailed_weather_impact(
                        location, product_type=product_name, lead_time=14  # 2 weeks for inventory planning
                    )
                    
                    # Use overall impact factor for order quantity adjustment
                    impact_factor = impact_analysis['overall_impact_factor']
                    
                    # Adjust order quantity based on weather (inverse impact to maintain stock)
                    weather_adjusted_qty = int(order_qty * (1 / impact_factor))
                    
                    # Add weather info to display
                    if impact_factor < 1.0:
                        # Create more detailed weather impact information
                        impact_direction = "increased" if weather_adjusted_qty > order_qty else "decreased"
                        impact_percent = abs(int((weather_adjusted_qty - order_qty) / order_qty * 100))
                        
                        # Create impact summary table
                        impact_rows = []
                        for day in impact_analysis['daily_impacts'][:7]:  # Show first week
                            impact_rows.append(
                                html.Tr([
                                    html.Td(day['date']),
                                    html.Td(f"{day['weather_category']} ({day['temperature']:.1f}°F)"),
                                    html.Td(
                                        html.Span(
                                            f"{day['impact_direction'].capitalize()} {day['impact_percent']}",
                                            className=f"text-{'danger' if day['impact_direction'] == 'decrease' else 'success' if day['impact_direction'] == 'increase' else 'muted'}"
                                        )
                                    )
                                ])
                            )
                        
                        weather_info = html.Div([
                            html.P(f"Weather Impact Analysis for {product_name}", 
                                   className="mt-2 mb-1 fw-bold"),
                            html.P([
                                "Order quantity ", 
                                html.Span(f"{impact_direction} by {impact_percent}%", 
                                          className=f"{'text-danger' if impact_direction == 'increased' else 'text-success'} fw-bold"),
                                " based on 2-week weather forecast."
                            ]),
                            html.P("Daily Weather Impact:", className="mb-1 mt-2"),
                            dbc.Table(
                                [
                                    html.Thead(
                                        html.Tr([
                                            html.Th("Date"),
                                            html.Th("Weather"),
                                            html.Th("Impact")
                                        ])
                                    ),
                                    html.Tbody(impact_rows)
                                ],
                                bordered=True,
                                hover=True,
                                responsive=True,
                                size="sm"
                            )
                        ])
                    else:
                        weather_info = html.P(f"Weather ({current_weather['weather_category']}) has no significant impact on order quantity.", 
                                            className="text-muted mt-2")
                except Exception as e:
                    print(f"Error applying weather adjustment to order quantity: {e}")
            
            recommendations.append(
                dbc.Card([
                    dbc.CardHeader(html.H5("Inventory Recommendation", className="text-center")),
                    dbc.CardBody([
                        html.H6(f"Current Stock Status: {stock_status}"),
                        html.P(f"Weeks of Supply: {weeks_supply:.1f}"),
                        html.P([
                            "Recommended Order Quantity: ",
                            html.Span(
                                f"{weather_adjusted_qty if apply_weather and location else order_qty} units", 
                                className="fw-bold text-primary"
                            ),
                            " ",
                            html.Small(f"(Base: {order_qty} units)", className="text-muted") if weather_adjusted_qty != order_qty else ""
                        ]),
                        html.P(inv_rec['Recommendation_Reason'].iloc[0]),
                        weather_info if apply_weather and location else ""
                    ])
                ])
            )
    

# Callback to update the inventory summary chart
'''@app.callback(
    Output("inventory-summary", "figure"),
    [Input("apply-filters", "n_clicks")],
    [State("store-dropdown", "value")]
)'''
'''
def update_inventory_summary(n_clicks, store):
    if inventory_recs is None or store is None:
        return go.Figure()
    
    # Filter data for selected store
    store_data = inventory_recs[inventory_recs['Store_Id'] == store]
    
    # Group by stock status
    status_counts = store_data.groupby('Stock_Status').size()
    
    # Create the figure
    fig = go.Figure(data=[
        go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=.3,
            marker_colors=['red', 'green', 'orange']
        )
    ])
    
    fig.update_layout(
        title=f"Inventory Status Distribution (Store {store})",
        template="plotly_white"
    )
    
    return fig
'''

# Callback to update the price impact summary chart
@app.callback(
    Output("price-impact-summary", "figure"),
    [Input("apply-filters", "n_clicks")],
    [State("store-dropdown", "value")]
)
def update_price_impact_summary(n_clicks, store):
    if profit_impact is None or store is None:
        return go.Figure()
    
    # Filter data for selected store
    store_data = profit_impact[profit_impact['Store_Id'] == store]
    
    # Sort by profit impact
    top_products = store_data.sort_values('Profit_Impact', ascending=False).head(10)
    
    # Create the figure
    fig = go.Figure()
    
    # Add bar chart with profit impact
    fig.add_trace(
        go.Bar(
            y=[f"{row['Product'][:25]}..." for _, row in top_products.iterrows()],
            x=top_products['Profit_Impact'],
            orientation='h',
            marker_color=['green' if x > 0 else 'red' for x in top_products['Profit_Impact']],
            name="Profit Impact",
            text=[f"${x:.2f}" for x in top_products['Profit_Impact']],
            textposition="auto"
        )
    )
    
    # Add profit margin indicator as a secondary visualization
    if 'Profit_Impact_Pct' in top_products.columns:
        fig.add_trace(
            go.Scatter(
                y=[f"{row['Product'][:25]}..." for _, row in top_products.iterrows()],
                x=top_products['Profit_Impact'],
                mode='markers',
                marker=dict(
                    color=top_products['Profit_Impact_Pct'],
                    size=16,
                    colorscale='RdYlGn',
                    colorbar=dict(title="Profit Impact %"),
                    showscale=True,
                    cmin=0,
                    cmax=max(0.5, top_products['Profit_Impact_Pct'].max())
                ),
                text=[f"{x:.1f}%" for x in top_products['Profit_Impact_Pct']],
                hoverinfo='text',
                name="Profit Impact %"
            )
        )
    
    # Calculate total impact for title
    total_impact = top_products['Profit_Impact'].sum()
    
    fig.update_layout(
        title=f"Top 10 Products by Profit Impact (Store {store}) - Total: ${total_impact:.2f}",
        xaxis_title="30-Day Profit Impact ($)",
        yaxis_title="Product",
        template="plotly_white",
        height=600,
        margin=dict(l=200, r=20, t=60, b=40)
    )
    
    return fig

# Callback to update the lead time weather analysis
@app.callback(
    [
        Output("lead-time-weather-chart", "figure"),
        Output("lead-time-weather-details", "children")
    ],
    [Input("apply-filters", "n_clicks")],
    [State("store-dropdown", "value"),
     State("product-dropdown", "value"),
     State("weather-toggle", "value"),
     State("location-input", "value")]
)
def update_lead_time_weather_chart(n_clicks, store, product, apply_weather, location):
    # Default empty figure and no details
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="No Weather Data Available",
        xaxis_title="Date",
        yaxis_title="Impact",
        template="plotly_white",
        height=500,
        annotations=[
            dict(
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                text="Toggle weather adjustments and enter a ZIP code to see analysis",
                showarrow=False,
                font=dict(size=16)
            )
        ]
    )
    
    no_details = html.P("No weather analysis available. Toggle weather adjustments and enter a location.")
    
    if not apply_weather or not location or store is None or product is None:
        return empty_fig, no_details
    
    try:
        # Get product name
        product_data = combined_data[(combined_data['Store_Id'] == store) & 
                                      (combined_data['Item'] == product)]
        product_name = product_data['Product'].iloc[0] if len(product_data) > 0 else "Product"
        
        # Get detailed weather impact analysis for different lead times
        lead_times = [7, 14, 30]  # 1 week, 2 weeks, 1 month
        lead_time_impacts = {}
        
        for lead_time in lead_times:
            lead_time_impacts[lead_time] = weather_service.get_detailed_weather_impact(
                location, product_type=product_name, lead_time=lead_time
            )
        
        # Create a figure showing impact over different lead times
        fig = go.Figure()
        
        # Check if any lead time has None data
        has_valid_data = all(impact_data is not None for impact_data in lead_time_impacts.values())
        
        if not has_valid_data:
            # If any lead time has None data, return an error figure
            fig.add_annotation(
                text="Error retrieving weather data. Please check the location and try again.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red")
            )
        else:
            # Prepare data for the chart
            for lead_time in lead_times:
                impact_data = lead_time_impacts[lead_time]
                
                # Extract dates and impact factors
                dates = [day['date'] for day in impact_data['daily_impacts']]
                impact_factors = [day['impact_factor'] for day in impact_data['daily_impacts']]
                
                # Add trace for this lead time
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=impact_factors,
                        mode='lines+markers',
                        name=f"{lead_time}-Day Lead Time",
                        hovertemplate=(
                            "<b>Date:</b> %{x}<br>"
                            "<b>Impact Factor:</b> %{y:.2f}<br>"
                        )
                    )
                )
        
        # Add reference line for no impact
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="No Impact (1.0)")
        
        # Update layout
        fig.update_layout(
            title=f"Weather Impact on {product_name} by Lead Time for {location}",
            xaxis_title="Date",
            yaxis_title="Impact Factor",
            yaxis=dict(
                tickformat=".2f",
                range=[0.4, 1.2]  # Adjust as needed based on your impact factors
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            height=500
        )
        
        # Create detailed analysis table
        lead_time_rows = []
        
        if has_valid_data:
            for lead_time in lead_times:
                impact_data = lead_time_impacts[lead_time]
                lead_time_rows.append(
                    html.Tr([
                        html.Td(f"{lead_time} days"),
                        html.Td(f"{impact_data['overall_impact_factor']:.3f}"),
                        html.Td(
                            html.Span(
                                f"{impact_data['overall_impact_direction'].capitalize()} by {impact_data['overall_impact_percent']}",
                                className=f"text-{'danger' if impact_data['overall_impact_direction'] == 'decrease' else 'success' if impact_data['overall_impact_direction'] == 'increase' else 'muted'}"
                            )
                        ),
                        html.Td([
                            html.Span(
                                f"Demand {'reduced' if impact_data['overall_impact_factor'] < 1 else 'increased'}", 
                                className="fw-bold"
                            ),
                            ", order should be ",
                            html.Span(
                                f"{'increased' if impact_data['overall_impact_factor'] < 1 else 'reduced'}",
                                className="fw-bold"
                            ),
                            f" to maintain adequate stock over {lead_time} days."
                        ])
                    ])
                )
        else:
            # Add a single row explaining the error
            lead_time_rows.append(
                html.Tr([
                    html.Td(colspan=4, className="text-center text-danger", 
                           children="Unable to retrieve weather impact data. Please check the location and try again.")
                ])
            )
        
        analysis_table = dbc.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("Lead Time"),
                        html.Th("Impact Factor"),
                        html.Th("Effect"),
                        html.Th("Recommendation")
                    ])
                ),
                html.Tbody(lead_time_rows)
            ],
            bordered=True,
            hover=True,
            responsive=True
        )
        
        # Create additional insights
        # Check if impact_data is None before accessing its values
        if impact_data is None:
            weather_insights = dbc.Card([
                dbc.CardHeader(html.H5("Supply Chain Weather Insights")),
                dbc.CardBody([
                    html.P("Unable to retrieve weather data for this location. Please check the location and try again.")
                ])
            ])
        else:
            current_weather = impact_data['current_weather']
            weather_insights = dbc.Card([
                dbc.CardHeader(html.H5("Supply Chain Weather Insights")),
                dbc.CardBody([
                    html.P([
                        "Current weather in ", 
                        html.B(location), 
                        ": ", 
                        f"{current_weather['temperature_f']:.1f}°F, {current_weather['weather_category']}"
                    ]),
                    html.P([
                        "Weather has a ",
                        html.B(f"{lead_time_impacts[7]['overall_impact_direction']}"),
                        f" effect on demand for {product_name} over the next 7 days."
                    ]),
                    html.P([
                        "Suggested action: ",
                        html.B(
                            f"{'Increase' if lead_time_impacts[14]['overall_impact_factor'] < 1 else 'Decrease'} order quantities by approximately {abs(int((1 - lead_time_impacts[14]['overall_impact_factor']) * 100))}%"
                        ),
                        " for the next 14-day supply period."
                    ])
                ])
            ])
        
        return fig, html.Div([
            dbc.Row([
                dbc.Col(weather_insights, width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H5("Lead Time Impact Analysis", className="mt-3"),
                    html.P("How weather affects demand and ordering over different time periods:"),
                    analysis_table
                ], width=12)
            ])
        ])
    except Exception as e:
        print(f"Error updating lead time weather chart: {e}")
        return empty_fig, html.Div([
            dbc.Alert(
                [
                    html.H5("Error Analyzing Weather Impact"),
                    html.P(str(e))
                ],
                color="danger"
            )
        ])

# Register all inventory callbacks

# Callback to update the inventory chart
@app.callback(
    Output("inventory-chart", "figure"),
    [Input("apply-filters", "n_clicks"),
     Input("apply-stock-adjustment", "n_clicks")],
    [State("store-dropdown", "value"),
     State("product-dropdown", "value"),
     State("weather-toggle", "value"),
     State("location-input", "value"),
     State("stock-adjustment-input", "value"),
     State("stock-adjustment-date", "date")]
)
def inventory_chart_callback(n_clicks, adjust_clicks, store, product, apply_weather, location, stock_adjustment, adjustment_date):
    # Apply stock adjustments using centralized function
    adjusted_data, _ = apply_stock_adjustments(
        combined_data, store, product, stock_adjustment, adjust_clicks, adjustment_date
    )
    
    # Call with all the parameters, including adjustment_date and adjusted data
    # We need to modify the function call since we already have the adjusted data
    try:
        # Import required modules within the function to use the modified data
        from plotly_dashboard import app
        
        # Proceed with the same logic as update_inventory_chart but use our adjusted_data
        return update_inventory_chart(n_clicks, adjust_clicks, store, product, apply_weather, 
                                    location, stock_adjustment, adjustment_date)
    except Exception as e:
        print(f"Error in inventory chart callback: {e}")
        # Return an empty figure with error message
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error Loading Inventory Data",
            annotations=[{
                'text': f"Error: {str(e)}",
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5
            }]
        )
        return empty_fig
                                
# Register summary dashboard callbacks
summary_dashboard.register_summary_callbacks(app, combined_data, profit_impact, inventory_projection, pytorch_forecasts)

# Callback to update stock recommendations
@app.callback(
    Output("stock-recommendations", "children"),
    [Input("apply-filters", "n_clicks"),
     Input("apply-stock-adjustment", "n_clicks")],
    [State("store-dropdown", "value"),
     State("product-dropdown", "value"),
     State("stock-adjustment-input", "value"),
     State("stock-adjustment-date", "date")]
)
def stock_recommendations_callback(n_clicks, adjust_clicks, store, product, stock_adjustment, adjustment_date):
    # Apply stock adjustments using centralized function
    adjusted_data, _ = apply_stock_adjustments(
        combined_data, store, product, stock_adjustment, adjust_clicks, adjustment_date
    )
    
    return update_stock_recommendations(n_clicks, adjust_clicks, store, product, 
                                      stock_adjustment, adjusted_data, html, dbc, app, adjustment_date)

# Callback to update inventory turnover chart
@app.callback(
    Output("inventory-turnover-chart", "figure"),
    [Input("apply-filters", "n_clicks"),
     Input("apply-stock-adjustment", "n_clicks")],
    [State("store-dropdown", "value"),
     State("product-dropdown", "value"),
     State("stock-adjustment-input", "value"),
     State("stock-adjustment-date", "date")]
)
def inventory_turnover_callback(n_clicks, adjust_clicks, store, product, stock_adjustment, adjustment_date):
    # Apply stock adjustments using centralized function
    adjusted_data, _ = apply_stock_adjustments(
        combined_data, store, product, stock_adjustment, adjust_clicks, adjustment_date
    )
    
    return update_stock_velocity_chart(n_clicks, adjust_clicks, store, product, adjusted_data, go, np, stock_adjustment, adjustment_date)

# Callback to update inventory cost impact chart
@app.callback(
    Output("inventory-cost-impact-chart", "figure"),
    [Input("apply-filters", "n_clicks"),
     Input("apply-stock-adjustment", "n_clicks")],
    [State("store-dropdown", "value"),
     State("product-dropdown", "value"),
     State("stock-adjustment-input", "value"),
     State("stock-adjustment-date", "date")]
)
def inventory_cost_impact_callback(n_clicks, adjust_clicks, store, product, stock_adjustment, adjustment_date):
    # Apply stock adjustments using centralized function
    adjusted_data, _ = apply_stock_adjustments(
        combined_data, store, product, stock_adjustment, adjust_clicks, adjustment_date
    )
    
    return update_stock_penalty_chart(n_clicks, adjust_clicks, store, product, adjusted_data, go, np, stock_adjustment, adjustment_date)

# Callback to update inventory summary statistics
@app.callback(
    Output("inventory-summary-stats", "children"),
    [Input("apply-filters", "n_clicks"),
     Input("apply-stock-adjustment", "n_clicks")],
    [State("store-dropdown", "value"),
     State("product-dropdown", "value"),
     State("stock-adjustment-input", "value"),
     State("stock-adjustment-date", "date")]
)
def inventory_summary_callback(n_clicks, adjust_clicks, store, product, stock_adjustment, adjustment_date):
    # Apply stock adjustments using centralized function
    adjusted_data, _ = apply_stock_adjustments(
        combined_data, store, product, stock_adjustment, adjust_clicks, adjustment_date
    )
    
    return update_inventory_summary_stats(n_clicks, adjust_clicks, store, product, stock_adjustment, adjusted_data, html, dbc, app, adjustment_date)

# Callback to update the profit comparison chart
@app.callback(
    Output("profit-comparison-chart", "figure"),
    [Input("apply-filters", "n_clicks")],
    [State("store-dropdown", "value")]
)
def update_profit_comparison_chart(n_clicks, store):
    if profit_impact is None or store is None:
        return go.Figure()
    
    try:
        # Filter data for selected store
        store_data = profit_impact[profit_impact['Store_Id'] == store]
        
        # Calculate overall profit comparison
        baseline_total = store_data['Baseline_Profit'].sum()
        projected_total = store_data['Projected_Profit'].sum()
        profit_impact_total = projected_total - baseline_total
        profit_impact_pct = (profit_impact_total / baseline_total * 100) if baseline_total > 0 else 0
        
        # Create a more balanced and attractive figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.5, 0.5],
            specs=[[{"type": "domain"}, {"type": "domain"}]],
            subplot_titles=["Baseline Profit", "Projected Profit"]
        )
        
        # Add baseline profit gauge
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=baseline_total,
                number={
                    'prefix': "$",
                    'valueformat': ".2f",
                    'font': {"size": 50, "color": "#2c3e50"}
                },
                title={
                    'text': "Current",
                    'font': {"size": 20, "color": "#7f8c8d"}
                },
                domain={'row': 0, 'column': 0}
            ),
            row=1, col=1
        )
        
        # Add projected profit gauge with delta
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=projected_total,
                delta={
                    'reference': baseline_total,
                    'relative': False,
                    'valueformat': "+$.2f",
                    'font': {"size": 20},
                    'increasing': {'color': "#27ae60"},
                    'decreasing': {'color': "#e74c3c"}
                },
                number={
                    'prefix': "$",
                    'valueformat': ".2f",
                    'font': {"size": 50, "color": "#2c3e50"}
                },
                title={
                    'text': "Optimized",
                    'font': {"size": 20, "color": "#7f8c8d"}
                },
                domain={'row': 0, 'column': 1}
            ),
            row=1, col=2
        )
        
        # Add percentage change as a circular gauge in the center
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=abs(profit_impact_pct),  # Use absolute value for gauge
                gauge={
                    'axis': {'range': [None, max(10, abs(profit_impact_pct) * 1.5)]},  # Dynamic range
                    'bar': {'color': "#27ae60" if profit_impact_pct >= 0 else "#e74c3c"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#7f8c8d",
                    'steps': [
                        {'range': [0, abs(profit_impact_pct) * 0.5], 'color': "#edf2f7"},
                        {'range': [abs(profit_impact_pct) * 0.5, abs(profit_impact_pct)], 'color': "#e2e8f0"}
                    ]
                },
                number={
                    'suffix': "%",
                    'valueformat': ".1f",
                    'font': {"size": 30, "color": "#27ae60" if profit_impact_pct >= 0 else "#e74c3c"}
                },
                title={
                    'text': "Profit Change",
                    'font': {"size": 16}
                },
                domain={'x': [0.35, 0.65], 'y': [0.15, 0.55]}
            )
        )
        
        # Update layout for the entire figure
        fig.update_layout(
            title={
                'text': f"Overall Profit Comparison - Store {store}",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {"size": 24, "color": "#1e3a8a"}
            },
            template="plotly_white",
            height=450,  # Slightly taller for better proportions
            margin=dict(l=20, r=20, t=100, b=50),
            showlegend=False,
            annotations=[
                # Clean, centered impact annotation
                dict(
                    text=f"Total Profit Impact: ${profit_impact_total:.2f}",
                    x=0.5,
                    y=0.02,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(
                        size=18,
                        color="#27ae60" if profit_impact_pct >= 0 else "#e74c3c"
                    )
                )
            ]
        )
        
        return fig
        
    except Exception as e:
        print(f"Error updating profit comparison chart: {e}")
        # Return an empty figure with error message
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error Loading Profit Comparison Data",
            annotations=[{
                'text': f"Error: {str(e)}",
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5
            }]
        )
        return empty_fig

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)