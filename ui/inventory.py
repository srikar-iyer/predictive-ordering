"""
Inventory management UI components for the Pizza Predictive Ordering dashboard.
"""
import sys
import os
import logging
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core UI components
from ui.core import (
    load_dashboard_data, create_store_product_selectors,
    create_date_range_slider, create_toggle_switch,
    create_error_message, create_info_card, apply_stock_adjustments
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('inventory_ui')

# Import settings if available
try:
    from config.settings import (
        MIN_STOCK_WEEKS, TARGET_STOCK_WEEKS, MAX_STOCK_WEEKS,
        INVENTORY_PROJECTION_FILE, INVENTORY_RECOMMENDATIONS_FILE
    )
except ImportError:
    # Default values
    MIN_STOCK_WEEKS = 1
    TARGET_STOCK_WEEKS = 2
    MAX_STOCK_WEEKS = 3
    
    # Default paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INVENTORY_PROJECTION_FILE = os.path.join(ROOT_DIR, "inventory_projection.csv")
    INVENTORY_RECOMMENDATIONS_FILE = os.path.join(ROOT_DIR, "inventory_recommendations.csv")


def create_inventory_tab_content():
    """
    Create content for the inventory management tab.
    
    Returns:
        html.Div: Tab content
    """
    return html.Div([
        dbc.Tabs([
            dbc.Tab(label="Inventory Status", tab_id="inventory-status-tab",
                   children=create_inventory_status_content()),
            dbc.Tab(label="Order Recommendations", tab_id="order-recommendations-tab",
                   children=create_order_recommendations_content()),
            dbc.Tab(label="Inventory Analysis", tab_id="inventory-analysis-tab",
                   children=create_inventory_analysis_content())
        ], id="inventory-tabs")
    ])


def create_inventory_status_content():
    """
    Create content for the inventory status subtab.
    
    Returns:
        html.Div: Subtab content
    """
    return html.Div([
        # Controls
        dbc.Row([
            dbc.Col([
                # Store and product selectors
                create_store_product_selectors('inventory-status', [], [])
            ], width=12)
        ]),
        
        # Enhanced Inventory adjustment with more options
        dbc.Card([
            dbc.CardHeader(
                html.H5([
                    html.I(className="fas fa-edit mr-2"), 
                    "Manual Inventory Adjustment"
                ], className="m-0")
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Current Inventory Count:"),
                        dbc.InputGroup([
                            dbc.Input(
                                id="inventory-count-input",
                                type="number",
                                placeholder="Enter current count",
                                min=0,
                                step=1
                            ),
                            dbc.Button("Update", id="inventory-adjust-button", color="primary")
                        ])
                    ], width=12, md=6),
                    dbc.Col([
                        html.Label("Adjustment Date:"),
                        dcc.DatePickerSingle(
                            id="inventory-adjust-date",
                            display_format="YYYY-MM-DD",
                            placeholder="Select date",
                            className="w-100"
                        )
                    ], width=12, md=6)
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.Label("Item Lookup:"),
                        dcc.Dropdown(
                            id="item-lookup-dropdown",
                            options=[],
                            placeholder="Search for any item...",
                            className="mb-2"
                        )
                    ], width=12, md=6),
                    dbc.Col([
                        html.Label("Store Lookup:"),
                        dcc.Dropdown(
                            id="store-lookup-dropdown",
                            options=[],
                            placeholder="Select any store...",
                            className="mb-2"
                        )
                    ], width=12, md=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div(id="lookup-item-info", className="mt-3")
                    ], width=12)
                ])
            ])
        ], className="mb-4 shadow-sm"),
        
        # Inventory metrics
        dbc.Row([
            dbc.Col(html.Div(id="inventory-current-metrics"), width=12)
        ], className="mb-4"),
        
        # Inventory chart
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="loading-inventory-chart",
                    type="circle",
                    children=[
                        dcc.Graph(
                            id="inventory-chart",
                            figure=go.Figure(),
                            style={'height': '50vh'}
                        )
                    ]
                )
            ], width=12)
        ], className="mb-4")
    ])


def create_order_recommendations_content():
    """
    Create content for the order recommendations subtab.
    
    Returns:
        html.Div: Subtab content
    """
    return html.Div([
        # Controls
        dbc.Row([
            dbc.Col([
                # Store selector
                html.Label("Select Store:"),
                dcc.Dropdown(
                    id="inventory-recs-store-dropdown",
                    options=[],
                    clearable=False
                )
            ], width=12)
        ], className="mb-4"),
        
        # Stock policy controls
        dbc.Row([
            dbc.Col([
                html.H5("Stock Policy Settings"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Min Weeks of Stock:"),
                        dbc.Input(
                            id="min-stock-weeks-input",
                            type="number",
                            value=MIN_STOCK_WEEKS,
                            min=0.1,
                            max=4,
                            step=0.1
                        )
                    ], width=4),
                    dbc.Col([
                        html.Label("Target Weeks of Stock:"),
                        dbc.Input(
                            id="target-stock-weeks-input",
                            type="number",
                            value=TARGET_STOCK_WEEKS,
                            min=0.1,
                            max=8,
                            step=0.1
                        )
                    ], width=4),
                    dbc.Col([
                        html.Label("Max Weeks of Stock:"),
                        dbc.Input(
                            id="max-stock-weeks-input",
                            type="number",
                            value=MAX_STOCK_WEEKS,
                            min=0.1,
                            max=12,
                            step=0.1
                        )
                    ], width=4)
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Recommendations
        dbc.Row([
            dbc.Col([
                html.H5("Order Recommendations"),
                html.Div(id="inventory-recommendations")
            ], width=12)
        ])
    ])


def create_inventory_analysis_content():
    """
    Create content for the inventory analysis subtab.
    
    Returns:
        html.Div: Subtab content
    """
    return html.Div([
        # Controls
        dbc.Row([
            dbc.Col([
                # Store and product selectors
                create_store_product_selectors('inventory-analysis', [], [])
            ], width=12)
        ]),
        
        # Display Options
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.H5([
                            html.I(className="fas fa-sliders-h mr-2"),
                            "Display Options"
                        ], className="m-0")
                    ),
                    dbc.CardBody([
                        dbc.Row([
                            # Toggle for demand forecast overlay
                            dbc.Col([
                                create_toggle_switch(
                                    'demand-forecast-overlay',
                                    "Show Demand Forecast",
                                    False,
                                    icon="chart-line",
                                    description="Overlay demand forecast on inventory charts"
                                )
                            ], width=12, md=6),
                            # Toggle for confidence intervals
                            dbc.Col([
                                create_toggle_switch(
                                    'forecast-confidence',
                                    "Show Confidence Intervals",
                                    False,
                                    icon="chart",
                                    description="Display prediction uncertainty ranges"
                                )
                            ], width=12, md=6)
                        ])
                    ])
                ], className="mb-4")
            ], width=12)
        ]),
        
        # Analysis charts
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="loading-stock-velocity",
                    type="circle",
                    children=[
                        dcc.Graph(
                            id="stock-velocity-chart",
                            figure=go.Figure(),
                            style={'height': '40vh'}
                        )
                    ]
                )
            ], width=12)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="loading-stock-penalty",
                    type="circle",
                    children=[
                        dcc.Graph(
                            id="stock-penalty-chart",
                            figure=go.Figure(),
                            style={'height': '40vh'}
                        )
                    ]
                )
            ], width=12)
        ], className="mb-4")
    ])


def update_inventory_chart(app, data_dict, store_id, item_id, adjust_clicks=None, stock_adjustment=None, adjustment_date=None, show_item_numbers=True, show_forecast=False, show_confidence=False):
    """
    Update the inventory chart based on selections.
    
    Args:
        app: Dash app instance with stock adjustments
        data_dict: Dictionary with all loaded data
        store_id: Selected store ID
        item_id: Selected product ID
        adjust_clicks: Number of clicks on adjust button
        stock_adjustment: Manual stock adjustment value
        adjustment_date: Date for the adjustment
        show_item_numbers: Whether to show item numbers in labels
        show_forecast: Whether to overlay demand forecast
        show_confidence: Whether to show forecast confidence intervals
        
    Returns:
        go.Figure: Updated inventory chart
    """
    try:
        # Check for required data
        if store_id is None or item_id is None:
            logger.warning(f"Missing required parameters for inventory chart: store_id={store_id}, item_id={item_id}")
            return go.Figure()
        
        # Get inventory projection data
        inventory_projection = data_dict.get('inventory_projection')
        if inventory_projection is None:
            logger.warning("No inventory projection data available for chart")
            return go.Figure()
        
        # Apply stock adjustments
        try:
            adjusted_projection, adjustment_applied = apply_stock_adjustments(
                inventory_projection, store_id, item_id, stock_adjustment, adjust_clicks,
                adjustment_date, app.manual_stock_adjustments_with_dates
            )
            if adjustment_applied:
                logger.info(f"Applied stock adjustment for store {store_id}, product {item_id}: {stock_adjustment}")
        except Exception as adj_error:
            logger.error(f"Error applying stock adjustments: {str(adj_error)}")
            adjusted_projection = inventory_projection
        
        # Filter for the selected store and item
        projection_data = adjusted_projection[
            (adjusted_projection['Store_Id'] == store_id) &
            (adjusted_projection['Item'] == item_id)
        ]
        
        if len(projection_data) == 0:
            logger.warning(f"No inventory projection data found for store {store_id}, product {item_id}")
            return go.Figure()
    except Exception as e:
        logger.error(f"Error preparing inventory chart data: {str(e)}")
        return go.Figure()
    
    # Get product details
    product_name = projection_data['Product'].iloc[0] if 'Product' in projection_data.columns else f"Item {item_id}"
    
    # Format chart data
    dates = projection_data['Date']
    stock_levels = projection_data['Stock_Level'] if 'Stock_Level' in projection_data.columns else projection_data['Current_Stock']
    
    # Get forecast data
    forecasts = data_dict.get('forecasts')
    forecast_data = None
    if forecasts is not None:
        forecast_data = forecasts[
            (forecasts['Store_Id'] == store_id) &
            (forecasts['Item'] == item_id)
        ]
    
    # Calculate reference lines
    avg_daily_sales = projection_data['Recent_Daily_Sales'].mean() if 'Recent_Daily_Sales' in projection_data.columns else 1.0
    safety_stock = avg_daily_sales * 7 * MIN_STOCK_WEEKS
    target_stock = avg_daily_sales * 7 * TARGET_STOCK_WEEKS
    
    # Create figure
    fig = go.Figure()
    
    # Add stock level
    fig.add_trace(go.Scatter(
        x=dates,
        y=stock_levels,
        mode='lines+markers',
        name='Stock Level',
        line=dict(color='blue', width=2)
    ))
    
    # Add safety stock reference
    fig.add_trace(go.Scatter(
        x=dates,
        y=[safety_stock] * len(dates),
        mode='lines',
        name=f'Safety Stock ({MIN_STOCK_WEEKS} weeks)',
        line=dict(color='red', dash='dash', width=1)
    ))
    
    # Add target stock reference
    fig.add_trace(go.Scatter(
        x=dates,
        y=[target_stock] * len(dates),
        mode='lines',
        name=f'Target Stock ({TARGET_STOCK_WEEKS} weeks)',
        line=dict(color='green', dash='dash', width=1)
    ))
    
    # Add forecast if available
    if forecast_data is not None and len(forecast_data) > 0:
        # Calculate cumulative forecast
        forecast_dates = forecast_data['Date']
        forecast_values = forecast_data['Predicted_Demand'] if 'Predicted_Demand' in forecast_data.columns else forecast_data['Forecast']
        forecast_cum = forecast_values.cumsum()
        
        # Find the last stock level as starting point
        last_stock = stock_levels.iloc[-1] if not stock_levels.empty else 0
        
        # Calculate projected stock without reordering
        projected_stock = [last_stock - val for val in forecast_cum]
        
        # Add to chart
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=projected_stock,
            mode='lines',
            name='Projected Stock (No Orders)',
            line=dict(color='orange', dash='dot', width=2)
        ))
        
        # Add demand forecast overlay if requested
        if show_forecast:
            # Add the demand forecast line
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                mode='lines',
                name='Daily Demand Forecast',
                line=dict(color='purple', width=2),
                yaxis="y2"  # Use secondary y-axis
            ))
            
            # Add confidence intervals if requested
            if show_confidence and ('Lower_Bound' in forecast_data.columns and 'Upper_Bound' in forecast_data.columns):
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_data['Upper_Bound'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(width=0),
                    yaxis="y2",
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_data['Lower_Bound'],
                    mode='lines',
                    name='Lower Bound',
                    fill='tonexty',
                    fillcolor='rgba(128,0,128,0.2)',
                    line=dict(width=0),
                    yaxis="y2",
                    showlegend=False
                ))
    
    # Update layout
    title_text = f"Inventory Projection for {product_name}" + (f" ({item_id})" if show_item_numbers else "")
    layout_updates = {
        'title': title_text,
        'xaxis_title': "Date",
        'yaxis_title': "Stock Level (Units)",
        'legend': dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    }
    
    # Add second y-axis for demand forecast if needed
    if show_forecast:
        layout_updates.update({
            'yaxis2': dict(
                title="Daily Demand (Units)",
                titlefont=dict(color='purple'),
                tickfont=dict(color='purple'),
                overlaying='y',
                side='right'
            )
        })
    
    fig.update_layout(**layout_updates)
    
    return fig


def update_stock_velocity_chart(data_dict, store_id, item_id, show_item_numbers=True):
    """
    Update the stock velocity chart based on selections.
    
    Args:
        data_dict: Dictionary with all loaded data
        store_id: Selected store ID
        item_id: Selected product ID
        show_item_numbers: Whether to show item numbers in labels
        
    Returns:
        go.Figure: Updated stock velocity chart
    """
    # Check for required data
    if store_id is None or item_id is None:
        return go.Figure()
    
    # Get historical data
    combined_data = data_dict.get('combined_data')
    if combined_data is None:
        return go.Figure()
    
    # Filter for the selected store and item
    hist_data = combined_data[
        (combined_data['Store_Id'] == store_id) &
        (combined_data['Item'] == item_id)
    ]
    
    if len(hist_data) == 0:
        return go.Figure()
    
    # Get product details
    product_name = hist_data['Product'].iloc[0] if 'Product' in hist_data.columns else f"Item {item_id}"
    
    # Calculate rolling averages
    hist_data = hist_data.sort_values('Date')
    hist_data['7D_Avg'] = hist_data['Sales'].rolling(window=7, min_periods=1).mean()
    hist_data['14D_Avg'] = hist_data['Sales'].rolling(window=14, min_periods=1).mean()
    hist_data['28D_Avg'] = hist_data['Sales'].rolling(window=28, min_periods=1).mean()
    
    # Create figure
    fig = go.Figure()
    
    # Add daily sales
    fig.add_trace(go.Scatter(
        x=hist_data['Date'],
        y=hist_data['Sales'],
        mode='markers',
        name='Daily Sales',
        marker=dict(color='gray', size=5),
        opacity=0.5
    ))
    
    # Add rolling averages
    fig.add_trace(go.Scatter(
        x=hist_data['Date'],
        y=hist_data['7D_Avg'],
        mode='lines',
        name='7-Day Average',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=hist_data['Date'],
        y=hist_data['14D_Avg'],
        mode='lines',
        name='14-Day Average',
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=hist_data['Date'],
        y=hist_data['28D_Avg'],
        mode='lines',
        name='28-Day Average',
        line=dict(color='red', width=2)
    ))
    
    # Update layout
    title_text = f"Sales Velocity for {product_name}" + (f" ({item_id})" if show_item_numbers else "")
    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title="Units Sold",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def update_stock_penalty_chart(data_dict, store_id, item_id, show_item_numbers=True):
    """
    Update the stock penalty chart based on selections.
    
    Args:
        data_dict: Dictionary with all loaded data
        store_id: Selected store ID
        item_id: Selected product ID
        show_item_numbers: Whether to show item numbers in labels
        
    Returns:
        go.Figure: Updated stock penalty chart
    """
    # Check for required data
    if store_id is None or item_id is None:
        return go.Figure()
    
    # Get historical data
    combined_data = data_dict.get('combined_data')
    if combined_data is None:
        return go.Figure()
    
    # Filter for the selected store and item
    hist_data = combined_data[
        (combined_data['Store_Id'] == store_id) &
        (combined_data['Item'] == item_id)
    ]
    
    if len(hist_data) == 0:
        return go.Figure()
    
    # Get product details
    product_name = hist_data['Product'].iloc[0] if 'Product' in hist_data.columns else f"Item {item_id}"
    
    # Calculate costs and penalties
    hist_data = hist_data.sort_values('Date')
    
    # Get relevant metrics
    unit_price = hist_data['Price'].mean()
    unit_cost = hist_data['Cost'].mean() if 'Cost' in hist_data.columns else unit_price * 0.6  # Estimate cost
    unit_margin = unit_price - unit_cost
    
    # Calculate average daily sales
    avg_daily_sales = hist_data['Sales'].mean()
    
    # Calculate carrying cost (assume 20% annual cost, so daily rate is 20%/365)
    daily_carrying_cost_rate = 0.2 / 365
    
    # Calculate stockout cost (assume 50% margin loss on missed sales)
    stockout_penalty_rate = 0.5 * unit_margin
    
    # Calculate optimal stock level
    service_level = 0.95  # 95% service level
    from scipy.stats import norm
    import math
    
    # Calculate safety stock for 95% service level
    # Use coefficient of variation as a proxy for demand variability
    cv = hist_data['Sales'].std() / max(1, hist_data['Sales'].mean())
    lead_time_days = 3  # Assume 3-day lead time
    safety_factor = norm.ppf(service_level)
    safety_stock = safety_factor * math.sqrt(lead_time_days) * cv * avg_daily_sales
    
    # Generate range of stock levels (0 to 4 weeks of stock)
    max_stock = int(avg_daily_sales * 28) + 1  # 4 weeks of average sales
    stock_levels = np.arange(0, max_stock, max(1, max_stock // 50))  # About 50 points
    
    # Calculate costs for each stock level
    carrying_costs = []
    stockout_costs = []
    total_costs = []
    
    for stock in stock_levels:
        # Carrying cost (daily rate * inventory value)
        carrying_cost = stock * unit_cost * daily_carrying_cost_rate
        
        # Stockout cost (probability of stockout * penalty)
        # Use normal approximation for demand distribution
        if hist_data['Sales'].std() > 0:
            z_score = (stock - avg_daily_sales) / hist_data['Sales'].std()
            stockout_prob = 1 - norm.cdf(z_score)
        else:
            stockout_prob = 0 if stock >= avg_daily_sales else 1
            
        stockout_cost = stockout_prob * stockout_penalty_rate * avg_daily_sales
        
        # Total cost
        total_cost = carrying_cost + stockout_cost
        
        carrying_costs.append(carrying_cost)
        stockout_costs.append(stockout_cost)
        total_costs.append(total_cost)
    
    # Find optimal stock level
    optimal_index = np.argmin(total_costs)
    optimal_stock = stock_levels[optimal_index]
    
    # Create figure
    fig = go.Figure()
    
    # Add cost curves
    fig.add_trace(go.Scatter(
        x=stock_levels,
        y=carrying_costs,
        mode='lines',
        name='Carrying Cost',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=stock_levels,
        y=stockout_costs,
        mode='lines',
        name='Stockout Cost',
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=stock_levels,
        y=total_costs,
        mode='lines',
        name='Total Cost',
        line=dict(color='green', width=3)
    ))
    
    # Add optimal point
    fig.add_trace(go.Scatter(
        x=[optimal_stock],
        y=[total_costs[optimal_index]],
        mode='markers',
        name='Optimal Stock',
        marker=dict(color='green', size=10, symbol='star')
    ))
    
    # Add safety stock reference
    fig.add_trace(go.Scatter(
        x=[safety_stock, safety_stock],
        y=[0, max(total_costs)],
        mode='lines',
        name=f'Safety Stock ({service_level*100:.0f}% Service)',
        line=dict(color='purple', dash='dash', width=2)
    ))
    
    # Update layout
    title_text = f"Inventory Cost Analysis for {product_name}" + (f" ({item_id})" if show_item_numbers else "")
    fig.update_layout(
        title=title_text,
        xaxis_title="Stock Level (Units)",
        yaxis_title="Daily Cost ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def update_inventory_summary_stats(app, data_dict, store_id, item_id, adjust_clicks=None, stock_adjustment=None, adjustment_date=None):
    """
    Update inventory summary statistics.
    
    Args:
        app: Dash app instance with stock adjustments
        data_dict: Dictionary with all loaded data
        store_id: Selected store ID
        item_id: Selected product ID
        adjust_clicks: Number of clicks on adjust button
        stock_adjustment: Manual stock adjustment value
        adjustment_date: Date for the adjustment
        
    Returns:
        tuple: (current_stock, coverage, status, turnover)
    """
    try:
        # Check for required data
        if store_id is None or item_id is None:
            logger.warning(f"Missing required parameters for inventory summary stats: store_id={store_id}, item_id={item_id}")
            return None, None, None, None
        
        # Get inventory projection data
        inventory_projection = data_dict.get('inventory_projection')
        if inventory_projection is None:
            logger.warning("No inventory projection data available for summary stats")
            return None, None, None, None
        
        # Apply stock adjustments
        try:
            adjusted_projection, adjustment_applied = apply_stock_adjustments(
                inventory_projection, store_id, item_id, stock_adjustment, adjust_clicks,
                adjustment_date, app.manual_stock_adjustments_with_dates
            )
            if adjustment_applied:
                logger.info(f"Applied stock adjustment for summary stats: store {store_id}, product {item_id}: {stock_adjustment}")
        except Exception as adj_error:
            logger.error(f"Error applying stock adjustments for summary stats: {str(adj_error)}")
            adjusted_projection = inventory_projection
        
        # Filter for the selected store and item
        projection_data = adjusted_projection[
            (adjusted_projection['Store_Id'] == store_id) &
            (adjusted_projection['Item'] == item_id)
        ]
        
        if len(projection_data) == 0:
            logger.warning(f"No inventory projection data found for summary stats: store {store_id}, product {item_id}")
            return None, None, None, None
    except Exception as e:
        logger.error(f"Error preparing inventory summary stats: {str(e)}")
        return None, None, None, None
    
    # Get current stock level (most recent date)
    projection_data = projection_data.sort_values('Date')
    current_stock = projection_data['Stock_Level'].iloc[-1] if 'Stock_Level' in projection_data.columns else projection_data['Current_Stock'].iloc[-1]
    
    # Calculate weeks of coverage
    avg_daily_sales = projection_data['Recent_Daily_Sales'].mean() if 'Recent_Daily_Sales' in projection_data.columns else 1.0
    coverage_weeks = current_stock / (avg_daily_sales * 7) if avg_daily_sales > 0 else 4.0
    
    # Determine status
    if coverage_weeks < MIN_STOCK_WEEKS:
        status = "Low"
    elif coverage_weeks <= MAX_STOCK_WEEKS:
        status = "Adequate"
    else:
        status = "Excess"
    
    # Calculate turnover (annual)
    historical_data = data_dict.get('combined_data')
    if historical_data is not None:
        hist_data = historical_data[
            (historical_data['Store_Id'] == store_id) &
            (historical_data['Item'] == item_id)
        ]
        
        if len(hist_data) > 0:
            # Calculate annual turnover
            annual_sales = hist_data['Sales'].mean() * 365  # Estimate annual sales
            turnover = annual_sales / max(1, current_stock)  # Annual turnover
        else:
            turnover = None
    else:
        turnover = None
    
    return current_stock, coverage_weeks, status, turnover


def update_stock_recommendations(app, data_dict, store_id, min_weeks=None, target_weeks=None, max_weeks=None, show_item_numbers=True):
    """
    Update stock recommendations.
    
    Args:
        app: Dash app instance with stock adjustments
        data_dict: Dictionary with all loaded data
        store_id: Selected store ID
        min_weeks: Minimum weeks of stock
        target_weeks: Target weeks of stock
        max_weeks: Maximum weeks of stock
        show_item_numbers: Whether to show item numbers in labels
        
    Returns:
        dash component or None
    """
    # Check for required data
    if store_id is None:
        return None
    
    # Get inventory data
    inventory_recs = data_dict.get('inventory_recs')
    inventory_projection = data_dict.get('inventory_projection')
    
    if inventory_projection is None:
        return None
    
    # Apply global stock adjustments (store-level)
    adjustment_keys = [key for key in app.manual_stock_adjustments.keys() if key.startswith(f"{store_id}_")]
    
    if adjustment_keys:
        # Apply all adjustments for this store
        adjusted_projection = inventory_projection.copy()
        
        for key in adjustment_keys:
            parts = key.split("_")
            if len(parts) == 2:
                # Extract store and item IDs
                adj_store = int(parts[0])
                adj_item = float(parts[1])
                
                # Apply the adjustment to the stock level
                stock_col = 'Stock_Level' if 'Stock_Level' in adjusted_projection.columns else 'Current_Stock'
                if stock_col in adjusted_projection.columns:
                    # Update records for this store-item combination
                    mask = (adjusted_projection['Store_Id'] == adj_store) & (adjusted_projection['Item'] == adj_item)
                    if len(adjusted_projection[mask]) > 0:
                        # Set the stock level to the adjustment value
                        adjusted_projection.loc[mask, stock_col] = app.manual_stock_adjustments[key]
    else:
        adjusted_projection = inventory_projection
    
    # Filter for the selected store
    projection_data = adjusted_projection[adjusted_projection['Store_Id'] == store_id]
    
    if len(projection_data) == 0:
        return None
    
    # Get the most recent data for each product
    projection_data = projection_data.sort_values('Date')
    latest_data = projection_data.groupby('Item').last().reset_index()
    
    # Calculate weeks of coverage
    if 'Recent_Daily_Sales' in latest_data.columns and 'Stock_Level' in latest_data.columns:
        latest_data['Coverage_Weeks'] = latest_data.apply(
            lambda row: row['Stock_Level'] / (row['Recent_Daily_Sales'] * 7) 
                if row['Recent_Daily_Sales'] > 0 else 4.0,
            axis=1
        )
    else:
        # Use existing Weeks_Of_Stock if available
        if 'Weeks_Of_Stock' not in latest_data.columns:
            latest_data['Coverage_Weeks'] = 2.0  # Default
        else:
            latest_data['Coverage_Weeks'] = latest_data['Weeks_Of_Stock']
    
    # Apply custom stock policy if provided
    min_weeks = min_weeks or MIN_STOCK_WEEKS
    target_weeks = target_weeks or TARGET_STOCK_WEEKS
    max_weeks = max_weeks or MAX_STOCK_WEEKS
    
    # Determine status for each product
    def get_status(coverage):
        if coverage < min_weeks:
            return "Low"
        elif coverage <= max_weeks:
            return "Adequate"
        else:
            return "Excess"
    
    latest_data['Status'] = latest_data['Coverage_Weeks'].apply(get_status)
    
    # Calculate order recommendations
    def calculate_order(row):
        if row['Status'] == "Low":
            # Order up to target stock level
            daily_sales = row['Recent_Daily_Sales'] if 'Recent_Daily_Sales' in row else 1.0
            target_stock = daily_sales * 7 * target_weeks
            current_stock = row['Stock_Level'] if 'Stock_Level' in row else row['Current_Stock']
            return max(0, round(target_stock - current_stock))
        else:
            return 0
    
    latest_data['Order_Recommendation'] = latest_data.apply(calculate_order, axis=1)
    
    # Create recommendation table
    recs = []
    
    for _, row in latest_data.iterrows():
        if row['Order_Recommendation'] > 0:
            product_name = row['Product'] if 'Product' in row else f"Item {row['Item']}"
            if show_item_numbers:
                product_name += f" ({row['Item']})"
            
            current_stock = row['Stock_Level'] if 'Stock_Level' in row else row['Current_Stock']
            coverage = row['Coverage_Weeks']
            
            recs.append({
                'Product': product_name,
                'Current_Stock': int(current_stock),
                'Coverage_Weeks': f"{coverage:.1f} weeks",
                'Status': row['Status'],
                'Order': int(row['Order_Recommendation'])
            })
    
    # If we have recommendations, create the table
    if recs:
        return html.Div([
            html.P(f"Found {len(recs)} products that need ordering at Store {store_id}"),
            dash_table.DataTable(
                id="recommendations-table",
                columns=[
                    {"name": "Product", "id": "Product"},
                    {"name": "Current Stock", "id": "Current_Stock"},
                    {"name": "Coverage", "id": "Coverage_Weeks"},
                    {"name": "Status", "id": "Status"},
                    {"name": "Order Quantity", "id": "Order"}
                ],
                data=recs,
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px'
                },
                style_header={
                    'backgroundColor': 'lightgrey',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Status', 'filter_query': '{Status} eq "Low"'},
                        'backgroundColor': '#FFCDD2',
                        'color': 'black'
                    }
                ]
            )
        ])
    else:
        return html.P(f"No products need ordering at Store {store_id}")


def register_inventory_callbacks(app, data_dict):
    """
    Register inventory-related callbacks.
    
    Args:
        app: Dash app instance
        data_dict: Dictionary of loaded data
    """
    # Populate store and product dropdowns
    @app.callback(
        [
            Output("inventory-status-store-dropdown", "options"),
            Output("inventory-status-product-dropdown", "options"),
            Output("inventory-analysis-store-dropdown", "options"),
            Output("inventory-analysis-product-dropdown", "options"),
            Output("inventory-recs-store-dropdown", "options")
        ],
        Input("show-item-numbers-store", "data")
    )
    def update_inventory_dropdowns(show_item_numbers):
        """Update inventory dropdown options"""
        # Get the data sources
        combined_data = data_dict.get('combined_data')
        inventory_projection = data_dict.get('inventory_projection')
        
        # Use whatever data is available
        if inventory_projection is not None:
            data_source = inventory_projection
        elif combined_data is not None:
            data_source = combined_data
        else:
            # No data available
            return [], [], [], [], []
        
        # Get store and product options
        from ui.core import get_store_product_options
        store_options, product_options = get_store_product_options(data_source, show_item_numbers)
        
        # Return the options for all dropdowns
        return store_options, product_options, store_options, product_options, store_options
    
    # Inventory chart callback
    @app.callback(
        [
            Output("inventory-chart", "figure"),
            Output("inventory-current-metrics", "children")
        ],
        [
            Input("inventory-status-store-dropdown", "value"),
            Input("inventory-status-product-dropdown", "value"),
            Input("inventory-adjust-button", "n_clicks"),
            Input("show-item-numbers-store", "data")
        ],
        [
            State("inventory-count-input", "value"),
            State("inventory-adjust-date", "date")
        ]
    )
    def update_inventory_status(store, product, adjust_clicks, show_item_numbers, stock_value, adjust_date):
        """Update inventory status components"""
        # Update inventory chart
        fig = update_inventory_chart(
            app, data_dict, store, product,
            adjust_clicks, stock_value, adjust_date,
            show_item_numbers
        )
        
        # Update inventory metrics
        current_stock, coverage, status, turnover = update_inventory_summary_stats(
            app, data_dict, store, product,
            adjust_clicks, stock_value, adjust_date
        )
        
        # Create metrics cards
        if current_stock is not None:
            metrics = dbc.Row([
                dbc.Col(create_info_card("Current Stock", f"{int(current_stock)} units", "primary", "box"), width=3),
                dbc.Col(create_info_card("Coverage", f"{coverage:.1f} weeks", "info", "calendar"), width=3),
                dbc.Col(create_info_card("Status", status, 
                                        "danger" if status == "Low" else 
                                        "success" if status == "Adequate" else "warning", 
                                        "exclamation-triangle"), width=3),
                dbc.Col(create_info_card("Annual Turnover", f"{turnover:.1f}x" if turnover else "N/A", "secondary", "sync"), width=3)
            ])
        else:
            metrics = create_error_message("No inventory data available for this product")
        
        return fig, metrics
    
    # Stock velocity chart callback
    @app.callback(
        Output("stock-velocity-chart", "figure"),
        [
            Input("inventory-analysis-store-dropdown", "value"),
            Input("inventory-analysis-product-dropdown", "value"),
            Input("show-item-numbers-store", "data")
        ]
    )
    def update_velocity_chart(store, product, show_item_numbers):
        """Update stock velocity chart"""
        return update_stock_velocity_chart(data_dict, store, product, show_item_numbers)
    
    # Stock penalty chart callback
    @app.callback(
        Output("stock-penalty-chart", "figure"),
        [
            Input("inventory-analysis-store-dropdown", "value"),
            Input("inventory-analysis-product-dropdown", "value"),
            Input("show-item-numbers-store", "data")
        ]
    )
    def update_penalty_chart(store, product, show_item_numbers):
        """Update stock penalty chart"""
        return update_stock_penalty_chart(data_dict, store, product, show_item_numbers)
    
    # Stock recommendations callback
    @app.callback(
        Output("inventory-recommendations", "children"),
        [
            Input("inventory-recs-store-dropdown", "value"),
            Input("min-stock-weeks-input", "value"),
            Input("target-stock-weeks-input", "value"),
            Input("max-stock-weeks-input", "value"),
            Input("show-item-numbers-store", "data")
        ]
    )
    def update_recommendations(store, min_weeks, target_weeks, max_weeks, show_item_numbers):
        """Update stock recommendations"""
        return update_stock_recommendations(
            app, data_dict, store, 
            min_weeks, target_weeks, max_weeks,
            show_item_numbers
        )