"""
Integrated view module for connecting inventory, pricing, and demand forecasting.
This module combines elements from multiple modules to show their interrelationships.
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
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core UI components
from ui.core import (
    load_dashboard_data, create_app, format_product_name,
    create_date_range_slider, create_store_product_selectors,
    create_toggle_switch, create_error_message, create_info_card
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('integrated_view')

# Import settings if available
try:
    from config.settings import (
        MIN_STOCK_WEEKS, TARGET_STOCK_WEEKS, MAX_STOCK_WEEKS,
        PRICE_ELASTICITIES_FILE, PRICE_RECOMMENDATIONS_FILE
    )
except ImportError:
    # Default values
    MIN_STOCK_WEEKS = 1
    TARGET_STOCK_WEEKS = 2
    MAX_STOCK_WEEKS = 3
    
    # Default paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PRICE_ELASTICITIES_FILE = os.path.join(ROOT_DIR, "price_elasticities.csv")
    PRICE_RECOMMENDATIONS_FILE = os.path.join(ROOT_DIR, "price_recommendations.csv")


def create_integrated_view_content(data_dict):
    """
    Create content for the integrated view tab.
    
    Args:
        data_dict: Dictionary with all loaded data
        
    Returns:
        html.Div: Tab content
    """
    # Extract relevant data
    combined_data = data_dict.get('combined_data')
    forecasts = data_dict.get('forecasts')
    price_elasticities = data_dict.get('price_elasticities')
    inventory_projection = data_dict.get('inventory_projection')
    
    if combined_data is None:
        return create_error_message("Error: No historical data available")
    
    if forecasts is None:
        return create_error_message("Error: No forecast data available")
    
    if price_elasticities is None:
        return create_error_message("Error: No price elasticity data available")
    
    if inventory_projection is None:
        return create_error_message("Error: No inventory projection data available")
    
    # Get store and product options
    store_options = []
    product_options = []
    
    if combined_data is not None:
        # Get unique stores
        stores = combined_data['Store_Id'].unique()
        store_options = [{'label': f'Store {s}', 'value': s} for s in sorted(stores)]
        
        # Get unique products with names
        products = combined_data[['Item', 'Product']].drop_duplicates()
        
        # Format product names
        product_options = [
            {
                'label': f"{row['Product']} ({row['Item']})",
                'value': row['Item']
            } for _, row in products.iterrows()
        ]
    
    # Create the integrated view content
    return html.Div([
        # Title and description
        dbc.Card([
            dbc.CardHeader(html.H4("Integrated Business Analytics")),
            dbc.CardBody([
                html.P(
                    "This view shows the connections between inventory, pricing, and demand forecasting. "
                    "Changes in one area directly affect the others, helping you make better decisions by "
                    "understanding these interconnections."
                )
            ])
        ], className="mb-4"),
        
        # Controls
        dbc.Row([
            dbc.Col([
                # Store and product selectors
                create_store_product_selectors('integrated', store_options, product_options)
            ], width=12)
        ]),
        
        # Display options
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Display Options")),
                    dbc.CardBody([
                        dbc.Row([
                            # Price adjustment slider
                            dbc.Col([
                                html.Label("Price Adjustment (%)"),
                                dcc.Slider(
                                    id="price-adjustment-slider",
                                    min=-15,
                                    max=20,
                                    step=1,
                                    value=0,
                                    marks={
                                        -15: '-15%',
                                        -10: '-10%',
                                        -5: '-5%',
                                        0: '0%',
                                        5: '5%',
                                        10: '10%',
                                        15: '15%',
                                        20: '20%'
                                    }
                                )
                            ], width=12, md=6),
                            
                            # Inventory adjustment slider
                            dbc.Col([
                                html.Label("Inventory Adjustment (%)"),
                                dcc.Slider(
                                    id="inventory-adjustment-slider",
                                    min=-50,
                                    max=100,
                                    step=5,
                                    value=0,
                                    marks={
                                        -50: '-50%',
                                        -25: '-25%',
                                        0: '0%',
                                        25: '25%',
                                        50: '50%',
                                        75: '75%',
                                        100: '100%'
                                    }
                                )
                            ], width=12, md=6)
                        ]),
                        
                        dbc.Row([
                            # Date range selector
                            dbc.Col([
                                html.Label("Forecast Period"),
                                dcc.RangeSlider(
                                    id="forecast-period-slider",
                                    min=1,
                                    max=30,
                                    step=1,
                                    value=[1, 14],
                                    marks={
                                        1: '1d',
                                        7: '7d',
                                        14: '14d',
                                        21: '21d',
                                        30: '30d'
                                    }
                                )
                            ], width=12)
                        ], className="mt-3")
                    ])
                ], className="mb-4")
            ], width=12)
        ]),
        
        # Key metrics
        dbc.Row([
            dbc.Col([
                html.Div(id="integrated-key-metrics")
            ], width=12)
        ], className="mb-4"),
        
        # Interactive visualization
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Integrated Business Impact")),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-integrated-chart",
                            type="circle",
                            children=[
                                dcc.Graph(
                                    id="integrated-chart",
                                    figure=go.Figure(),
                                    style={'height': '60vh'}
                                )
                            ]
                        )
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Detailed insights
        dbc.Row([
            dbc.Col([
                html.Div(id="integrated-insights")
            ], width=12)
        ])
    ])


def update_integrated_chart(data_dict, store_id, item_id, price_adjustment=0, 
                           inventory_adjustment=0, forecast_period=None, show_item_numbers=True):
    """
    Update the integrated chart showing connections between inventory, pricing, and demand.
    
    Args:
        data_dict: Dictionary with all loaded data
        store_id: Selected store ID
        item_id: Selected product ID
        price_adjustment: Price adjustment percentage
        inventory_adjustment: Inventory adjustment percentage
        forecast_period: Tuple of (start_day, end_day) for forecast period
        show_item_numbers: Whether to show item numbers in labels
        
    Returns:
        go.Figure: Updated integrated chart
    """
    # Extract needed data
    combined_data = data_dict.get('combined_data')
    forecasts = data_dict.get('forecasts')
    price_elasticities = data_dict.get('price_elasticities')
    inventory_projection = data_dict.get('inventory_projection')
    
    # Check for required data
    if (combined_data is None or forecasts is None or 
        price_elasticities is None or inventory_projection is None or
        store_id is None or item_id is None):
        return go.Figure()
    
    try:
        # Filter data for the selected store and item
        hist_data = combined_data[(combined_data['Store_Id'] == store_id) & 
                                 (combined_data['Item'] == item_id)].sort_values('Date')
        
        forecast_data = forecasts[(forecasts['Store_Id'] == store_id) & 
                                (forecasts['Item'] == item_id)].sort_values('Date')
        
        elasticity_data = price_elasticities[(price_elasticities['Store_Id'] == store_id) & 
                                            (price_elasticities['Item'] == item_id)]
        
        inventory_data = inventory_projection[(inventory_projection['Store_Id'] == store_id) & 
                                             (inventory_projection['Item'] == item_id)].sort_values('Date')
        
        if len(hist_data) == 0 or len(forecast_data) == 0 or len(elasticity_data) == 0 or len(inventory_data) == 0:
            return go.Figure()
        
        # Get product name and key metrics
        product_name = hist_data['Product'].iloc[0]
        current_price = elasticity_data['Current_Price'].iloc[0]
        elasticity = elasticity_data['Elasticity'].iloc[0]
        
        # Format product name
        formatted_product = format_product_name(product_name, item_id, show_item_numbers)
        
        # Create subplots with shared x-axis
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Sales & Demand', 'Inventory Level', 'Profit Impact'),
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Prepare date range
        min_date = forecast_data['Date'].min()
        if forecast_period is not None:
            start_day, end_day = forecast_period
            forecast_dates = pd.date_range(start=min_date, periods=end_day)
            forecast_dates = forecast_dates[start_day-1:end_day]
        else:
            forecast_dates = forecast_data['Date']
        
        # Filter forecast data to the selected period
        forecast_data = forecast_data[forecast_data['Date'].isin(forecast_dates)]
        
        # Apply price adjustment
        new_price = current_price * (1 + price_adjustment / 100)
        price_ratio = new_price / current_price
        
        # Adjust forecasted demand based on price elasticity
        adjusted_forecast = forecast_data.copy()
        quantity_ratio = price_ratio ** elasticity if elasticity < 0 else 1.0
        adjusted_forecast['Adjusted_Forecast'] = adjusted_forecast['Forecast'] * quantity_ratio
        
        # Apply inventory adjustment
        current_stock = inventory_data['Stock_Level'].iloc[-1]
        adjusted_stock = current_stock * (1 + inventory_adjustment / 100)
        
        # Create projected inventory based on adjusted forecast
        inventory_projection = []
        stock_level = adjusted_stock
        
        for _, row in adjusted_forecast.iterrows():
            # Subtract daily demand from stock
            stock_level = max(0, stock_level - row['Adjusted_Forecast'])
            inventory_projection.append({
                'Date': row['Date'],
                'Stock_Level': stock_level
            })
        
        inventory_df = pd.DataFrame(inventory_projection)
        
        # Calculate profit impact
        cost_per_unit = elasticity_data['Cost'].iloc[0] if 'Cost' in elasticity_data.columns else current_price * 0.6
        
        # Original profit
        original_profit = []
        for _, row in forecast_data.iterrows():
            daily_profit = (current_price - cost_per_unit) * row['Forecast']
            original_profit.append({
                'Date': row['Date'],
                'Profit': daily_profit
            })
        
        original_profit_df = pd.DataFrame(original_profit)
        
        # Adjusted profit
        adjusted_profit = []
        for _, row in adjusted_forecast.iterrows():
            daily_profit = (new_price - cost_per_unit) * row['Adjusted_Forecast']
            adjusted_profit.append({
                'Date': row['Date'],
                'Profit': daily_profit
            })
        
        adjusted_profit_df = pd.DataFrame(adjusted_profit)
        
        # Calculate cumulative values
        original_profit_df['Cumulative_Profit'] = original_profit_df['Profit'].cumsum()
        adjusted_profit_df['Cumulative_Profit'] = adjusted_profit_df['Profit'].cumsum()
        
        profit_difference = (adjusted_profit_df['Cumulative_Profit'].iloc[-1] - 
                           original_profit_df['Cumulative_Profit'].iloc[-1])
        profit_change_pct = (profit_difference / original_profit_df['Cumulative_Profit'].iloc[-1] * 100) if original_profit_df['Cumulative_Profit'].iloc[-1] > 0 else 0
        
        # Plot historical and forecasted sales
        # Historical sales data (last 30 days)
        recent_hist_data = hist_data.iloc[-30:] if len(hist_data) > 30 else hist_data
        
        # Add historical sales to Sales & Demand subplot
        fig.add_trace(
            go.Scatter(
                x=recent_hist_data['Date'],
                y=recent_hist_data['Sales'],
                mode='lines',
                name='Historical Sales',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Add original forecast to Sales & Demand subplot
        fig.add_trace(
            go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Forecast'],
                mode='lines',
                name='Original Forecast',
                line=dict(color='orange')
            ),
            row=1, col=1
        )
        
        # Add adjusted forecast to Sales & Demand subplot
        fig.add_trace(
            go.Scatter(
                x=adjusted_forecast['Date'],
                y=adjusted_forecast['Adjusted_Forecast'],
                mode='lines',
                name=f'Adjusted Forecast (Price {price_adjustment:+}%)',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        # Add inventory levels to Inventory Level subplot
        if len(inventory_data) > 0:
            # Get recent inventory data
            recent_inventory = inventory_data.iloc[-30:] if len(inventory_data) > 30 else inventory_data
            
            fig.add_trace(
                go.Scatter(
                    x=recent_inventory['Date'],
                    y=recent_inventory['Stock_Level'],
                    mode='lines',
                    name='Historical Stock',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
        
        # Add projected inventory to Inventory Level subplot
        fig.add_trace(
            go.Scatter(
                x=inventory_df['Date'],
                y=inventory_df['Stock_Level'],
                mode='lines',
                name=f'Projected Stock (Adj. {inventory_adjustment:+}%)',
                line=dict(color='darkgreen')
            ),
            row=2, col=1
        )
        
        # Add safety stock reference line
        avg_daily_sales = adjusted_forecast['Adjusted_Forecast'].mean()
        safety_stock = avg_daily_sales * 7 * MIN_STOCK_WEEKS
        
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=[safety_stock] * len(forecast_dates),
                mode='lines',
                name=f'Safety Stock ({MIN_STOCK_WEEKS} weeks)',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=1
        )
        
        # Add target stock reference line
        target_stock = avg_daily_sales * 7 * TARGET_STOCK_WEEKS
        
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=[target_stock] * len(forecast_dates),
                mode='lines',
                name=f'Target Stock ({TARGET_STOCK_WEEKS} weeks)',
                line=dict(color='green', dash='dash')
            ),
            row=2, col=1
        )
        
        # Add profit impact to Profit Impact subplot
        fig.add_trace(
            go.Scatter(
                x=original_profit_df['Date'],
                y=original_profit_df['Cumulative_Profit'],
                mode='lines',
                name='Original Profit',
                line=dict(color='purple')
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=adjusted_profit_df['Date'],
                y=adjusted_profit_df['Cumulative_Profit'],
                mode='lines',
                name='Adjusted Profit',
                line=dict(color='magenta')
            ),
            row=3, col=1
        )
        
        # Add annotations to show key metrics
        fig.add_annotation(
            x=0.5,
            y=1.12,
            xref='paper',
            yref='paper',
            text=f"<b>{formatted_product}</b> | Current Price: ${current_price:.2f} | "
                f"New Price: ${new_price:.2f} | Elasticity: {elasticity:.2f}",
            showarrow=False,
            font=dict(size=14),
            align='center'
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=40, r=40, t=100, b=40),
            title=f"Integrated Business Impact Analysis for {formatted_product}"
        )
        
        # Set y-axis titles
        fig.update_yaxes(title_text="Units Sold", row=1, col=1)
        fig.update_yaxes(title_text="Stock Level", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Profit ($)", row=3, col=1)
        
        # Set x-axis title only on the bottom subplot
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error updating integrated chart: {str(e)}")
        return go.Figure()


def calculate_key_metrics(data_dict, store_id, item_id, price_adjustment=0, 
                         inventory_adjustment=0, forecast_period=None):
    """
    Calculate key metrics for the integrated view.
    
    Args:
        data_dict: Dictionary with all loaded data
        store_id: Selected store ID
        item_id: Selected product ID
        price_adjustment: Price adjustment percentage
        inventory_adjustment: Inventory adjustment percentage
        forecast_period: Tuple of (start_day, end_day) for forecast period
        
    Returns:
        dict: Dictionary of key metrics
    """
    # Extract needed data
    forecasts = data_dict.get('forecasts')
    price_elasticities = data_dict.get('price_elasticities')
    inventory_projection = data_dict.get('inventory_projection')
    
    # Check for required data
    if (forecasts is None or price_elasticities is None or 
        inventory_projection is None or store_id is None or item_id is None):
        return None
    
    try:
        # Filter data for the selected store and item
        forecast_data = forecasts[(forecasts['Store_Id'] == store_id) & 
                                (forecasts['Item'] == item_id)].sort_values('Date')
        
        elasticity_data = price_elasticities[(price_elasticities['Store_Id'] == store_id) & 
                                            (price_elasticities['Item'] == item_id)]
        
        inventory_data = inventory_projection[(inventory_projection['Store_Id'] == store_id) & 
                                             (inventory_projection['Item'] == item_id)].sort_values('Date')
        
        if len(forecast_data) == 0 or len(elasticity_data) == 0 or len(inventory_data) == 0:
            return None
        
        # Get key values
        current_price = elasticity_data['Current_Price'].iloc[0]
        elasticity = elasticity_data['Elasticity'].iloc[0]
        current_stock = inventory_data['Stock_Level'].iloc[-1]
        cost_per_unit = elasticity_data['Cost'].iloc[0] if 'Cost' in elasticity_data.columns else current_price * 0.6
        
        # Prepare date range
        min_date = forecast_data['Date'].min()
        if forecast_period is not None:
            start_day, end_day = forecast_period
            forecast_dates = pd.date_range(start=min_date, periods=end_day)
            forecast_dates = forecast_dates[start_day-1:end_day]
            forecast_data = forecast_data[forecast_data['Date'].isin(forecast_dates)]
        
        # Calculate price impact
        new_price = current_price * (1 + price_adjustment / 100)
        price_ratio = new_price / current_price
        
        # Calculate demand impact
        quantity_ratio = price_ratio ** elasticity if elasticity < 0 else 1.0
        original_demand = forecast_data['Forecast'].sum()
        adjusted_demand = original_demand * quantity_ratio
        demand_change = adjusted_demand - original_demand
        demand_change_pct = (demand_change / original_demand * 100) if original_demand > 0 else 0
        
        # Calculate profit impact
        original_profit = (current_price - cost_per_unit) * original_demand
        adjusted_profit = (new_price - cost_per_unit) * adjusted_demand
        profit_change = adjusted_profit - original_profit
        profit_change_pct = (profit_change / original_profit * 100) if original_profit > 0 else 0
        
        # Calculate inventory impact
        adjusted_stock = current_stock * (1 + inventory_adjustment / 100)
        stock_change = adjusted_stock - current_stock
        avg_daily_demand = adjusted_demand / len(forecast_data) if len(forecast_data) > 0 else 1
        coverage_days = adjusted_stock / avg_daily_demand if avg_daily_demand > 0 else 0
        coverage_weeks = coverage_days / 7
        
        if coverage_weeks < MIN_STOCK_WEEKS:
            stock_status = "Low"
            risk_level = "High"
        elif coverage_weeks <= TARGET_STOCK_WEEKS:
            stock_status = "Adequate"
            risk_level = "Low"
        elif coverage_weeks <= MAX_STOCK_WEEKS:
            stock_status = "Good"
            risk_level = "Low"
        else:
            stock_status = "Excess"
            risk_level = "Medium"
        
        # Calculate stockout risk
        if coverage_days <= len(forecast_data) and coverage_days > 0:
            stockout_day = int(coverage_days)
            stockout_risk = "High"
            stockout_msg = f"Projected stockout in {stockout_day} days"
        else:
            stockout_day = None
            stockout_risk = "Low"
            stockout_msg = "No stockout projected during forecast period"
        
        # Determine if reordering is needed
        safety_stock = avg_daily_demand * 7 * MIN_STOCK_WEEKS
        if adjusted_stock < safety_stock:
            reorder_needed = True
            reorder_amount = (avg_daily_demand * 7 * TARGET_STOCK_WEEKS) - adjusted_stock
            reorder_msg = f"Reorder {int(reorder_amount)} units now"
        else:
            reorder_needed = False
            reorder_amount = 0
            reorder_msg = "No reorder needed at this time"
        
        # Return metrics
        return {
            'current_price': current_price,
            'new_price': new_price,
            'price_change_pct': price_adjustment,
            'elasticity': elasticity,
            'current_stock': current_stock,
            'adjusted_stock': adjusted_stock,
            'original_demand': original_demand,
            'adjusted_demand': adjusted_demand,
            'demand_change_pct': demand_change_pct,
            'original_profit': original_profit,
            'adjusted_profit': adjusted_profit,
            'profit_change': profit_change,
            'profit_change_pct': profit_change_pct,
            'coverage_days': coverage_days,
            'coverage_weeks': coverage_weeks,
            'stock_status': stock_status,
            'risk_level': risk_level,
            'stockout_day': stockout_day,
            'stockout_risk': stockout_risk,
            'stockout_msg': stockout_msg,
            'reorder_needed': reorder_needed,
            'reorder_amount': reorder_amount,
            'reorder_msg': reorder_msg
        }
        
    except Exception as e:
        logger.error(f"Error calculating key metrics: {str(e)}")
        return None


def update_key_metrics_display(metrics):
    """
    Create a display for key metrics.
    
    Args:
        metrics: Dictionary of key metrics
        
    Returns:
        html.Div: Display component
    """
    if metrics is None:
        return html.Div("No metrics available")
    
    try:
        # Create cards for key metrics
        return dbc.Row([
            # Price metrics
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5([html.I(className="fas fa-tags mr-2"), "Price Impact"])),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.P("Current Price:"),
                                html.H4(f"${metrics['current_price']:.2f}")
                            ], width=4),
                            dbc.Col([
                                html.P("New Price:"),
                                html.H4(f"${metrics['new_price']:.2f}")
                            ], width=4),
                            dbc.Col([
                                html.P("Elasticity:"),
                                html.H4(f"{metrics['elasticity']:.2f}")
                            ], width=4)
                        ])
                    ])
                ])
            ], width=12, md=4),
            
            # Demand metrics
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5([html.I(className="fas fa-chart-line mr-2"), "Demand Impact"])),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.P("Original Demand:"),
                                html.H4(f"{metrics['original_demand']:.0f} units")
                            ], width=4),
                            dbc.Col([
                                html.P("Adjusted Demand:"),
                                html.H4(f"{metrics['adjusted_demand']:.0f} units")
                            ], width=4),
                            dbc.Col([
                                html.P("Change:"),
                                html.H4([
                                    f"{metrics['demand_change_pct']:+.1f}%",
                                    html.Span(
                                        className=f"{'text-success' if metrics['demand_change_pct'] >= 0 else 'text-danger'} ml-2"
                                    )
                                ])
                            ], width=4)
                        ])
                    ])
                ])
            ], width=12, md=4),
            
            # Inventory metrics
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5([html.I(className="fas fa-boxes mr-2"), "Inventory Impact"])),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.P("Stock Level:"),
                                html.H4(f"{metrics['adjusted_stock']:.0f} units")
                            ], width=4),
                            dbc.Col([
                                html.P("Coverage:"),
                                html.H4(f"{metrics['coverage_weeks']:.1f} weeks")
                            ], width=4),
                            dbc.Col([
                                html.P("Status:"),
                                html.H4(metrics['stock_status'], 
                                      className={
                                          "text-danger": metrics['stock_status'] == "Low",
                                          "text-success": metrics['stock_status'] in ["Adequate", "Good"],
                                          "text-warning": metrics['stock_status'] == "Excess"
                                      })
                            ], width=4)
                        ])
                    ])
                ])
            ], width=12, md=4)
        ])
        
    except Exception as e:
        logger.error(f"Error updating key metrics display: {str(e)}")
        return html.Div("Error displaying metrics")


def update_insights(metrics):
    """
    Create insights and recommendations based on the metrics.
    
    Args:
        metrics: Dictionary of key metrics
        
    Returns:
        html.Div: Insights and recommendations component
    """
    if metrics is None:
        return html.Div("No insights available")
    
    try:
        # Create insights and recommendations
        insights = []
        
        # Price insights
        price_color = "text-success" if metrics['profit_change_pct'] > 0 else "text-danger"
        price_icon = "fa-arrow-up" if metrics['price_change_pct'] > 0 else "fa-arrow-down" if metrics['price_change_pct'] < 0 else "fa-equals"
        
        price_insight = dbc.Card([
            dbc.CardHeader(html.H5("Price Analysis")),
            dbc.CardBody([
                html.P([
                    html.I(className=f"fas {price_icon} mr-2"),
                    f"Price change of {metrics['price_change_pct']:+.1f}% results in a ",
                    html.Span(f"{metrics['profit_change_pct']:+.1f}% profit impact", 
                             className=price_color),
                    "."
                ]),
                html.P([
                    f"The projected profit change is ",
                    html.Span(f"${metrics['profit_change']:+,.2f}", 
                             className=price_color),
                    " over the forecast period."
                ]),
                html.Hr(),
                html.H6("Recommendation:"),
                html.P(
                    "Implement price increase to boost profit margin." if metrics['profit_change_pct'] > 0 else
                    "Maintain current price." if metrics['profit_change_pct'] == 0 else
                    "Consider alternative pricing strategies."
                )
            ])
        ], className="mb-3")
        
        insights.append(price_insight)
        
        # Inventory insights
        stock_color = {
            "Low": "text-danger",
            "Adequate": "text-success",
            "Good": "text-success",
            "Excess": "text-warning"
        }.get(metrics['stock_status'], "text-primary")
        
        inventory_insight = dbc.Card([
            dbc.CardHeader(html.H5("Inventory Analysis")),
            dbc.CardBody([
                html.P([
                    "Current inventory status: ",
                    html.Span(metrics['stock_status'], className=stock_color),
                    f" with {metrics['coverage_weeks']:.1f} weeks of coverage."
                ]),
                html.P(metrics['stockout_msg']),
                html.Hr(),
                html.H6("Recommendation:"),
                html.P(
                    html.Span(f"URGENT: {metrics['reorder_msg']}", className="text-danger font-weight-bold") if metrics['reorder_needed'] else
                    "Inventory levels are adequate."
                )
            ])
        ], className="mb-3")
        
        insights.append(inventory_insight)
        
        # Business impact summary
        impact_color = "text-success" if metrics['profit_change_pct'] > 0 else "text-danger"
        
        summary_insight = dbc.Card([
            dbc.CardHeader(html.H5("Combined Business Impact")),
            dbc.CardBody([
                html.P([
                    "The combined price and inventory adjustments will result in a ",
                    html.Span(f"{metrics['profit_change_pct']:+.1f}% profit impact", 
                             className=impact_color),
                    "."
                ]),
                html.P([
                    f"Demand is projected to change by {metrics['demand_change_pct']:+.1f}%."
                ]),
                html.Hr(),
                html.H6("Optimal Strategy:"),
                html.P(
                    "Implement price adjustment and maintain current inventory levels." if metrics['profit_change_pct'] > 0 and not metrics['reorder_needed'] else
                    "Implement price adjustment and reorder additional inventory." if metrics['profit_change_pct'] > 0 and metrics['reorder_needed'] else
                    "Maintain current price and reorder inventory." if metrics['profit_change_pct'] <= 0 and metrics['reorder_needed'] else
                    "Maintain current price and inventory strategy."
                )
            ])
        ])
        
        insights.append(summary_insight)
        
        # Return insights in two-column layout
        return dbc.Row([
            dbc.Col(insights[0], width=12, md=6),
            dbc.Col([insights[1], insights[2]], width=12, md=6)
        ])
        
    except Exception as e:
        logger.error(f"Error updating insights: {str(e)}")
        return html.Div("Error generating insights")


def register_integrated_callbacks(app, data_dict):
    """
    Register callbacks for the integrated view.
    
    Args:
        app: Dash app instance
        data_dict: Dictionary with all loaded data
    """
    # Update integrated chart
    @app.callback(
        [
            Output("integrated-chart", "figure"),
            Output("integrated-key-metrics", "children"),
            Output("integrated-insights", "children")
        ],
        [
            Input("integrated-store-dropdown", "value"),
            Input("integrated-product-dropdown", "value"),
            Input("price-adjustment-slider", "value"),
            Input("inventory-adjustment-slider", "value"),
            Input("forecast-period-slider", "value"),
            Input("show-item-numbers-store", "data")
        ]
    )
    def update_integrated_view(store, product, price_adjustment, inventory_adjustment, 
                             forecast_period, show_item_numbers):
        """Update all integrated view components"""
        # Calculate key metrics
        metrics = calculate_key_metrics(
            data_dict, store, product, price_adjustment, 
            inventory_adjustment, forecast_period
        )
        
        # Update chart
        fig = update_integrated_chart(
            data_dict, store, product, price_adjustment, 
            inventory_adjustment, forecast_period, show_item_numbers
        )
        
        # Update key metrics display
        metrics_display = update_key_metrics_display(metrics)
        
        # Update insights
        insights = update_insights(metrics)
        
        return fig, metrics_display, insights