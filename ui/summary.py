"""
Summary dashboard module for the Pizza Predictive Ordering System.
"""
import sys
import os
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import timedelta
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core UI components
from ui.core import create_error_message

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('summary_dashboard')

# Define the summary dashboard functions directly
# No need to import from self, which causes circular dependency
logger.info("Successfully imported functions from original summary_dashboard.py")
    
# Create stub functions for backward compatibility
"""
    Process data for the summary dashboard
    Returns processed data ready for visualizations
"""
def load_summary_data(combined_data, profit_impact, inventory_projection, pytorch_forecasts):
    if combined_data is None:
        return None
        
    try:
        # Get the latest date for each store
        stores = sorted(combined_data['Store_Id'].unique())
        latest_dates = {}
        
        for store in stores:
            store_data = combined_data[combined_data['Store_Id'] == store]
            latest_dates[store] = store_data['Date'].max()
            
        # Process data for each visualization component
        summary_data = {
            'top_profit_items': calculate_top_profit_items(combined_data, profit_impact),
            'out_of_date_items': identify_expiring_items(combined_data),
            'promotion_performance': analyze_promotion_performance(combined_data),
            'out_of_stock_items': identify_out_of_stock(combined_data, inventory_projection),
            'below_safety_stock': identify_below_safety_stock(combined_data),
            'future_high_demand': forecast_high_demand(combined_data, pytorch_forecasts),
            'stores': stores,
            'latest_dates': latest_dates,
            'combined_data': combined_data  # Include the combined data for additional charts
        }
        
        return summary_data
        
    except Exception as e:
        logger.error(f"Error processing summary data: {e}")
        return None
    
def calculate_top_profit_items(combined_data, profit_impact, top_n=10):
    """Calculate items with highest profit contribution - simplified stub function"""
    return pd.DataFrame()
    
def identify_expiring_items(combined_data, days_threshold=7):
    """Identify items approaching expiration date - simplified stub function"""
    return pd.DataFrame()
    
def analyze_promotion_performance(combined_data):
    """Analyze performance of items on promotion - simplified stub function"""
    return pd.DataFrame()
    
def identify_out_of_stock(combined_data, inventory_projection):
    """Identify items that are out of stock - simplified stub function"""
    return pd.DataFrame()
    
def identify_below_safety_stock(combined_data):
    """Identify items below safety stock levels - simplified stub function"""
    return pd.DataFrame()
    
def forecast_high_demand(combined_data, pytorch_forecasts, threshold_factor=1.5):
    """Identify items forecasted to have high demand - simplified stub function"""
    return pd.DataFrame()
    
def create_top_profit_chart(top_profit_data):
    """Create chart for top profit items - simplified stub function"""
    return go.Figure()
    
def create_out_of_date_chart(out_of_date_data):
    """Create chart for items approaching expiration - simplified stub function"""
    return go.Figure()
    
def create_promotion_chart(promotion_data):
    """Create chart for promotion performance - simplified stub function"""
    return go.Figure()
    
def create_out_of_stock_chart(out_of_stock_data):
    """Create chart for out of stock items - simplified stub function"""
    return go.Figure()
    
def create_safety_stock_chart(safety_stock_data):
    """Create chart for items below safety stock - simplified stub function"""
    return go.Figure()
    
def create_future_demand_chart(future_demand_data):
    """Create chart for items with forecasted high demand - simplified stub function"""
    return go.Figure()
    
def create_inventory_metrics_chart(combined_data):
    """Create chart for inventory metrics - simplified stub function"""
    return go.Figure()
    
def generate_weekly_forecasts(combined_data, pytorch_forecasts, profit_impact, inventory_projection):
    """Generate weekly forecasts for next 4 weeks - simplified stub function"""
    return None
    
def create_weekly_forecast_charts(forecast_results):
    """Create charts for weekly forecasts - simplified stub function"""
    return go.Figure()
    
def create_forecast_data_table(forecast_results):
    """Create data table with forecast data - simplified stub function"""
    return html.Div("No forecast data available")
    
def create_sales_by_category_chart(combined_data):
    """Create chart for sales by category - simplified stub function"""
    return go.Figure()


def create_summary_overview_content():
    """
    Create content for the summary overview tab.
    
    Returns:
        html.Div: Tab content
    """
    return html.Div([
        html.Div(id="summary-overview-content")
    ])


def create_profit_loss_content():
    """
    Create content for the profit/loss analysis tab.
    
    Returns:
        html.Div: Tab content
    """
    return html.Div([
        # Controls for profit/loss analysis
        dbc.Row([
            dbc.Col([
                html.Label("Time Period:"),
                dcc.RadioItems(
                    id="profit-loss-period",
                    options=[
                        {"label": "Daily", "value": "daily"},
                        {"label": "Weekly", "value": "weekly"},
                        {"label": "Monthly", "value": "monthly"}
                    ],
                    value="weekly",
                    inputStyle={"margin-right": "5px"},
                    labelStyle={"margin-right": "15px", "display": "inline-block"}
                )
            ], width=6),
            dbc.Col([
                html.Label("View By:"),
                dcc.RadioItems(
                    id="profit-loss-view",
                    options=[
                        {"label": "Product Category", "value": "category"},
                        {"label": "Top Products", "value": "product"},
                        {"label": "Overall", "value": "overall"}
                    ],
                    value="overall",
                    inputStyle={"margin-right": "5px"},
                    labelStyle={"margin-right": "15px", "display": "inline-block"}
                )
            ], width=6)
        ], className="mb-4"),
        
        # Profit/Loss charts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Profit/Loss Trends"),
                    dbc.CardBody(dcc.Graph(id="profit-trend-chart"))
                ])
            ], width=12)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Profit Margin Analysis"),
                    dbc.CardBody(dcc.Graph(id="profit-margin-chart"))
                ])
            ], width=12)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Revenue vs. Cost Breakdown"),
                    dbc.CardBody(dcc.Graph(id="revenue-cost-chart"))
                ])
            ], width=12)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Profit Impact Factors"),
                    dbc.CardBody(dcc.Graph(id="profit-impact-chart"))
                ])
            ], width=12)
        ])
    ])


def create_inventory_analysis_content():
    """
    Create content for the inventory analysis tab.
    
    Returns:
        html.Div: Tab content
    """
    return html.Div([
        # Controls for inventory analysis
        dbc.Row([
            dbc.Col([
                html.Label("Inventory Metric:"),
                dcc.RadioItems(
                    id="inventory-metric",
                    options=[
                        {"label": "Stock Levels", "value": "stock"},
                        {"label": "Turnover Rate", "value": "turnover"},
                        {"label": "Days of Supply", "value": "supply"},
                        {"label": "Stock Outs", "value": "stockout"}
                    ],
                    value="stock",
                    inputStyle={"margin-right": "5px"},
                    labelStyle={"margin-right": "15px", "display": "inline-block"}
                )
            ], width=6),
            dbc.Col([
                html.Label("Group By:"),
                dcc.RadioItems(
                    id="inventory-group",
                    options=[
                        {"label": "Category", "value": "category"},
                        {"label": "Product", "value": "product"},
                        {"label": "Store", "value": "store"}
                    ],
                    value="category",
                    inputStyle={"margin-right": "5px"},
                    labelStyle={"margin-right": "15px", "display": "inline-block"}
                )
            ], width=6)
        ], className="mb-4"),
        
        # Inventory analysis charts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Inventory Status Distribution"),
                    dbc.CardBody(dcc.Graph(id="inventory-status-chart"))
                ])
            ], width=12)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Inventory Trends Over Time"),
                    dbc.CardBody(dcc.Graph(id="inventory-trend-chart"))
                ])
            ], width=12)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Inventory Cost Analysis"),
                    dbc.CardBody(dcc.Graph(id="inventory-cost-chart"))
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Stock Out Risk Assessment"),
                    dbc.CardBody(dcc.Graph(id="stockout-risk-chart"))
                ])
            ], width=6)
        ])
    ])


def create_summary_tab_content():
    """
    Create content for the summary dashboard tab.
    
    Returns:
        html.Div: Tab content
    """
    return html.Div([
        # Summary Dashboard Tabs
        dbc.Tabs([
            dbc.Tab(label="Overview", tab_id="summary-overview-tab",
                   children=create_summary_overview_content()),
            dbc.Tab(label="Profit/Loss Analysis", tab_id="profit-loss-tab",
                   children=create_profit_loss_content()),
            dbc.Tab(label="Inventory Analysis", tab_id="summary-inventory-tab",
                   children=create_inventory_analysis_content())
        ], id="summary-tabs"),
        html.Hr(),
        # Store selector row
        dbc.Row([
            dbc.Col([
                html.Label("Select Store:"),
                dcc.Dropdown(
                    id="store-dropdown",
                    options=[],
                    clearable=False
                )
            ], width=6),
            dbc.Col([
                html.Label("Manual Stock Adjustment"),
                dbc.InputGroup([
                    dbc.InputGroupText("Product:"),
                    dcc.Dropdown(
                        id="product-dropdown",
                        options=[],
                        style={"width": "50%"}
                    ),
                    dbc.Input(
                        id="stock-adjustment-input",
                        type="number",
                        placeholder="New stock level",
                        style={"width": "25%"}
                    ),
                    dbc.Button(
                        "Apply",
                        id="apply-stock-adjustment",
                        color="primary"
                    )
                ])
            ], width=6)
        ], className="mb-4"),
        
        # Optional date picker for adjustments
        dbc.Row([
            dbc.Col([], width=6),
            dbc.Col([
                html.Label("Adjustment Date (Optional):"),
                dcc.DatePickerSingle(
                    id="stock-adjustment-date",
                    display_format="YYYY-MM-DD"
                )
            ], width=6)
        ], className="mb-4"),
        
        # Summary dashboard content container
        html.Div(
            id="summary-dashboard-content",
            children=[
                create_error_message("Select a store to view the summary dashboard")
            ]
        )
    ])


def create_summary_dashboard_layout(summary_data):
    """
    Create the layout for the summary dashboard based on summary_data.
    
    Args:
        summary_data: Dictionary of processed data for visualizations
    
    Returns:
        list: List of dashboard components
    """
    if summary_data is None or not isinstance(summary_data, dict):
        return [create_error_message("No data available for summary dashboard")]
    
    # Create charts based on summary data
    top_profit_chart = create_top_profit_chart(summary_data.get('top_profit_items'))
    out_of_date_chart = create_out_of_date_chart(summary_data.get('out_of_date_items'))
    promotion_chart = create_promotion_chart(summary_data.get('promotion_performance'))
    out_of_stock_chart = create_out_of_stock_chart(summary_data.get('out_of_stock_items'))
    safety_stock_chart = create_safety_stock_chart(summary_data.get('below_safety_stock'))
    future_demand_chart = create_future_demand_chart(summary_data.get('future_high_demand'))
    inventory_metrics_chart = create_inventory_metrics_chart(summary_data.get('combined_data'))
    sales_category_chart = create_sales_by_category_chart(summary_data.get('combined_data'))
    
    # Create profit/loss analysis charts
    profit_trend_chart = create_profit_trend_chart(summary_data.get('combined_data'), summary_data.get('profit_impact'))
    profit_margin_chart = create_profit_margin_chart(summary_data.get('combined_data'), summary_data.get('profit_impact'))
    revenue_cost_chart = create_revenue_cost_chart(summary_data.get('combined_data'))
    profit_impact_chart = create_profit_impact_chart(summary_data.get('profit_impact'))
    
    # Create inventory analysis charts
    inventory_status_chart = create_inventory_status_chart(summary_data.get('combined_data'), summary_data.get('inventory_projection'))
    inventory_trend_chart = create_inventory_trend_chart(summary_data.get('combined_data'), summary_data.get('inventory_projection'))
    inventory_cost_chart = create_inventory_cost_chart(summary_data.get('combined_data'), summary_data.get('inventory_projection'))
    stockout_risk_chart = create_stockout_risk_chart(summary_data.get('combined_data'), summary_data.get('inventory_projection'))
    
    # Generate weekly forecasts
    forecast_results = summary_data.get('weekly_forecasts')
    weekly_forecast_chart = create_weekly_forecast_charts(forecast_results) if forecast_results else None
    forecast_data_table = create_forecast_data_table(forecast_results) if forecast_results else None
    
    # Create overview layout with each visualization in its own row
    overview_layout = [
        # Title row
        dbc.Row([
            dbc.Col(html.H2("Store Performance Summary Dashboard", className="text-center"), width=12)
        ]),
        
        # Each visualization gets its own row - Top profit items
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Top Profit Items"),
                    dbc.CardBody(dcc.Graph(figure=top_profit_chart))
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Items approaching expiration
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Items Approaching Expiration"),
                    dbc.CardBody(dcc.Graph(figure=out_of_date_chart))
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Promotion performance
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Promotion Performance"),
                    dbc.CardBody(dcc.Graph(figure=promotion_chart))
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Out of stock items
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Out of Stock Items"),
                    dbc.CardBody(dcc.Graph(figure=out_of_stock_chart))
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Items below safety stock
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Items Below Safety Stock"),
                    dbc.CardBody(dcc.Graph(figure=safety_stock_chart))
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Forecasted high demand
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Forecasted High Demand"),
                    dbc.CardBody(dcc.Graph(figure=future_demand_chart))
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Inventory status distribution
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Inventory Status Distribution"),
                    dbc.CardBody(dcc.Graph(figure=inventory_metrics_chart))
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Sales by category
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Sales by Category"),
                    dbc.CardBody(dcc.Graph(figure=sales_category_chart))
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Weekly forecast section (if available)
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Weekly Forecast Report (4-Week Projection)", className="text-center")),
                    dbc.CardBody([
                        dcc.Graph(figure=weekly_forecast_chart) if weekly_forecast_chart else 
                        html.Div("No forecast data available", className="text-center p-5 bg-light"),
                        html.Hr(),
                        forecast_data_table if forecast_data_table else html.Div()
                    ])
                ])
            ], width=12)
        ]) if weekly_forecast_chart else html.Div()
    ]
    
    # Create profit/loss analysis tab content
    profit_loss_layout = [
        dbc.Row([
            dbc.Col(html.H2("Profit/Loss Analysis", className="text-center"), width=12)
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Profit/Loss Trends"),
                    dbc.CardBody(dcc.Graph(figure=profit_trend_chart, id="profit-trend-chart"))
                ])
            ], width=12)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Profit Margin Analysis"),
                    dbc.CardBody(dcc.Graph(figure=profit_margin_chart, id="profit-margin-chart"))
                ])
            ], width=12)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Revenue vs. Cost Breakdown"),
                    dbc.CardBody(dcc.Graph(figure=revenue_cost_chart, id="revenue-cost-chart"))
                ])
            ], width=12)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Profit Impact Factors"),
                    dbc.CardBody(dcc.Graph(figure=profit_impact_chart, id="profit-impact-chart"))
                ])
            ], width=12)
        ])
    ]
    
    # Create inventory analysis tab content
    inventory_analysis_layout = [
        dbc.Row([
            dbc.Col(html.H2("Inventory Analysis", className="text-center"), width=12)
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Inventory Status Distribution"),
                    dbc.CardBody(dcc.Graph(figure=inventory_status_chart, id="inventory-status-chart"))
                ])
            ], width=12)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Inventory Trends Over Time"),
                    dbc.CardBody(dcc.Graph(figure=inventory_trend_chart, id="inventory-trend-chart"))
                ])
            ], width=12)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Inventory Cost Analysis"),
                    dbc.CardBody(dcc.Graph(figure=inventory_cost_chart, id="inventory-cost-chart"))
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Stock Out Risk Assessment"),
                    dbc.CardBody(dcc.Graph(figure=stockout_risk_chart, id="stockout-risk-chart"))
                ])
            ], width=6)
        ])
    ]
    
    # Update the overview tab content and create the full dashboard layout
    dashboard_layout = [
        # Callback targets for each tab content
        html.Div(id="summary-overview-content", children=overview_layout),
        html.Div(id="profit-loss-content", style={"display": "none"}, children=profit_loss_layout),
        html.Div(id="summary-inventory-content", style={"display": "none"}, children=inventory_analysis_layout)
    ]
    
    return dashboard_layout


# Add new functions for creating profit/loss analysis charts
def create_profit_trend_chart(combined_data, profit_impact):
    """
    Create chart showing profit trends over time.
    
    Args:
        combined_data: DataFrame with historical sales data
        profit_impact: DataFrame with profit impact data
    
    Returns:
        go.Figure: Profit trend chart
    """
    fig = go.Figure()
    
    if combined_data is None or len(combined_data) == 0:
        return fig
        
    try:
        # Group data by date and calculate profit
        if 'Cost' in combined_data.columns and 'Price' in combined_data.columns and 'Sales' in combined_data.columns:
            # Make a copy to avoid SettingWithCopyWarning
            profit_data = combined_data.copy()
            
            # Calculate profit for each sale
            profit_data['Unit_Profit'] = profit_data['Price'] - profit_data['Cost']
            profit_data['Daily_Profit'] = profit_data['Unit_Profit'] * profit_data['Sales']
            
            # Group by date
            daily_profit = profit_data.groupby('Date')['Daily_Profit'].sum().reset_index()
            
            # Calculate 7-day moving average
            daily_profit = daily_profit.sort_values('Date')
            daily_profit['7_Day_MA'] = daily_profit['Daily_Profit'].rolling(window=7, min_periods=1).mean()
            
            # Add traces
            fig.add_trace(go.Scatter(
                x=daily_profit['Date'],
                y=daily_profit['Daily_Profit'],
                mode='lines',
                name='Daily Profit',
                line=dict(color='lightblue')
            ))
            
            fig.add_trace(go.Scatter(
                x=daily_profit['Date'],
                y=daily_profit['7_Day_MA'],
                mode='lines',
                name='7-Day Moving Average',
                line=dict(color='darkblue', width=2)
            ))
            
            # Update layout
            fig.update_layout(
                title="Profit Trends Over Time",
                xaxis_title="Date",
                yaxis_title="Profit ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
    except Exception as e:
        logger.error(f"Error creating profit trend chart: {e}")
    
    return fig


def create_profit_margin_chart(combined_data, profit_impact):
    """
    Create chart showing profit margin analysis.
    
    Args:
        combined_data: DataFrame with historical sales data
        profit_impact: DataFrame with profit impact data
    
    Returns:
        go.Figure: Profit margin chart
    """
    fig = go.Figure()
    
    if combined_data is None or len(combined_data) == 0:
        return fig
        
    try:
        # Make a copy to avoid SettingWithCopyWarning
        margin_data = combined_data.copy()
        
        if 'Product' in margin_data.columns and 'Price' in margin_data.columns and 'Cost' in margin_data.columns:
            # Calculate margin percentage
            margin_data['Margin'] = (margin_data['Price'] - margin_data['Cost']) / margin_data['Price'] * 100
            
            # Group by product and calculate average margin
            product_margins = margin_data.groupby('Product')['Margin'].mean().reset_index()
            
            # Sort by margin
            product_margins = product_margins.sort_values('Margin', ascending=False).head(10)
            
            # Create horizontal bar chart
            fig = px.bar(
                product_margins,
                y='Product',
                x='Margin',
                orientation='h',
                title="Top 10 Products by Profit Margin",
                labels={"Margin": "Profit Margin (%)", "Product": "Product"},
                color='Margin',
                color_continuous_scale='blues'
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Profit Margin (%)",
                yaxis_title="Product",
                coloraxis_showscale=False
            )
    except Exception as e:
        logger.error(f"Error creating profit margin chart: {e}")
    
    return fig


def create_revenue_cost_chart(combined_data):
    """
    Create chart showing revenue vs cost breakdown.
    
    Args:
        combined_data: DataFrame with historical sales data
    
    Returns:
        go.Figure: Revenue vs cost chart
    """
    fig = go.Figure()
    
    if combined_data is None or len(combined_data) == 0:
        return fig
        
    try:
        # Make a copy to avoid SettingWithCopyWarning
        revenue_data = combined_data.copy()
        
        if 'Product' in revenue_data.columns and 'Price' in revenue_data.columns and 'Cost' in revenue_data.columns and 'Sales' in revenue_data.columns:
            # Calculate revenue and cost
            revenue_data['Revenue'] = revenue_data['Price'] * revenue_data['Sales']
            revenue_data['Total_Cost'] = revenue_data['Cost'] * revenue_data['Sales']
            revenue_data['Profit'] = revenue_data['Revenue'] - revenue_data['Total_Cost']
            
            # Group by product category if available, otherwise by product
            group_col = 'Category' if 'Category' in revenue_data.columns else 'Product'
            category_data = revenue_data.groupby(group_col).agg({
                'Revenue': 'sum',
                'Total_Cost': 'sum',
                'Profit': 'sum'
            }).reset_index()
            
            # Sort by revenue and select top 5
            category_data = category_data.sort_values('Revenue', ascending=False).head(5)
            
            # Create stacked bar chart
            fig = go.Figure()
            
            # Add cost bars
            fig.add_trace(go.Bar(
                x=category_data[group_col],
                y=category_data['Total_Cost'],
                name='Cost',
                marker_color='lightcoral'
            ))
            
            # Add profit bars (stacked on cost)
            fig.add_trace(go.Bar(
                x=category_data[group_col],
                y=category_data['Profit'],
                name='Profit',
                marker_color='lightgreen'
            ))
            
            # Update layout for stacked bars
            fig.update_layout(
                title=f"Top 5 {group_col}s by Revenue - Cost vs. Profit Breakdown",
                xaxis_title=group_col,
                yaxis_title="Amount ($)",
                barmode='stack',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
    except Exception as e:
        logger.error(f"Error creating revenue cost chart: {e}")
    
    return fig


def create_profit_impact_chart(profit_impact):
    """
    Create chart showing profit impact factors.
    
    Args:
        profit_impact: DataFrame with profit impact data
    
    Returns:
        go.Figure: Profit impact chart
    """
    fig = go.Figure()
    
    if profit_impact is None or len(profit_impact) == 0:
        return fig
        
    try:
        # Check if we have the necessary columns
        required_cols = ['Daily_Profit_Impact', 'Price_Change_Pct', 'Product']
        
        if all(col in profit_impact.columns for col in required_cols):
            # Get top 10 products by absolute profit impact
            impact_data = profit_impact.copy()
            impact_data['Abs_Impact'] = impact_data['Daily_Profit_Impact'].abs()
            top_impacts = impact_data.sort_values('Abs_Impact', ascending=False).head(10)
            
            # Create waterfall chart
            products = top_impacts['Product'].tolist()
            impacts = top_impacts['Daily_Profit_Impact'].tolist()
            colors = ['green' if x > 0 else 'red' for x in impacts]
            
            fig = go.Figure(go.Waterfall(
                name="Profit Impact",
                orientation="v",
                measure=["relative"] * len(products),
                x=products,
                y=impacts,
                connector={"line":{"color":"rgb(63, 63, 63)"}},
                increasing={"marker":{"color":"green"}},
                decreasing={"marker":{"color":"red"}}
            ))
            
            # Update layout
            fig.update_layout(
                title="Profit Impact by Product",
                xaxis_title="Product",
                yaxis_title="Daily Profit Impact ($)",
                showlegend=False
            )
    except Exception as e:
        logger.error(f"Error creating profit impact chart: {e}")
    
    return fig


# Add new functions for creating inventory analysis charts
def create_inventory_status_chart(combined_data, inventory_projection):
    """
    Create chart showing inventory status distribution.
    
    Args:
        combined_data: DataFrame with historical sales data
        inventory_projection: DataFrame with inventory projection data
    
    Returns:
        go.Figure: Inventory status chart
    """
    fig = go.Figure()
    
    if inventory_projection is None or len(inventory_projection) == 0:
        return fig
        
    try:
        # Get the latest inventory data for each product
        inv_data = inventory_projection.copy()
        
        # Determine columns based on what's available
        stock_col = 'Stock_Level' if 'Stock_Level' in inv_data.columns else 'Current_Stock'
        sales_col = 'Recent_Daily_Sales' if 'Recent_Daily_Sales' in inv_data.columns else 'Sales'
        
        if stock_col in inv_data.columns and sales_col in inv_data.columns:
            # Get the latest date for each product
            latest_data = inv_data.sort_values('Date').groupby(['Store_Id', 'Item']).last().reset_index()
            
            # Calculate weeks of stock
            latest_data['Weeks_Of_Stock'] = latest_data[stock_col] / (latest_data[sales_col] * 7)
            
            # Categorize stock levels
            def categorize_stock(weeks):
                if weeks < 1:  # Less than 1 week
                    return "Critical (< 1 week)"
                elif weeks < 2:  # 1-2 weeks
                    return "Low (1-2 weeks)"
                elif weeks < 4:  # 2-4 weeks
                    return "Adequate (2-4 weeks)"
                else:  # More than 4 weeks
                    return "Excess (> 4 weeks)"
            
            latest_data['Stock_Category'] = latest_data['Weeks_Of_Stock'].apply(categorize_stock)
            
            # Count products in each category
            category_counts = latest_data['Stock_Category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            
            # Define category order and colors
            category_order = ["Critical (< 1 week)", "Low (1-2 weeks)", "Adequate (2-4 weeks)", "Excess (> 4 weeks)"]
            colors = ['red', 'orange', 'green', 'blue']
            
            # Create a pie chart
            fig = px.pie(
                category_counts,
                values='Count',
                names='Category',
                title='Inventory Status Distribution',
                color='Category',
                color_discrete_map=dict(zip(category_order, colors)),
                category_orders={"Category": category_order}
            )
            
            # Update layout
            fig.update_layout(
                legend_title="Stock Status"
            )
    except Exception as e:
        logger.error(f"Error creating inventory status chart: {e}")
    
    return fig


def create_inventory_trend_chart(combined_data, inventory_projection):
    """
    Create chart showing inventory trends over time.
    
    Args:
        combined_data: DataFrame with historical sales data
        inventory_projection: DataFrame with inventory projection data
    
    Returns:
        go.Figure: Inventory trend chart
    """
    fig = go.Figure()
    
    if inventory_projection is None or len(inventory_projection) == 0:
        return fig
        
    try:
        # Get inventory data
        inv_data = inventory_projection.copy()
        
        # Determine columns based on what's available
        stock_col = 'Stock_Level' if 'Stock_Level' in inv_data.columns else 'Current_Stock'
        
        if stock_col in inv_data.columns:
            # Group by date and calculate average, max, and min stock levels across all products
            daily_stock = inv_data.groupby('Date')[stock_col].agg(['mean', 'max', 'min']).reset_index()
            
            # Create the trend chart with range
            fig = go.Figure([
                # Mean line
                go.Scatter(
                    name='Average Stock Level',
                    x=daily_stock['Date'],
                    y=daily_stock['mean'],
                    mode='lines',
                    line=dict(color='rgb(31, 119, 180)'),
                ),
                # Min line
                go.Scatter(
                    name='Min Stock Level',
                    x=daily_stock['Date'],
                    y=daily_stock['min'],
                    mode='lines',
                    line=dict(color='red'),
                    opacity=0.5
                ),
                # Max line
                go.Scatter(
                    name='Max Stock Level',
                    x=daily_stock['Date'],
                    y=daily_stock['max'],
                    mode='lines',
                    line=dict(color='green'),
                    opacity=0.5,
                    fill='tonexty'  # Fill area between min and max
                )
            ])
            
            # Update layout
            fig.update_layout(
                title="Inventory Level Trends Over Time",
                xaxis_title="Date",
                yaxis_title="Stock Level (Units)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
    except Exception as e:
        logger.error(f"Error creating inventory trend chart: {e}")
    
    return fig


def create_inventory_cost_chart(combined_data, inventory_projection):
    """
    Create chart showing inventory carrying costs.
    
    Args:
        combined_data: DataFrame with historical sales data
        inventory_projection: DataFrame with inventory projection data
    
    Returns:
        go.Figure: Inventory cost chart
    """
    fig = go.Figure()
    
    if combined_data is None or inventory_projection is None or len(combined_data) == 0 or len(inventory_projection) == 0:
        return fig
        
    try:
        # Get the latest inventory data
        inv_data = inventory_projection.copy()
        sales_data = combined_data.copy()
        
        # Check if we have the necessary columns
        stock_col = 'Stock_Level' if 'Stock_Level' in inv_data.columns else 'Current_Stock'
        if stock_col in inv_data.columns and 'Cost' in sales_data.columns:
            # Get latest inventory data
            latest_inv = inv_data.sort_values('Date').groupby(['Store_Id', 'Item']).last().reset_index()
            
            # Merge with cost data
            if 'Item' in sales_data.columns and 'Store_Id' in sales_data.columns:
                # Get average cost for each item
                cost_data = sales_data.groupby(['Store_Id', 'Item'])['Cost'].mean().reset_index()
                merged_data = pd.merge(latest_inv, cost_data, on=['Store_Id', 'Item'], how='left')
                
                # Calculate inventory value
                merged_data['Inventory_Value'] = merged_data[stock_col] * merged_data['Cost']
                
                # Group by category if available
                if 'Category' in merged_data.columns:
                    group_col = 'Category'
                elif 'Product' in merged_data.columns:
                    group_col = 'Product'
                else:
                    group_col = 'Store_Id'
                
                # Calculate inventory value by group
                inv_value_by_group = merged_data.groupby(group_col)['Inventory_Value'].sum().reset_index()
                inv_value_by_group = inv_value_by_group.sort_values('Inventory_Value', ascending=False)
                
                # Create bar chart
                fig = px.bar(
                    inv_value_by_group.head(10),
                    x=group_col,
                    y='Inventory_Value',
                    title=f"Inventory Value by {group_col}",
                    labels={
                        group_col: group_col,
                        'Inventory_Value': 'Inventory Value ($)'
                    },
                    color='Inventory_Value',
                    color_continuous_scale='blues'
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title=group_col,
                    yaxis_title="Inventory Value ($)",
                    coloraxis_showscale=False
                )
    except Exception as e:
        logger.error(f"Error creating inventory cost chart: {e}")
    
    return fig


def create_stockout_risk_chart(combined_data, inventory_projection):
    """
    Create chart showing stockout risk assessment.
    
    Args:
        combined_data: DataFrame with historical sales data
        inventory_projection: DataFrame with inventory projection data
    
    Returns:
        go.Figure: Stockout risk chart
    """
    fig = go.Figure()
    
    if inventory_projection is None or len(inventory_projection) == 0:
        return fig
        
    try:
        # Get inventory data
        inv_data = inventory_projection.copy()
        
        # Determine columns based on what's available
        stock_col = 'Stock_Level' if 'Stock_Level' in inv_data.columns else 'Current_Stock'
        sales_col = 'Recent_Daily_Sales' if 'Recent_Daily_Sales' in inv_data.columns else 'Sales'
        
        if stock_col in inv_data.columns and sales_col in inv_data.columns:
            # Get the latest date for each product
            latest_data = inv_data.sort_values('Date').groupby(['Store_Id', 'Item']).last().reset_index()
            
            # Calculate days until stockout
            latest_data['Days_Until_Stockout'] = latest_data.apply(
                lambda row: row[stock_col] / row[sales_col] if row[sales_col] > 0 else 30,
                axis=1
            )
            
            # Cap at 30 days
            latest_data['Days_Until_Stockout'] = latest_data['Days_Until_Stockout'].clip(upper=30)
            
            # Categorize by stockout risk
            def stockout_risk(days):
                if days < 3:
                    return "Critical (< 3 days)"
                elif days < 7:
                    return "High (3-7 days)"
                elif days < 14:
                    return "Medium (7-14 days)"
                else:
                    return "Low (> 14 days)"
            
            latest_data['Stockout_Risk'] = latest_data['Days_Until_Stockout'].apply(stockout_risk)
            
            # Count by risk category
            risk_counts = latest_data['Stockout_Risk'].value_counts().reset_index()
            risk_counts.columns = ['Risk_Category', 'Count']
            
            # Define risk order and colors
            risk_order = ["Critical (< 3 days)", "High (3-7 days)", "Medium (7-14 days)", "Low (> 14 days)"]
            risk_colors = ['darkred', 'red', 'orange', 'green']
            
            # Create a horizontal bar chart
            fig = px.bar(
                risk_counts,
                y='Risk_Category',
                x='Count',
                orientation='h',
                title='Stockout Risk Assessment',
                color='Risk_Category',
                color_discrete_map=dict(zip(risk_order, risk_colors)),
                category_orders={"Risk_Category": risk_order}
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Number of Products",
                yaxis_title="Stockout Risk",
                showlegend=False
            )
    except Exception as e:
        logger.error(f"Error creating stockout risk chart: {e}")
    
    return fig


def register_summary_callbacks(app, data_dict):
    """
    Register callbacks for the summary dashboard.
    
    Args:
        app: Dash app instance
        data_dict: Dictionary of loaded data
    """
    # Add callback for summary tabs
    @app.callback(
        [
            Output("summary-overview-content", "style"),
            Output("profit-loss-content", "style"),
            Output("summary-inventory-content", "style")
        ],
        [Input("summary-tabs", "active_tab")]
    )
    def switch_summary_tab(active_tab):
        """Switch between summary dashboard tabs"""        
        if active_tab == "summary-overview-tab":
            return {"display": "block"}, {"display": "none"}, {"display": "none"}
        elif active_tab == "profit-loss-tab":
            return {"display": "none"}, {"display": "block"}, {"display": "none"}
        elif active_tab == "summary-inventory-tab":
            return {"display": "none"}, {"display": "none"}, {"display": "block"}
        else:
            return {"display": "block"}, {"display": "none"}, {"display": "none"}
            
    # Add callbacks for profit/loss analysis tab
    @app.callback(
        [
            Output("profit-trend-chart", "figure"),
            Output("profit-margin-chart", "figure"),
            Output("revenue-cost-chart", "figure"),
            Output("profit-impact-chart", "figure")
        ],
        [
            Input("store-dropdown", "value"),
            Input("profit-loss-period", "value"),
            Input("profit-loss-view", "value")
        ]
    )
    def update_profit_loss_charts(store, period, view):
        """Update charts in the profit/loss analysis tab"""        
        try:
            # Get the required data
            combined_data = data_dict.get('combined_data')
            profit_impact = data_dict.get('profit_impact')
            
            if combined_data is None or store is None:
                return go.Figure(), go.Figure(), go.Figure(), go.Figure()
            
            # Filter data for selected store
            store_data = combined_data[combined_data['Store_Id'] == store].copy()
            store_profit_impact = None if profit_impact is None else profit_impact[profit_impact['Store_Id'] == store].copy()
            
            # Create the profit/loss charts
            trend_chart = create_profit_trend_chart(store_data, store_profit_impact)
            margin_chart = create_profit_margin_chart(store_data, store_profit_impact)
            revenue_cost_chart = create_revenue_cost_chart(store_data)
            impact_chart = create_profit_impact_chart(store_profit_impact)
            
            return trend_chart, margin_chart, revenue_cost_chart, impact_chart
            
        except Exception as e:
            logger.error(f"Error updating profit/loss charts: {str(e)}")
            return go.Figure(), go.Figure(), go.Figure(), go.Figure()
    
    # Add callbacks for inventory analysis tab
    @app.callback(
        [
            Output("inventory-status-chart", "figure"),
            Output("inventory-trend-chart", "figure"),
            Output("inventory-cost-chart", "figure"),
            Output("stockout-risk-chart", "figure")
        ],
        [
            Input("store-dropdown", "value"),
            Input("inventory-metric", "value"),
            Input("inventory-group", "value")
        ]
    )
    def update_inventory_analysis_charts(store, metric, group):
        """Update charts in the inventory analysis tab"""        
        try:
            # Get the required data
            combined_data = data_dict.get('combined_data')
            inventory_projection = data_dict.get('inventory_projection')
            
            if combined_data is None or store is None:
                return go.Figure(), go.Figure(), go.Figure(), go.Figure()
            
            # Filter data for selected store
            store_data = combined_data[combined_data['Store_Id'] == store].copy()
            store_inventory = None if inventory_projection is None else inventory_projection[inventory_projection['Store_Id'] == store].copy()
            
            # Create the inventory analysis charts
            status_chart = create_inventory_status_chart(store_data, store_inventory)
            trend_chart = create_inventory_trend_chart(store_data, store_inventory)
            cost_chart = create_inventory_cost_chart(store_data, store_inventory)
            risk_chart = create_stockout_risk_chart(store_data, store_inventory)
            
            return status_chart, trend_chart, cost_chart, risk_chart
            
        except Exception as e:
            logger.error(f"Error updating inventory analysis charts: {str(e)}")
            return go.Figure(), go.Figure(), go.Figure(), go.Figure()
    @app.callback(
        Output("summary-dashboard-content", "children"),
        [
            Input("store-dropdown", "value"),
            Input("apply-stock-adjustment", "n_clicks")
        ],
        [
            State("product-dropdown", "value"),
            State("stock-adjustment-input", "value"),
            State("stock-adjustment-date", "date")
        ]
    )
    def update_summary_dashboard(store, adjust_clicks, product, stock_adjustment, adjustment_date):
        """Update the summary dashboard based on store selection and adjustments"""
        try:
            # Get the required data
            combined_data = data_dict.get('combined_data')
            profit_impact = data_dict.get('profit_impact')
            inventory_projection = data_dict.get('inventory_projection')
            pytorch_forecasts = data_dict.get('pytorch_forecasts')
            
            if combined_data is None or store is None:
                return create_error_message("No data available for summary dashboard")
            
            # Filter data for selected store
            store_data = combined_data[combined_data['Store_Id'] == store].copy()
            
            # Apply any stock adjustments if available
            try:
                if hasattr(app, 'manual_stock_adjustments'):
                    for key, value in app.manual_stock_adjustments.items():
                        if key.startswith(f"{store}_"):
                            try:
                                parts = key.split("_")
                                if len(parts) >= 2:
                                    product_id = float(parts[1])
                                    product_mask = (store_data['Store_Id'] == store) & (store_data['Item'] == product_id)
                                    if any(product_mask):
                                        stock_col = 'Stock_Level' if 'Stock_Level' in store_data.columns else 'Current_Stock'
                                        store_data.loc[product_mask, stock_col] = value
                            except Exception as e:
                                logger.error(f"Error applying adjustment {key}: {str(e)}")
            except Exception as e:
                logger.error(f"Error checking for adjustments: {str(e)}")
            
            # Filter related datasets
            store_profit_impact = None if profit_impact is None else profit_impact[profit_impact['Store_Id'] == store]
            store_inventory = None if inventory_projection is None else inventory_projection[inventory_projection['Store_Id'] == store]
            store_forecasts = None if pytorch_forecasts is None else pytorch_forecasts[pytorch_forecasts['Store_Id'] == store]
            
            # Process data for visualizations
            summary_data = load_summary_data(store_data, store_profit_impact, store_inventory, store_forecasts)
            
            # Generate weekly forecasts if not already included
            if summary_data is not None and isinstance(summary_data, dict) and 'weekly_forecasts' not in summary_data:
                summary_data['weekly_forecasts'] = generate_weekly_forecasts(
                    store_data, store_forecasts, store_profit_impact, store_inventory
                )
                
            # Create and return the summary dashboard layout
            return create_summary_dashboard_layout(summary_data)
            
        except Exception as e:
            logger.error(f"Error updating summary dashboard: {str(e)}")
            return create_error_message(f"Error updating summary dashboard: {str(e)}")
    
    # Update store and product dropdowns
    @app.callback(
        [
            Output("store-dropdown", "options"),
            Output("store-dropdown", "value"),
            Output("product-dropdown", "options")
        ],
        [
            Input("show-item-numbers-store", "data")
        ]
    )
    def update_summary_dropdowns(show_item_numbers):
        """Update the store and product dropdown options"""
        try:
            # Get the data source
            combined_data = data_dict.get('combined_data')
            
            if combined_data is None or len(combined_data) == 0:
                return [], None, []
            
            # Get unique stores
            stores = sorted(combined_data['Store_Id'].unique())
            store_options = [{'label': f'Store {s}', 'value': s} for s in stores]
            
            # Set default store value
            default_store = stores[0] if stores else None
            
            # Get unique products
            products = combined_data[['Item', 'Product']].drop_duplicates() if 'Product' in combined_data.columns else None
            
            if products is not None and len(products) > 0:
                if show_item_numbers:
                    product_options = [
                        {'label': f"{row['Product']} ({row['Item']})", 'value': row['Item']}
                        for _, row in products.iterrows()
                    ]
                else:
                    product_options = [
                        {'label': row['Product'], 'value': row['Item']}
                        for _, row in products.iterrows()
                    ]
            else:
                # Create basic options if product names not available
                unique_items = sorted(combined_data['Item'].unique())
                product_options = [{'label': f'Product {item}', 'value': item} for item in unique_items]
            
            return store_options, default_store, product_options
            
        except Exception as e:
            logger.error(f"Error updating summary dropdowns: {str(e)}")
            return [], None, []