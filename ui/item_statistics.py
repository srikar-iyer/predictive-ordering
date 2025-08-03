"""
UI component for item statistics and extended forecasting visualizations.
This module provides Dash/Plotly components for visualizing item statistics
and extended forecasting data.
"""
import os
import sys
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ast
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.core import format_product_name, create_error_message
from config.settings import (
    MODELS_DIR, STATIC_DIR, ITEM_STATISTICS_FILE, EXTENDED_FORECASTS_FILE
)
from src.models.plotly_visualizations import PlotlyVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ui_item_statistics')

# Log the file paths
logger.info(f"Using item statistics file: {ITEM_STATISTICS_FILE}")
logger.info(f"Using extended forecasts file: {EXTENDED_FORECASTS_FILE}")

# Initialize PlotlyVisualizer for creating interactive charts
plotly_viz = PlotlyVisualizer(os.path.join(STATIC_DIR, 'plotly_visualizations'))

def create_item_statistics_tab_content():
    """
    Create content for the item statistics tab.
    
    Returns:
        html.Div: Tab content
    """
    return html.Div([
        # Title and description
        dbc.Row([
            dbc.Col([
                html.H2("Item Statistics and Extended Forecasting"),
                html.P(
                    "Comprehensive item-level analysis with extended forecasting capabilities beyond the current time range.",
                    className="text-muted"
                ),
            ])
        ], className="mb-4"),
        
        # Controls
        dbc.Row([
            # Store selector
            dbc.Col([
                html.Label("Select Store:"),
                dcc.Dropdown(
                    id="item-stats-store-dropdown",
                    options=[],  # Will be populated in callback
                    placeholder="Select a store"
                )
            ], width=3),
            
            # Item selector
            dbc.Col([
                html.Label("Select Item:"),
                dcc.Dropdown(
                    id="item-stats-item-dropdown",
                    options=[],  # Will be populated in callback
                    placeholder="Select an item"
                )
            ], width=5),
            
            # View type selector
            dbc.Col([
                html.Label("View Type:"),
                dcc.RadioItems(
                    id="item-stats-view-type",
                    options=[
                        {'label': 'Basic Statistics', 'value': 'basic'},
                        {'label': 'Detailed Analysis', 'value': 'detailed'},
                        {'label': 'Extended Forecast', 'value': 'forecast'}
                    ],
                    value='basic',
                    labelStyle={'display': 'block', 'margin-bottom': '5px'}
                )
            ], width=4)
        ], className="mb-4"),
        
        # Forecast controls (only visible when view type is 'forecast')
        dbc.Row([
            dbc.Col([
                html.Div([
                    # Date range selector
                    html.Label("Date Range:"),
                    dcc.DatePickerRange(
                        id="item-stats-date-range",
                        start_date=None,  # Will be set in callback
                        end_date=None,    # Will be set in callback
                        min_date_allowed=None,  # Will be set in callback
                        max_date_allowed=None,  # Will be set in callback
                        className="mb-2"
                    ),
                    
                    # Forecast options
                    dbc.Checklist(
                        id="item-stats-forecast-options",
                        options=[
                            {"label": "Show Original Forecast", "value": "original"},
                            {"label": "Show Extended Forecast", "value": "extended"},
                            {"label": "Show Confidence Intervals", "value": "ci"}
                        ],
                        value=["original", "extended", "ci"],
                        inline=True,
                        switch=True,
                        className="mb-2"
                    ),
                ], id="item-stats-forecast-controls", style={'display': 'none'})
            ])
        ], className="mb-4"),
        
        # Content area - will display different visualizations based on selection
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="item-stats-loading",
                    type="circle",
                    children=[
                        # Basic stats card
                        html.Div([
                            dbc.Card([
                                dbc.CardHeader(html.H4("Item Basic Statistics", id="item-stats-title")),
                                dbc.CardBody(id="item-stats-basic-content")
                            ])
                        ], id="item-stats-basic-container"),
                        
                        # Detailed analysis container
                        html.Div([
                            dbc.Tabs([
                                dbc.Tab(
                                    dcc.Graph(id="item-stats-sales-pattern", figure=go.Figure()),
                                    label="Sales Pattern"
                                ),
                                dbc.Tab(
                                    dcc.Graph(id="item-stats-weekly-pattern", figure=go.Figure()),
                                    label="Weekly Pattern"
                                ),
                                dbc.Tab(
                                    dcc.Graph(id="item-stats-monthly-pattern", figure=go.Figure()),
                                    label="Monthly Pattern"
                                ),
                                dbc.Tab(
                                    dcc.Graph(id="item-stats-price-elasticity", figure=go.Figure()),
                                    label="Price Sensitivity"
                                ),
                                dbc.Tab(
                                    dcc.Graph(id="item-stats-weather-impact", figure=go.Figure()),
                                    label="Weather Impact"
                                )
                            ], id="item-stats-detailed-tabs")
                        ], id="item-stats-detailed-container", style={'display': 'none'}),
                        
                        # Extended forecast container
                        html.Div([
                            dcc.Graph(
                                id="item-stats-extended-forecast",
                                figure=go.Figure(),
                                style={'height': '500px'}
                            ),
                            html.Div([
                                html.H5("Interactive Visualizations", className="mt-4 mb-3"),
                                html.Div(id="item-stats-interactive-links")
                            ])
                        ], id="item-stats-forecast-container", style={'display': 'none'}),
                        
                        # Error message container
                        html.Div(id="item-stats-error-container")
                    ]
                )
            ])
        ], className="mb-4"),
        
        # Item comparison section
        dbc.Row([
            dbc.Col([
                html.H4("Item Comparison"),
                dbc.RadioItems(
                    id="item-stats-comparison-type",
                    options=[
                        {'label': 'Sales Volume', 'value': 'sales'},
                        {'label': 'Profit', 'value': 'profit'},
                        {'label': 'Sales Variability', 'value': 'variability'},
                        {'label': 'Stock Coverage', 'value': 'stock'}
                    ],
                    value='sales',
                    inline=True,
                    className="mb-3"
                ),
                dcc.Loading(
                    id="item-stats-comparison-loading",
                    type="circle",
                    children=[
                        dcc.Graph(
                            id="item-stats-comparison-chart",
                            figure=go.Figure(),
                            style={'height': '500px'}
                        )
                    ]
                )
            ])
        ])
    ])

def load_item_statistics(file_path=ITEM_STATISTICS_FILE):
    """
    Load item statistics from file
    
    Args:
        file_path: Path to item statistics file
        
    Returns:
        DataFrame: Item statistics
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"Item statistics file not found: {file_path}")
            return None
            
        df = pd.read_csv(file_path)
        
        # Convert string representations of dictionaries back to dictionaries
        for col in ['Weekly_Pattern', 'Monthly_Pattern', 'Weather_Impact', 'Day_Of_Week_Effect']:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
                
        logger.info(f"Loaded item statistics with {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error loading item statistics: {e}")
        return None

def load_extended_forecasts(file_path=EXTENDED_FORECASTS_FILE):
    """
    Load extended forecasts from file
    
    Args:
        file_path: Path to extended forecasts file
        
    Returns:
        DataFrame: Extended forecasts
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"Extended forecasts file not found: {file_path}")
            return None
            
        df = pd.read_csv(file_path)
        
        # Convert date to datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            
        # Make sure Is_Extended column exists (for legacy compatibility)
        if 'Is_Extended' not in df.columns:
            df['Is_Extended'] = False
            
        logger.info(f"Loaded extended forecasts with {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error loading extended forecasts: {e}")
        return None

def format_item_statistics_card(stats):
    """
    Format item statistics into a card layout
    
    Args:
        stats: Dictionary with item statistics
        
    Returns:
        html.Div: Formatted card content
    """
    if stats is None:
        return html.P("No statistics available for this item.")
        
    # Create cards for different stat categories
    
    # Basic information
    basic_info = dbc.Card([
        dbc.CardHeader(html.H5("Basic Information")),
        dbc.CardBody([
            html.P(f"Product: {stats['Product']}", className="mb-1"),
            html.P(f"Data Range: {stats['First_Date']} to {stats['Last_Date']}", className="mb-1"),
            html.P(f"Data Points: {stats['Days_Of_Data']} days", className="mb-1")
        ])
    ], className="mb-3")
    
    # Sales statistics
    sales_stats = dbc.Card([
        dbc.CardHeader(html.H5("Sales Statistics")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.P(f"Total Sales: {stats['Total_Sales']:.1f} units", className="mb-1"),
                    html.P(f"Daily Average: {stats['Avg_Daily_Sales']:.2f} units", className="mb-1"),
                    html.P(f"Variability (CV): {stats['Sales_CV']:.2f}", className="mb-1"),
                ], width=6),
                dbc.Col([
                    html.P(f"Minimum: {stats['Sales_Min']:.1f} units", className="mb-1"),
                    html.P(f"Maximum: {stats['Sales_Max']:.1f} units", className="mb-1"),
                    html.P(f"Zero Sales Days: {stats['Zero_Sales_Pct']:.1f}%", className="mb-1"),
                ], width=6)
            ])
        ])
    ], className="mb-3")
    
    # Trend and seasonality
    trend_stats = dbc.Card([
        dbc.CardHeader(html.H5("Trend & Seasonality")),
        dbc.CardBody([
            html.P(f"Trend: {stats['Trend_Interpretation']}", className="mb-1"),
            html.P(f"Seasonality: {'Detected' if stats['Seasonality_Detected'] else 'Not detected'}", className="mb-1"),
            html.P(f"Seasonal Period: {stats['Seasonal_Period'] if stats['Seasonality_Detected'] else 'N/A'} days", className="mb-1"),
        ])
    ], className="mb-3")
    
    # Price statistics
    price_stats = dbc.Card([
        dbc.CardHeader(html.H5("Price & Profit")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.P(f"Average Price: ${stats['Avg_Price']:.2f}", className="mb-1"),
                    html.P(f"Price Range: ${stats['Price_Min']:.2f} - ${stats['Price_Max']:.2f}", className="mb-1"),
                ], width=6),
                dbc.Col([
                    html.P(f"Total Profit: ${stats['Total_Profit']:.2f}", className="mb-1"),
                    html.P(f"Profit Margin: {stats['Profit_Margin']*100:.1f}%", className="mb-1"),
                ], width=6)
            ])
        ])
    ], className="mb-3")
    
    # Inventory statistics
    inventory_stats = dbc.Card([
        dbc.CardHeader(html.H5("Inventory")),
        dbc.CardBody([
            html.P(f"Current Stock: {stats['Current_Stock']:.1f} units", className="mb-1"),
            html.P(f"Stock Coverage: {stats['Stock_Coverage_Weeks']:.1f} weeks", className="mb-1"),
            html.P(f"Average Stock: {stats['Avg_Stock']:.1f} units", className="mb-1"),
        ])
    ], className="mb-3")
    
    return html.Div([
        basic_info,
        dbc.Row([
            dbc.Col([sales_stats], width=6),
            dbc.Col([trend_stats], width=6)
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([price_stats], width=6),
            dbc.Col([inventory_stats], width=6)
        ])
    ])

def register_callbacks(app, data_dict):
    """
    Register all callbacks for the item statistics tab
    
    Args:
        app: Dash app
        data_dict: Dictionary with all loaded data
    """
    # Load item statistics and extended forecasts
    item_stats_df = load_item_statistics()
    extended_forecasts_df = load_extended_forecasts()
    
    # Add to data dict if available
    if item_stats_df is not None:
        data_dict['item_stats'] = item_stats_df
    
    if extended_forecasts_df is not None:
        data_dict['extended_forecasts'] = extended_forecasts_df
    
    @app.callback(
        [
            Output("item-stats-store-dropdown", "options"),
            Output("item-stats-store-dropdown", "value")
        ],
        Input("item-stats-view-type", "value")
    )
    def update_store_options(view_type):
        """Update store options based on available data"""
        # Use combined data or item stats depending on what's available
        if 'item_stats' in data_dict and data_dict['item_stats'] is not None:
            df = data_dict['item_stats']
            stores = sorted(df['Store_Id'].unique())
        elif 'combined_data' in data_dict and data_dict['combined_data'] is not None:
            df = data_dict['combined_data']
            stores = sorted(df['Store_Id'].unique())
        else:
            return [], None
            
        # Create options
        options = [{'label': f'Store {s}', 'value': s} for s in stores]
        default_value = stores[0] if len(stores) > 0 else None
        
        return options, default_value
    
    @app.callback(
        [
            Output("item-stats-item-dropdown", "options"),
            Output("item-stats-item-dropdown", "value")
        ],
        [
            Input("item-stats-store-dropdown", "value"),
            Input("show-item-numbers-store", "data")
        ]
    )
    def update_item_options(store_id, show_item_numbers):
        """Update item options based on selected store"""
        if store_id is None:
            return [], None
            
        # Use item stats if available, otherwise combined data
        if 'item_stats' in data_dict and data_dict['item_stats'] is not None:
            df = data_dict['item_stats']
            if len(df) > 0:
                store_items = df[df['Store_Id'] == store_id]
                # Sort by total sales in descending order
                store_items = store_items.sort_values('Total_Sales', ascending=False)
                
                options = [
                    {
                        'label': format_product_name(row['Product'], row['Item'], show_item_numbers),
                        'value': row['Item']
                    }
                    for _, row in store_items.iterrows()
                ]
                
                default_value = store_items['Item'].iloc[0] if len(store_items) > 0 else None
                return options, default_value
                
        # Fall back to combined data
        elif 'combined_data' in data_dict and data_dict['combined_data'] is not None:
            df = data_dict['combined_data']
            store_items = df[df['Store_Id'] == store_id][['Item', 'Product']].drop_duplicates()
            
            options = [
                {
                    'label': format_product_name(row['Product'], row['Item'], show_item_numbers),
                    'value': row['Item']
                }
                for _, row in store_items.iterrows()
            ]
            
            default_value = store_items['Item'].iloc[0] if len(store_items) > 0 else None
            return options, default_value
        
        return [], None
    
    @app.callback(
        [
            Output("item-stats-basic-container", "style"),
            Output("item-stats-detailed-container", "style"),
            Output("item-stats-forecast-container", "style"),
            Output("item-stats-forecast-controls", "style")
        ],
        Input("item-stats-view-type", "value")
    )
    def toggle_view_containers(view_type):
        """Toggle visibility of content containers based on selected view type"""
        basic_style = {'display': 'block'} if view_type == 'basic' else {'display': 'none'}
        detailed_style = {'display': 'block'} if view_type == 'detailed' else {'display': 'none'}
        forecast_style = {'display': 'block'} if view_type == 'forecast' else {'display': 'none'}
        forecast_controls_style = {'display': 'block'} if view_type == 'forecast' else {'display': 'none'}
        
        return basic_style, detailed_style, forecast_style, forecast_controls_style
    
    @app.callback(
        [
            Output("item-stats-title", "children"),
            Output("item-stats-basic-content", "children"),
            Output("item-stats-error-container", "children")
        ],
        [
            Input("item-stats-store-dropdown", "value"),
            Input("item-stats-item-dropdown", "value")
        ]
    )
    def update_basic_stats(store_id, item_id):
        """Update basic statistics card"""
        if store_id is None or item_id is None:
            return "Item Statistics", html.P("Please select a store and item to view statistics."), ""
            
        if 'item_stats' not in data_dict or data_dict['item_stats'] is None:
            return (
                "Item Statistics", 
                html.P("No item statistics data available. Please run item statistics analysis first."),
                create_error_message("Item statistics data not available")
            )
            
        # Get statistics for this item
        df = data_dict['item_stats']
        item_stats = df[(df['Store_Id'] == store_id) & (df['Item'] == item_id)]
        
        if len(item_stats) == 0:
            return (
                "Item Statistics", 
                html.P(f"No statistics available for Store {store_id}, Item {item_id}."),
                create_error_message("No data available for selected item")
            )
            
        # Get item details
        stats = item_stats.iloc[0].to_dict()
        product_name = stats['Product']
        
        # Format statistics card
        card_content = format_item_statistics_card(stats)
        
        return f"Statistics for {product_name}", card_content, ""
    
    @app.callback(
        [
            Output("item-stats-sales-pattern", "figure"),
            Output("item-stats-weekly-pattern", "figure"),
            Output("item-stats-monthly-pattern", "figure"),
            Output("item-stats-price-elasticity", "figure"),
            Output("item-stats-weather-impact", "figure")
        ],
        [
            Input("item-stats-store-dropdown", "value"),
            Input("item-stats-item-dropdown", "value")
        ]
    )
    def update_detailed_stats(store_id, item_id):
        """Update detailed statistics visualizations"""
        if store_id is None or item_id is None:
            # Return empty figures
            empty_fig = go.Figure().update_layout(
                title="Please select a store and item",
                xaxis={'visible': False},
                yaxis={'visible': False}
            )
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
            
        if 'item_stats' not in data_dict or data_dict['item_stats'] is None:
            # Return empty figures with error message
            empty_fig = go.Figure().update_layout(
                title="No item statistics data available",
                xaxis={'visible': False},
                yaxis={'visible': False}
            )
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
            
        # Get statistics for this item
        df = data_dict['item_stats']
        item_stats = df[(df['Store_Id'] == store_id) & (df['Item'] == item_id)]
        
        if len(item_stats) == 0:
            # Return empty figures with no data message
            empty_fig = go.Figure().update_layout(
                title=f"No data available for Store {store_id}, Item {item_id}",
                xaxis={'visible': False},
                yaxis={'visible': False}
            )
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
            
        # Get combined data for this item if available
        combined_data = None
        if 'combined_data' in data_dict and data_dict['combined_data'] is not None:
            combined_data = data_dict['combined_data']
            combined_data = combined_data[(combined_data['Store_Id'] == store_id) & (combined_data['Item'] == item_id)]
        
        # Get item details
        stats = item_stats.iloc[0].to_dict()
        product_name = stats['Product']
        
        # 1. Sales Pattern Figure
        sales_fig = go.Figure()
        
        if combined_data is not None and len(combined_data) > 0:
            # Use combined data for sales pattern
            combined_data = combined_data.sort_values('Date')
            
            # Add sales data
            sales_fig.add_trace(go.Scatter(
                x=combined_data['Date'],
                y=combined_data['Sales'],
                mode='lines',
                name='Daily Sales',
                line=dict(color='blue')
            ))
            
            # Add trend line if available
            if not pd.isna(stats['Trend_Coefficient']):
                x_numeric = np.arange(len(combined_data))
                trend = stats['Trend_Coefficient'] * x_numeric + combined_data['Sales'].mean()
                sales_fig.add_trace(go.Scatter(
                    x=combined_data['Date'],
                    y=trend,
                    mode='lines',
                    name=f"Trend ({stats['Trend_Interpretation']})",
                    line=dict(color='red', dash='dash')
                ))
            
            # Add moving average
            window = 7
            ma = combined_data['Sales'].rolling(window=window, min_periods=1).mean()
            sales_fig.add_trace(go.Scatter(
                x=combined_data['Date'],
                y=ma,
                mode='lines',
                name=f'{window}-Day Moving Average',
                line=dict(color='green')
            ))
        
        sales_fig.update_layout(
            title=f"Sales Pattern for {product_name}",
            xaxis_title="Date",
            yaxis_title="Sales",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # 2. Weekly Pattern Figure with enhanced visualization
        weekly_fig = go.Figure()
        
        if 'Day_Of_Week_Effect' in stats and stats['Day_Of_Week_Effect'] is not None:
            dow_effect = stats['Day_Of_Week_Effect']
            
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_values = []
            colors = []
            
            for i in range(7):
                value = dow_effect.get(i, 1.0)
                day_values.append(value)
                
                # Color bars based on whether they are above or below average
                if value > 1.05:
                    colors.append('rgba(99, 169, 255, 0.7)')  # Blue for above average
                elif value < 0.95:
                    colors.append('rgba(255, 99, 132, 0.7)')  # Red for below average
                else:
                    colors.append('rgba(120, 120, 120, 0.7)')  # Gray for average
            
            # Add bars with improved styling
            weekly_fig.add_trace(go.Bar(
                x=days,
                y=day_values,
                marker_color=colors,
                text=[f"{v:.2f}" for v in day_values],
                textposition='auto',
                hovertemplate='%{x}<br>Sales relative to average: %{y:.2f}<extra></extra>'
            ))
            
            # Add reference line at 1.0 with annotation
            weekly_fig.add_shape(
                type="line",
                x0=-0.5,
                y0=1.0,
                x1=6.5,
                y1=1.0,
                line=dict(color="black", width=2, dash="dash")
            )
            
            weekly_fig.add_annotation(
                x=6.5,
                y=1.0,
                text="Average",
                showarrow=False,
                yshift=10,
                font=dict(size=10)
            )
        
        weekly_fig.update_layout(
            title=f"Weekly Sales Pattern for {product_name}",
            xaxis_title="Day of Week",
            yaxis_title="Relative Sales (1.0 = Average)"
        )
        
        # 3. Monthly Pattern Figure with seasonal highlighting
        monthly_fig = go.Figure()
        
        if 'Monthly_Pattern' in stats and stats['Monthly_Pattern'] is not None:
            monthly_pattern = stats['Monthly_Pattern']
            
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_values = []
            colors = []
            annotations = []
            
            # Find peak and trough months for annotations
            peak_month = 1
            trough_month = 1
            peak_value = 0
            trough_value = 99
            
            for i in range(1, 13):
                value = monthly_pattern.get(i, 1.0)
                month_values.append(value)
                
                # Track peak and trough
                if value > peak_value:
                    peak_value = value
                    peak_month = i - 1  # Convert to 0-indexed
                if value < trough_value:
                    trough_value = value
                    trough_month = i - 1  # Convert to 0-indexed
                
                # Seasonal coloring
                if i in [12, 1, 2]:  # Winter
                    colors.append('rgba(0, 191, 255, 0.7)')  # Light blue
                elif i in [3, 4, 5]:  # Spring
                    colors.append('rgba(50, 205, 50, 0.7)')  # Green
                elif i in [6, 7, 8]:  # Summer
                    colors.append('rgba(255, 165, 0, 0.7)')  # Orange
                else:  # Fall
                    colors.append('rgba(139, 69, 19, 0.7)')  # Brown
            
            # Add bars with seasonal coloring
            monthly_fig.add_trace(go.Bar(
                x=months,
                y=month_values,
                marker_color=colors,
                text=[f"{v:.2f}" for v in month_values],
                textposition='auto',
                hovertemplate='%{x}<br>Sales relative to average: %{y:.2f}<extra></extra>'
            ))
            
            # Add reference line at 1.0
            monthly_fig.add_shape(
                type="line",
                x0=-0.5,
                y0=1.0,
                x1=11.5,
                y1=1.0,
                line=dict(color="black", width=2, dash="dash")
            )
            
            # Add annotations for peak and trough
            monthly_fig.add_annotation(
                x=months[peak_month],
                y=peak_value,
                text="Peak",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-30,
                font=dict(size=10, color="darkgreen")
            )
            
            monthly_fig.add_annotation(
                x=months[trough_month],
                y=trough_value,
                text="Trough",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=30,
                font=dict(size=10, color="darkred")
            )
        
        monthly_fig.update_layout(
            title=f"Monthly Sales Pattern for {product_name}",
            xaxis_title="Month",
            yaxis_title="Relative Sales (1.0 = Average)"
        )
        
        # 4. Price Elasticity Figure
        elasticity_fig = go.Figure()
        
        # Use simple elasticity visualization based on available data
        if combined_data is not None and len(combined_data) > 0:
            # Group by price and calculate average sales
            price_groups = combined_data.groupby('Price')['Sales'].mean().reset_index()
            
            if len(price_groups) > 1:
                elasticity_fig.add_trace(go.Scatter(
                    x=price_groups['Price'],
                    y=price_groups['Sales'],
                    mode='markers',
                    marker=dict(size=10, color='blue')
                ))
                
                # Try to fit a simple elasticity curve
                if len(price_groups) >= 3:
                    try:
                        # Log-log regression for elasticity
                        log_price = np.log(price_groups['Price'])
                        log_sales = np.log(price_groups['Sales'].replace(0, 0.01))  # Avoid log(0)
                        
                        # Simple linear regression
                        coeffs = np.polyfit(log_price, log_sales, 1)
                        elasticity = coeffs[0]
                        
                        # Generate points for the elasticity curve
                        price_range = np.linspace(price_groups['Price'].min() * 0.9, 
                                                 price_groups['Price'].max() * 1.1, 
                                                 100)
                        base_price = price_groups['Price'].median()
                        base_sales = price_groups.loc[price_groups['Price'].idxmin(), 'Sales']
                        
                        # Calculate curve points using elasticity
                        curve_sales = base_sales * (price_range / base_price) ** elasticity
                        
                        elasticity_fig.add_trace(go.Scatter(
                            x=price_range,
                            y=curve_sales,
                            mode='lines',
                            name=f'Elasticity: {elasticity:.2f}',
                            line=dict(color='red')
                        ))
                    except Exception as e:
                        logger.warning(f"Error calculating elasticity curve: {e}")
            
        elasticity_fig.update_layout(
            title=f"Price Sensitivity for {product_name}",
            xaxis_title="Price ($)",
            yaxis_title="Average Sales"
        )
        
        # 5. Weather Impact Figure
        weather_fig = go.Figure()
        
        if 'Weather_Impact' in stats and stats['Weather_Impact'] is not None:
            weather_impact = stats['Weather_Impact']
            
            if isinstance(weather_impact, dict) and len(weather_impact) > 0:
                weather_types = []
                impact_values = []
                colors = []
                
                for weather, impact in weather_impact.items():
                    weather_types.append(weather)
                    impact_values.append(impact)
                    
                    # Color based on impact (green for positive, red for negative)
                    if impact > 1.05:
                        colors.append('green')
                    elif impact < 0.95:
                        colors.append('red')
                    else:
                        colors.append('grey')
                
                weather_fig.add_trace(go.Bar(
                    x=weather_types,
                    y=impact_values,
                    marker_color=colors
                ))
                
                # Add reference line at 1.0
                weather_fig.add_shape(
                    type="line",
                    x0=-0.5,
                    y0=1.0,
                    x1=len(weather_types) - 0.5,
                    y1=1.0,
                    line=dict(color="black", width=2, dash="dash")
                )
        
        weather_fig.update_layout(
            title=f"Weather Impact on Sales for {product_name}",
            xaxis_title="Weather Condition",
            yaxis_title="Relative Sales (1.0 = Normal Weather)"
        )
        
        return sales_fig, weekly_fig, monthly_fig, elasticity_fig, weather_fig
    
    @app.callback(
        [
            Output("item-stats-date-range", "min_date_allowed"),
            Output("item-stats-date-range", "max_date_allowed"),
            Output("item-stats-date-range", "start_date"),
            Output("item-stats-date-range", "end_date")
        ],
        Input("item-stats-view-type", "value")
    )
    def update_date_range(view_type):
        """Update date range picker with available forecast dates"""
        if 'extended_forecasts' not in data_dict or data_dict['extended_forecasts'] is None:
            # Set default dates
            today = datetime.today().date()
            min_date = today - timedelta(days=30)
            max_date = today + timedelta(days=30)
            start_date = today
            end_date = today + timedelta(days=30)
            
            return min_date, max_date, start_date, end_date
            
        # Use extended forecasts for date range
        df = data_dict['extended_forecasts']
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        # Set default range to show both original and extended forecasts
        midpoint = df[~df['Is_Extended']]['Date'].max() if 'Is_Extended' in df.columns else df['Date'].median()
        start_date = (midpoint - timedelta(days=15)).date()
        end_date = (midpoint + timedelta(days=45)).date()
        
        # Ensure start/end are within valid range
        start_date = max(min_date, start_date)
        end_date = min(max_date, end_date)
        
        return min_date, max_date, start_date, end_date
    
    @app.callback(
        Output("item-stats-extended-forecast", "figure"),
        [
            Input("item-stats-store-dropdown", "value"),
            Input("item-stats-item-dropdown", "value"),
            Input("item-stats-date-range", "start_date"),
            Input("item-stats-date-range", "end_date"),
            Input("item-stats-forecast-options", "value")
        ]
    )
    def update_extended_forecast(store_id, item_id, start_date, end_date, options):
        """Update extended forecast visualization"""
        if store_id is None or item_id is None:
            # Return empty figure
            return go.Figure().update_layout(
                title="Please select a store and item",
                xaxis={'visible': False},
                yaxis={'visible': False}
            )
            
        if 'extended_forecasts' not in data_dict or data_dict['extended_forecasts'] is None:
            # Return empty figure with error message
            return go.Figure().update_layout(
                title="No forecast data available",
                xaxis={'visible': False},
                yaxis={'visible': False}
            )
            
        # Get extended forecasts for this item
        df = data_dict['extended_forecasts']
        item_forecasts = df[(df['Store_Id'] == store_id) & (df['Item'] == item_id)]
        
        if len(item_forecasts) == 0:
            # Return empty figure with no data message
            return go.Figure().update_layout(
                title=f"No forecast data available for Store {store_id}, Item {item_id}",
                xaxis={'visible': False},
                yaxis={'visible': False}
            )
            
        # Convert dates to datetime
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
        if end_date is not None:
            end_date = pd.to_datetime(end_date)
            
        # Filter by date range
        if start_date is not None and end_date is not None:
            item_forecasts = item_forecasts[
                (item_forecasts['Date'] >= start_date) &
                (item_forecasts['Date'] <= end_date)
            ]
        
        # Get historical data if available
        historical_data = None
        if 'combined_data' in data_dict and data_dict['combined_data'] is not None:
            historical_data = data_dict['combined_data']
            historical_data = historical_data[
                (historical_data['Store_Id'] == store_id) & 
                (historical_data['Item'] == item_id)
            ]
            
            # Filter by date range
            if start_date is not None:
                historical_data = historical_data[historical_data['Date'] < start_date]
                # Limit to most recent data
                historical_data = historical_data.sort_values('Date').tail(90)
        
        # Get product name
        product_name = item_forecasts['Product'].iloc[0] if len(item_forecasts) > 0 else f"Item {item_id}"
        
        # Create figure
        fig = go.Figure()
        
        # Split into original and extended forecasts
        if 'Is_Extended' in item_forecasts.columns:
            original_forecasts = item_forecasts[~item_forecasts['Is_Extended']]
            extended_forecasts = item_forecasts[item_forecasts['Is_Extended']]
        else:
            original_forecasts = item_forecasts
            extended_forecasts = pd.DataFrame()  # Empty DataFrame
        
        # Add historical data if available
        if historical_data is not None and len(historical_data) > 0:
            fig.add_trace(go.Scatter(
                x=historical_data['Date'],
                y=historical_data['Sales'],
                mode='lines',
                name='Historical Sales',
                line=dict(color='gray')
            ))
        
        # Add original forecast
        if len(original_forecasts) > 0 and 'original' in options:
            fig.add_trace(go.Scatter(
                x=original_forecasts['Date'],
                y=original_forecasts['Forecast'],
                mode='lines',
                name='Original Forecast',
                line=dict(color='blue')
            ))
            
            # Add confidence intervals
            if 'ci' in options and 'Lower_Bound' in original_forecasts.columns and 'Upper_Bound' in original_forecasts.columns:
                fig.add_trace(go.Scatter(
                    x=original_forecasts['Date'],
                    y=original_forecasts['Upper_Bound'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=original_forecasts['Date'],
                    y=original_forecasts['Lower_Bound'],
                    mode='lines',
                    name='Lower Bound',
                    fill='tonexty',
                    fillcolor='rgba(0, 0, 255, 0.2)',
                    line=dict(width=0),
                    showlegend=False
                ))
        
        # Add extended forecast
        if len(extended_forecasts) > 0 and 'extended' in options:
            fig.add_trace(go.Scatter(
                x=extended_forecasts['Date'],
                y=extended_forecasts['Forecast'],
                mode='lines',
                name='Extended Forecast',
                line=dict(color='red')
            ))
            
            # Add confidence intervals
            if 'ci' in options and 'Lower_Bound' in extended_forecasts.columns and 'Upper_Bound' in extended_forecasts.columns:
                fig.add_trace(go.Scatter(
                    x=extended_forecasts['Date'],
                    y=extended_forecasts['Upper_Bound'],
                    mode='lines',
                    name='Extended Upper Bound',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=extended_forecasts['Date'],
                    y=extended_forecasts['Lower_Bound'],
                    mode='lines',
                    name='Extended Lower Bound',
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    line=dict(width=0),
                    showlegend=False
                ))
        
        # Add transition point between original and extended
        if len(original_forecasts) > 0 and len(extended_forecasts) > 0 and 'original' in options and 'extended' in options:
            transition_date = original_forecasts['Date'].max()
            
            fig.add_vline(
                x=transition_date,
                line_dash="dash",
                line_color="black",
                opacity=0.5,
                annotation_text="Forecast Extension Start",
                annotation_position="top right"
            )
        
        # Update layout
        fig.update_layout(
            title=f"Extended Forecast for {product_name} (Store {store_id})",
            xaxis_title="Date",
            yaxis_title="Sales",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    @app.callback(
        Output("item-stats-interactive-links", "children"),
        [
            Input("item-stats-store-dropdown", "value"),
            Input("item-stats-item-dropdown", "value")
        ]
    )
    def update_interactive_links(store_id, item_id):
        """Update links to interactive Plotly visualizations"""
        if store_id is None or item_id is None:
            return html.P("Select a store and item to view interactive visualizations.")
            
        # Check if we have interactive visualizations available
        plotly_dir = os.path.join(STATIC_DIR, 'plotly_visualizations')
        if not os.path.exists(plotly_dir):
            return html.P("No interactive visualizations available.")
        
        # Check for available visualizations for this item
        links = []
        
        # Forecast visualization
        forecast_file = os.path.join(plotly_dir, f"forecast_{store_id}_{item_id}.html")
        if os.path.exists(forecast_file):
            rel_path = os.path.relpath(forecast_file, STATIC_DIR)
            links.append(html.Div([
                html.A(
                    "Interactive Sales Forecast", 
                    href=f"/static/{rel_path}",
                    target="_blank",
                    className="btn btn-primary btn-sm m-1"
                ),
                html.Span(" View detailed interactive forecast with confidence intervals")
            ], className="mb-2"))
        
        # Price sensitivity visualization
        sensitivity_file = os.path.join(plotly_dir, f"price_sensitivity_{store_id}_{item_id}.html")
        if os.path.exists(sensitivity_file):
            rel_path = os.path.relpath(sensitivity_file, STATIC_DIR)
            links.append(html.Div([
                html.A(
                    "Price Sensitivity Curve", 
                    href=f"/static/{rel_path}",
                    target="_blank",
                    className="btn btn-info btn-sm m-1"
                ),
                html.Span(" Explore price-demand relationship and revenue optimization")
            ], className="mb-2"))
        
        # Total sales forecast
        total_forecast = os.path.join(plotly_dir, "total_forecast.html")
        if os.path.exists(total_forecast):
            rel_path = os.path.relpath(total_forecast, STATIC_DIR)
            links.append(html.Div([
                html.A(
                    "Total Sales Forecast", 
                    href=f"/static/{rel_path}",
                    target="_blank", 
                    className="btn btn-success btn-sm m-1"
                ),
                html.Span(" View aggregated sales forecast across all products")
            ], className="mb-2"))
        
        if not links:
            return html.P("No interactive visualizations available for this item.")
        else:
            return html.Div([
                html.P("Click the links below to open interactive visualizations in a new tab:"),
                html.Div(links)
            ])
    
    @app.callback(
        Output("item-stats-comparison-chart", "figure"),
        Input("item-stats-comparison-type", "value")
    )
    def update_comparison_chart(comparison_type):
        """Update item comparison chart"""
        if 'item_stats' not in data_dict or data_dict['item_stats'] is None:
            # Return empty figure with error message
            return go.Figure().update_layout(
                title="No item statistics data available",
                xaxis={'visible': False},
                yaxis={'visible': False}
            )
            
        # Get top items based on comparison type
        df = data_dict['item_stats']
        
        if comparison_type == 'sales':
            df = df.sort_values('Total_Sales', ascending=False).head(15)
            x_col = 'Total_Sales'
            title = 'Top Items by Total Sales'
            xaxis_title = 'Total Sales (Units)'
        elif comparison_type == 'profit':
            df = df.sort_values('Total_Profit', ascending=False).head(15)
            x_col = 'Total_Profit'
            title = 'Top Items by Total Profit'
            xaxis_title = 'Total Profit ($)'
        elif comparison_type == 'variability':
            df = df.sort_values('Sales_CV', ascending=False).head(15)
            x_col = 'Sales_CV'
            title = 'Top Items by Sales Variability'
            xaxis_title = 'Coefficient of Variation'
        elif comparison_type == 'stock':
            df = df.sort_values('Stock_Coverage_Weeks').head(15)
            x_col = 'Stock_Coverage_Weeks'
            title = 'Items with Lowest Stock Coverage'
            xaxis_title = 'Weeks of Stock'
        else:
            # Default to sales
            df = df.sort_values('Total_Sales', ascending=False).head(15)
            x_col = 'Total_Sales'
            title = 'Top Items by Total Sales'
            xaxis_title = 'Total Sales (Units)'
        
        # Create enhanced bar chart with more interactive features
        fig = px.bar(
            df,
            x=x_col,
            y='Product',
            color='Store_Id',
            orientation='h',
            text=x_col,
            title=title,
            hover_data=['Total_Sales', 'Avg_Price', 'Sales_CV'],  # Add more data for hover
            color_continuous_scale=px.colors.sequential.Blues if comparison_type != 'variability' else px.colors.sequential.Reds,
            height=600
        )
        
        # Update layout with more sophisticated styling
        fig.update_layout(
            xaxis_title=xaxis_title,
            yaxis_title='Product',
            yaxis={'categoryorder':'total ascending'},
            legend_title_text='Store',
            template="plotly_white",
            margin=dict(l=50, r=20, t=80, b=20),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
            # Add annotations for insights
            annotations=[
                dict(
                    x=0.5,
                    y=-0.15,
                    xref="paper",
                    yref="paper",
                    text=f"Top products by {comparison_type} shown. Click legend items to filter.",
                    showarrow=False,
                    font=dict(size=12)
                )
            ]
        )
        
        # Update text position and formatting
        fig.update_traces(
            texttemplate='%{text:.2s}', 
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>' +
                         f'{xaxis_title}: %{{x:.2f}}<br>' +
                         'Sales: %{customdata[0]:.1f}<br>' +
                         'Price: $%{customdata[1]:.2f}<br>' +
                         'Variability: %{customdata[2]:.2f}<extra></extra>'
        )
        
        return fig