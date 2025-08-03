"""
Business impact UI module for the Pizza Predictive Ordering System.
This module provides visualization and analysis of profit, loss, and revenue metrics.
"""
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core UI components
from ui.core import (
    load_dashboard_data, create_app, format_product_name,
    create_date_range_slider, create_store_product_selectors,
    create_toggle_switch, create_error_message, create_info_card
)

# Setup logging
logger = logging.getLogger('business_impact')


def create_profit_tab_content(data_dict):
    """
    Create content for the profit analysis tab.
    
    Args:
        data_dict: Dictionary with all loaded data
        
    Returns:
        html.Div: Tab content
    """
    # Extract relevant data
    combined_data = data_dict.get('combined_data')
    price_elasticities = data_dict.get('price_elasticities')
    price_recommendations = data_dict.get('price_recommendations')
    profit_impact = data_dict.get('profit_impact')
    
    if combined_data is None:
        return create_error_message("Error: No historical data available")
        
    if profit_impact is None:
        return create_error_message("Error: No profit impact data available")
    
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
    
    # Calculate summary metrics
    if profit_impact is not None:
        total_profit_impact = profit_impact['Total_Profit_Difference'].sum()
        avg_profit_change_pct = profit_impact['Profit_Change_Pct'].mean()
        positive_impact_count = len(profit_impact[profit_impact['Total_Profit_Difference'] > 0])
        negative_impact_count = len(profit_impact[profit_impact['Total_Profit_Difference'] < 0])
    else:
        total_profit_impact = 0
        avg_profit_change_pct = 0
        positive_impact_count = 0
        negative_impact_count = 0
    
    # Create the profit tab content
    return html.Div([
        # Controls
        dbc.Row([
            dbc.Col([
                # Store selector
                html.Label("Select Store:"),
                dcc.Dropdown(
                    id="profit-store-dropdown",
                    options=store_options,
                    value=store_options[0]['value'] if store_options else None,
                    clearable=False
                )
            ], width=6),
            dbc.Col([
                # Time range selector
                html.Label("Time Range:"),
                dcc.Dropdown(
                    id="profit-time-range",
                    options=[
                        {'label': 'Last 7 days', 'value': '7d'},
                        {'label': 'Last 30 days', 'value': '30d'},
                        {'label': 'Last 90 days', 'value': '90d'},
                        {'label': 'Year to date', 'value': 'ytd'},
                        {'label': 'All time', 'value': 'all'}
                    ],
                    value='30d',
                    clearable=False
                )
            ], width=6)
        ], className="mb-4"),
        
        # KPI Cards
        dbc.Row([
            dbc.Col([
                create_info_card(
                    "Total Profit Impact",
                    f"${total_profit_impact:.2f}",
                    color="success" if total_profit_impact > 0 else "danger",
                    icon="chart-line",
                    subtitle=f"{avg_profit_change_pct:.1f}% Average Change"
                )
            ], width=3),
            dbc.Col([
                create_info_card(
                    "Products with Positive Impact",
                    f"{positive_impact_count}",
                    color="success",
                    icon="plus-circle",
                    subtitle=f"{positive_impact_count / (positive_impact_count + negative_impact_count) * 100:.1f}% of Products"
                )
            ], width=3),
            dbc.Col([
                create_info_card(
                    "Products with Negative Impact",
                    f"{negative_impact_count}",
                    color="warning",
                    icon="minus-circle",
                    subtitle=f"{negative_impact_count / (positive_impact_count + negative_impact_count) * 100:.1f}% of Products"
                )
            ], width=3),
            dbc.Col([
                create_info_card(
                    "Highest Profit Item",
                    profit_impact['Product'].iloc[0] if len(profit_impact) > 0 else "N/A",
                    color="info",
                    icon="trophy",
                    subtitle=f"${profit_impact['Total_Profit_Difference'].iloc[0]:.2f}" if len(profit_impact) > 0 else "N/A"
                )
            ], width=3)
        ], className="mb-4"),
        
        # Profit Impact Waterfall Chart
        dbc.Row([
            dbc.Col([
                html.H4("Profit Impact by Product"),
                dcc.Loading(
                    id="loading-profit-waterfall",
                    type="circle",
                    children=[
                        dcc.Graph(
                            id="profit-waterfall-chart",
                            figure=go.Figure(),
                            style={'height': '50vh'}
                        )
                    ]
                )
            ], width=12)
        ], className="mb-4"),
        
        # Profit Trend Chart
        dbc.Row([
            dbc.Col([
                html.H4("Profit Trend Analysis"),
                dcc.Loading(
                    id="loading-profit-trend",
                    type="circle",
                    children=[
                        dcc.Graph(
                            id="profit-trend-chart",
                            figure=go.Figure(),
                            style={'height': '40vh'}
                        )
                    ]
                )
            ], width=12)
        ], className="mb-4")
    ])


def create_revenue_tab_content(data_dict):
    """
    Create content for the revenue analysis tab.
    
    Args:
        data_dict: Dictionary with all loaded data
        
    Returns:
        html.Div: Tab content
    """
    # Extract relevant data
    combined_data = data_dict.get('combined_data')
    price_elasticities = data_dict.get('price_elasticities')
    profit_impact = data_dict.get('profit_impact')
    
    if combined_data is None:
        return create_error_message("Error: No historical data available")
    
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
    
    # Calculate summary metrics if profit_impact is available
    if profit_impact is not None and len(profit_impact) > 0:
        baseline_revenue = profit_impact['Total_Current_Profit'].sum() / 0.25  # Assuming 25% margin
        projected_revenue = profit_impact['Total_New_Profit'].sum() / 0.25  # Assuming 25% margin
        revenue_change = projected_revenue - baseline_revenue
        revenue_change_pct = (revenue_change / baseline_revenue * 100) if baseline_revenue > 0 else 0
    else:
        baseline_revenue = 0
        projected_revenue = 0
        revenue_change = 0
        revenue_change_pct = 0
    
    # Create the revenue tab content
    return html.Div([
        # Controls
        dbc.Row([
            dbc.Col([
                # Store selector
                html.Label("Select Store:"),
                dcc.Dropdown(
                    id="revenue-store-dropdown",
                    options=store_options,
                    value=store_options[0]['value'] if store_options else None,
                    clearable=False
                )
            ], width=6),
            dbc.Col([
                # Time range selector
                html.Label("Time Range:"),
                dcc.Dropdown(
                    id="revenue-time-range",
                    options=[
                        {'label': 'Last 7 days', 'value': '7d'},
                        {'label': 'Last 30 days', 'value': '30d'},
                        {'label': 'Last 90 days', 'value': '90d'},
                        {'label': 'Year to date', 'value': 'ytd'},
                        {'label': 'All time', 'value': 'all'}
                    ],
                    value='30d',
                    clearable=False
                )
            ], width=6)
        ], className="mb-4"),
        
        # KPI Cards
        dbc.Row([
            dbc.Col([
                create_info_card(
                    "Baseline Revenue",
                    f"${baseline_revenue:.2f}",
                    color="info",
                    icon="dollar-sign",
                    subtitle="Current Revenue Estimate"
                )
            ], width=3),
            dbc.Col([
                create_info_card(
                    "Projected Revenue",
                    f"${projected_revenue:.2f}",
                    color="success" if projected_revenue > baseline_revenue else "danger",
                    icon="chart-line",
                    subtitle=f"{revenue_change_pct:.1f}% Change"
                )
            ], width=3),
            dbc.Col([
                create_info_card(
                    "Revenue Change",
                    f"${revenue_change:.2f}",
                    color="success" if revenue_change > 0 else "danger",
                    icon="exchange-alt",
                    subtitle="Projected Impact"
                )
            ], width=3),
            dbc.Col([
                create_info_card(
                    "Top Revenue Product",
                    combined_data.groupby('Product')['Sales'].sum().idxmax() if len(combined_data) > 0 else "N/A",
                    color="info",
                    icon="trophy",
                    subtitle="By Total Revenue"
                )
            ], width=3)
        ], className="mb-4"),
        
        # Revenue Breakdown Chart
        dbc.Row([
            dbc.Col([
                html.H4("Revenue Breakdown by Product Category"),
                dcc.Loading(
                    id="loading-revenue-breakdown",
                    type="circle",
                    children=[
                        dcc.Graph(
                            id="revenue-breakdown-chart",
                            figure=go.Figure(),
                            style={'height': '50vh'}
                        )
                    ]
                )
            ], width=12)
        ], className="mb-4"),
        
        # Revenue Trend Chart
        dbc.Row([
            dbc.Col([
                html.H4("Revenue Trend Analysis"),
                dcc.Loading(
                    id="loading-revenue-trend",
                    type="circle",
                    children=[
                        dcc.Graph(
                            id="revenue-trend-chart",
                            figure=go.Figure(),
                            style={'height': '40vh'}
                        )
                    ]
                )
            ], width=12)
        ], className="mb-4")
    ])


def create_loss_prevention_tab_content(data_dict):
    """
    Create content for the loss prevention tab.
    
    Args:
        data_dict: Dictionary with all loaded data
        
    Returns:
        html.Div: Tab content
    """
    # Extract relevant data
    combined_data = data_dict.get('combined_data')
    inventory_projection = data_dict.get('inventory_projection')
    
    if combined_data is None:
        return create_error_message("Error: No historical data available")
    
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
    
    # Calculate estimated losses if inventory projection is available
    estimated_waste = 0
    stockout_cost = 0
    
    if inventory_projection is not None and len(inventory_projection) > 0:
        # Calculate waste as expired inventory (oversupply)
        if 'Expired_Units' in inventory_projection.columns:
            expired_units = inventory_projection['Expired_Units'].sum()
            avg_cost = combined_data['Cost'].mean() if 'Cost' in combined_data.columns else 5.0
            estimated_waste = expired_units * avg_cost
        
        # Calculate stockout cost (lost sales due to understocking)
        if 'Lost_Sales' in inventory_projection.columns:
            lost_sales = inventory_projection['Lost_Sales'].sum()
            avg_price = combined_data['Price'].mean() if 'Price' in combined_data.columns else 7.0
            avg_cost = combined_data['Cost'].mean() if 'Cost' in combined_data.columns else 5.0
            avg_margin = avg_price - avg_cost
            stockout_cost = lost_sales * avg_margin
    
    # Create the loss prevention tab content
    return html.Div([
        # Controls
        dbc.Row([
            dbc.Col([
                # Store selector
                html.Label("Select Store:"),
                dcc.Dropdown(
                    id="loss-store-dropdown",
                    options=store_options,
                    value=store_options[0]['value'] if store_options else None,
                    clearable=False
                )
            ], width=6),
            dbc.Col([
                # Time range selector
                html.Label("Time Range:"),
                dcc.Dropdown(
                    id="loss-time-range",
                    options=[
                        {'label': 'Last 7 days', 'value': '7d'},
                        {'label': 'Last 30 days', 'value': '30d'},
                        {'label': 'Last 90 days', 'value': '90d'},
                        {'label': 'Year to date', 'value': 'ytd'},
                        {'label': 'All time', 'value': 'all'}
                    ],
                    value='30d',
                    clearable=False
                )
            ], width=6)
        ], className="mb-4"),
        
        # KPI Cards
        dbc.Row([
            dbc.Col([
                create_info_card(
                    "Estimated Waste Cost",
                    f"${estimated_waste:.2f}",
                    color="danger",
                    icon="trash-alt",
                    subtitle="Cost of Expired Products"
                )
            ], width=3),
            dbc.Col([
                create_info_card(
                    "Stockout Cost",
                    f"${stockout_cost:.2f}",
                    color="warning",
                    icon="exclamation-triangle",
                    subtitle="Lost Profit Due to Stockouts"
                )
            ], width=3),
            dbc.Col([
                create_info_card(
                    "Total Loss Impact",
                    f"${estimated_waste + stockout_cost:.2f}",
                    color="danger",
                    icon="dollar-sign",
                    subtitle="Combined Loss Impact"
                )
            ], width=3),
            dbc.Col([
                create_info_card(
                    "Loss Prevention Opportunity",
                    f"${(estimated_waste + stockout_cost) * 0.8:.2f}",
                    color="success",
                    icon="lightbulb",
                    subtitle="Potential Savings with Optimization"
                )
            ], width=3)
        ], className="mb-4"),
        
        # Loss Breakdown Chart
        dbc.Row([
            dbc.Col([
                html.H4("Loss Breakdown Analysis"),
                dcc.Loading(
                    id="loading-loss-breakdown",
                    type="circle",
                    children=[
                        dcc.Graph(
                            id="loss-breakdown-chart",
                            figure=go.Figure(),
                            style={'height': '50vh'}
                        )
                    ]
                )
            ], width=12)
        ], className="mb-4"),
        
        # Loss Prevention Opportunities
        dbc.Row([
            dbc.Col([
                html.H4("Loss Prevention Opportunities"),
                dcc.Loading(
                    id="loading-loss-prevention",
                    type="circle",
                    children=[
                        dcc.Graph(
                            id="loss-prevention-chart",
                            figure=go.Figure(),
                            style={'height': '40vh'}
                        )
                    ]
                )
            ], width=12)
        ], className="mb-4")
    ])


def register_callbacks(app, data_dict):
    """
    Register callbacks for the business impact tabs.
    
    Args:
        app: Dash app instance
        data_dict: Dictionary with all loaded data
    """
    # Profit Waterfall Chart Callback
    @app.callback(
        Output("profit-waterfall-chart", "figure"),
        [
            Input("profit-store-dropdown", "value"),
            Input("profit-time-range", "value")
        ]
    )
    def update_profit_waterfall(store, time_range):
        """Update the profit impact waterfall chart"""
        if store is None:
            return go.Figure()
        
        profit_impact = data_dict.get('profit_impact')
        if profit_impact is None or len(profit_impact) == 0:
            return go.Figure()
        
        # Filter by store
        store_impact = profit_impact[profit_impact['Store_Id'] == store]
        
        if len(store_impact) == 0:
            return go.Figure()
        
        # Sort by profit impact and get top products
        top_items = store_impact.sort_values('Total_Profit_Difference', ascending=False).head(10)
        
        # Create waterfall chart
        fig = go.Figure()
        
        # Start with baseline profit
        total_current_profit = top_items['Total_Current_Profit'].sum()
        fig.add_trace(go.Bar(
            name="Current Profit",
            y=["Total"],
            x=[total_current_profit],
            marker={"color": "blue"},
            text=f"${total_current_profit:.2f}",
            textposition="outside"
        ))
        
        # Add each product's contribution
        for i, (_, row) in enumerate(top_items.iterrows()):
            product = row['Product']
            profit_diff = row['Total_Profit_Difference']
            
            # Only include significant impacts
            if abs(profit_diff) > total_current_profit * 0.01:
                fig.add_trace(go.Bar(
                    name=product,
                    y=["Total"],
                    x=[profit_diff],
                    marker={"color": "green" if profit_diff > 0 else "red"},
                    text=f"${profit_diff:.2f}",
                    textposition="outside"
                ))
        
        # Add final value
        total_new_profit = top_items['Total_New_Profit'].sum()
        fig.add_trace(go.Bar(
            name="New Profit",
            y=["Total"],
            x=[0],  # This will be calculated using cumulative sum
            marker={"color": "blue"},
            text=f"${total_new_profit:.2f}",
            textposition="outside"
        ))
        
        # Configure layout for waterfall effect
        fig.update_layout(
            title=f"Profit Impact Waterfall for Store {store}",
            barmode="stack",
            xaxis_title="Profit ($)",
            height=500,
            bargap=0.1,
            margin=dict(t=50, b=50, l=50, r=50),
            annotations=[
                dict(
                    x=0.5,
                    y=1.1,
                    text=f"Total Profit Change: ${total_new_profit - total_current_profit:.2f} ({(total_new_profit/total_current_profit - 1)*100:.1f}%)",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    font=dict(size=14, color="green" if total_new_profit > total_current_profit else "red")
                )
            ]
        )
        
        return fig
    
    # Profit Trend Chart Callback
    @app.callback(
        Output("profit-trend-chart", "figure"),
        [
            Input("profit-store-dropdown", "value"),
            Input("profit-time-range", "value")
        ]
    )
    def update_profit_trend_chart(store, time_range):
        """Update the profit trend chart"""
        if store is None:
            return go.Figure()
        
        combined_data = data_dict.get('combined_data')
        if combined_data is None or len(combined_data) == 0:
            return go.Figure()
        
        # Filter by store
        store_data = combined_data[combined_data['Store_Id'] == store].copy()
        
        if len(store_data) == 0:
            return go.Figure()
        
        # Apply time range filter
        end_date = datetime.now()
        if time_range == '7d':
            start_date = end_date - timedelta(days=7)
        elif time_range == '30d':
            start_date = end_date - timedelta(days=30)
        elif time_range == '90d':
            start_date = end_date - timedelta(days=90)
        elif time_range == 'ytd':
            start_date = datetime(end_date.year, 1, 1)
        else:  # 'all'
            start_date = store_data['Date'].min()
        
        # Filter by date range
        store_data = store_data[(store_data['Date'] >= start_date) & (store_data['Date'] <= end_date)]
        
        if len(store_data) == 0:
            return go.Figure()
        
        # Calculate profit (assuming cost column exists, otherwise use 70% margin)
        if 'Cost' in store_data.columns:
            store_data['Profit'] = (store_data['Price'] - store_data['Cost']) * store_data['Sales']
        else:
            store_data['Profit'] = store_data['Price'] * 0.3 * store_data['Sales']
        
        # Group by date for trend analysis
        daily_profit = store_data.groupby('Date')['Profit'].sum().reset_index()
        
        # Create trend figure
        fig = go.Figure()
        
        # Add profit line
        fig.add_trace(go.Scatter(
            x=daily_profit['Date'],
            y=daily_profit['Profit'],
            mode='lines+markers',
            name='Daily Profit',
            line=dict(color='green', width=2)
        ))
        
        # Add trend line (7-day moving average)
        daily_profit['MA7'] = daily_profit['Profit'].rolling(window=7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=daily_profit['Date'],
            y=daily_profit['MA7'],
            mode='lines',
            name='7-Day Moving Average',
            line=dict(color='blue', width=2, dash='dash')
        ))
        
        # Configure layout
        fig.update_layout(
            title=f"Profit Trend Analysis for Store {store}",
            xaxis_title="Date",
            yaxis_title="Profit ($)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    # Revenue Breakdown Chart Callback
    @app.callback(
        Output("revenue-breakdown-chart", "figure"),
        [
            Input("revenue-store-dropdown", "value"),
            Input("revenue-time-range", "value")
        ]
    )
    def update_revenue_breakdown_chart(store, time_range):
        """Update the revenue breakdown chart"""
        if store is None:
            return go.Figure()
        
        combined_data = data_dict.get('combined_data')
        if combined_data is None or len(combined_data) == 0:
            return go.Figure()
        
        # Filter by store
        store_data = combined_data[combined_data['Store_Id'] == store].copy()
        
        if len(store_data) == 0:
            return go.Figure()
        
        # Apply time range filter
        end_date = datetime.now()
        if time_range == '7d':
            start_date = end_date - timedelta(days=7)
        elif time_range == '30d':
            start_date = end_date - timedelta(days=30)
        elif time_range == '90d':
            start_date = end_date - timedelta(days=90)
        elif time_range == 'ytd':
            start_date = datetime(end_date.year, 1, 1)
        else:  # 'all'
            start_date = store_data['Date'].min()
        
        # Filter by date range
        store_data = store_data[(store_data['Date'] >= start_date) & (store_data['Date'] <= end_date)]
        
        if len(store_data) == 0:
            return go.Figure()
        
        # Calculate revenue
        store_data['Revenue'] = store_data['Price'] * store_data['Sales']
        
        # Categorize products into groups
        # This is a simplified example - you might want to create better categorization
        store_data['Product_Category'] = store_data['Product'].apply(lambda x: 
            'Supreme' if 'SUPREME' in str(x).upper() else
            'Pepperoni' if 'PEPPERONI' in str(x).upper() else
            'Cheese' if 'CHEESE' in str(x).upper() else
            'Specialty' if any(kw in str(x).upper() for kw in ['BBQ', 'HAWAIIAN', 'CHICKEN']) else
            'Other'
        )
        
        # Group by category
        category_revenue = store_data.groupby('Product_Category')['Revenue'].sum().reset_index()
        
        # Create donut chart
        fig = go.Figure(data=[go.Pie(
            labels=category_revenue['Product_Category'],
            values=category_revenue['Revenue'],
            hole=.4,
            textinfo='label+percent',
            marker=dict(colors=px.colors.qualitative.Set2)
        )])
        
        # Configure layout
        fig.update_layout(
            title=f"Revenue Breakdown by Product Category for Store {store}",
            annotations=[dict(
                text=f'${category_revenue["Revenue"].sum():.2f}',
                x=0.5, y=0.5,
                font_size=20,
                showarrow=False
            )]
        )
        
        return fig
    
    # Revenue Trend Chart Callback
    @app.callback(
        Output("revenue-trend-chart", "figure"),
        [
            Input("revenue-store-dropdown", "value"),
            Input("revenue-time-range", "value")
        ]
    )
    def update_revenue_trend_chart(store, time_range):
        """Update the revenue trend chart"""
        if store is None:
            return go.Figure()
        
        combined_data = data_dict.get('combined_data')
        if combined_data is None or len(combined_data) == 0:
            return go.Figure()
        
        # Filter by store
        store_data = combined_data[combined_data['Store_Id'] == store].copy()
        
        if len(store_data) == 0:
            return go.Figure()
        
        # Apply time range filter
        end_date = datetime.now()
        if time_range == '7d':
            start_date = end_date - timedelta(days=7)
        elif time_range == '30d':
            start_date = end_date - timedelta(days=30)
        elif time_range == '90d':
            start_date = end_date - timedelta(days=90)
        elif time_range == 'ytd':
            start_date = datetime(end_date.year, 1, 1)
        else:  # 'all'
            start_date = store_data['Date'].min()
        
        # Filter by date range
        store_data = store_data[(store_data['Date'] >= start_date) & (store_data['Date'] <= end_date)]
        
        if len(store_data) == 0:
            return go.Figure()
        
        # Calculate revenue
        store_data['Revenue'] = store_data['Price'] * store_data['Sales']
        
        # Group by date for trend analysis
        daily_revenue = store_data.groupby('Date')['Revenue'].sum().reset_index()
        
        # Create trend figure
        fig = go.Figure()
        
        # Add revenue line
        fig.add_trace(go.Scatter(
            x=daily_revenue['Date'],
            y=daily_revenue['Revenue'],
            mode='lines+markers',
            name='Daily Revenue',
            line=dict(color='blue', width=2)
        ))
        
        # Add trend line (7-day moving average)
        daily_revenue['MA7'] = daily_revenue['Revenue'].rolling(window=7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=daily_revenue['Date'],
            y=daily_revenue['MA7'],
            mode='lines',
            name='7-Day Moving Average',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        # Configure layout
        fig.update_layout(
            title=f"Revenue Trend Analysis for Store {store}",
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    # Loss Breakdown Chart Callback
    @app.callback(
        Output("loss-breakdown-chart", "figure"),
        [
            Input("loss-store-dropdown", "value"),
            Input("loss-time-range", "value")
        ]
    )
    def update_loss_breakdown_chart(store, time_range):
        """Update the loss breakdown chart"""
        if store is None:
            return go.Figure()
        
        combined_data = data_dict.get('combined_data')
        inventory_projection = data_dict.get('inventory_projection')
        
        if combined_data is None or len(combined_data) == 0:
            return go.Figure()
        
        # Create default figure with simulated data if inventory_projection is not available
        if inventory_projection is None or len(inventory_projection) == 0:
            # Create simulated loss data for visualization purposes
            loss_types = ['Expired Products', 'Stockouts', 'Pricing Errors', 'Ordering Inefficiencies']
            loss_values = [12500, 8700, 4300, 3200]
            
            # Create the bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=loss_types,
                    y=loss_values,
                    marker_color=['crimson', 'darkorange', 'gold', 'lightcoral'],
                    text=[f"${x:,.2f}" for x in loss_values],
                    textposition='auto'
                )
            ])
            
            # Configure layout
            fig.update_layout(
                title=f"Loss Breakdown Analysis for Store {store} (Simulated Data)",
                xaxis_title="Loss Category",
                yaxis_title="Loss Amount ($)",
                yaxis=dict(
                    rangemode='nonnegative'
                )
            )
            
            return fig
        
        # Filter by store
        store_data = combined_data[combined_data['Store_Id'] == store].copy()
        
        if len(store_data) == 0:
            return go.Figure()
        
        # Apply time range filter
        end_date = datetime.now()
        if time_range == '7d':
            start_date = end_date - timedelta(days=7)
        elif time_range == '30d':
            start_date = end_date - timedelta(days=30)
        elif time_range == '90d':
            start_date = end_date - timedelta(days=90)
        elif time_range == 'ytd':
            start_date = datetime(end_date.year, 1, 1)
        else:  # 'all'
            start_date = store_data['Date'].min()
        
        # Filter by date range
        store_data = store_data[(store_data['Date'] >= start_date) & (store_data['Date'] <= end_date)]
        
        if len(store_data) == 0:
            return go.Figure()
        
        # Calculate approximate losses (simulated for demonstration)
        avg_cost = store_data['Cost'].mean() if 'Cost' in store_data.columns else 5.0
        avg_price = store_data['Price'].mean() if 'Price' in store_data.columns else 7.0
        avg_margin = avg_price - avg_cost
        
        # Calculate stock-based metrics (if available)
        if 'Stock_Level' in store_data.columns:
            # Expired products (assumption: 5% of stock expires)
            expired_units = store_data['Stock_Level'].sum() * 0.05
            expired_cost = expired_units * avg_cost
            
            # Stockouts (assumption: 10% lost sales when stock is low)
            potential_stockouts = store_data[store_data['Stock_Level'] < 5]['Sales'].sum() * 0.1
            stockout_cost = potential_stockouts * avg_margin
            
            # Price errors (assumption: 2% of transactions have price errors)
            price_error_cost = store_data['Revenue'].sum() * 0.02
            
            # Ordering inefficiencies (assumption: 3% of stock is inefficient)
            ordering_cost = store_data['Stock_Level'].sum() * 0.03 * avg_cost
        else:
            # Fallback to simulated data if stock information is not available
            total_revenue = (store_data['Price'] * store_data['Sales']).sum()
            expired_cost = total_revenue * 0.04
            stockout_cost = total_revenue * 0.035
            price_error_cost = total_revenue * 0.02
            ordering_cost = total_revenue * 0.015
        
        # Create loss breakdown
        loss_types = ['Expired Products', 'Stockouts', 'Pricing Errors', 'Ordering Inefficiencies']
        loss_values = [expired_cost, stockout_cost, price_error_cost, ordering_cost]
        
        # Create the bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=loss_types,
                y=loss_values,
                marker_color=['crimson', 'darkorange', 'gold', 'lightcoral'],
                text=[f"${x:,.2f}" for x in loss_values],
                textposition='auto'
            )
        ])
        
        # Configure layout
        fig.update_layout(
            title=f"Loss Breakdown Analysis for Store {store}",
            xaxis_title="Loss Category",
            yaxis_title="Loss Amount ($)",
            yaxis=dict(
                rangemode='nonnegative'
            )
        )
        
        return fig
    
    # Loss Prevention Chart Callback
    @app.callback(
        Output("loss-prevention-chart", "figure"),
        [
            Input("loss-store-dropdown", "value"),
            Input("loss-time-range", "value")
        ]
    )
    def update_loss_prevention_chart(store, time_range):
        """Update the loss prevention opportunities chart"""
        if store is None:
            return go.Figure()
        
        combined_data = data_dict.get('combined_data')
        
        if combined_data is None or len(combined_data) == 0:
            return go.Figure()
        
        # Filter by store
        store_data = combined_data[combined_data['Store_Id'] == store].copy()
        
        if len(store_data) == 0:
            return go.Figure()
        
        # Apply time range filter
        end_date = datetime.now()
        if time_range == '7d':
            start_date = end_date - timedelta(days=7)
        elif time_range == '30d':
            start_date = end_date - timedelta(days=30)
        elif time_range == '90d':
            start_date = end_date - timedelta(days=90)
        elif time_range == 'ytd':
            start_date = datetime(end_date.year, 1, 1)
        else:  # 'all'
            start_date = store_data['Date'].min()
        
        # Filter by date range
        store_data = store_data[(store_data['Date'] >= start_date) & (store_data['Date'] <= end_date)]
        
        if len(store_data) == 0:
            return go.Figure()
        
        # Create simulated loss prevention opportunities
        # This is for demonstration purposes
        savings_categories = ['Improved Forecasting', 'Optimized Pricing', 'Just-in-time Ordering', 'Better Stock Management']
        current_losses = [15000, 12000, 9000, 8000]
        potential_savings = [12000, 9000, 6500, 6000]
        
        # Create the grouped bar chart
        fig = go.Figure(data=[
            go.Bar(
                name='Current Losses',
                x=savings_categories,
                y=current_losses,
                marker_color='firebrick',
                text=[f"${x:,.2f}" for x in current_losses],
                textposition='auto'
            ),
            go.Bar(
                name='Potential Savings',
                x=savings_categories,
                y=potential_savings,
                marker_color='forestgreen',
                text=[f"${x:,.2f}" for x in potential_savings],
                textposition='auto'
            )
        ])
        
        # Calculate total potential savings
        total_savings = sum(potential_savings)
        total_losses = sum(current_losses)
        savings_percentage = (total_savings / total_losses) * 100 if total_losses > 0 else 0
        
        # Configure layout
        fig.update_layout(
            title=f"Loss Prevention Opportunities for Store {store}",
            xaxis_title="Opportunity Categories",
            yaxis_title="Amount ($)",
            barmode='group',
            annotations=[
                dict(
                    x=0.5,
                    y=1.1,
                    text=f"Total Potential Savings: ${total_savings:,.2f} ({savings_percentage:.1f}% of Current Losses)",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    font=dict(size=14, color="green")
                )
            ]
        )
        
        return fig