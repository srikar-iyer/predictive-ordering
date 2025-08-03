"""
Enhanced integrated view module for connecting inventory, pricing, and demand forecasting.

This module provides a unified interface that shows the connections between
inventory, pricing, and demand forecasting components and how they affect each other.
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

# Import the unified data model and visualizations
from src.models.unified_data_model import UnifiedDataModel
from src.models.integrated_visualizations import (
    create_integrated_chart, create_impact_heatmap,
    create_kpi_indicators, create_recommendations_table
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('enhanced_integrated_view')

# Import settings if available
try:
    from config.settings import (
        MIN_STOCK_WEEKS, TARGET_STOCK_WEEKS, MAX_STOCK_WEEKS,
        MAX_PRICE_INCREASE, MAX_PRICE_DECREASE
    )
except ImportError:
    # Default values
    MIN_STOCK_WEEKS = 1
    TARGET_STOCK_WEEKS = 2
    MAX_STOCK_WEEKS = 3
    MAX_PRICE_INCREASE = 15
    MAX_PRICE_DECREASE = 10

def create_enhanced_integrated_view_content(data_dict):
    """
    Create content for the enhanced integrated view tab.
    
    Args:
        data_dict: Dictionary with all loaded data
        
    Returns:
        html.Div: Tab content
    """
    # Check if we have required data
    required_data = ['combined_data', 'forecasts', 'price_elasticities', 'inventory_projection']
    missing_data = [key for key in required_data if key not in data_dict or data_dict[key] is None]
    
    if missing_data:
        missing_str = ", ".join(missing_data)
        return create_error_message(f"Missing required data: {missing_str}")
    
    # Get store and product options
    store_options = []
    product_options = []
    
    if data_dict['combined_data'] is not None:
        # Get unique stores
        stores = data_dict['combined_data']['Store_Id'].unique()
        store_options = [{'label': f'Store {s}', 'value': s} for s in sorted(stores)]
        
        # Get unique products with names
        products = data_dict['combined_data'][['Item', 'Product']].drop_duplicates()
        
        # Format product names
        product_options = [
            {
                'label': f"{row['Product']} ({row['Item']})",
                'value': row['Item']
            } for _, row in products.iterrows()
        ]
    
    # Create the enhanced integrated view content
    return html.Div([
        # Introduction section
        dbc.Card([
            dbc.CardHeader(
                html.H4([
                    html.I(className="fas fa-project-diagram mr-2"),
                    "Integrated Business Analytics"
                ])
            ),
            dbc.CardBody([
                html.P([
                    "This view shows how ",
                    html.Strong("pricing"), ", ",
                    html.Strong("inventory"), ", and ",
                    html.Strong("demand forecasting"),
                    " are connected and how changes in one area affect others."
                ], className="lead"),
                html.P([
                    "Use the controls below to explore different scenarios and see the impact on your business metrics."
                ])
            ])
        ], className="mb-4"),
        
        # Controls section
        dbc.Card([
            dbc.CardHeader(
                html.H5([
                    html.I(className="fas fa-sliders-h mr-2"),
                    "Scenario Controls"
                ])
            ),
            dbc.CardBody([
                # Store and product selectors
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Store:"),
                        dcc.Dropdown(
                            id="integrated-store-dropdown",
                            options=store_options,
                            value=store_options[0]['value'] if store_options else None,
                            clearable=False,
                            className="mb-3"
                        )
                    ], width=12, md=6),
                    dbc.Col([
                        html.Label("Select Product:"),
                        dcc.Dropdown(
                            id="integrated-product-dropdown",
                            options=product_options,
                            value=product_options[0]['value'] if product_options else None,
                            clearable=False,
                            className="mb-3"
                        )
                    ], width=12, md=6)
                ], className="mb-3"),
                
                # Scenario adjustment sliders
                dbc.Row([
                    dbc.Col([
                        html.Label("Price Adjustment (%)"),
                        dcc.Slider(
                            id="price-adjustment-slider",
                            min=-MAX_PRICE_DECREASE,
                            max=MAX_PRICE_INCREASE,
                            step=1,
                            value=0,
                            marks={
                                -MAX_PRICE_DECREASE: f'-{MAX_PRICE_DECREASE}%',
                                -MAX_PRICE_DECREASE//2: f'-{MAX_PRICE_DECREASE//2}%',
                                0: '0%',
                                MAX_PRICE_INCREASE//2: f'+{MAX_PRICE_INCREASE//2}%',
                                MAX_PRICE_INCREASE: f'+{MAX_PRICE_INCREASE}%'
                            },
                            className="mb-3"
                        )
                    ], width=12, md=6),
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
                                25: '+25%',
                                50: '+50%',
                                75: '+75%',
                                100: '+100%'
                            },
                            className="mb-3"
                        )
                    ], width=12, md=6)
                ]),
                
                # Forecast period slider
                dbc.Row([
                    dbc.Col([
                        html.Label("Forecast Period (Days)"),
                        dcc.RangeSlider(
                            id="forecast-period-slider",
                            min=1,
                            max=30,
                            step=1,
                            value=[1, 14],
                            marks={
                                1: '1',
                                7: '7',
                                14: '14',
                                21: '21',
                                30: '30'
                            },
                            className="mb-3"
                        )
                    ], width=12)
                ]),
                
                # Analysis view options
                dbc.Row([
                    dbc.Col([
                        html.Label("Analysis View"),
                        dbc.RadioItems(
                            id="analysis-view-radio",
                            options=[
                                {"label": "Integrated Chart", "value": "chart"},
                                {"label": "Impact Heatmap", "value": "heatmap"},
                                {"label": "KPI Dashboard", "value": "kpi"}
                            ],
                            value="chart",
                            inline=True,
                            className="mb-3"
                        )
                    ], width=12, md=8),
                    dbc.Col([
                        dbc.Button(
                            [html.I(className="fas fa-sync-alt mr-2"), "Reset Scenario"],
                            id="reset-scenario-button",
                            color="secondary",
                            className="float-right"
                        )
                    ], width=12, md=4)
                ])
            ])
        ], className="mb-4"),
        
        # Key metrics section
        dbc.Card([
            dbc.CardHeader(
                html.H5([
                    html.I(className="fas fa-chart-line mr-2"),
                    "Key Performance Indicators"
                ])
            ),
            dbc.CardBody([
                dbc.Row(id="integrated-metrics-row")
            ])
        ], className="mb-4"),
        
        # Main visualization section
        dbc.Card([
            dbc.CardHeader(
                html.H5([
                    html.I(className="fas fa-chart-area mr-2"),
                    "Business Impact Visualization",
                    html.Small(id="viz-subtitle", className="text-muted ml-2")
                ])
            ),
            dbc.CardBody([
                dcc.Loading(
                    id="loading-integrated-visualization",
                    type="circle",
                    children=[
                        html.Div(id="integrated-visualization")
                    ]
                )
            ])
        ], className="mb-4"),
        
        # Recommendations section
        dbc.Card([
            dbc.CardHeader(
                html.H5([
                    html.I(className="fas fa-lightbulb mr-2"),
                    "Business Recommendations"
                ])
            ),
            dbc.CardBody([
                dcc.Loading(
                    id="loading-recommendations",
                    type="circle",
                    children=[
                        html.Div(id="integrated-recommendations")
                    ]
                )
            ])
        ])
    ])

def register_enhanced_integrated_callbacks(app, data_dict):
    """
    Register callbacks for the enhanced integrated view.
    
    Args:
        app: Dash app instance
        data_dict: Dictionary with all loaded data
    """
    # Create a unified data model instance
    unified_model = UnifiedDataModel(data_dict)
    
    # Callback for updating metrics and visualizations
    @app.callback(
        [
            Output("integrated-metrics-row", "children"),
            Output("integrated-visualization", "children"),
            Output("integrated-recommendations", "children"),
            Output("viz-subtitle", "children")
        ],
        [
            Input("integrated-store-dropdown", "value"),
            Input("integrated-product-dropdown", "value"),
            Input("price-adjustment-slider", "value"),
            Input("inventory-adjustment-slider", "value"),
            Input("forecast-period-slider", "value"),
            Input("analysis-view-radio", "value"),
            Input("reset-scenario-button", "n_clicks"),
            Input("show-item-numbers-store", "data")
        ]
    )
    def update_integrated_view(store, product, price_adj, inventory_adj, 
                             forecast_period, analysis_view, reset_clicks,
                             show_item_numbers):
        """Update all integrated view components"""
        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Check if reset button was clicked
        if triggered_id == "reset-scenario-button" and reset_clicks:
            # Create a fresh model instance
            nonlocal unified_model
            unified_model = UnifiedDataModel(data_dict)
            
            # Reset adjustments
            price_adj = 0
            inventory_adj = 0
        
        # Check if we have required inputs
        if store is None or product is None:
            return (
                html.Div("Please select a store and product to view metrics"),
                html.Div("Please select a store and product to view visualizations"),
                html.Div("Please select a store and product to view recommendations"),
                ""
            )
        
        try:
            # Apply adjustments to the model
            if price_adj != 0:
                unified_model.adjust_price(store, product, price_adj)
                
            if inventory_adj != 0:
                unified_model.adjust_inventory(store, product, inventory_adj)
            
            # Calculate metrics
            metrics = unified_model.calculate_metrics(store, product)
            
            # Create KPI cards
            kpi_cards = create_kpi_cards(metrics) if metrics else []
            
            # Create the appropriate visualization
            visualization = None
            subtitle = ""
            
            if analysis_view == "chart":
                visualization = dcc.Graph(
                    figure=create_integrated_chart(
                        unified_model, store, product, price_adj, inventory_adj, 
                        forecast_period, show_item_numbers
                    ),
                    style={'height': '70vh'}
                )
                subtitle = "Integrated Chart View"
                
            elif analysis_view == "heatmap":
                visualization = dcc.Graph(
                    figure=create_impact_heatmap(
                        unified_model, store, product, 
                        price_range=(-MAX_PRICE_DECREASE, MAX_PRICE_INCREASE),
                        inventory_range=(-50, 100),
                        steps=10,
                        metric='profit'
                    ),
                    style={'height': '70vh'}
                )
                subtitle = "Impact Heatmap View"
                
            elif analysis_view == "kpi":
                # Create KPI dashboard with indicators
                kpi_figs = create_kpi_indicators(metrics)
                if kpi_figs:
                    kpi_divs = [
                        dbc.Col(
                            dcc.Graph(figure=fig, style={'height': '30vh'}),
                            width=12, md=6, lg=3
                        ) for fig in kpi_figs
                    ]
                    visualization = dbc.Row(kpi_divs)
                else:
                    visualization = html.Div("No metrics available for KPI dashboard")
                subtitle = "KPI Dashboard View"
            
            # Create recommendations
            recommendations = dcc.Graph(
                figure=create_recommendations_table(metrics),
                config={'displayModeBar': False}
            ) if metrics else html.Div("No recommendations available")
            
            return kpi_cards, visualization, recommendations, subtitle
            
        except Exception as e:
            logger.error(f"Error updating integrated view: {str(e)}")
            return (
                html.Div(f"Error loading metrics: {str(e)}"),
                html.Div(f"Error loading visualization: {str(e)}"),
                html.Div(f"Error loading recommendations: {str(e)}"),
                "Error"
            )

def create_kpi_cards(metrics):
    """
    Create KPI metric cards based on metrics data.
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics()
        
    Returns:
        list: List of dbc.Col components with KPI cards
    """
    kpi_cards = []
    
    try:
        if not metrics:
            return [html.Div("No metrics available")]
        
        # Profit impact card
        if ('integrated' in metrics and 
            'price_change_impact' in metrics['integrated'] and
            'profit_diff_pct' in metrics['integrated']['price_change_impact']):
            
            profit_diff_pct = metrics['integrated']['price_change_impact']['profit_diff_pct']
            profit_diff = metrics['integrated']['price_change_impact']['profit_diff']
            
            color = "success" if profit_diff_pct >= 0 else "danger"
            icon = "arrow-up" if profit_diff_pct > 0 else "arrow-down" if profit_diff_pct < 0 else "equals"
            
            kpi_cards.append(
                dbc.Col(
                    create_info_card(
                        "Profit Impact",
                        [
                            f"{profit_diff_pct:+.1f}%",
                            html.Br(),
                            html.Small(f"${profit_diff:+,.2f}")
                        ],
                        color,
                        icon
                    ),
                    width=12, md=6, lg=3
                )
            )
        
        # Demand impact card
        if ('integrated' in metrics and 
            'price_change_impact' in metrics['integrated'] and
            'forecast_diff_pct' in metrics['integrated']['price_change_impact']):
            
            forecast_diff_pct = metrics['integrated']['price_change_impact']['forecast_diff_pct']
            adjusted_forecast = metrics['integrated']['price_change_impact']['adjusted_forecast']
            original_forecast = metrics['integrated']['price_change_impact']['original_forecast']
            
            color = "info" if forecast_diff_pct >= 0 else "warning"
            icon = "chart-line"
            
            kpi_cards.append(
                dbc.Col(
                    create_info_card(
                        "Demand Impact",
                        [
                            f"{forecast_diff_pct:+.1f}%",
                            html.Br(),
                            html.Small(f"From {original_forecast:.0f} to {adjusted_forecast:.0f} units")
                        ],
                        color,
                        icon
                    ),
                    width=12, md=6, lg=3
                )
            )
        
        # Inventory status card
        if 'inventory' in metrics:
            status = metrics['inventory'].get('status', 'Unknown')
            coverage_weeks = metrics['inventory'].get('coverage_weeks', 0)
            current_stock = metrics['inventory'].get('current_stock', 0)
            
            if status == "Low":
                color = "danger"
                icon = "exclamation-triangle"
            elif status == "Adequate":
                color = "success"
                icon = "check-circle"
            elif status == "Good":
                color = "success"
                icon = "thumbs-up"
            else:  # Excess
                color = "warning"
                icon = "exclamation-circle"
            
            kpi_cards.append(
                dbc.Col(
                    create_info_card(
                        "Inventory Status",
                        [
                            status,
                            html.Br(),
                            html.Small(f"{coverage_weeks:.1f} weeks coverage ({int(current_stock)} units)")
                        ],
                        color,
                        icon
                    ),
                    width=12, md=6, lg=3
                )
            )
        
        # Business impact score card
        if 'integrated' in metrics and 'business_impact_score' in metrics['integrated']:
            score = metrics['integrated']['business_impact_score']
            recommendation = metrics['integrated'].get('recommendation', '')
            
            if score >= 75:
                color = "success"
                icon = "trophy"
            elif score >= 60:
                color = "success"
                icon = "thumbs-up"
            elif score >= 40:
                color = "secondary"
                icon = "balance-scale"
            elif score >= 25:
                color = "warning"
                icon = "thumbs-down"
            else:
                color = "danger"
                icon = "times-circle"
            
            kpi_cards.append(
                dbc.Col(
                    create_info_card(
                        "Business Impact",
                        [
                            f"Score: {score:.0f}/100",
                            html.Br(),
                            html.Small(recommendation)
                        ],
                        color,
                        icon
                    ),
                    width=12, md=6, lg=3
                )
            )
    
    except Exception as e:
        logger.error(f"Error creating KPI cards: {str(e)}")
        kpi_cards = [html.Div(f"Error creating metrics: {str(e)}")]
    
    return kpi_cards