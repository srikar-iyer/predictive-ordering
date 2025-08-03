"""
Main dashboard module for the Pizza Predictive Ordering System.
"""
import sys
import traceback
import os
import logging
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pathlib

# Import chart export utilities
from src.models.chart_export_utils import (
    configure_chart_export_buttons,
    configure_dash_app_for_exports
)

# Import pandas and numpy for data processing
import pandas as pd
import numpy as np

# Import interactive features
from ui.interactive_callbacks import register_interactive_callbacks, create_interactive_components

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core UI components
from ui.core import (
    load_dashboard_data, create_app, format_product_name,
    create_date_range_slider, create_store_product_selectors,
    create_toggle_switch, create_info_card,
    create_tab_layout, create_tab_content, apply_stock_adjustments
)

# Custom enhanced error message component
def create_error_message(message, details=None, traceback_info=None):
    """Create an error message component with optional traceback."""
    components = [
        html.H4("Error", className="text-danger"),
        html.P(message),
        html.P(details, className="text-muted") if details else None
    ]
    
    # Add traceback information if available
    if traceback_info:
        components.extend([
            html.Hr(),
            html.Details([
                html.Summary("Technical Error Details", className="text-danger small"),
                html.Pre(traceback_info, 
                         className="small bg-light p-2 mt-2",
                         style={"whiteSpace": "pre-wrap", "maxHeight": "300px", "overflow": "auto"})
            ])
        ])
    
    return html.Div(components, className="p-3 bg-light border rounded shadow-sm")

# Import pricing module if available
try:
    from ui.pricing import create_pricing_tab_content, register_pricing_callbacks
except ImportError:
    # Create stub functions if pricing module is not available
    def create_pricing_tab_content(data_dict):
        return html.Div("Enhanced pricing module not available")
    def register_pricing_callbacks(app, data_dict):
        pass

# Import integrated view module if available
try:
    from ui.enhanced_integrated_view import create_enhanced_integrated_view_content, register_enhanced_integrated_callbacks
    
    # Use the enhanced version
    def create_integrated_view_content(data_dict):
        return create_enhanced_integrated_view_content(data_dict)
    
    def register_integrated_callbacks(app, data_dict):
        register_enhanced_integrated_callbacks(app, data_dict)
except ImportError:
    try:
        from ui.integrated_view import create_integrated_view_content, register_integrated_callbacks
    except ImportError:
        # Create stub functions if integrated view module is not available
        def create_integrated_view_content(data_dict):
            return html.Div("Integrated view module not available")
        def register_integrated_callbacks(app, data_dict):
            pass

# Import inventory components if available
try:
    from ui.inventory import (
        create_inventory_tab_content, update_inventory_chart,
        update_stock_velocity_chart, update_stock_penalty_chart,
        update_inventory_summary_stats, update_stock_recommendations
    )
except ImportError:
    try:
        from plotly_dashboard_inventory_new import (
            update_inventory_chart, update_stock_velocity_chart,
            update_stock_penalty_chart, update_inventory_summary_stats,
            update_stock_recommendations
        )
    except ImportError:
        # Create stub functions if inventory components are not available
        def create_inventory_tab_content(data_dict=None):
            return html.Div("Inventory components not available")
            
        def update_inventory_chart(*args, **kwargs):
            return go.Figure()
            
        def update_stock_velocity_chart(*args, **kwargs):
            return go.Figure()
            
        def update_stock_penalty_chart(*args, **kwargs):
            return go.Figure()
            
        def update_inventory_summary_stats(*args, **kwargs):
            return None, None, None, None
            
        def update_stock_recommendations(*args, **kwargs):
            return None

# Import summary dashboard if available
try:
    from ui.summary import create_summary_tab_content
except ImportError:
    try:
        import summary_dashboard
        def create_summary_tab_content(data_dict=None):
            return summary_dashboard.create_layout()
    except ImportError:
        # Create stub function if summary dashboard is not available
        def create_summary_tab_content(data_dict=None):
            return html.Div("Summary dashboard not available")
            
# Import item statistics if available
try:
    from ui.item_statistics import create_item_statistics_tab_content, register_callbacks as register_item_stats_callbacks
except ImportError:
    # Create stub function if item statistics is not available
    def create_item_statistics_tab_content(data_dict=None):
        return html.Div("Item statistics not available")
    def register_item_stats_callbacks(app, data_dict):
        pass
        
# Import business impact module if available
try:
    from ui.business_impact import (
        create_profit_tab_content, create_revenue_tab_content, 
        create_loss_prevention_tab_content, register_callbacks as register_business_impact_callbacks
    )
except ImportError:
    # Create stub functions if business impact module is not available
    def create_profit_tab_content(data_dict):
        return html.Div("Profit analysis not available")
    def create_revenue_tab_content(data_dict):
        return html.Div("Revenue analysis not available")
    def create_loss_prevention_tab_content(data_dict):
        return html.Div("Loss prevention analysis not available")
    def register_business_impact_callbacks(app, data_dict):
        pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dashboard')


def create_forecast_tab_content(data_dict):
    """
    Create enhanced content for the forecast tab with improved styling.
    
    Args:
        data_dict: Dictionary with all loaded data
        
    Returns:
        html.Div: Tab content with improved styling
    """
    # Extract relevant data
    combined_data = data_dict.get('combined_data')
    pytorch_forecasts = data_dict.get('pytorch_forecasts')
    rf_forecasts = data_dict.get('rf_forecasts')
    
    if combined_data is None:
        return create_error_message("No historical data available. Please check your data sources.")
        
    if pytorch_forecasts is None and rf_forecasts is None:
        return create_error_message("No forecast data available. Please check your data sources.")
    
    # Use whatever forecast data is available
    forecasts = pytorch_forecasts if pytorch_forecasts is not None else rf_forecasts
    
    # Get store and product options
    store_options, product_options = get_store_product_options(combined_data)
    
    # Create the enhanced forecast tab content
    return html.Div([
        # Introduction section
        dbc.Card([
            dbc.CardBody([
                html.H4([
                    html.I(className="fas fa-chart-line mr-2 text-primary"),
                    "Demand Forecast Analysis"
                ], className="mb-3"),
                html.P(
                    "Analyze predicted demand across different time periods and compare with historical data. "
                    "Adjust for weather conditions and view confidence intervals to improve decision-making.",
                    className="text-muted"
                ),
                html.Div([
                    html.P([
                        html.I(className="fas fa-file-export mr-2 text-primary"),
                        "Export Options: ",
                        "Charts can be downloaded as PNG, SVG, or PDF, and data can be exported to CSV or Excel. "
                        "Use the menu in the top-right of any chart to access export options."
                    ], className="text-muted mt-2 small")
                ])
            ])
        ], className="mb-3 mb-md-4 border-0 rounded shadow-sm"),
        
        # Controls Row - Full width on mobile, side by side on larger screens
        dbc.Row([
            # Left Card - Data Selection
            dbc.Col([
                # Store and product selectors
                create_store_product_selectors('forecast', store_options, product_options)
            ], width=12, lg=8),
            
            # Right Card - Model Selection
            dbc.Col([
                # Model selection card
                dbc.Card([
                    dbc.CardHeader(
                        html.H5([
                            html.I(className="fas fa-cogs mr-2"),
                            "Model Configuration"
                        ], className="m-0"),
                        className="bg-white border-bottom border-light"
                    ),
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-brain mr-2 text-purple"),
                                "Select Forecast Model:"
                            ], className="font-weight-bold mb-3 text-dark d-flex align-items-center"),
                            dcc.RadioItems(
                                id="forecast-model-selector",
                                options=[
                                    {
                                        'label': html.Span([
                                            html.I(className="fas fa-network-wired mr-2", style={"color": "#3498db"}),
                                            "PyTorch Time Series Model"
                                        ], className="d-flex align-items-center"),
                                        'value': 'pytorch'
                                    },
                                    {
                                        'label': html.Span([
                                            html.I(className="fas fa-tree mr-2", style={"color": "#2ecc71"}),
                                            "Random Forest Model"
                                        ], className="d-flex align-items-center"),
                                        'value': 'rf'
                                    }
                                ],
                                value='pytorch' if pytorch_forecasts is not None else 'rf',
                                labelClassName="d-flex align-items-center mb-3 cursor-pointer p-2 rounded",
                                className="forecast-model-selector"
                            )
                        ])
                    ])
                ], className="shadow-sm border-0 rounded h-100 mb-3 mb-lg-0")
            ], width=12, lg=4)
        ]),
        
        # Date Range Row
        dbc.Row([
            dbc.Col([
                create_date_range_slider('forecast')
            ], width=12)
        ]),
        
        # Display Options Row
        dbc.Row([
            dbc.Col([
                # Options Card
                dbc.Card([
                    dbc.CardHeader(
                        html.H5([
                            html.I(className="fas fa-sliders-h mr-2"),
                            "Display Options"
                        ], className="m-0"),
                        className="bg-white border-bottom border-light"
                    ),
                    dbc.CardBody([
                        dbc.Row([
                            # Weather adjustment toggle
                            dbc.Col([
                                create_toggle_switch(
                                    'forecast', 
                                    "Adjust for Weather", 
                                    False, 
                                    icon="weather", 
                                    description="Apply weather-based demand adjustment to forecasts"
                                )
                            ], width=12, sm=6, md=4),
                            
                            # Historical vs Forecast toggle
                            dbc.Col([
                                create_toggle_switch(
                                    'history', 
                                    "Show Historical Data", 
                                    True, 
                                    icon="history",
                                    description="Compare forecasts with historical sales data"
                                )
                            ], width=12, sm=6, md=4),
                            
                            # Display confidence intervals toggle
                            dbc.Col([
                                create_toggle_switch(
                                    'confidence', 
                                    "Show Confidence Intervals", 
                                    True, 
                                    icon="chart",
                                    description="Display prediction uncertainty ranges"
                                )
                            ], width=12, sm=12, md=4)
                        ], className="toggle-control-row")
                    ])
                ], className="shadow-sm border-0 rounded mb-3 mb-md-4")
            ], width=12)
        ]),
        
        # Forecast Chart Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.Div([
                            html.H5([
                                html.I(className="fas fa-chart-area mr-2"),
                                "Forecast Visualization"
                            ], className="m-0 d-inline-block"),
                            html.Small(id="forecast-chart-title", className="text-muted d-block d-md-inline-block mt-1 mt-md-0 ml-md-2")
                        ], className="d-flex flex-column flex-md-row align-items-md-center"),
                        className="bg-white border-bottom border-light"
                    ),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-forecast",
                            type="circle",
                            children=[
                                dcc.Graph(
                                    id="forecast-chart",
                                    figure=go.Figure(),
                                    className="chart-container forecast-chart interactive-chart",  # Add interactive-chart class for JS detection
                                    config=configure_chart_export_buttons({
                                        'displayModeBar': True,
                                        'responsive': True,
                                        'toImageButtonOptions': {
                                            'format': 'png',
                                            'filename': 'sales_forecast',
                                            'scale': 2
                                        },
                                        'scrollZoom': True,  # Enable scroll zoom
                                        'modeBarButtonsToAdd': ['select2d', 'lasso2d'],  # Add selection tools
                                        'modeBarButtonsToRemove': ['autoScale2d']  # Remove some default buttons
                                    })
                                )
                            ]
                        )
                    ])
                ], className="shadow-sm border-0 rounded")
            ], width=12)
        ], className="mb-3 mb-md-4"),
        
        # Forecast Metrics Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.H5([
                            html.I(className="fas fa-calculator mr-2"),
                            "Forecast Metrics"
                        ], className="m-0"),
                        className="bg-white border-bottom border-light"
                    ),
                    dbc.CardBody([
                        html.Div(id="forecast-metrics", className="metrics-container")
                    ])
                ], className="shadow-sm border-0 rounded")
            ], width=12)
        ])
    ], className="forecast-tab-content")


def create_pricing_tab_content(data_dict):
    """
    Create content for the pricing tab.
    
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
        
    if price_elasticities is None:
        return create_error_message("Error: No price elasticity data available")
    
    # Get store and product options
    store_options, product_options = get_store_product_options(combined_data)
    
    # Create the pricing tab content
    return html.Div([
        # Introduction section with export info
        dbc.Card([
            dbc.CardBody([
                html.H4([
                    html.I(className="fas fa-tags mr-2 text-secondary"),
                    "Price Optimization"
                ], className="mb-3"),
                html.P(
                    "Analyze price elasticity and optimize pricing strategies to maximize profit. "
                    "View recommendations based on elasticity analysis and projected impact on sales.",
                    className="text-muted"
                ),
                html.Div([
                    html.P([
                        html.I(className="fas fa-file-export mr-2 text-secondary"),
                        "Export Options: ",
                        "All charts can be downloaded as PNG, SVG, or PDF, and data can be exported to CSV or Excel formats."
                    ], className="text-muted mt-2 small")
                ])
            ])
        ], className="mb-4 border-0 rounded shadow-sm"),
        
        # Controls
        dbc.Row([
            dbc.Col([
                # Store and product selectors
                create_store_product_selectors('pricing', store_options, product_options)
            ], width=12)
        ]),
        
        # Price elasticity visualization
        dbc.Row([
            dbc.Col([
                html.H4("Price Elasticity Analysis"),
                dcc.Loading(
                    id="loading-elasticity",
                    type="circle",
                    children=[
                        dcc.Graph(
                            id="elasticity-chart",
                            figure=go.Figure(),
                            className="chart-container-medium interactive-chart",  # Add interactive-chart class for JS detection
                            config=configure_chart_export_buttons()
                        )
                    ]
                )
            ], width=12)
        ], className="mb-4"),
        
        # Price optimization
        dbc.Row([
            dbc.Col([
                html.H4("Price Optimization"),
                dcc.Loading(
                    id="loading-price-optimization",
                    type="circle",
                    children=[
                        dcc.Graph(
                            id="price-optimization-chart",
                            figure=go.Figure(),
                            className="chart-container-medium interactive-chart",  # Add interactive-chart class for JS detection
                            config=configure_chart_export_buttons()
                        )
                    ]
                )
            ], width=12)
        ], className="mb-4"),
        
        # Price recommendations
        dbc.Row([
            dbc.Col([
                html.H4("Price Recommendations"),
                html.Div(id="price-recommendations")
            ], width=12)
        ])
    ])


def create_export_help_card():
    """
    Create a card that explains the export capabilities.
    
    Returns:
        dbc.Card: Card component with export help information
    """
    return dbc.Card([
        dbc.CardHeader(
            html.H5([
                html.I(className="fas fa-file-export mr-2"),
                "Export Capabilities"
            ], className="m-0"),
            style={
                "backgroundColor": "white",
                "borderBottom": "1px solid #eaecef"
            }
        ),
        dbc.CardBody([
            html.P("All charts in this dashboard support the following export options:"),
            html.Ul([
                html.Li([
                    html.Strong("Image Export: "),
                    "Click the camera icon in the chart toolbar to download the chart as a PNG image."
                ]),
                html.Li([
                    html.Strong("Additional Image Formats: "),
                    "Use the export dropdown menu in the top-right corner of each chart to download as SVG or PDF."
                ]),
                html.Li([
                    html.Strong("Data Export: "),
                    "Use the export dropdown menu to download the chart data in CSV or Excel format."
                ])
            ]),
            html.P([
                html.I(className="fas fa-info-circle mr-1"),
                "Tip: Hover over a chart and click the menu button in the top-right to see all available export options."
            ], className="text-muted mt-2")
        ])
    ], style={
        "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.07)",
        "border": "none",
        "borderRadius": "6px",
        "marginBottom": "1rem"
    })

def create_settings_tab_content():
    """
    Create content for the settings tab.
    
    Returns:
        html.Div: Tab content
    """
    return html.Div([
        # Add export help card
        dbc.Row([
            dbc.Col([
                create_export_help_card()
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H4("Display Settings"),
                # Toggle item numbers
                create_toggle_switch('item-numbers', "Show Item Numbers in Product Names", True),
                
                html.Hr(),
                
                html.H4("Weather Settings"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Store ZIP Code:"),
                        dbc.Input(
                            id="zipcode-input",
                            type="text",
                            placeholder="Enter ZIP code",
                            value="10001"  # Default to NYC
                        )
                    ], width=6),
                    dbc.Col([
                        html.Button(
                            "Update Weather",
                            id="update-weather-button",
                            className="btn btn-primary"
                        ),
                        html.Div(id="weather-status", className="mt-2")
                    ], width=6)
                ], className="mb-4"),
                
                html.Hr(),
                
                html.H4("Data Management"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Fallback Data Directory:"),
                        dbc.Input(
                            id="fallback-dir-input",
                            type="text",
                            placeholder="Enter fallback directory path",
                            value=""
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Button(
                            "Reload Data",
                            id="reload-data-button",
                            color="primary",
                            className="mb-3"
                        ),
                        html.Div(id="data-reload-status", className="mt-2")
                    ], width=6)
                ], className="mb-4"),
                
                html.Hr(),
                
                html.H4("About"),
                html.P("Pizza Predictive Ordering System"),
                html.P("This dashboard provides insights into sales forecasts, inventory management, and price optimization for pizza retailers."),
                
                html.Hr(),
                
                html.H4("Debug Information"),
                dbc.Button(
                    "Show Data Info",
                    id="debug-button",
                    color="secondary",
                    className="mb-3"
                ),
                html.Div(id="debug-output")
            ], width=12)
        ])
    ])


def create_category_summary_section(data_dict):
    """
    Create a summary section with statistics for the entire category.
    
    Args:
        data_dict: Dictionary with all loaded data
        
    Returns:
        dbc.Card: Card containing category summary statistics
    """
    # Get combined data
    combined_data = data_dict.get('combined_data')
    if combined_data is None or len(combined_data) == 0:
        return dbc.Card(
            dbc.CardBody([
                html.H5("Category Summary"),
                html.P("No data available for category statistics")
            ])
        )
    
    # Calculate category statistics
    try:
        # Total sales and revenue
        total_sales = combined_data['Sales'].sum()
        total_revenue = combined_data['Retail_Revenue'].sum() if 'Retail_Revenue' in combined_data.columns else combined_data['Sales'].sum() * combined_data['Price'].mean()
        
        # Profit metrics
        if 'Profit' in combined_data.columns:
            total_profit = combined_data['Profit'].sum()
            profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
        else:
            # Estimate profit if not directly available
            if 'Cost' in combined_data.columns:
                estimated_profit = ((combined_data['Price'] - combined_data['Cost']) * combined_data['Sales']).sum()
                total_profit = estimated_profit
                profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
            else:
                total_profit = None
                profit_margin = None
        
        # Inventory metrics
        current_stock = combined_data['Stock_Level'].sum() if 'Stock_Level' in combined_data.columns else None
        avg_weekly_sales = combined_data.groupby('Store_Id')['Sales'].sum().mean() * 7 if 'Sales' in combined_data.columns else None
        weeks_of_stock = current_stock / avg_weekly_sales if current_stock is not None and avg_weekly_sales and avg_weekly_sales > 0 else None
        
        # Product counts
        unique_products = combined_data['Product'].nunique() if 'Product' in combined_data.columns else 0
        unique_stores = combined_data['Store_Id'].nunique() if 'Store_Id' in combined_data.columns else 0
        
        # Create the summary card
        return dbc.Card([
            dbc.CardHeader(
                html.H5([html.I(className="fas fa-chart-pie mr-2"), "Frozen Pizza Category Summary"], className="m-0")
            ),
            dbc.CardBody([
                dbc.Row([
                    # Sales metrics
                    dbc.Col([
                        html.H6("Sales Metrics", className="border-bottom pb-2 mb-3"),
                        html.Div([
                            html.P(["Total Units Sold: ", html.Strong(f"{total_sales:,.0f}")]),
                            html.P(["Total Revenue: ", html.Strong(f"${total_revenue:,.2f}")]),
                            html.P(["Total Profit: ", html.Strong(f"${total_profit:,.2f}") if total_profit is not None else "Not available"]),
                            html.P(["Profit Margin: ", html.Strong(f"{profit_margin:.1f}%") if profit_margin is not None else "Not available"])
                        ])
                    ], width=12, md=4),
                    
                    # Inventory metrics
                    dbc.Col([
                        html.H6("Inventory Metrics", className="border-bottom pb-2 mb-3"),
                        html.Div([
                            html.P(["Current Stock: ", html.Strong(f"{current_stock:,.0f}") if current_stock is not None else "Not available"]),
                            html.P(["Avg Weekly Sales: ", html.Strong(f"{avg_weekly_sales:,.1f}") if avg_weekly_sales is not None else "Not available"]),
                            html.P(["Weeks of Stock: ", html.Strong(f"{weeks_of_stock:.1f}") if weeks_of_stock is not None else "Not available"])
                        ])
                    ], width=12, md=4),
                    
                    # Category metrics
                    dbc.Col([
                        html.H6("Category Metrics", className="border-bottom pb-2 mb-3"),
                        html.Div([
                            html.P(["Unique Products: ", html.Strong(f"{unique_products}")]),
                            html.P(["Stores with Data: ", html.Strong(f"{unique_stores}")]),
                            html.P(["Date Range: ", html.Strong(f"{combined_data['Date'].min().strftime('%Y-%m-%d')} to {combined_data['Date'].max().strftime('%Y-%m-%d')}")])
                        ])
                    ], width=12, md=4)
                ])
            ])
        ], className="mb-4 shadow-sm")
    except Exception as e:
        logger.error(f"Error calculating category statistics: {str(e)}", exc_info=True)
        return dbc.Card(
            dbc.CardBody([
                html.H5("Category Summary"),
                html.P(f"Error calculating category statistics: {str(e)}")
            ])
        )

def create_dashboard_layout(data_dict):
    """
    Create the main dashboard layout with improved debugging support.
    
    Args:
        data_dict: Dictionary with all loaded data
        
    Returns:
        html.Div: Dashboard layout
    """
    # Create the tabs
    tabs = create_tab_layout()
    
    # Create the tab contents
    tab_contents = create_tab_content()
    
    # Ensure there's at least some forecast data to work with
    if 'forecasts' not in data_dict or data_dict['forecasts'] is None:
        if 'pytorch_forecasts' in data_dict and data_dict['pytorch_forecasts'] is not None:
            data_dict['forecasts'] = data_dict['pytorch_forecasts']
            logger.info("Using PyTorch forecasts as default forecasts")
        elif 'rf_forecasts' in data_dict and data_dict['rf_forecasts'] is not None:
            data_dict['forecasts'] = data_dict['rf_forecasts']
            logger.info("Using RF forecasts as default forecasts")
        elif 'arima_forecasts' in data_dict and data_dict['arima_forecasts'] is not None:
            data_dict['forecasts'] = data_dict['arima_forecasts']
            logger.info("Using ARIMA forecasts as default forecasts")
    
    # Check if any critical data is missing
    data_status = check_data_status(data_dict)
    
    # Return the complete layout with enhanced styling
    interactive_components = create_interactive_components()
    
    return html.Div([
        # Enhanced Header
        dbc.Navbar(
            dbc.Container([
                html.A(
                    dbc.Row([
                        dbc.Col(
                            html.Div([
                                html.Img(src="/assets/docker-logo.png", height="30px", className="mr-2") if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'docker-logo.png')) else 
                                html.I(className="fab fa-docker", style={"fontSize": "1.5rem", "color": "#0db7ed", "marginRight": "10px"})
                            ]),
                            width="auto",
                            className="d-flex align-items-center"
                        ),
                        dbc.Col(
                            dbc.NavbarBrand([
                                "Pizza ", 
                                html.Span("Predictive", style={"color": "#0db7ed", "fontWeight": "600"}),
                                " Ordering",
                                html.Span(" System", className="d-none d-md-inline")
                            ], 
                            style={"fontSize": "1.1rem", "letterSpacing": "0.5px"},
                            className="ml-2"), 
                            width="auto"
                        )
                    ], align="center", className="g-0"),
                    href="#",
                    style={"textDecoration": "none"}
                ),
                dbc.NavbarToggler(id="navbar-toggler", className="ms-auto"),
                dbc.Collapse(
                    dbc.Row([
                        dbc.Col(
                            dbc.NavItem(
                                html.Div(
                                    id="current-weather", 
                                    className="text-light d-flex align-items-center",
                                    style={"fontSize": "0.9rem"}
                                )
                            ),
                            width="auto"
                        ),
                        dbc.Col(
                            dbc.NavItem(
                                html.Div(
                                    id="data-status-indicator", 
                                    children=data_status,
                                    className="d-flex align-items-center ms-3",
                                    style={"fontSize": "0.9rem"}
                                )
                            ),
                            width="auto"
                        )
                    ], className="g-0 ml-auto flex-nowrap mt-3 mt-md-0", align="center"),
                    id="navbar-collapse",
                    navbar=True
                )
            ], fluid=True),
            color="dark",
            dark=True,
            className="mb-3 mb-md-4 py-2 shadow-sm",
            style={
                "background": "linear-gradient(135deg, #0db7ed, #002b36)",
                "borderBottom": "2px solid #eaecef"
            }
        ),
        
        # Data loading error alert with enhanced styling (only shown if there are critical errors)
        dbc.Alert(
            html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-circle me-2 me-md-3", style={"fontSize": "1.25rem"}),
                    html.Div([
                        html.H5("Data Loading Warning", className="alert-heading mb-1 fs-6"),
                        html.P("Some data files could not be loaded. Functionality may be limited. Check Settings tab for details.", 
                              className="mb-0 small")
                    ])
                ], className="d-flex align-items-center")
            ]),
            id="data-loading-alert",
            color="warning",
            dismissable=True,
            is_open=not all(v is not None for k, v in data_dict.items() if k in ['combined_data', 'pytorch_forecasts', 'rf_forecasts', 'price_elasticities']),
            className="mb-3 mb-md-4 mx-2 mx-md-4 shadow-sm",
            style={
                "borderRadius": "6px",
                "border": "none"
            }
        ),
        
        # Main container with enhanced styling
        dbc.Container([
            # Category Summary Statistics
            create_category_summary_section(data_dict),
            # Introduction header (only shown on first load)
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H3([
                            "Welcome to the ", 
                            html.Span("Pizza Predictive Ordering", className="text-danger"),
                            " Dashboard"
                        ], className="mb-2 mb-md-3 fs-4 fs-md-3"),
                        html.P([
                            "Explore sales forecasts, optimize inventory, and maximize profits with our advanced analytics platform.",
                            html.Br(),
                            "Select a tab below to get started."
                        ], className="lead text-muted small")
                    ], width=12)
                ], className="mb-3 pb-2 border-bottom")
            ], id="dashboard-intro", className="mb-3"),
            
            # Enhanced Tabs - with horizontal scrolling on mobile - Docker Hub style
            dbc.Tabs(
                        [
                            tabs['forecast'],
                            tabs['inventory'],
                            tabs['pricing'],
                            tabs['integrated'],
                            tabs['item_stats'],
                            tabs['profit'],
                            tabs['revenue'],
                            tabs['loss'],
                            tabs['summary'],
                            tabs['settings']
                        ],
                        id="main-tabs",
                        active_tab="tab-forecast",
                        className="mb-3"
),
            
            
            # Tab content - using default Plotly styling
            html.Div(
                id="tab-content", 
                className="mt-3"
            )
        ], fluid=True, className="pb-4 pb-md-5"),
        
        # Footer - simplified for mobile - Docker Hub style
        html.Footer(
            dbc.Container([
                html.Hr(className="mb-2 mb-md-3"),
                dbc.Row([
                    dbc.Col([
                        html.P([
                            html.I(className="fab fa-docker mr-1 mr-md-2", style={"color": "#0db7ed"}),
                            "Pizza Predictive Ordering Dashboard ",
                            html.Small("v1.0", className="text-muted")
                        ], className="mb-0 text-center small")
                    ], width=12)
                ])
            ], fluid=True),
            className="py-2 py-md-3 mt-4 mt-md-5",
            style={"backgroundColor": "#f8f9fa", "borderTop": "1px solid #eaecef"}
        ),
        
        # Store components for maintaining state
        dcc.Store(id="show-item-numbers-store", data=True),
        dcc.Store(id="weather-data-store", data=None),
        dcc.Store(id="fallback-dir-store", data=None),
        dcc.Store(id="error-log-store", data=[]),
        dcc.Store(id="debug-log-store", data=[]),
        
        # Debug console logger
        html.Div(id="console-log-trigger"),
        
        # Add interactive components
        *interactive_components
    ], className="dashboard-container")


def get_store_product_options(data_df, show_item_numbers=True):
    """
    Get store and product options for dropdown menus.
    
    Args:
        data_df: DataFrame with store and product data
        show_item_numbers: Whether to include item numbers in product names
        
    Returns:
        tuple: (store_options, product_options)
    """
    if data_df is None or len(data_df) == 0:
        return [], []
        
    try:
        # Get unique stores
        stores = data_df['Store_Id'].unique()
        store_options = [{'label': f'Store {s}', 'value': s} for s in sorted(stores)]
        
        # Get unique products with names
        products = data_df[['Item', 'Product']].drop_duplicates()
        
        # Format product names based on setting
        product_options = [
            {
                'label': format_product_name(row['Product'], row['Item'], show_item_numbers),
                'value': row['Item']
            } for _, row in products.iterrows()
        ]
        
        return store_options, product_options
        
    except Exception as e:
        logger.error(f"Error generating store/product options: {str(e)}")
        return [], []


def register_callbacks(app, data_dict):
    """
    Register all callbacks for the dashboard with improved error handling.
    
    Args:
        app: Dash app instance
        data_dict: Dictionary with all loaded data
    """
    # Register global error handler for all callbacks
    @app.callback(
        Output("error-log-store", "data"),
        [Input("main-tabs", "active_tab"),
         Input("forecast-chart", "figure"),
         Input("elasticity-chart", "figure"),
         Input("price-optimization-chart", "figure")],
        [State("error-log-store", "data")]
    )
    def log_callback_errors(active_tab, forecast_figure, elasticity_figure, price_figure, current_errors):
        """Global error handler for callbacks"""
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_errors if current_errors else []
            
        # Get the callback that triggered this
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Check if there's an error in the callback context
        error_data = current_errors if current_errors else []
        
        if hasattr(ctx, '_dash_error') and ctx._dash_error:
            # Log the error
            error_entry = {
                'timestamp': datetime.now().isoformat(),
                'callback': trigger,
                'error': str(ctx._dash_error),
                'stack': getattr(ctx._dash_error, '__traceback__', None)
            }
            error_data.append(error_entry)
            logger.error(f"Callback error in {trigger}: {ctx._dash_error}")
            
            # Add browser console logging for client-side debugging
            app.clientside_callback(
                """
                function(errorData) {
                    if (errorData && errorData.length > 0) {
                        const latestError = errorData[errorData.length - 1];
                        console.error(
                            `Dash callback error in ${latestError.callback}: ${latestError.error}`, 
                            latestError
                        );
                    }
                    return window.dash_clientside.no_update;
                }
                """,
                Output("error-log-store", "data", allow_duplicate=True),
                Input("error-log-store", "data"),
                prevent_initial_call=True
            )
        
        return error_data
    # Tab selection callback with enhanced error handling
    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "active_tab"),
        State("show-item-numbers-store", "data")
    )
    def render_tab_content(active_tab, show_item_numbers):
        """Render content for the active tab with robust error handling"""
        try:
            # Initial logging
            logger.info(f"Rendering tab content for {active_tab}")
            
            # Add pre-callback validation to detect common issues
            if active_tab is None:
                logger.warning("No active tab selected, defaulting to forecast tab")
                active_tab = "tab-forecast"
            
            # Add client-side console logging
            app.clientside_callback(
                """
                function(tab) {
                    console.log(`Tab changed to: ${tab}`);
                    return window.dash_clientside.no_update;
                }
                """,
                Output("tab-content", "id", allow_duplicate=True),
                Input("main-tabs", "active_tab"),
                prevent_initial_call=True
            )
            
            # Verify that the data dictionary exists and has necessary keys
            if not data_dict:
                error_stack = traceback.format_exc()
                logger.error(f"Data dictionary is empty or missing\n{error_stack}")
                return create_error_message(
                    "Data is not available. Please reload the application.", 
                    details="The data dictionary could not be loaded properly.",
                    traceback_info=error_stack
                )
                
            # Add more detailed logging
            logger.info(f"Rendering content for tab: {active_tab}")
            
            # Use a dictionary to map tab IDs to their content creation functions
            tab_content_creators = {
                "tab-forecast": lambda: create_forecast_tab_content(data_dict),
                "tab-inventory": lambda: create_inventory_tab_content(),
                "tab-pricing": lambda: create_pricing_tab_content(data_dict),
                "tab-integrated": lambda: create_integrated_view_content(data_dict),
                "tab-item-statistics": lambda: create_item_statistics_tab_content(),
                "tab-profit": lambda: create_profit_tab_content(data_dict),
                "tab-revenue": lambda: create_revenue_tab_content(data_dict),
                "tab-loss": lambda: create_loss_prevention_tab_content(data_dict),
                "tab-summary": lambda: create_summary_tab_content(),
                "tab-settings": lambda: create_settings_tab_content()
            }
            
            # Get the appropriate content creator function
            content_creator = tab_content_creators.get(active_tab)
            
            if content_creator:
                # Try to create the content with specific error handling
                try:
                    logger.info(f"Creating content for {active_tab}")
                    content = content_creator()
                    
                    # Add wrapper to ensure content is visible
                    if content:
                        # Add timestamp to force re-render and console logging
                        timestamp = int(time.time())
                        rendered_content = html.Div([
                            # Add a timestamp to force re-render
                            html.Div(id=f"tab-timestamp-{timestamp}", style={"display": "none"}),
                            
                            # Add a debug div with render timestamp
                            html.Div(id="debug-render-info", **{
                                'data-render-time': timestamp,
                                'data-tab-id': active_tab
                            }, style={"display": "none"}),
                            
                            # Actual content
                            content
                        ], className="tab-content-wrapper", style={"display": "block", "width": "100%"})
                        
                        # Add client-side logging for tab render completion
                        app.clientside_callback(
                            """
                            function(tabId, timestamp) {
                                console.log(`Tab content rendered: ${tabId} at ${timestamp}`);
                                // Trigger additional event to fix tab visibility
                                if (window.fixTabVisibility) {
                                    console.log("Running post-render tab visibility fix...");
                                    window.fixTabVisibility();
                                    
                                    // Set multiple delayed fixes to ensure everything is visible
                                    setTimeout(() => window.fixTabVisibility(), 200);
                                    setTimeout(() => window.fixTabVisibility(), 500);
                                }
                                return window.dash_clientside.no_update;
                            }
                            """,
                            Output(f"tab-timestamp-{timestamp}", "id", allow_duplicate=True),
                            Input("debug-render-info", "data-tab-id"),
                            State("debug-render-info", "data-render-time"),
                            prevent_initial_call=True
                        )
                        
                        return rendered_content
                    else:
                        logger.error(f"Content creator for {active_tab} returned None or empty content")
                        return create_error_message(f"Unable to load content for {active_tab}. The content may be missing or unavailable.")
                except Exception as tab_error:
                    error_stack = traceback.format_exc()
                    logger.error(f"Error in content creator for {active_tab}: {str(tab_error)}\n{error_stack}")
                    return create_error_message(
                        f"Error loading {active_tab.replace('tab-', '')} content: {str(tab_error)}",
                        details=f"Function: {content_creator.__name__ if hasattr(content_creator, '__name__') else 'unknown'}",
                        traceback_info=error_stack
                    )
            else:
                logger.warning(f"Unknown tab ID: {active_tab}, defaulting to forecast tab")
                # Default to forecast tab
                return tab_content_creators["tab-forecast"]()
        except Exception as e:
            logger.error(f"Critical error rendering tab content: {str(e)}", exc_info=True)
            # Return a comprehensive error message with debugging information
            return html.Div([
                html.H4("Error Loading Tab Content", className="text-danger"),
                html.P([
                    "There was an error loading content for this tab: ",
                    html.Strong(str(e))
                ]),
                html.Hr(),
                html.Details([
                    html.Summary("Technical Details (for support)"),
                    html.Pre(traceback.format_exc(), style={"whiteSpace": "pre-wrap", "fontSize": "0.8rem"})
                ]),
                html.Hr(),
                html.Div([
                    html.P("Try one of the following:"),
                    html.Ul([
                        html.Li("Reload the page"),
                        html.Li("Select a different tab"),
                        html.Li("Check browser console for errors (F12)")
                    ])
                ]),
                # Add a retry button
                html.Button(
                    "Retry Loading Content",
                    id="retry-tab-button",
                    className="btn btn-primary mt-3",
                    n_clicks=0
                )
            ], className="p-4 border rounded bg-light")
    
    # Add callback for retry button
    @app.callback(
        Output("tab-content", "children", allow_duplicate=True),
        Input("retry-tab-button", "n_clicks"),
        State("main-tabs", "active_tab"),
        State("show-item-numbers-store", "data"),
        prevent_initial_call=True
    )
    def retry_loading_content(n_clicks, active_tab, show_item_numbers):
        """Retry loading the content when the retry button is clicked"""
        if n_clicks > 0:
            logger.info(f"Retrying content loading for tab {active_tab}")
            # Add console logging for the retry attempt
            app.clientside_callback(
                """
                function() {
                    console.log("Retrying tab content loading...");
                    return window.dash_clientside.no_update;
                }
                """,
                Output("retry-tab-button", "n_clicks", allow_duplicate=True),
                Input("retry-tab-button", "n_clicks"),
                prevent_initial_call=True
            )
            # Call the main render function
            return render_tab_content(active_tab, show_item_numbers)
        return dash.no_update

    # Show/hide item numbers callback
    @app.callback(
        Output("show-item-numbers-store", "data"),
        Input("item-numbers-toggle", "value")
    )
    def update_show_item_numbers(toggle_value):
        """Update show item numbers setting"""
        show_numbers = bool(toggle_value and 1 in toggle_value)
        return show_numbers
    
    # Forecast chart and title callbacks
    @app.callback(
        [
            Output("forecast-chart", "figure"),
            Output("forecast-chart-title", "children")
        ],
        [
            Input("forecast-store-dropdown", "value"),
            Input("forecast-product-dropdown", "value"),
            Input("forecast-model-selector", "value"),
            Input("forecast-date-range", "start_date"),
            Input("forecast-date-range", "end_date"),
            Input("forecast-toggle", "value"),  # Weather adjustment
            Input("history-toggle", "value"),   # Show history
            Input("confidence-toggle", "value"),  # Show confidence intervals
            Input("weather-data-store", "data"),
            Input("show-item-numbers-store", "data")
        ]
    )
    def update_forecast_chart(store, product, model, start_date, end_date, 
                             weather_toggle, history_toggle, confidence_toggle,
                             weather_data, show_item_numbers):
        """Update the forecast chart based on selections"""
        try:
            # Select the appropriate forecast data
            if model == 'pytorch':
                forecast_df = data_dict.get('pytorch_forecasts')
            else:
                forecast_df = data_dict.get('rf_forecasts')
            
            # Use either forecast if available
            if forecast_df is None:
                if model == 'pytorch':
                    forecast_df = data_dict.get('rf_forecasts')
                else:
                    forecast_df = data_dict.get('pytorch_forecasts')
            
            # If still none, try ARIMA as last resort
            if forecast_df is None:
                forecast_df = data_dict.get('arima_forecasts')
                if forecast_df is not None:
                    logger.info("Using ARIMA forecasts as fallback")
            
            # If still none, return empty figure with error message
            if forecast_df is None:
                logger.error("No forecast data available for chart")
                fig = go.Figure()
                fig.add_annotation(
                    text="Error: No forecast data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=20, color="red")
                )
                return fig, "No forecast data available"
            
            # Filter data by store and product
            if store is not None and product is not None:
                forecast_data = forecast_df[(forecast_df['Store_Id'] == store) & (forecast_df['Item'] == product)]
                if len(forecast_data) == 0:
                    # Generate detailed error with traceback
                    error_stack = traceback.format_exc()
                    logger.error(f"No forecast data found for store {store}, product {product}\n{error_stack}")
                    fig = go.Figure()
                    fig.add_annotation(
                        text=f"No forecast data found for store {store}, product {product}",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False,
                        font=dict(size=16, color="red")
                    )
                    return fig, f"No data for store {store}, product {product}"
            else:
                # Generate detailed error with traceback
                error_stack = traceback.format_exc()
                logger.error(f"Missing store or product selection for forecast\n{error_stack}")
                fig = go.Figure()
                fig.add_annotation(
                    text="Missing store or product selection",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16, color="red")
                )
                return fig, "Missing store/product selection"
        except Exception as e:
            # Generate detailed error with traceback
            error_stack = traceback.format_exc()
            logger.error(f"Error preparing forecast data: {str(e)}\n{error_stack}")
            
            # Create error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.7, showarrow=False,
                font=dict(size=16, color="red")
            )
            
            # Add file and line information from traceback
            tb_lines = error_stack.split('\n')
            file_line_info = [line for line in tb_lines if 'File "' in line]
            if file_line_info:
                fig.add_annotation(
                    text=file_line_info[-1].strip(),
                    xref="paper", yref="paper",
                    x=0.5, y=0.3, showarrow=False,
                    font=dict(size=12, color="gray")
                )
            
            return fig, f"Error: {str(e)}"
        
        # Get historical data if needed
        historical_data = None
        if history_toggle and 1 in history_toggle:
            combined_data = data_dict.get('combined_data')
            if combined_data is not None:
                historical_data = combined_data[(combined_data['Store_Id'] == store) & (combined_data['Item'] == product)]
        
        # Apply weather adjustment if needed
        if weather_toggle and 1 in weather_toggle and app.weather_service is not None and weather_data:
            try:
                # Use the ZIP code from weather data
                zipcode = weather_data.get('zipcode', '10001')
                
                # Adjust forecast based on weather
                forecast_data = app.weather_service.adjust_demand_forecast(
                    forecast_data, zipcode, product_type="frozen_pizza")
            except Exception as e:
                logger.error(f"Error applying weather adjustment: {str(e)}")
        
        # Create figure with error handling
        try:
            fig = go.Figure()
            
            # Format dates
            if start_date:
                start_date = pd.to_datetime(start_date)
            if end_date:
                end_date = pd.to_datetime(end_date)
            
            # Filter by date range
            if start_date and end_date:
                forecast_data = forecast_data[(forecast_data['Date'] >= start_date) & (forecast_data['Date'] <= end_date)]
                if historical_data is not None:
                    historical_data = historical_data[(historical_data['Date'] >= start_date) & (historical_data['Date'] <= end_date)]
        except Exception as e:
            # Generate detailed error with traceback
            error_stack = traceback.format_exc()
            logger.error(f"Error filtering date range: {str(e)}\n{error_stack}")
            
            # Create error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Date range error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig, f"Date range error: {str(e)}"
        
        # Get product name
        product_name = "Unknown Product"
        if len(forecast_data) > 0 and 'Product' in forecast_data.columns:
            product_name = forecast_data['Product'].iloc[0]
        elif historical_data is not None and len(historical_data) > 0 and 'Product' in historical_data.columns:
            product_name = historical_data['Product'].iloc[0]
        
        # Format product name
        formatted_product = format_product_name(product_name, product, show_item_numbers)
        
        # Add historical data
        if historical_data is not None and len(historical_data) > 0:
            fig.add_trace(go.Scatter(
                x=historical_data['Date'],
                y=historical_data['Sales'],
                mode='lines',
                name='Historical Sales',
                line=dict(color='blue')
            ))
        
        # Add forecast data
        if len(forecast_data) > 0:
            # Check for different column names used in different forecast types
            forecast_col = 'Forecast'
            if 'Predicted_Demand' in forecast_data.columns:
                forecast_col = 'Predicted_Demand'
            elif 'Forecast' in forecast_data.columns:
                forecast_col = 'Forecast'
            elif 'Predicted_Sales' in forecast_data.columns:
                forecast_col = 'Predicted_Sales'
            else:
                # Try to find any suitable numeric column as a fallback
                numeric_cols = forecast_data.select_dtypes(include=[np.number]).columns.tolist()
                potential_forecast_cols = [col for col in numeric_cols 
                                         if col not in ['Store_Id', 'Item', 'Days_In_Future'] 
                                         and not col.startswith('Lower_') 
                                         and not col.startswith('Upper_')]
                if potential_forecast_cols:
                    forecast_col = potential_forecast_cols[0]
                    logger.info(f"Using '{forecast_col}' as fallback forecast column")
                    # Add standardized column names for future compatibility
                    forecast_data['Forecast'] = forecast_data[forecast_col]
                    forecast_col = 'Forecast'
            
            if forecast_col in forecast_data.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_data['Date'],
                    y=forecast_data[forecast_col],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red')
                ))
            else:
                logger.warning(f"No valid forecast column found in data. Available columns: {forecast_data.columns}")
            
            # Add confidence intervals
            if confidence_toggle and 1 in confidence_toggle:
                # Generate confidence intervals if not present in data
                if 'Lower_Bound' not in forecast_data.columns or 'Upper_Bound' not in forecast_data.columns:
                    # Create confidence intervals (approx 20% margin of error)
                    if forecast_col in forecast_data.columns:
                        demand = forecast_data[forecast_col]
                        std_dev = demand * 0.1  # Assuming 10% standard deviation
                        forecast_data['Upper_Bound'] = demand + 1.96 * std_dev
                        forecast_data['Lower_Bound'] = np.maximum(0, demand - 1.96 * std_dev)
                    else:
                        logger.warning(f"Cannot generate confidence intervals: forecast column '{forecast_col}' not found")
                
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
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(width=0),
                    showlegend=False
                ))
        
        # Set chart title and labels
        title = f"Demand Forecast for {formatted_product} at Store {store}"
        
        # Enhanced chart layout
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Units Demanded",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=20, b=50, l=60, r=40),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(
                family="Roboto, Arial, sans-serif",
                size=12,
                color="#333333"
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor="#eaecef",
                tickfont=dict(size=11)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#eaecef",
                tickfont=dict(size=11)
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Roboto, Arial, sans-serif"
            ),
            hovermode="closest"
        )
        
        # Create chart title with model info
        model_name = model.upper()
        # Use model name without item numbers as requested
        model_display = "PyTorch Model" if model == 'pytorch' else "RF Model"
        chart_title = f"(Store {store} | {formatted_product} | {model_display})"
        if weather_toggle and 1 in weather_toggle:
            chart_title += " - Weather Adjusted"
            
        # Add metadata for interactive features
        if not hasattr(fig.layout, 'metadata'):
            fig.layout.metadata = {}
            
        # Add drill-down capabilities
        fig.layout.metadata['drill_down'] = {
            'callback_data': {
                'chart_type': f"Forecast: {formatted_product}",
                'store_id': store,
                'product_id': product
            },
            'chart_title': f"Demand Forecast for {formatted_product}"
        }
        
        # Add zoom sync capabilities
        fig.layout.metadata['zoom_sync'] = {
            'linked_charts': ['elasticity-chart', 'price-optimization-chart'],
            'sync_x': True,
            'sync_y': False
        }
        
        # Return both the figure and the title for the chart header
        try:
            return fig, chart_title
        except Exception as e:
            # Generate detailed error with traceback
            error_stack = traceback.format_exc()
            logger.error(f"Final error in forecast chart: {str(e)}\n{error_stack}")
            
            # Create error figure if something went wrong
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig, f"Error: {str(e)}"
    
    # Price elasticity chart callback
    @app.callback(
        Output("elasticity-chart", "figure"),
        [
            Input("pricing-store-dropdown", "value"),
            Input("pricing-product-dropdown", "value"),
            Input("show-item-numbers-store", "data")
        ]
    )
    def update_elasticity_chart(store, product, show_item_numbers):
        """Update the price elasticity chart based on selections"""
        try:
            price_elasticities = data_dict.get('price_elasticities')
            
            if price_elasticities is None:
                logger.warning("No price elasticity data available")
                return go.Figure()
            
            if store is None or product is None:
                logger.warning(f"Missing required parameters for elasticity chart: store={store}, product={product}")
                return go.Figure()
            
            # Filter by store and product
            elasticity_data = price_elasticities[
                (price_elasticities['Store_Id'] == store) &
                (price_elasticities['Item'] == product)
            ]
            
            if len(elasticity_data) == 0:
                logger.warning(f"No elasticity data found for store {store}, product {product}")
                return go.Figure()
        except Exception as e:
            logger.error(f"Error preparing elasticity data: {str(e)}")
            return go.Figure()
        
        # Get elasticity value and product info
        elasticity = elasticity_data['Elasticity'].iloc[0]
        product_name = elasticity_data['Product'].iloc[0]
        current_price = elasticity_data['Avg_Price'].iloc[0]
        
        # Format product name
        formatted_product = format_product_name(product_name, product, show_item_numbers)
        
        # Create price range (80% to 120% of current price)
        price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
        
        # Calculate quantity at each price point
        quantity = []
        for price in price_range:
            price_ratio = price / current_price
            quantity_ratio = price_ratio ** elasticity
            quantity.append(quantity_ratio)
        
        # Create figure
        fig = go.Figure()
        
        # Add price sensitivity curve
        fig.add_trace(go.Scatter(
            x=price_range,
            y=quantity,
            mode='lines',
            name=f'Price Sensitivity (e={elasticity:.2f})'
        ))
        
        # Add reference lines
        fig.add_shape(
            type="line",
            x0=current_price,
            y0=0,
            x1=current_price,
            y1=1,
            line=dict(color="red", width=1, dash="dash"),
            name="Current Price"
        )
        
        fig.add_shape(
            type="line",
            x0=price_range[0],
            y0=1,
            x1=price_range[-1],
            y1=1,
            line=dict(color="green", width=1, dash="dash"),
            name="Current Quantity"
        )
        
        # Set chart title and labels
        fig.update_layout(
            title=f"Price Sensitivity for {formatted_product} at Store {store}",
            xaxis_title="Price ($)",
            yaxis_title="Relative Quantity (Current = 1.0)"
        )
        
        # Add metadata for interactive features
        if not hasattr(fig.layout, 'metadata'):
            fig.layout.metadata = {}
            
        # Add drill-down capabilities
        fig.layout.metadata['drill_down'] = {
            'callback_data': {
                'chart_type': f"Price Sensitivity: {formatted_product}",
                'store_id': store,
                'product_id': product,
                'elasticity': elasticity
            },
            'chart_title': f"Price Sensitivity for {formatted_product}"
        }
        
        return fig
    
    # Price optimization chart callback
    @app.callback(
        Output("price-optimization-chart", "figure"),
        [
            Input("pricing-store-dropdown", "value"),
            Input("pricing-product-dropdown", "value"),
            Input("show-item-numbers-store", "data")
        ]
    )
    def update_price_optimization_chart(store, product, show_item_numbers):
        """Update the price optimization chart based on selections"""
        price_elasticities = data_dict.get('price_elasticities')
        price_recommendations = data_dict.get('price_recommendations')
        
        if (price_elasticities is None or price_recommendations is None or 
            store is None or product is None):
            return go.Figure()
        
        # Filter by store and product
        elasticity_data = price_elasticities[
            (price_elasticities['Store_Id'] == store) &
            (price_elasticities['Item'] == product)
        ]
        
        recommendation_data = price_recommendations[
            (price_recommendations['Store_Id'] == store) &
            (price_recommendations['Item'] == product)
        ]
        
        if len(elasticity_data) == 0 or len(recommendation_data) == 0:
            return go.Figure()
        
        # Get data values
        elasticity = elasticity_data['Elasticity'].iloc[0]
        product_name = elasticity_data['Product'].iloc[0]
        current_price = elasticity_data['Avg_Price'].iloc[0]
        cost = elasticity_data['Cost'].iloc[0]
        optimal_price = recommendation_data['Optimal_Price'].iloc[0]
        
        # Format product name
        formatted_product = format_product_name(product_name, product, show_item_numbers)
        
        # Create price range (70% to 130% of current price)
        price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
        
        # Calculate profit at each price point
        profit = []
        for price in price_range:
            price_ratio = price / current_price
            quantity_ratio = price_ratio ** elasticity
            profit_value = (price - cost) * quantity_ratio
            profit.append(profit_value)
        
        # Normalize profit (current profit = 1.0)
        current_profit = (current_price - cost)
        profit_normalized = [p / current_profit for p in profit]
        
        # Create figure
        fig = go.Figure()
        
        # Add profit curve
        fig.add_trace(go.Scatter(
            x=price_range,
            y=profit_normalized,
            mode='lines',
            name='Relative Profit'
        ))
        
        # Add reference lines for current price
        fig.add_shape(
            type="line",
            x0=current_price,
            y0=0,
            x1=current_price,
            y1=max(profit_normalized),
            line=dict(color="blue", width=1, dash="dash"),
            name="Current Price"
        )
        
        # Add reference line for optimal price
        fig.add_shape(
            type="line",
            x0=optimal_price,
            y0=0,
            x1=optimal_price,
            y1=max(profit_normalized),
            line=dict(color="green", width=1, dash="dash"),
            name="Optimal Price"
        )
        
        # Set chart title and labels
        fig.update_layout(
            title=f"Profit Optimization for {formatted_product} at Store {store}",
            xaxis_title="Price ($)",
            yaxis_title="Relative Profit (Current = 1.0)"
        )
        
        # Add annotations
        fig.add_annotation(
            x=current_price,
            y=1,
            text="Current Price",
            showarrow=True,
            arrowhead=1
        )
        
        fig.add_annotation(
            x=optimal_price,
            y=profit_normalized[np.argmin(np.abs(price_range - optimal_price))],
            text="Optimal Price",
            showarrow=True,
            arrowhead=1
        )
        
        # Add metadata for interactive features
        if not hasattr(fig.layout, 'metadata'):
            fig.layout.metadata = {}
            
        # Add drill-down capabilities
        fig.layout.metadata['drill_down'] = {
            'callback_data': {
                'chart_type': f"Profit Optimization: {formatted_product}",
                'store_id': store,
                'product_id': product,
                'optimal_price': optimal_price,
                'current_price': current_price,
                'elasticity': elasticity
            },
            'chart_title': f"Profit Optimization for {formatted_product}"
        }
        
        # Add crossfilter capabilities to connect with elasticity chart
        fig.layout.metadata['crossfilter'] = {
            'chart_id': 'price-optimization-chart',
            'linked_charts': ['elasticity-chart'],
            'filter_dimensions': {'price': ['elasticity-chart']}
        }
        
        return fig
    
    # Price recommendations callback
    @app.callback(
        Output("price-recommendations", "children"),
        [
            Input("pricing-store-dropdown", "value"),
            Input("pricing-product-dropdown", "value"),
            Input("show-item-numbers-store", "data")
        ]
    )
    def update_price_recommendations(store, product, show_item_numbers):
        """Update the price recommendations based on selections"""
        try:
            price_recommendations = data_dict.get('price_recommendations')
            profit_impact = data_dict.get('profit_impact')
            
            if price_recommendations is None:
                logger.warning("No price recommendations data available")
                return html.P("No price recommendations available")
            
            if store is None or product is None:
                logger.warning(f"Missing required parameters for price recommendations: store={store}, product={product}")
                return html.P("Please select a store and product to view recommendations")
            
            # Filter by store and product
            recommendation_data = price_recommendations[
                (price_recommendations['Store_Id'] == store) &
                (price_recommendations['Item'] == product)
            ]
            
            if len(recommendation_data) == 0:
                logger.warning(f"No price recommendations found for store {store}, product {product}")
                return html.P("No price recommendations available for this product")
        except Exception as e:
            logger.error(f"Error preparing price recommendations: {str(e)}")
            return html.P(f"Error loading recommendations: {str(e)}")
        
        # Get impact data if available
        impact_data = None
        if profit_impact is not None:
            impact_data = profit_impact[
                (profit_impact['Store_Id'] == store) &
                (profit_impact['Item'] == product)
            ]
        
        # Create recommendation card
        rec = recommendation_data.iloc[0]
        
        # Create card components
        components = [
            html.H5(f"Price Recommendation: {rec['Recommendation']}"),
            html.P(f"Current Price: ${rec['Avg_Price']:.2f}"),
            html.P(f"Optimal Price: ${rec['Optimal_Price']:.2f}"),
            html.P(f"Price Change: {rec['Price_Change_Pct']:.1f}%"),
            html.P(f"Profit Improvement: {rec['Profit_Improvement_Pct']:.1f}%"),
            html.P(f"Current Margin: {rec['Current_Margin']*100:.1f}%"),
            html.P(f"New Margin: {rec['Optimal_Margin']*100:.1f}%"),
        ]
        
        # Add profit impact if available
        if impact_data is not None and len(impact_data) > 0:
            impact = impact_data.iloc[0]
            components.extend([
                html.Hr(),
                html.H5("Profit Impact"),
                html.P(f"Daily Profit Impact: ${impact['Daily_Profit_Impact']:.2f}"),
                html.P(f"Total Profit Impact: ${impact['Total_Profit_Difference']:.2f}"),
                html.P(f"Profit Change: {impact['Profit_Change_Pct']:.1f}%")
            ])
        
        # Create card
        return dbc.Card(dbc.CardBody(components), className="mb-4")
    
    # Debug button callback
    @app.callback(
        Output("debug-output", "children"),
        [Input("debug-button", "n_clicks"),
         Input("fallback-dir-input", "value")]
    )
    def show_debug_info(n_clicks, fallback_dir):
        """Show debug information about loaded data"""
        if not n_clicks:
            return ""
            
        # Create debug info
        debug_info = []
        debug_info.append(html.H5("Loaded Data"))
        
        # Show basic data info
        for name, df in data_dict.items():
            if df is not None:
                row_count = len(df)
                col_count = df.shape[1]
                
                # Determine status color
                if row_count == 0:
                    status_class = "text-danger"
                    status = "Empty"
                elif row_count < 10 and name not in ['inventory_recs', 'price_recommendations']:
                    status_class = "text-warning"
                    status = "Limited"
                else:
                    status_class = "text-success"
                    status = "OK"
                
                info = html.P([
                    f"{name}: ",
                    html.Span(f"{row_count} rows, {col_count} columns", className=status_class),
                    f" - {status}"
                ])
            else:
                info = html.P([
                    f"{name}: ",
                    html.Span("Not loaded", className="text-danger")
                ])
            debug_info.append(info)
        
        # Add environment info
        debug_info.append(html.Hr())
        debug_info.append(html.H5("Environment Info"))
        
        # Show data file paths
        debug_info.append(html.P(f"COMBINED_DATA_FILE: {COMBINED_DATA_FILE}"))
        debug_info.append(html.P(f"PYTORCH_FORECASTS_FILE: {PYTORCH_FORECASTS_FILE}"))
        debug_info.append(html.P(f"RF_FORECASTS_FILE: {RF_FORECASTS_FILE}"))
        debug_info.append(html.P(f"PRICE_ELASTICITIES_FILE: {PRICE_ELASTICITIES_FILE}"))
        debug_info.append(html.P(f"Current fallback directory: {fallback_dir if fallback_dir else 'None'}"))
        
        # Add cache status
        debug_info.append(html.Hr())
        debug_info.append(html.H5("Cache Status"))
        cached_items = len(_DATA_CACHE) if '_DATA_CACHE' in globals() else 0
        debug_info.append(html.P(f"Items in cache: {cached_items}"))
        
        # Add weather service status
        debug_info.append(html.Hr())
        debug_info.append(html.H5("Services"))
        weather_status = "Active" if hasattr(app, 'weather_service') and app.weather_service is not None else "Disabled"
        debug_info.append(html.P(f"Weather service: {weather_status}"))
        
        return html.Div(debug_info)
    
    # Weather update callback
    @app.callback(
        [
            Output("weather-data-store", "data"),
            Output("weather-status", "children"),
            Output("current-weather", "children")
        ],
        Input("update-weather-button", "n_clicks"),
        State("zipcode-input", "value")
    )
    def update_weather(n_clicks, zipcode):
        """Update weather data"""
        if not n_clicks or app.weather_service is None:
            return None, "", ""
            
        try:
            # Get current weather
            weather = app.weather_service.get_current_weather(zipcode)
            
            # Store weather data
            weather_data = {
                'zipcode': zipcode,
                'condition': weather['condition'],
                'temperature_f': weather['temperature_f'],
                'timestamp': weather['timestamp']
            }
            
            # Create weather status message
            status = html.Div([
                html.P(f"Weather updated: {weather['condition']}"),
                html.P(f"Temperature: {weather['temperature_f']:.1f}°F"),
                html.P(f"As of: {weather['timestamp']}")
            ])
            
            # Create weather header display
            header_weather = html.Span([
                f"{weather['condition']} {weather['temperature_f']:.1f}°F"
            ])
            
            return weather_data, status, header_weather
            
        except Exception as e:
            logger.error(f"Error updating weather: {str(e)}")
            return None, html.P(f"Error: {str(e)}"), ""


def create_dashboard(data=None, use_weather=True, fallback_root=None):
    """
    Create and configure the dashboard application.
    
    Args:
        data: Dictionary with loaded data (will be loaded if None)
        use_weather: Whether to use weather service
        fallback_root: Path to directory with fallback data files
        
    Returns:
        dash.Dash: Configured dashboard application
    """
    # Create the app and load data
    app, data_dict = create_app(data, use_weather, fallback_root)
    
    # Configure app for exports
    try:
        # Make sure there are dummy components for the export functionality
        if not hasattr(app, 'layout') or app.layout is None:
            app.layout = html.Div([
                html.Div(id='dummy-output'),
                html.Div(id='dummy-input')
            ])
        app = configure_dash_app_for_exports(app)
        logger.info("Successfully configured Dash app for exports")
    except Exception as e:
        logger.error(f"Failed to configure Dash app for exports: {str(e)}")
        # Continue without export functionality rather than crashing the app
    
    # Set the layout
    app.layout = create_dashboard_layout(data_dict)
    
    # Register callbacks
    register_callbacks(app, data_dict)
    
    # Register interactive features callbacks
    register_interactive_callbacks(app)
    
    # Register item statistics callbacks
    register_item_stats_callbacks(app, data_dict)
    
    # Register enhanced pricing callbacks
    register_pricing_callbacks(app, data_dict)
    
    # Register business impact callbacks
    register_business_impact_callbacks(app, data_dict)
    
    # Register integrated view callbacks
    register_integrated_callbacks(app, data_dict)
    
    # Add debug console logging
    app.clientside_callback(
        """
        function(msg) {
            console.log("Dashboard initialized", {
                'version': '1.0',
                'timestamp': new Date().toISOString(),
                'debug': true
            });
            
            // Create global console logger for the dashboard
            window.dashboardLogger = {
                log: function(message, data) {
                    console.log(`[Dashboard] ${message}`, data || '');
                },
                warn: function(message, data) {
                    console.warn(`[Dashboard] ${message}`, data || '');
                },
                error: function(message, data) {
                    console.error(`[Dashboard] ${message}`, data || '');
                },
                info: function(message, data) {
                    console.info(`[Dashboard] ${message}`, data || '');
                },
                group: function(name) {
                    console.group(`[Dashboard] ${name}`);
                },
                groupEnd: function() {
                    console.groupEnd();
                }
            };
            
            // Log browser information for debugging
            window.dashboardLogger.group('Environment Info');
            window.dashboardLogger.log('User Agent', navigator.userAgent);
            window.dashboardLogger.log('Window Dimensions', `${window.innerWidth}x${window.innerHeight}`);
            window.dashboardLogger.log('Device Pixel Ratio', window.devicePixelRatio);
            window.dashboardLogger.log('Plotly Version', window.Plotly ? window.Plotly.version : 'Not Available');
            window.dashboardLogger.groupEnd();
            
            return window.dash_clientside.no_update;
        }
        """,
        Output("console-log-trigger", "children"),
        Input("console-log-trigger", "id"),
    )
    
    # Add callback to toggle intro section
    @app.callback(
        Output("dashboard-intro", "style"),
        [Input("main-tabs", "active_tab")]
    )
    def toggle_intro(active_tab):
        """Hide intro after first tab selection"""
        if active_tab != "tab-forecast":
            # Hide intro when user navigates to other tabs
            return {"display": "none"}
        return {"display": "block"}
    
    # Add callback for reloading data
    @app.callback(
        Output("data-reload-status", "children"),
        [Input("reload-data-button", "n_clicks"),
         State("fallback-dir-input", "value")]
    )
    def reload_data(n_clicks, fallback_dir):
        if not n_clicks:
            return ""
            
        try:
            # Try to use the specified fallback directory
            fallback = pathlib.Path(fallback_dir) if fallback_dir else None
            if fallback and not fallback.exists():
                return html.P(f"Fallback directory not found: {fallback_dir}", className="text-danger")
            
            # Reload data
            nonlocal data_dict
            from ui.core import load_dashboard_data, clear_data_cache
            
            # Clear cache and reload data
            clear_data_cache()
            new_data = load_dashboard_data(reload=True, fallback_root=fallback)
            
            # Update data_dict with new data
            for key, value in new_data.items():
                if key in data_dict:
                    data_dict[key] = value
            
            return html.Div([
                html.P("Data reloaded successfully!", className="text-success"),
                html.P(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            ])
            
        except Exception as e:
            logger.error(f"Error reloading data: {str(e)}", exc_info=True)
            return html.P(f"Error: {str(e)}", className="text-danger")
    
    return app


def check_data_status(data_dict):
    """
    Check the status of loaded data and return an indicator.
    
    Args:
        data_dict: Dictionary with all loaded data
        
    Returns:
        html.Div: Status indicator element
    """
    # Define critical datasets
    critical_datasets = {
        'combined_data': 'Historical Data',
        'forecasts': 'Forecast Data',
        'price_elasticities': 'Price Data'
    }
    
    # Check if critical datasets are loaded
    missing = [name for key, name in critical_datasets.items() 
               if (key not in data_dict or data_dict[key] is None or len(data_dict[key]) == 0) 
               and not (key == 'forecasts' and ('pytorch_forecasts' in data_dict or 'rf_forecasts' in data_dict))]
    
    if not missing:
        return html.Span(
            [html.I(className="fas fa-check-circle mr-1", style={"color": "#28a745"}), "Data OK"],
            className="text-light"
        )
    else:
        return html.Span(
            [html.I(className="fas fa-exclamation-circle mr-1", style={"color": "#ffc107"}), 
             f"Missing: {', '.join(missing)}"],
            className="text-warning"
        )


def main():
    """
    Main function to run the dashboard.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Pizza Predictive Ordering Dashboard')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the dashboard on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the dashboard on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--no-weather', action='store_true', help='Disable weather service')
    parser.add_argument('--fallback-dir', type=str, help='Directory with fallback data files')
    
    args = parser.parse_args()
    
    # Process fallback directory if provided
    fallback_dir = None
    if args.fallback_dir:
        fallback_dir = pathlib.Path(args.fallback_dir)
        if not fallback_dir.exists():
            logger.warning(f"Fallback directory {fallback_dir} does not exist")
            fallback_dir = None
    
    # Create the dashboard
    try:
        app = create_dashboard(use_weather=not args.no_weather, fallback_root=fallback_dir)
        
        # Run the server - always bind to 0.0.0.0 to ensure it's accessible from other machines
        host = '0.0.0.0'  # Force binding to all network interfaces
        logger.info(f"Starting dashboard on {host}:{args.port} (will be accessible from any network interface)")
        app.run(host=host, port=args.port, debug=args.debug)
    except Exception as e:
        logger.critical(f"Critical error starting dashboard: {e}")
        traceback.print_exc()
        print(f"\nERROR: Failed to start dashboard: {e}\n")
        print("Check log for details.")


def run_dashboard():
    """
    Run the dashboard as a standalone application.
    This function is used when the dashboard is run directly.
    """
    return main()

if __name__ == '__main__':
    run_dashboard()