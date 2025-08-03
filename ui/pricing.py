"""
Price optimization UI module for the Pizza Predictive Ordering System.
"""
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core UI components
from ui.core import (
    load_dashboard_data, create_app, format_product_name,
    create_date_range_slider, create_store_product_selectors,
    create_toggle_switch, create_error_message, create_info_card
)

# Import settings
from config.settings import (
    MAX_PRICE_INCREASE, MAX_PRICE_DECREASE, 
    MIN_MARGIN, ELASTICITY_CONSTRAINT
)


def create_pricing_tab_content(data_dict):
    """
    Create enhanced content for the pricing tab with elasticity constraints.
    
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
    
    # Create the pricing tab content
    return html.Div([
        # Controls
        dbc.Row([
            dbc.Col([
                # Store and product selectors
                create_store_product_selectors('pricing', store_options, product_options)
            ], width=12)
        ]),
        
        # Optimization parameter controls
        dbc.Row([
            dbc.Col([
                html.H4("Optimization Parameters"),
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Max Price Increase (%):"),
                                dbc.Input(
                                    id="max-price-increase",
                                    type="number",
                                    min=1,
                                    max=50,
                                    step=1,
                                    value=MAX_PRICE_INCREASE
                                )
                            ], width=3),
                            dbc.Col([
                                html.Label("Max Price Decrease (%):"),
                                dbc.Input(
                                    id="max-price-decrease",
                                    type="number",
                                    min=1,
                                    max=50,
                                    step=1,
                                    value=MAX_PRICE_DECREASE
                                )
                            ], width=3),
                            dbc.Col([
                                html.Label("Min Margin (%):"),
                                dbc.Input(
                                    id="min-margin",
                                    type="number",
                                    min=5,
                                    max=60,
                                    step=1,
                                    value=MIN_MARGIN
                                )
                            ], width=3),
                            dbc.Col([
                                html.Label("Elasticity Constraint (%):"),
                                dbc.Input(
                                    id="elasticity-constraint",
                                    type="number",
                                    min=1,
                                    max=20,
                                    step=0.5,
                                    value=ELASTICITY_CONSTRAINT
                                )
                            ], width=3)
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Elasticity Threshold:"),
                                dbc.Input(
                                    id="elasticity-threshold",
                                    type="number",
                                    min=0.5,
                                    max=3.0,
                                    step=0.1,
                                    value=1.5,
                                    tooltip="Products with elasticity magnitude above this threshold will use tighter constraints"
                                )
                            ], width=3),
                            dbc.Col([
                                html.Button(
                                    "Apply Parameters",
                                    id="apply-price-params-button",
                                    className="btn btn-primary mt-4"
                                )
                            ], width=3),
                            dbc.Col([
                                html.Div(id="optimization-status", className="mt-4")
                            ], width=6)
                        ])
                    ])
                ], className="mb-4")
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
                            style={'height': '40vh'}
                        )
                    ]
                )
            ], width=12)
        ], className="mb-4"),
        
        # Price sensitivity visualization
        dbc.Row([
            dbc.Col([
                html.H4("Price Sensitivity Analysis"),
                dcc.Loading(
                    id="loading-price-sensitivity",
                    type="circle",
                    children=[
                        dcc.Graph(
                            id="price-sensitivity-chart",
                            figure=go.Figure(),
                            style={'height': '40vh'}
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
                            style={'height': '40vh'}
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


def register_pricing_callbacks(app, data_dict):
    """
    Register callbacks for the pricing tab.
    
    Args:
        app: Dash app instance
        data_dict: Dictionary with all loaded data
    """
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
                return go.Figure()
            
            if store is None or product is None:
                return go.Figure()
            
            # Filter by store and product
            elasticity_data = price_elasticities[
                (price_elasticities['Store_Id'] == store) &
                (price_elasticities['Item'] == product)
            ]
            
            if len(elasticity_data) == 0:
                return go.Figure()
        except Exception as e:
            return go.Figure()
        
        # Get elasticity value and product info
        elasticity = elasticity_data['Elasticity'].iloc[0]
        product_name = elasticity_data['Product'].iloc[0]
        current_price = elasticity_data['Current_Price'].iloc[0]
        r_squared = elasticity_data['R_Squared'].iloc[0] if 'R_Squared' in elasticity_data.columns else 0
        
        # Format product name
        formatted_product = format_product_name(product_name, product, show_item_numbers)
        
        # Create elasticity distribution plot
        # First, get all elasticities for this product across stores
        all_product_elasticities = price_elasticities[
            price_elasticities['Item'] == product
        ]['Elasticity'].values
        
        fig = go.Figure()
        
        # Add histogram of elasticities
        fig.add_trace(go.Histogram(
            x=all_product_elasticities,
            opacity=0.7,
            name='Elasticity Distribution',
            nbinsx=15,
            marker=dict(color='blue')
        ))
        
        # Add vertical line for this store's elasticity
        fig.add_shape(
            type="line",
            x0=elasticity,
            y0=0,
            x1=elasticity,
            y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash")
        )
        
        # Add annotation for this store's elasticity
        fig.add_annotation(
            x=elasticity,
            y=0.95,
            yref="paper",
            text=f"Store {store} Elasticity: {elasticity:.2f}",
            showarrow=True,
            arrowhead=1
        )
        
        # Set chart title and labels
        fig.update_layout(
            title=f"Price Elasticity for {formatted_product} (R² = {r_squared:.2f})",
            xaxis_title="Elasticity Value",
            yaxis_title="Count",
            bargap=0.05,
            showlegend=False
        )
        
        # Add elasticity interpretation text
        interpretation = ""
        if elasticity >= -0.5:
            interpretation = "Inelastic: Demand is not very sensitive to price changes"
        elif elasticity >= -1.0:
            interpretation = "Unit Elastic: Demand changes proportionally to price"
        elif elasticity >= -1.5:
            interpretation = "Elastic: Demand is sensitive to price changes"
        else:
            interpretation = "Highly Elastic: Demand is very sensitive to price changes"
        
        fig.add_annotation(
            x=0.5,
            y=1.05,
            xref="paper",
            yref="paper",
            text=interpretation,
            showarrow=False,
            font=dict(size=14)
        )
        
        return fig
    
    # Price sensitivity chart callback
    @app.callback(
        Output("price-sensitivity-chart", "figure"),
        [
            Input("pricing-store-dropdown", "value"),
            Input("pricing-product-dropdown", "value"),
            Input("show-item-numbers-store", "data")
        ]
    )
    def update_price_sensitivity_chart(store, product, show_item_numbers):
        """Update the price sensitivity chart based on selections"""
        try:
            price_elasticities = data_dict.get('price_elasticities')
            
            if price_elasticities is None:
                return go.Figure()
            
            if store is None or product is None:
                return go.Figure()
            
            # Filter by store and product
            elasticity_data = price_elasticities[
                (price_elasticities['Store_Id'] == store) &
                (price_elasticities['Item'] == product)
            ]
            
            if len(elasticity_data) == 0:
                return go.Figure()
        except Exception as e:
            return go.Figure()
        
        # Get elasticity value and product info
        elasticity = elasticity_data['Elasticity'].iloc[0]
        product_name = elasticity_data['Product'].iloc[0]
        current_price = elasticity_data['Current_Price'].iloc[0]
        
        # Format product name
        formatted_product = format_product_name(product_name, product, show_item_numbers)
        
        # Create price range (70% to 130% of current price)
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
            name=f'Demand (e={elasticity:.2f})',
            line=dict(color='blue', width=3)
        ))
        
        # Add reference line for current price
        fig.add_shape(
            type="line",
            x0=current_price,
            y0=0,
            x1=current_price,
            y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash")
        )
        
        # Add reference line for current quantity
        fig.add_shape(
            type="line",
            x0=price_range[0],
            y0=1,
            x1=price_range[-1],
            y1=1,
            line=dict(color="green", width=2, dash="dash")
        )
        
        # Add shaded areas for elasticity constraints
        elasticity_constraint = ELASTICITY_CONSTRAINT / 100  # Convert percentage to decimal
        lower_bound = current_price * (1 - elasticity_constraint)
        upper_bound = current_price * (1 + elasticity_constraint)
        
        # Add shaded area for constraint bounds
        x_fill = list(price_range)
        y_fill = [0] * len(price_range)
        
        # Find indices for the constraint bounds
        lower_idx = np.argmin(np.abs(price_range - lower_bound))
        upper_idx = np.argmin(np.abs(price_range - upper_bound))
        
        # Add vertical lines for constraints
        fig.add_shape(
            type="line",
            x0=lower_bound,
            y0=0,
            x1=lower_bound,
            y1=1,
            yref="paper",
            line=dict(color="purple", width=2, dash="dot")
        )
        
        fig.add_shape(
            type="line",
            x0=upper_bound,
            y0=0,
            x1=upper_bound,
            y1=1,
            yref="paper",
            line=dict(color="purple", width=2, dash="dot")
        )
        
        # Add annotations
        fig.add_annotation(
            x=current_price,
            y=1.05,
            text=f"Current Price: ${current_price:.2f}",
            showarrow=True,
            arrowhead=1,
            arrowcolor="red"
        )
        
        fig.add_annotation(
            x=lower_bound,
            y=0.9,
            text=f"-{ELASTICITY_CONSTRAINT}% Bound",
            showarrow=True,
            arrowhead=1,
            arrowcolor="purple"
        )
        
        fig.add_annotation(
            x=upper_bound,
            y=0.8,
            text=f"+{ELASTICITY_CONSTRAINT}% Bound",
            showarrow=True,
            arrowhead=1,
            arrowcolor="purple"
        )
        
        # Set chart title and labels
        fig.update_layout(
            title=f"Price Sensitivity for {formatted_product} at Store {store}",
            xaxis_title="Price ($)",
            yaxis_title="Relative Demand (Current = 1.0)"
        )
        
        return fig
    
    # Price optimization chart callback
    @app.callback(
        Output("price-optimization-chart", "figure"),
        [
            Input("pricing-store-dropdown", "value"),
            Input("pricing-product-dropdown", "value"),
            Input("elasticity-constraint", "value"),
            Input("show-item-numbers-store", "data")
        ]
    )
    def update_price_optimization_chart(store, product, elasticity_constraint, show_item_numbers):
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
        current_price = elasticity_data['Current_Price'].iloc[0]
        cost = elasticity_data['Cost'].iloc[0]
        optimal_price = recommendation_data['Optimal_Price'].iloc[0]
        
        # Format product name
        formatted_product = format_product_name(product_name, product, show_item_numbers)
        
        # Create price range (70% to 130% of current price)
        price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
        
        # Calculate profit at each price point
        profit = []
        revenue = []
        demand = []
        
        for price in price_range:
            price_ratio = price / current_price
            quantity_ratio = price_ratio ** elasticity
            profit_value = (price - cost) * quantity_ratio
            profit.append(profit_value)
            revenue.append(price * quantity_ratio)
            demand.append(quantity_ratio)
        
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
            name='Profit',
            line=dict(color='blue', width=3)
        ))
        
        # Add revenue curve
        current_revenue = current_price
        revenue_normalized = [r / current_revenue for r in revenue]
        
        fig.add_trace(go.Scatter(
            x=price_range,
            y=revenue_normalized,
            mode='lines',
            name='Revenue',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        # Add demand curve
        fig.add_trace(go.Scatter(
            x=price_range,
            y=demand,
            mode='lines',
            name='Demand',
            line=dict(color='red', width=2, dash='dot')
        ))
        
        # Add reference lines for current price
        fig.add_shape(
            type="line",
            x0=current_price,
            y0=0,
            x1=current_price,
            y1=max(profit_normalized) * 1.1,
            line=dict(color="gray", width=2, dash="dash")
        )
        
        # Add reference line for optimal price
        fig.add_shape(
            type="line",
            x0=optimal_price,
            y0=0,
            x1=optimal_price,
            y1=max(profit_normalized) * 1.1,
            line=dict(color="green", width=2, dash="dash")
        )
        
        # Add elasticity constraint bounds
        if elasticity_constraint is not None:
            elasticity_constraint_decimal = elasticity_constraint / 100
            lower_bound = current_price * (1 - elasticity_constraint_decimal)
            upper_bound = current_price * (1 + elasticity_constraint_decimal)
            
            fig.add_shape(
                type="rect",
                x0=lower_bound,
                y0=0,
                x1=upper_bound,
                y1=max(profit_normalized) * 1.1,
                fillcolor="rgba(128, 0, 128, 0.1)",
                line=dict(color="purple", width=1, dash="dot"),
                name="Elasticity Constraint"
            )
        
        # Set chart title and labels
        fig.update_layout(
            title=f"Profit Optimization for {formatted_product} at Store {store}",
            xaxis_title="Price ($)",
            yaxis_title="Relative Value (Current = 1.0)"
        )
        
        # Add annotations
        fig.add_annotation(
            x=current_price,
            y=1,
            text="Current Price",
            showarrow=True,
            arrowhead=1
        )
        
        optimal_profit_idx = np.argmin(np.abs(price_range - optimal_price))
        fig.add_annotation(
            x=optimal_price,
            y=profit_normalized[optimal_profit_idx],
            text="Optimal Price",
            showarrow=True,
            arrowhead=1
        )
        
        # Add elasticity constraint annotation
        if elasticity_constraint is not None:
            fig.add_annotation(
                x=(lower_bound + upper_bound) / 2,
                y=max(profit_normalized) * 0.5,
                text=f"Elasticity\nConstraint\n±{elasticity_constraint}%",
                showarrow=False,
                font=dict(color="purple")
            )
        
        return fig
    
    # Price recommendations callback with updated constraints
    @app.callback(
        [Output("price-recommendations", "children"),
         Output("optimization-status", "children")],
        [Input("apply-price-params-button", "n_clicks"),
         Input("pricing-store-dropdown", "value"),
         Input("pricing-product-dropdown", "value")],
        [State("max-price-increase", "value"),
         State("max-price-decrease", "value"),
         State("min-margin", "value"),
         State("elasticity-constraint", "value"),
         State("elasticity-threshold", "value"),
         State("show-item-numbers-store", "data")]
    )
    def update_price_recommendations(n_clicks, store, product, 
                                    max_increase, max_decrease, min_margin, 
                                    elasticity_constraint, elasticity_threshold,
                                    show_item_numbers):
        """Update the price recommendations based on selections and parameters"""
        ctx = callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        
        # Initialize status
        status = html.Div("")
        
        # Get base recommendation data
        price_elasticities = data_dict.get('price_elasticities')
        price_recommendations = data_dict.get('price_recommendations')
        profit_impact = data_dict.get('profit_impact')
        
        if price_elasticities is None or price_recommendations is None:
            return html.P("No price recommendations available"), status
        
        if store is None or product is None:
            return html.P("Please select a store and product to view recommendations"), status
        
        # Default to the base recommendation
        recommendation_data = price_recommendations[
            (price_recommendations['Store_Id'] == store) &
            (price_recommendations['Item'] == product)
        ]
        
        if len(recommendation_data) == 0:
            return html.P("No price recommendations available for this product"), status
        
        # If the button was clicked, recalculate with new parameters
        if trigger_id == "apply-price-params-button" and n_clicks is not None and n_clicks > 0:
            # Get the selected product's elasticity data
            elasticity_data = price_elasticities[
                (price_elasticities['Store_Id'] == store) &
                (price_elasticities['Item'] == product)
            ].copy()
            
            if len(elasticity_data) > 0:
                try:
                    # Import the optimizer
                    from src.optimization.profit_optimizer import ProfitOptimizer
                    
                    # Create custom constraints
                    constraints = {
                        'max_price_increase': max_increase,
                        'max_price_decrease': max_decrease,
                        'min_margin': min_margin,
                        'elasticity_constraint': elasticity_constraint
                    }
                    
                    # Create optimizer instance
                    optimizer = ProfitOptimizer()
                    
                    # Run optimization for just this product with custom constraints
                    custom_recommendation = optimizer.optimize_prices(elasticity_data, constraints)
                    
                    # Update the recommendation data
                    recommendation_data = custom_recommendation
                    
                    # Update status
                    status = html.Div([
                        html.P("Optimization parameters applied successfully!", className="text-success"),
                        html.P([
                            "Using: ",
                            html.Span(f"Max Increase: {max_increase}%, "),
                            html.Span(f"Max Decrease: {max_decrease}%, "),
                            html.Span(f"Min Margin: {min_margin}%, "),
                            html.Span(f"Elasticity Constraint: {elasticity_constraint}%, "),
                            html.Span(f"Elasticity Threshold: {elasticity_threshold}")
                        ], className="text-info")
                    ])
                except Exception as e:
                    status = html.Div([
                        html.P("Error applying optimization parameters!", className="text-danger"),
                        html.P(str(e), className="text-danger")
                    ])
        
        # Create recommendation card
        rec = recommendation_data.iloc[0]
        
        # Create card components
        components = [
            html.H5(f"Price Recommendation: {rec['Recommendation']}"),
            html.P(f"Current Price: ${rec['Current_Price']:.2f}"),
            html.P(f"Optimal Price: ${rec['Optimal_Price']:.2f}"),
            html.P(f"Price Change: {rec['Price_Change_Pct']:.1f}%"),
            html.P(f"Profit Improvement: {rec['Expected_Profit_Change_Pct']:.1f}%"),
            html.P(f"Expected Sales Change: {rec['Expected_Sales_Change_Pct']:.1f}%"),
            html.P(f"Current Margin: {rec['Current_Margin_Pct']*100 if 'Current_Margin_Pct' in rec else 0:.1f}%"),
            html.P(f"New Margin: {rec['New_Margin_Pct']*100 if 'New_Margin_Pct' in rec else 0:.1f}%"),
            html.Hr(),
            html.H6("Price Elasticity Impact:"),
            html.P(f"Elasticity: {rec['Elasticity']:.2f}")
        ]
        
        # Add constraint information if available
        if 'Constraint_Applied' in rec:
            constraint_msg = rec['Constraint_Applied']
            if constraint_msg == "elasticity":
                constraint_text = "Elasticity constraint was applied due to high price sensitivity"
            elif constraint_msg == "elasticity_lower_bound":
                constraint_text = "Limited to elasticity constraint lower bound"
            elif constraint_msg == "elasticity_upper_bound":
                constraint_text = "Limited to elasticity constraint upper bound"
            elif constraint_msg == "margin":
                constraint_text = "Limited by minimum margin requirement"
            elif "conflict" in constraint_msg:
                constraint_text = "Constraints conflict - could not optimize"
            else:
                constraint_text = f"Constraint type: {constraint_msg}"
                
            components.append(html.P(f"Constraint: {constraint_text}", className="font-italic"))
        
        # Add profit impact if available
        if profit_impact is not None and len(profit_impact) > 0:
            impact_data = profit_impact[
                (profit_impact['Store_Id'] == store) &
                (profit_impact['Item'] == product)
            ]
            
            if len(impact_data) > 0:
                impact = impact_data.iloc[0]
                components.extend([
                    html.Hr(),
                    html.H6("Profit Impact"),
                    html.P(f"Daily Profit Impact: ${impact['Daily_Profit_Impact']:.2f}"),
                    html.P(f"Total Profit Impact: ${impact['Total_Profit_Difference']:.2f}"),
                    html.P(f"Profit Change: {impact['Profit_Change_Pct']:.1f}%")
                ])
        
        # Create card
        card = dbc.Card(dbc.CardBody(components), className="mb-4")
        
        return card, status


def update_elasticity_visualization():
    """
    Create elasticity visualization for the dashboard.
    """
    pass


def initialize_pricing_module():
    """
    Initialize the pricing module with necessary data.
    """
    pass
