"""
Integrated visualizations for connecting inventory, pricing, and demand forecasting.

This module provides visualization components that show the relationships
between inventory, pricing, and demand forecasting in a unified way.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging

# Import interactive features
from src.models.interactive_features import (
    enhance_integrated_chart,
    enhance_impact_heatmap,
    enhance_kpi_indicators,
    enhance_recommendations_table
)

# Import export utilities
from src.models.chart_export_utils import (
    enhance_figure_with_exports,
    configure_chart_export_buttons
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('integrated_visualizations')

def create_integrated_chart(model, store_id, item_id, price_adjustment=0, 
                          inventory_adjustment=0, forecast_period=None, show_item_numbers=True):
    """
    Create integrated chart showing connections between pricing, inventory, and demand.
    
    Args:
        model: UnifiedDataModel instance containing all data
        store_id: Store ID
        item_id: Product ID
        price_adjustment: Price adjustment percentage
        inventory_adjustment: Inventory adjustment percentage
        forecast_period: Tuple of (start_day, end_day) for forecast period
        show_item_numbers: Whether to show item numbers in product names
        
    Returns:
        plotly.graph_objects.Figure: Integrated chart
    """
    try:
        # Validate inputs
        if model is None:
            logger.error("No data model provided")
            return go.Figure()
            
        try:
            # Convert IDs to strings for consistent comparison
            store_id = str(store_id)
            item_id = str(item_id)
        except Exception as e:
            logger.error(f"Error converting IDs to strings: {str(e)}")
        
        # Apply adjustments through the unified model
        if price_adjustment != 0:
            try:
                model.adjust_price(store_id, item_id, price_adjustment)
            except Exception as e:
                logger.error(f"Error applying price adjustment: {str(e)}")
        
        if inventory_adjustment != 0:
            try:
                model.adjust_inventory(store_id, item_id, inventory_adjustment)
            except Exception as e:
                logger.error(f"Error applying inventory adjustment: {str(e)}")
        
        # Calculate metrics - this will generate fallback metrics if needed
        metrics = model.calculate_metrics(store_id, item_id)
        if not metrics:
            logger.warning(f"No metrics available for store {store_id}, product {item_id}")
            # Create empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for this product",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Get relevant data
        product_data = model.get_product_data(store_id, item_id)
        if not product_data:
            logger.warning(f"No data available for store {store_id}, product {item_id}")
            # We'll continue with metrics-only visualization
            # The metrics should contain fallback data from the model
        
        # Get forecast data
        forecast_data = model.get_data('adjusted_forecasts')
        if forecast_data is None:
            forecast_data = model.get_data('forecasts')
        
        if forecast_data is not None:
            # Make sure Store_Id and Item are strings for comparison
            try:
                forecast_df = forecast_data.copy()
                forecast_df['Store_Id'] = forecast_df['Store_Id'].astype(str)
                forecast_df['Item'] = forecast_df['Item'].astype(str)
                forecast_data = forecast_df[(forecast_df['Store_Id'] == store_id) & 
                                          (forecast_df['Item'] == item_id)]
            except Exception as e:
                logger.error(f"Error filtering forecast data: {str(e)}")
                forecast_data = None
        
        # Check if we have valid forecast data
        if forecast_data is None or len(forecast_data) == 0:
            logger.warning(f"No forecast data available for store {store_id}, product {item_id}")
            # We'll continue with metrics-only visualization if possible
        
        # Get inventory data
        inventory_data = model.get_data('adjusted_inventory')
        if inventory_data is None:
            inventory_data = model.get_data('inventory_projection')
        
        if inventory_data is not None:
            # Make sure Store_Id and Item are strings for comparison
            try:
                inventory_df = inventory_data.copy()
                inventory_df['Store_Id'] = inventory_df['Store_Id'].astype(str)
                inventory_df['Item'] = inventory_df['Item'].astype(str)
                inventory_data = inventory_df[(inventory_df['Store_Id'] == store_id) & 
                                            (inventory_df['Item'] == item_id)]
            except Exception as e:
                logger.error(f"Error filtering inventory data: {str(e)}")
                inventory_data = None
        
        # Check if we have valid inventory data
        if inventory_data is None or len(inventory_data) == 0:
            logger.warning(f"No inventory data available for store {store_id}, product {item_id}")
            # We'll continue with metrics-only visualization if possible
        
        # Get historical data
        historical_data = model.get_data('combined_data')
        if historical_data is not None:
            # Make sure Store_Id and Item are strings for comparison
            try:
                historical_df = historical_data.copy()
                historical_df['Store_Id'] = historical_df['Store_Id'].astype(str)
                historical_df['Item'] = historical_df['Item'].astype(str)
                historical_data = historical_df[(historical_df['Store_Id'] == store_id) & 
                                             (historical_df['Item'] == item_id)]
            except Exception as e:
                logger.error(f"Error filtering historical data: {str(e)}")
                historical_data = None
                
        # Check if we have valid historical data
        if historical_data is None or len(historical_data) == 0:
            logger.warning(f"No historical data available for store {store_id}, product {item_id}")
            # We'll continue with metrics-only visualization if possible
            
        # Get product name with robust error handling
        product_name = f"Product {item_id}"
        
        try:
            if forecast_data is not None and len(forecast_data) > 0 and 'Product' in forecast_data.columns:
                first_name = forecast_data['Product'].iloc[0]
                if first_name is not None and str(first_name).strip() != '':
                    product_name = first_name
            elif inventory_data is not None and len(inventory_data) > 0 and 'Product' in inventory_data.columns:
                first_name = inventory_data['Product'].iloc[0]
                if first_name is not None and str(first_name).strip() != '':
                    product_name = first_name
            elif historical_data is not None and len(historical_data) > 0 and 'Product' in historical_data.columns:
                first_name = historical_data['Product'].iloc[0]
                if first_name is not None and str(first_name).strip() != '':
                    product_name = first_name
        except Exception as e:
            logger.error(f"Error extracting product name: {str(e)}")
            
        # Try to get product name from metrics if all else fails
        if product_name == f"Product {item_id}" and metrics:
            try:
                # Check if price metrics has product name
                if 'price' in metrics and isinstance(metrics['price'], dict):
                    if 'product_name' in metrics['price'] and metrics['price']['product_name'] is not None:
                        product_name = metrics['price']['product_name']
            except Exception:
                pass
        
        # Format the product name
        formatted_product = product_name
        if show_item_numbers:
            formatted_product = f"{product_name} ({item_id})"
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                'Sales & Demand Forecast', 
                'Inventory Level',
                'Price & Elasticity Impact',
                'Profit Impact'
            ),
            row_heights=[0.3, 0.2, 0.2, 0.3]
        )
        
        # Add historical sales to the first subplot
        if historical_data is not None and len(historical_data) > 0:
            # Get recent history (last 30 days)
            recent_hist = historical_data.sort_values('Date').tail(30)
            
            fig.add_trace(
                go.Scatter(
                    x=recent_hist['Date'],
                    y=recent_hist['Sales'],
                    mode='lines',
                    name='Historical Sales',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        # Add forecast to the first subplot
        if forecast_data is not None and len(forecast_data) > 0:
            # Apply date filtering if forecast_period is provided
            if forecast_period:
                start_day, end_day = forecast_period
                min_date = forecast_data['Date'].min()
                forecast_dates = pd.date_range(start=min_date, periods=end_day)
                forecast_dates = forecast_dates[start_day-1:end_day]
                forecast_data = forecast_data[forecast_data['Date'].isin(forecast_dates)]
            
            # Sort by date
            forecast_data = forecast_data.sort_values('Date')
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_data['Date'],
                    y=forecast_data['Forecast'],
                    mode='lines',
                    name='Demand Forecast',
                    line=dict(color='orange', width=3)
                ),
                row=1, col=1
            )
            
            # Add confidence interval if available
            if 'Upper_Bound' in forecast_data.columns and 'Lower_Bound' in forecast_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_data['Date'],
                        y=forecast_data['Upper_Bound'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=forecast_data['Date'],
                        y=forecast_data['Lower_Bound'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255,165,0,0.2)',
                        name='Forecast Confidence',
                        showlegend=True
                    ),
                    row=1, col=1
                )
        
        # Add inventory levels to the second subplot
        if inventory_data is not None and len(inventory_data) > 0:
            # Sort by date
            inventory_data = inventory_data.sort_values('Date')
            
            # Get stock column name
            stock_col = 'Stock_Level' if 'Stock_Level' in inventory_data.columns else 'Current_Stock'
            
            fig.add_trace(
                go.Scatter(
                    x=inventory_data['Date'],
                    y=inventory_data[stock_col],
                    mode='lines',
                    name='Inventory Level',
                    line=dict(color='green', width=3)
                ),
                row=2, col=1
            )
            
            # Add safety stock and target stock reference lines
            if 'safety_stock' in metrics['inventory'] and 'target_stock' in metrics['inventory']:
                safety_stock = metrics['inventory']['safety_stock']
                target_stock = metrics['inventory']['target_stock']
                
                # Get date range
                dates = inventory_data['Date']
                
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=[safety_stock] * len(dates),
                        mode='lines',
                        name='Safety Stock',
                        line=dict(color='red', dash='dash', width=2)
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=[target_stock] * len(dates),
                        mode='lines',
                        name='Target Stock',
                        line=dict(color='green', dash='dash', width=2)
                    ),
                    row=2, col=1
                )
        
        # Add price elasticity impact to the third subplot
        price_metrics = metrics.get('price', {})
        if price_metrics:
            current_price = price_metrics.get('current_price')
            new_price = price_metrics.get('new_price')
            elasticity = price_metrics.get('elasticity')
            
            if current_price is not None and elasticity is not None:
                # Create price range (70% to 130% of current price)
                price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
                
                # Calculate demand response at each price point
                quantity_ratios = []
                for price in price_range:
                    price_ratio = price / current_price
                    quantity_ratio = price_ratio ** elasticity
                    quantity_ratios.append(quantity_ratio)
                
                # Add elasticity curve
                fig.add_trace(
                    go.Scatter(
                        x=price_range,
                        y=quantity_ratios,
                        mode='lines',
                        name=f'Price Elasticity ({elasticity:.2f})',
                        line=dict(color='purple', width=3)
                    ),
                    row=3, col=1
                )
                
                # Add current and new price markers
                fig.add_trace(
                    go.Scatter(
                        x=[current_price],
                        y=[1.0],  # Current quantity ratio is 1.0 by definition
                        mode='markers',
                        name='Current Price',
                        marker=dict(color='blue', size=10)
                    ),
                    row=3, col=1
                )
                
                if new_price != current_price:
                    # Calculate quantity ratio at new price
                    price_ratio = new_price / current_price
                    quantity_ratio = price_ratio ** elasticity
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[new_price],
                            y=[quantity_ratio],
                            mode='markers',
                            name='New Price',
                            marker=dict(color='red', size=10)
                        ),
                        row=3, col=1
                    )
        
        # Add profit impact to the fourth subplot
        integrated_metrics = metrics.get('integrated', {})
        price_change_impact = integrated_metrics.get('price_change_impact', {})
        
        if price_change_impact:
            # Create date range matching forecast
            if forecast_data is not None and len(forecast_data) > 0:
                dates = forecast_data['Date'].sort_values()
                
                # Calculate cumulative original profit
                original_forecast = forecast_data['Forecast'].values
                original_profit = np.array([
                    (price_metrics['current_price'] - price_metrics['cost']) * f
                    for f in original_forecast
                ])
                cum_original_profit = np.cumsum(original_profit)
                
                # Calculate cumulative new profit
                new_profit = np.array([
                    (price_metrics['new_price'] - price_metrics['cost']) * 
                    (f * price_metrics['quantity_ratio'])
                    for f in original_forecast
                ])
                cum_new_profit = np.cumsum(new_profit)
                
                # Add original profit curve
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=cum_original_profit,
                        mode='lines',
                        name='Original Profit',
                        line=dict(color='blue', width=3)
                    ),
                    row=4, col=1
                )
                
                # Add new profit curve
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=cum_new_profit,
                        mode='lines',
                        name='New Profit',
                        line=dict(color='red', width=3)
                    ),
                    row=4, col=1
                )
                
                # Add profit difference as a bar chart
                profit_diff = new_profit - original_profit
                
                fig.add_trace(
                    go.Bar(
                        x=dates,
                        y=profit_diff,
                        name='Profit Difference',
                        marker_color=np.where(profit_diff >= 0, 'green', 'red'),
                        opacity=0.5
                    ),
                    row=4, col=1
                )
        
        # Update layout with improved styling
        title = f"Integrated Business Impact Analysis: {formatted_product} (Store {store_id})"
        
        # Add subtitle with key metrics
        subtitle = ""
        if price_adjustment != 0:
            subtitle += f"Price Adjustment: {price_adjustment:+}% | "
        if inventory_adjustment != 0:
            subtitle += f"Inventory Adjustment: {inventory_adjustment:+}% | "
        
        # Add impact score if available
        if 'business_impact_score' in integrated_metrics:
            score = integrated_metrics['business_impact_score']
            subtitle += f"Business Impact Score: {score:.0f}/100 | "
        
        # Add recommendation if available
        if 'recommendation' in integrated_metrics:
            subtitle += f"Recommendation: {integrated_metrics['recommendation']}"
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=16)
            ),
            height=1600,
            width=1600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
                font=dict(size=12)
            ),
            margin=dict(t=100, b=20, l=60, r=40),
            hovermode="closest",
            annotations=[
                dict(
                    text=subtitle,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=1.01,
                    showarrow=False,
                    font=dict(size=12)
                )
            ]
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Units", row=1, col=1)
        fig.update_yaxes(title_text="Stock Units", row=2, col=1)
        fig.update_yaxes(title_text="Demand Ratio", row=3, col=1)
        fig.update_yaxes(title_text="Cumulative Profit ($)", secondary_y=False, row=4, col=1)
        
        # Update x-axis titles - only show on bottom subplot
        fig.update_xaxes(title_text="", row=1, col=1)
        fig.update_xaxes(title_text="", row=2, col=1)
        fig.update_xaxes(title_text="Price ($)", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=4, col=1)
        
        # Enhance with interactive features
        if fig is not None:
            fig = enhance_integrated_chart(fig, model, store_id, item_id)
        
        # Add export capabilities
        if fig is not None:
            title = f"Integrated_Business_Impact_{store_id}_{item_id}"
            fig = enhance_figure_with_exports(fig, title)
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating integrated chart: {str(e)}")
        return go.Figure()

def create_impact_heatmap(model, store_id, item_id, price_range=(-15, 20), 
                         inventory_range=(-30, 50), steps=10, metric='profit'):
    """
    Create a heatmap showing the impact of different price and inventory adjustments.
    
    Args:
        model: UnifiedDataModel instance containing all data
        store_id: Store ID
        item_id: Product ID
        price_range: Tuple of (min_pct, max_pct) for price adjustment percentages
        inventory_range: Tuple of (min_pct, max_pct) for inventory adjustments
        steps: Number of steps for each dimension
        metric: Metric to display ('profit', 'revenue', 'stockout_risk')
        
    Returns:
        plotly.graph_objects.Figure: Heatmap figure
    """
    try:
        # Generate price and inventory adjustment values
        price_adjustments = np.linspace(price_range[0], price_range[1], steps)
        inventory_adjustments = np.linspace(inventory_range[0], inventory_range[1], steps)
        
        # Initialize results matrix
        results = np.zeros((len(inventory_adjustments), len(price_adjustments)))
        
        # Create original metrics as baseline
        baseline_metrics = model.calculate_metrics(store_id, item_id)
        if not baseline_metrics:
            logger.warning(f"No baseline metrics available for store {store_id}, product {item_id}")
            return go.Figure()
        
        # Get baseline values for comparison
        baseline_profit = 0
        if 'integrated' in baseline_metrics and 'price_change_impact' in baseline_metrics['integrated']:
            baseline_profit = baseline_metrics['integrated']['price_change_impact'].get('original_profit', 0)
        
        baseline_stockout_risk = "Low"
        if 'inventory' in baseline_metrics:
            baseline_stockout_risk = baseline_metrics['inventory'].get('stockout_risk', "Low")
        
        # For each combination, calculate metrics
        for i, inv_adj in enumerate(inventory_adjustments):
            for j, price_adj in enumerate(price_adjustments):
                # Create a fresh model for each combination
                model_copy = type(model)(model.data_dict)
                
                # Apply adjustments
                if price_adj != 0:
                    model_copy.adjust_price(store_id, item_id, price_adj)
                
                if inv_adj != 0:
                    model_copy.adjust_inventory(store_id, item_id, inv_adj)
                
                # Calculate metrics
                metrics = model_copy.calculate_metrics(store_id, item_id)
                
                # Extract relevant metric
                value = 0
                if metric == 'profit':
                    if ('integrated' in metrics and 
                        'price_change_impact' in metrics['integrated']):
                        value = metrics['integrated']['price_change_impact'].get('profit_diff_pct', 0)
                
                elif metric == 'revenue':
                    if ('integrated' in metrics and 
                        'price_change_impact' in metrics['integrated']):
                        forecast_diff = metrics['integrated']['price_change_impact'].get('forecast_diff_pct', 0)
                        price_adj_pct = 1 + price_adj/100
                        # Revenue change is price change * demand change
                        value = price_adj_pct * (1 + forecast_diff/100) - 1
                        value *= 100  # Convert to percentage
                
                elif metric == 'stockout_risk':
                    if 'inventory' in metrics:
                        risk = metrics['inventory'].get('stockout_risk', "Low")
                        # Convert risk to numeric value
                        value = 1 if risk == "High" else 0.5 if risk == "Medium" else 0
                        # Compare to baseline
                        baseline_value = 1 if baseline_stockout_risk == "High" else 0.5 if baseline_stockout_risk == "Medium" else 0
                        value = (value - baseline_value) * 100  # Scale for visualization
                
                elif metric == 'impact_score':
                    if 'integrated' in metrics:
                        value = metrics['integrated'].get('business_impact_score', 50) - 50  # Center at 0
                
                # Store the result
                results[i, j] = value
        
        # Create heatmap
        price_labels = [f"{p:.0f}%" for p in price_adjustments]
        inventory_labels = [f"{i:.0f}%" for i in inventory_adjustments]
        
        # Determine colorscale based on metric
        if metric == 'stockout_risk':
            colorscale = [
                [0, 'green'],  # Improved (reduced risk)
                [0.5, 'white'],  # Neutral
                [1, 'red']  # Worsened (increased risk)
            ]
        elif metric in ['profit', 'revenue', 'impact_score']:
            colorscale = [
                [0, 'red'],  # Negative impact
                [0.5, 'white'],  # Neutral
                [1, 'green']  # Positive impact
            ]
        else:
            colorscale = 'RdBu'
        
        # Determine title and z-axis label based on metric
        if metric == 'profit':
            title = f"Profit Impact (%) - Store {store_id}, Product {item_id}"
            z_label = "Profit Change (%)"
        elif metric == 'revenue':
            title = f"Revenue Impact (%) - Store {store_id}, Product {item_id}"
            z_label = "Revenue Change (%)"
        elif metric == 'stockout_risk':
            title = f"Stockout Risk Impact - Store {store_id}, Product {item_id}"
            z_label = "Risk Change"
        elif metric == 'impact_score':
            title = f"Business Impact Score - Store {store_id}, Product {item_id}"
            z_label = "Impact Score"
        else:
            title = f"Impact Analysis - Store {store_id}, Product {item_id}"
            z_label = "Impact Value"
        
        # Create figure
        fig = go.Figure(data=go.Heatmap(
            z=results,
            x=price_labels,
            y=inventory_labels,
            colorscale=colorscale,
            colorbar=dict(title=z_label)
        ))
        
        # Add annotations to show optimal point
        if metric in ['profit', 'revenue', 'impact_score']:
            # Find the maximum value
            max_idx = np.unravel_index(np.argmax(results), results.shape)
            max_i, max_j = max_idx
            
            # Add annotation for maximum point
            optimal_price_adj = price_adjustments[max_j]
            optimal_inv_adj = inventory_adjustments[max_i]
            optimal_value = results[max_i, max_j]
            
            fig.add_annotation(
                x=price_labels[max_j],
                y=inventory_labels[max_i],
                text=f"Optimal:<br>Price: {optimal_price_adj:.1f}%<br>Inventory: {optimal_inv_adj:.1f}%<br>Value: {optimal_value:.1f}",
                showarrow=True,
                arrowhead=1
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Price Adjustment (%)",
            yaxis_title="Inventory Adjustment (%)",
            height=800,
            width=1600,
            margin=dict(t=80, b=40, l=60, r=40),
            xaxis=dict(
                tickmode='array',
                tickvals=price_labels,
                ticktext=price_labels
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=inventory_labels,
                ticktext=inventory_labels
            )
        )
        
        # Enhance with interactive features
        if fig is not None:
            fig = enhance_impact_heatmap(fig, model, store_id, item_id)
        
        # Add export capabilities
        if fig is not None:
            title = f"Impact_Heatmap_{store_id}_{item_id}_{metric}"
            fig = enhance_figure_with_exports(fig, title)
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating impact heatmap: {str(e)}")
        return go.Figure()

def create_kpi_indicators(metrics):
    """
    Create a set of KPI indicators based on the metrics.
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics()
        
    Returns:
        list: List of indicator figure objects
    """
    indicators = []
    
    try:
        if not metrics:
            return indicators
        
        # Profit impact indicator
        if ('integrated' in metrics and 
            'price_change_impact' in metrics['integrated'] and
            'profit_diff_pct' in metrics['integrated']['price_change_impact']):
            
            profit_change_pct = metrics['integrated']['price_change_impact']['profit_diff_pct']
            
            profit_fig = go.Figure()
            profit_fig.add_trace(go.Indicator(
                mode="number+delta",
                value=profit_change_pct,
                delta={"reference": 0, "valueformat": ".1f"},
                title={"text": "Profit Impact"},
                number={"suffix": "%", "valueformat": ".1f"},
                domain={"row": 0, "column": 0}
            ))
            
            indicators.append(profit_fig)
        
        # Inventory coverage indicator
        if 'inventory' in metrics and 'coverage_weeks' in metrics['inventory']:
            coverage_weeks = metrics['inventory']['coverage_weeks']
            
            # Reference is target weeks (ideally from config)
            target_weeks = 2
            if 'target_stock' in metrics['inventory'] and 'avg_daily_sales' in metrics['inventory']:
                target_stock = metrics['inventory']['target_stock']
                avg_daily_sales = metrics['inventory']['avg_daily_sales']
                if avg_daily_sales > 0:
                    target_weeks = target_stock / (avg_daily_sales * 7)
            
            inventory_fig = go.Figure()
            inventory_fig.add_trace(go.Indicator(
                mode="number+delta",
                value=coverage_weeks,
                delta={"reference": target_weeks, "valueformat": ".1f"},
                title={"text": "Inventory Coverage"},
                number={"suffix": " weeks", "valueformat": ".1f"},
                domain={"row": 0, "column": 0}
            ))
            
            indicators.append(inventory_fig)
        
        # Demand forecast indicator
        if ('forecast' in metrics and 
            'avg_daily_forecast' in metrics['forecast'] and
            'recent_avg_sales' in metrics['forecast'] and
            metrics['forecast']['recent_avg_sales'] is not None):
            
            forecast = metrics['forecast']['avg_daily_forecast']
            reference = metrics['forecast']['recent_avg_sales']
            
            forecast_fig = go.Figure()
            forecast_fig.add_trace(go.Indicator(
                mode="number+delta",
                value=forecast,
                delta={"reference": reference, "valueformat": ".1f"},
                title={"text": "Daily Demand Forecast"},
                number={"valueformat": ".1f"},
                domain={"row": 0, "column": 0}
            ))
            
            indicators.append(forecast_fig)
        
        # Business impact score
        if 'integrated' in metrics and 'business_impact_score' in metrics['integrated']:
            score = metrics['integrated']['business_impact_score']
            
            score_fig = go.Figure()
            score_fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=score,
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "darkgreen" if score >= 60 else "orange" if score >= 40 else "red"},
                    "steps": [
                        {"range": [0, 30], "color": "lightcoral"},
                        {"range": [30, 50], "color": "lightyellow"},
                        {"range": [50, 100], "color": "lightgreen"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": 60
                    }
                },
                title={"text": "Business Impact Score"},
                domain={"row": 0, "column": 0}
            ))
            
            indicators.append(score_fig)
    
    except Exception as e:
        logger.error(f"Error creating KPI indicators: {str(e)}")
    
    # Enhance with interactive features
    if indicators and all(indicator is not None for indicator in indicators):
        indicators = enhance_kpi_indicators(indicators)
    
    # Add export capabilities to each indicator
    for i, fig in enumerate(indicators):
        if fig is not None:
            title = f"KPI_Indicator_{i}"
            indicators[i] = enhance_figure_with_exports(fig, title)
    
    return indicators

def create_recommendations_table(metrics):
    """
    Create a recommendations table based on metrics.
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics()
        
    Returns:
        plotly.graph_objects.Figure: Table figure
    """
    try:
        if not metrics:
            return go.Figure()
        
        # Extract relevant metrics
        price_metrics = metrics.get('price', {})
        inventory_metrics = metrics.get('inventory', {})
        integrated_metrics = metrics.get('integrated', {})
        
        # Create recommendations list
        recommendations = []
        
        # Pricing recommendations
        if price_metrics:
            elasticity = price_metrics.get('elasticity', 0)
            price_change_pct = price_metrics.get('price_change_pct', 0)
            current_price = price_metrics.get('current_price')
            new_price = price_metrics.get('new_price')
            
            if elasticity < -1.5:  # Highly elastic
                if price_change_pct > 0:
                    recommendations.append({
                        "Area": "Pricing",
                        "Recommendation": "Price increase may reduce profit due to high elasticity",
                        "Priority": "High",
                        "Impact": "High"
                    })
                elif price_change_pct < 0:
                    recommendations.append({
                        "Area": "Pricing",
                        "Recommendation": f"Price reduction of {abs(price_change_pct):.1f}% could boost sales significantly",
                        "Priority": "Medium",
                        "Impact": "High"
                    })
            elif -1.5 <= elasticity <= -0.5:  # Moderate elasticity
                if price_change_pct > 0:
                    recommendations.append({
                        "Area": "Pricing",
                        "Recommendation": f"Moderate price increase of {price_change_pct:.1f}% is acceptable",
                        "Priority": "Medium",
                        "Impact": "Medium"
                    })
                elif price_change_pct < 0:
                    recommendations.append({
                        "Area": "Pricing",
                        "Recommendation": f"Consider price decrease to {new_price:.2f} to boost demand",
                        "Priority": "Medium",
                        "Impact": "Medium"
                    })
            else:  # Inelastic
                if price_change_pct == 0:
                    recommendations.append({
                        "Area": "Pricing",
                        "Recommendation": f"Consider price increase as product is price-inelastic",
                        "Priority": "Medium",
                        "Impact": "Medium"
                    })
                elif price_change_pct > 0:
                    recommendations.append({
                        "Area": "Pricing",
                        "Recommendation": f"Price increase to {new_price:.2f} should improve profit",
                        "Priority": "High",
                        "Impact": "High"
                    })
        
        # Inventory recommendations
        if inventory_metrics:
            status = inventory_metrics.get('status')
            coverage_weeks = inventory_metrics.get('coverage_weeks')
            reorder_needed = inventory_metrics.get('reorder_needed')
            reorder_amount = inventory_metrics.get('reorder_amount')
            
            if status == "Low" and reorder_needed:
                recommendations.append({
                    "Area": "Inventory",
                    "Recommendation": f"Order {int(reorder_amount)} units immediately to prevent stockout",
                    "Priority": "High",
                    "Impact": "High"
                })
            elif status == "Excess":
                recommendations.append({
                    "Area": "Inventory",
                    "Recommendation": f"Reduce inventory or consider promotional pricing",
                    "Priority": "Medium",
                    "Impact": "Medium"
                })
        
        # Integrated recommendations
        if integrated_metrics:
            price_change_impact = integrated_metrics.get('price_change_impact', {})
            inventory_health = integrated_metrics.get('inventory_health', {})
            
            profit_diff_pct = price_change_impact.get('profit_diff_pct', 0)
            balance_status = inventory_health.get('balance_status')
            
            if profit_diff_pct > 10 and balance_status == "Balanced":
                recommendations.append({
                    "Area": "Integrated",
                    "Recommendation": "Implement price change - inventory level is balanced and profit impact is significant",
                    "Priority": "High",
                    "Impact": "High"
                })
            elif profit_diff_pct > 10 and balance_status == "Understocked":
                recommendations.append({
                    "Area": "Integrated",
                    "Recommendation": "Increase inventory before implementing price change to avoid stockouts",
                    "Priority": "High",
                    "Impact": "High"
                })
            elif profit_diff_pct > 10 and balance_status == "Overstocked":
                recommendations.append({
                    "Area": "Integrated",
                    "Recommendation": "Implement price change to reduce excess inventory and increase profit",
                    "Priority": "High",
                    "Impact": "High"
                })
            elif profit_diff_pct < 0:
                recommendations.append({
                    "Area": "Integrated",
                    "Recommendation": "Avoid proposed price change - negative profit impact expected",
                    "Priority": "High",
                    "Impact": "High"
                })
        
        # If no recommendations were generated, add a default one
        if not recommendations:
            recommendations.append({
                "Area": "General",
                "Recommendation": "No specific recommendations at this time - current settings appear optimal",
                "Priority": "Low",
                "Impact": "Low"
            })
        
        # Create table figure
        headers = ["Area", "Recommendation", "Priority", "Impact"]
        cells = []
        
        # Extract data for each column
        for col in headers:
            cells.append([rec[col] for rec in recommendations])
        
        # Color mappings for priority and impact
        color_map = {
            "High": "rgba(255, 0, 0, 0.7)",
            "Medium": "rgba(255, 165, 0, 0.7)",
            "Low": "rgba(0, 128, 0, 0.7)"
        }
        
        # Create table figure
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=headers,
                fill_color='paleturquoise',
                align='left',
                font=dict(size=14)
            ),
            cells=dict(
                values=cells,
                fill_color=[[color_map.get(p, 'white') for p in cells[2]],
                           [color_map.get(i, 'white') for i in cells[3]]],
                align='left',
                font=dict(size=13)
            )
        )])
        
        # Update layout
        fig.update_layout(
            title="Business Recommendations",
            margin=dict(t=50, b=20, l=20, r=20),
            height=120 + 35 * len(recommendations)  # Adjust height based on number of rows
        )
        
        # Enhance with interactive features
        fig = enhance_recommendations_table(fig)
        
        # Add export capabilities
        title = "Business_Recommendations"
        fig = enhance_figure_with_exports(fig, title)
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating recommendations table: {str(e)}")
        return go.Figure()