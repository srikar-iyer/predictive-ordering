"""
Plotly visualization components for the Pizza Predictive Ordering System.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import logging

# Import interactive features
from src.models.interactive_features import (
    enhance_elasticity_distribution,
    enhance_price_sensitivity_curve,
    enhance_profit_impact_waterfall,
    enhance_elasticity_vs_margin_plot
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
logger = logging.getLogger('plotly_visualizations')


def create_elasticity_distribution_plot(elasticity_df, selected_elasticity=None):
    """
    Create a distribution plot of elasticity values.
    
    Args:
        elasticity_df: DataFrame with elasticity data
        selected_elasticity: Specific elasticity to highlight (optional)
        
    Returns:
        go.Figure: Plotly figure with elasticity distribution
    """
    try:
        if elasticity_df is None or len(elasticity_df) == 0:
            # Create an empty figure with an error message
            fig = go.Figure()
            fig.add_annotation(
                text="No elasticity data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Ensure elasticity_df has 'Elasticity' column
        if 'Elasticity' not in elasticity_df.columns:
            # Create an empty figure with an error message
            fig = go.Figure()
            fig.add_annotation(
                text="Elasticity column not found in data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Filter out invalid elasticities
        valid_elasticities = elasticity_df[
            (elasticity_df['Elasticity'] < 0) & 
            (elasticity_df['Elasticity'] > -10)  # Filter extreme outliers
        ]['Elasticity']
        
        valid_df = elasticity_df[
            (elasticity_df['Elasticity'] < 0) & 
            (elasticity_df['Elasticity'] > -10)
        ]
        
        if len(valid_elasticities) == 0:
            # Create an empty figure with a message
            fig = go.Figure()
            fig.add_annotation(
                text="No valid elasticities found in data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Calculate elasticity stats
        avg_elasticity = valid_elasticities.mean()
        median_elasticity = valid_elasticities.median()
        significant_count = valid_df['Is_Significant'].sum()
        total_count = len(valid_df)
        significant_pct = 100 * significant_count / total_count if total_count > 0 else 0
        
        # Create figure with subplots for histogram and bar chart
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=("Distribution of Price Elasticities", 
                                          "Elasticity Significance"),
                           specs=[[{"type": "histogram"}, {"type": "bar"}]],
                           column_widths=[0.7, 0.3])
        
        # Add histogram to first subplot
        fig.add_trace(go.Histogram(
            x=valid_elasticities,
            opacity=0.7,
            name='Elasticity Distribution',
            nbinsx=20,
            marker=dict(color='steelblue'),
            hovertemplate='Elasticity: %{x:.2f}<br>Count: %{y}<extra></extra>'
        ), row=1, col=1)
        
        # Add average line
        fig.add_shape(
            type="line",
            x0=avg_elasticity,
            y0=0,
            x1=avg_elasticity,
            y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash"),
            row=1, col=1
        )
        
        # Add median line
        fig.add_shape(
            type="line",
            x0=median_elasticity,
            y0=0,
            x1=median_elasticity,
            y1=1,
            yref="paper",
            line=dict(color="green", width=2, dash="dot"),
            row=1, col=1
        )
        
        # Add annotation for average and median
        fig.add_annotation(
            x=avg_elasticity,
            y=0.95,
            yref="paper",
            text=f"Average: {avg_elasticity:.2f}",
            showarrow=True,
            arrowhead=1,
            row=1, col=1
        )
        
        fig.add_annotation(
            x=median_elasticity,
            y=0.85,
            yref="paper",
            text=f"Median: {median_elasticity:.2f}",
            showarrow=True,
            arrowhead=1,
            row=1, col=1
        )
        
        # Highlight selected elasticity if provided
        if selected_elasticity is not None:
            fig.add_shape(
                type="line",
                x0=selected_elasticity,
                y0=0,
                x1=selected_elasticity,
                y1=0.75,
                yref="paper",
                line=dict(color="purple", width=2),
                row=1, col=1
            )
            
            fig.add_annotation(
                x=selected_elasticity,
                y=0.75,
                yref="paper",
                text=f"Selected: {selected_elasticity:.2f}",
                showarrow=True,
                arrowhead=1,
                row=1, col=1
            )
        
        # Add bar chart showing statistically significant vs not significant
        categories = ['Statistically Significant', 'Not Significant']
        values = [significant_count, total_count - significant_count]
        colors = ['rgba(0, 128, 0, 0.7)', 'rgba(255, 0, 0, 0.7)']
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v} ({v/total_count*100:.1f}%)" for v in values],
            textposition='auto',
            hovertemplate='%{x}<br>Count: %{y}<br>Percentage: %{text}<extra></extra>'
        ), row=1, col=2)
        
        # Add a table for elasticity summary
        fig.add_trace(go.Table(
            header=dict(values=['Elasticity Metric', 'Value'],
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=[['Average Elasticity', 'Median Elasticity', 'Statistically Significant', 'Total Products'],
                              [f"{avg_elasticity:.2f}", f"{median_elasticity:.2f}", f"{significant_count} ({significant_pct:.1f}%)", f"{total_count}"]],
                     fill_color='lavender',
                     align='left'),
            domain=dict(x=[0, 0.3], y=[0, 0.3])
        ))
        
        # Set chart title and labels
        fig.update_layout(
            title="Frozen Pizza Price Elasticity Analysis",
            xaxis_title="Elasticity Value",
            yaxis_title="Count",
            xaxis2_title="",
            yaxis2_title="Count",
            bargap=0.05,
            showlegend=False,
            height=800,
            width=1600,
            margin=dict(t=100, b=100),
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            xaxis=dict(
                tickfont=dict(size=12),
                title_font=dict(size=14)
            ),
            yaxis=dict(
                tickfont=dict(size=12),
                title_font=dict(size=14)
            ),
        )
        
        # Enhance with interactive features
        if fig is not None and elasticity_df is not None:
            fig = enhance_elasticity_distribution(fig, elasticity_df)
        
        # Add export capabilities
        title = "Distribution of Price Elasticities"
        if fig is not None:
            fig = enhance_figure_with_exports(fig, title)
        
        return fig
        
    except Exception as e:
        # Create an error figure if something goes wrong
        logger.error(f"Error creating elasticity distribution chart: {str(e)}")
        
        # Simple fallback figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating elasticity chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig


def create_price_sensitivity_curve(elasticity, current_price, cost=None, constraints=None, forecast_data=None, product_name=None):
    """
    Create a price sensitivity curve based on elasticity, integrated with forecast and inventory data.
    
    Args:
        elasticity: Elasticity value
        current_price: Current price
        cost: Unit cost (optional)
        constraints: Dictionary of constraints (optional)
        forecast_data: DataFrame with forecast data (optional)
        product_name: Name of the product (optional)
        
    Returns:
        go.Figure: Plotly figure with price sensitivity curve
    """
    try:
        # Validate inputs
        if current_price is None or current_price <= 0:
            fig = go.Figure()
            fig.add_annotation(
                text="Invalid price input (must be > 0)",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Set up subplots for sensitivity analysis
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Price Sensitivity (Demand Response)",
                "Revenue & Profit Impact", 
                "Price-Volume Curve", 
                "Optimal Price Analysis"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
            
        # Create price range (70% to 130% of current price)
        price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
        
        # Calculate quantity, revenue and profit at each price point
        quantity = []
        revenue = []
        profit = []
        margin_pct = []
        
        baseline_quantity = 1.0
        if forecast_data is not None and len(forecast_data) > 0:
            if 'Forecast' in forecast_data.columns:
                # Use average forecast as baseline quantity
                baseline_quantity = forecast_data['Forecast'].mean()
        
        for price in price_range:
            price_ratio = price / current_price
            quantity_ratio = price_ratio ** elasticity
            actual_quantity = baseline_quantity * quantity_ratio
            
            quantity.append(actual_quantity)
            revenue.append(price * actual_quantity)
            
            # Calculate profit if cost is provided
            if cost is not None:
                profit.append((price - cost) * actual_quantity)
                if price > 0:
                    margin_pct.append(((price - cost) / price) * 100)
                else:
                    margin_pct.append(0)
            else:
                # Assume 25% margin if cost isn't provided
                estimated_cost = price * 0.75
                profit.append((price - estimated_cost) * actual_quantity)
                margin_pct.append(25.0)
        
        # Subplot 1: Price Sensitivity (Demand Response)
        fig.add_trace(go.Scatter(
            x=price_range,
            y=quantity,
            mode='lines',
            name='Demand',
            line=dict(color='blue', width=3),
            hovertemplate="Price: $%{x:.2f}<br>Demand: %{y:.1f} units<extra></extra>"
        ), row=1, col=1)
        
        # Find optimal price point
        optimal_idx = 0
        if profit:
            optimal_idx = np.argmax(profit)
        
        # Mark optimal point on demand curve
        fig.add_trace(go.Scatter(
            x=[price_range[optimal_idx]],
            y=[quantity[optimal_idx]],
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                symbol='circle',
                line=dict(color='white', width=2)
            ),
            name='Optimal Price Point',
            hovertemplate="Optimal Price: $%{x:.2f}<br>Demand: %{y:.1f} units<extra></extra>"
        ), row=1, col=1)
        
        # Subplot 2: Revenue and Profit Impact
        fig.add_trace(go.Scatter(
            x=price_range,
            y=revenue,
            mode='lines',
            name='Revenue',
            line=dict(color='green', width=2),
            hovertemplate="Price: $%{x:.2f}<br>Revenue: $%{y:.2f}<extra></extra>"
        ), row=1, col=2)
        
        if profit:
            fig.add_trace(go.Scatter(
                x=price_range,
                y=profit,
                mode='lines',
                name='Profit',
                line=dict(color='red', width=2),
                hovertemplate="Price: $%{x:.2f}<br>Profit: $%{y:.2f}<extra></extra>"
            ), row=1, col=2)
        
        # Mark optimal point on profit curve
        if profit:
            fig.add_trace(go.Scatter(
                x=[price_range[optimal_idx]],
                y=[profit[optimal_idx]],
                mode='markers',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                name='Maximum Profit',
                hovertemplate="Optimal Price: $%{x:.2f}<br>Max Profit: $%{y:.2f}<extra></extra>",
                showlegend=False
            ), row=1, col=2)
        
        # Subplot 3: Price-Volume Curve (Demand Curve)
        fig.add_trace(go.Scatter(
            x=quantity,
            y=price_range,
            mode='lines',
            name='Price-Volume',
            line=dict(color='purple', width=2),
            hovertemplate="Demand: %{x:.1f} units<br>Price: $%{y:.2f}<extra></extra>"
        ), row=2, col=1)
        
        # Mark current price and optimal price on demand curve
        fig.add_trace(go.Scatter(
            x=[quantity[np.where(price_range == current_price)[0][0]] if current_price in price_range else quantity[50]],
            y=[current_price],
            mode='markers',
            marker=dict(size=10, color='blue', symbol='circle'),
            name='Current Price',
            hovertemplate="Current Price: $%{y:.2f}<br>Demand: %{x:.1f} units<extra></extra>"
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=[quantity[optimal_idx]],
            y=[price_range[optimal_idx]],
            mode='markers',
            marker=dict(size=10, color='red', symbol='circle'),
            name='Optimal Price',
            hovertemplate="Optimal Price: $%{y:.2f}<br>Demand: %{x:.1f} units<extra></extra>",
            showlegend=False
        ), row=2, col=1)
        
        # Subplot 4: Profit vs Price curve with margin data
        fig.add_trace(go.Scatter(
            x=price_range,
            y=profit if profit else [0] * len(price_range),
            mode='lines',
            name='Profit vs Price',
            line=dict(color='red', width=2),
            hovertemplate="Price: $%{x:.2f}<br>Profit: $%{y:.2f}<br>Margin: %{text:.1f}%<extra></extra>",
            text=margin_pct
        ), row=2, col=2)
        
        # Mark optimal point on profit vs price curve
        if profit:
            fig.add_trace(go.Scatter(
                x=[price_range[optimal_idx]],
                y=[profit[optimal_idx]],
                mode='markers',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                name='Optimal Price',
                hovertemplate="Optimal Price: $%{x:.2f}<br>Max Profit: $%{y:.2f}<br>Margin: %{text:.1f}%<extra></extra>",
                text=[margin_pct[optimal_idx]],
                showlegend=False
            ), row=2, col=2)
        
        # Add reference lines for current price
        for row, col in [(1, 1), (1, 2), (2, 2)]:
            fig.add_shape(
                type="line",
                x0=current_price,
                y0=0,
                x1=current_price,
                y1=1,
                xref=f"x{row if col == 1 else row + 2}",
                yref="paper",
                line=dict(color="black", width=1.5, dash="dash")
            )
        
        # Add constraints if provided
        if constraints is not None:
            # Add min/max price constraints on subplots 1, 2, and 4
            for row, col in [(1, 1), (1, 2), (2, 2)]:
                if 'max_price_increase' in constraints:
                    upper_bound = current_price * (1 + constraints['max_price_increase'] / 100)
                    fig.add_shape(
                        type="line",
                        x0=upper_bound,
                        y0=0,
                        x1=upper_bound,
                        y1=0.8,
                        xref=f"x{row if col == 1 else row + 2}",
                        yref="paper",
                        line=dict(color="orange", width=1.5, dash="dot")
                    )
                
                if 'max_price_decrease' in constraints:
                    lower_bound = current_price * (1 - constraints['max_price_decrease'] / 100)
                    fig.add_shape(
                        type="line",
                        x0=lower_bound,
                        y0=0,
                        x1=lower_bound,
                        y1=0.8,
                        xref=f"x{row if col == 1 else row + 2}",
                        yref="paper",
                        line=dict(color="orange", width=1.5, dash="dot")
                    )
        
        # Calculate optimal price and profit improvement
        optimal_price = price_range[optimal_idx]
        price_change_pct = ((optimal_price / current_price) - 1) * 100
        
        current_profit_idx = np.where(np.isclose(price_range, current_price))[0][0] if any(np.isclose(price_range, current_price)) else 50
        current_profit = profit[current_profit_idx] if profit else 0
        optimal_profit = profit[optimal_idx] if profit else 0
        profit_change_pct = ((optimal_profit / current_profit) - 1) * 100 if current_profit > 0 else 0
        
        # Add summary annotations
        summary_text = [
            f"<b>Elasticity:</b> {elasticity:.2f}",
            f"<b>Current Price:</b> ${current_price:.2f}",
            f"<b>Optimal Price:</b> ${optimal_price:.2f} ({'+' if price_change_pct >= 0 else ''}{price_change_pct:.1f}%)",
            f"<b>Profit Change:</b> {'+' if profit_change_pct >= 0 else ''}{profit_change_pct:.1f}%"
        ]
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            text="<br>".join(summary_text),
            showarrow=False,
            font=dict(size=12),
            align="center",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )
        
        # Set product name in title if provided
        title = f"Price Sensitivity Analysis for {product_name}" if product_name else f"Price Sensitivity Analysis (Elasticity: {elasticity:.2f})"
        
        # Set chart title and labels
        fig.update_layout(
            title={
                'text': title,
                'font': {'size': 16},
                'y': 0.98
            },
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            hovermode='closest'
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Demand (Units)", row=1, col=1)
        
        fig.update_xaxes(title_text="Price ($)", row=1, col=2)
        fig.update_yaxes(title_text="Value ($)", row=1, col=2)
        
        fig.update_xaxes(title_text="Demand (Units)", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=2, col=1)
        
        fig.update_xaxes(title_text="Price ($)", row=2, col=2)
        fig.update_yaxes(title_text="Profit ($)", row=2, col=2)
        
        # Enhance with interactive features
        try:
            if fig is not None:
                fig = enhance_price_sensitivity_curve(fig, None)
        except Exception as e:
            logger.warning(f"Could not enhance price sensitivity curve: {str(e)}")
        
        # Add export capabilities
        title_slug = f"Price_Sensitivity_{product_name.replace(' ', '_')}" if product_name else f"Price_Sensitivity_Elasticity_{elasticity:.2f}"
        if fig is not None:
            fig = enhance_figure_with_exports(fig, title_slug)
        
        return fig
        
    except Exception as e:
        # Create an error figure if something goes wrong
        logger.error(f"Error creating price sensitivity curve: {str(e)}")
        
        # Simple fallback figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating price sensitivity curve: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig


def create_profit_impact_waterfall(impact_df, limit=10):
    """
    Create a waterfall chart showing profit impact of price changes.
    
    Args:
        impact_df: DataFrame with profit impact data
        limit: Maximum number of products to include
        
    Returns:
        go.Figure: Plotly figure with waterfall chart
    """
    try:
        if impact_df is None or len(impact_df) == 0:
            # Create an empty figure with an error message
            fig = go.Figure()
            fig.add_annotation(
                text="No profit impact data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Use only products that have price changes and profit impact
        impact_df_with_changes = impact_df[impact_df['Price_Change_Pct'] != 0.0].copy()
        
        # Select top items by profit impact
        top_items = impact_df_with_changes.sort_values('Total_Profit_Difference', ascending=False).head(limit)
        
        # Create a subplot with waterfall and bar chart
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Profit Impact of Price Optimization", 
                           "Daily Profit Impact by Product"),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )
        
        # Calculate initial and final values
        initial_profit = top_items['Total_Current_Profit'].sum()
        final_profit = top_items['Total_New_Profit'].sum()
        profit_increase = final_profit - initial_profit
        profit_pct_increase = (profit_increase / initial_profit) * 100 if initial_profit > 0 else 0
        
        # Create data for waterfall chart
        x_labels = ['Current Profit']
        y_values = [initial_profit]
        measure = ['absolute']
        text_values = [f"${initial_profit:.2f}"]
        
        # Add items
        for _, row in top_items.iterrows():
            price_change = row.get('Price_Change_Pct', 0.0)
            price_direction = "↑" if price_change > 0 else "↓"
            elasticity = row.get('Elasticity', 0.0)
            prod_name = row['Product'].split(' ')
            if len(prod_name) > 2:
                # Shorten product name for display
                prod_name = f"{prod_name[0]} {prod_name[1]}..."
            else:
                prod_name = row['Product']
                
            x_labels.append(f"{prod_name} ({price_direction}{abs(price_change):.1f}%)")
            y_values.append(row['Total_Profit_Difference'])
            measure.append('relative')
            text_values.append(f"${row['Total_Profit_Difference']:.2f}")
        
        # Add final profit
        x_labels.append('New Profit')
        y_values.append(final_profit)
        measure.append('total')
        text_values.append(f"${final_profit:.2f}")
        
        # Create waterfall chart
        fig.add_trace(go.Waterfall(
            name="Profit Impact",
            orientation="v",
            measure=measure,
            x=x_labels,
            y=y_values,
            textposition="outside",
            text=text_values,
            connector={"line":{"color":"rgb(63, 63, 63)"}},
            increasing={"marker":{"color":"green"}},
            decreasing={"marker":{"color":"red"}},
            hovertemplate="<b>%{x}</b><br>Value: $%{y:.2f}<extra></extra>"
        ), row=1, col=1)
        
        # Create bar chart for daily profit impact
        daily_impact = top_items.sort_values('Daily_Profit_Impact', ascending=True).copy()
        bar_colors = ['green' if x > 0 else 'red' for x in daily_impact['Daily_Profit_Impact']]
        
        fig.add_trace(go.Bar(
            x=daily_impact['Daily_Profit_Impact'],
            y=daily_impact['Product'],
            orientation='h',
            marker_color=bar_colors,
            text=[f"${x:.2f}/day" for x in daily_impact['Daily_Profit_Impact']],
            textposition='auto',
            hovertemplate="<b>%{y}</b><br>Daily Impact: $%{x:.2f}<br>Elasticity: %{customdata[0]:.2f}<br>Current Price: $%{customdata[1]:.2f}<br>Optimal Price: $%{customdata[2]:.2f}<extra></extra>",
            customdata=daily_impact[['Elasticity', 'Current_Price', 'Optimal_Price']].values
        ), row=2, col=1)
        
        # Add summary annotations
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.98,
            text=f"Total Profit Increase: ${profit_increase:.2f} (+{profit_pct_increase:.1f}%)",
            showarrow=False,
            font=dict(size=14, color="green"),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="green",
            borderwidth=2,
            borderpad=4
        )
        
        forecast_days = top_items['Forecast_Days'].iloc[0] if 'Forecast_Days' in top_items.columns and len(top_items) > 0 else 30
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.99, y=0.98,
            text=f"Forecast Period: {forecast_days} days",
            showarrow=False,
            font=dict(size=12),
            align="right"
        )
        
        # Set chart title and labels
        fig.update_layout(
            title={
                'text': "Frozen Pizza Revenue Impact Analysis",
                'font': {'size': 16}
            },
            xaxis_title="",
            yaxis_title="",
            xaxis2_title="Daily Profit Impact ($)",
            yaxis2_title="",
            height=800,
            showlegend=False,
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            margin=dict(t=80, b=50, l=50, r=50)
        )
        
        # Enhance with interactive features
        try:
            if fig is not None:
                fig = enhance_profit_impact_waterfall(fig, impact_df)
        except Exception as e:
            logger.warning(f"Could not enhance profit impact waterfall: {str(e)}")
        
        # Add export capabilities
        title = "Profit_Impact_of_Price_Optimization"
        if fig is not None:
            fig = enhance_figure_with_exports(fig, title)
        
        return fig
    except Exception as e:
        # Create error figure if something goes wrong
        logger.error(f"Error creating profit impact waterfall chart: {str(e)}")
        
        # Simple fallback figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating profit impact chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig


def create_elasticity_vs_margin_plot(elasticity_df):
    """
    Create a scatter plot of elasticity vs margin.
    
    Args:
        elasticity_df: DataFrame with elasticity data
        
    Returns:
        go.Figure: Plotly figure with elasticity vs margin scatter plot
    """
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=elasticity_df['Elasticity'],
        y=elasticity_df['Margin_Pct'],
        mode='markers',
        text=elasticity_df['Product'],
        hoverinfo='text+x+y',
        marker=dict(
            size=10,
            color=elasticity_df['Elasticity'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Elasticity')
        )
    ))
    
    # Add quadrant lines
    fig.add_shape(
        type="line",
        x0=-1,
        y0=0,
        x1=-1,
        y1=max(elasticity_df['Margin_Pct']) * 1.1,
        line=dict(color="gray", width=1, dash="dash")
    )
    
    median_margin = elasticity_df['Margin_Pct'].median()
    fig.add_shape(
        type="line",
        x0=min(elasticity_df['Elasticity']) * 1.1,
        y0=median_margin,
        x1=0,
        y1=median_margin,
        line=dict(color="gray", width=1, dash="dash")
    )
    
    # Add quadrant labels
    fig.add_annotation(
        x=-0.5,
        y=median_margin * 1.5,
        text="High Margin, Low Elasticity<br>Premium Pricing",
        showarrow=False,
        font=dict(size=10)
    )
    
    fig.add_annotation(
        x=-2,
        y=median_margin * 1.5,
        text="High Margin, High Elasticity<br>Price Optimization Needed",
        showarrow=False,
        font=dict(size=10)
    )
    
    fig.add_annotation(
        x=-0.5,
        y=median_margin * 0.5,
        text="Low Margin, Low Elasticity<br>Increase Prices",
        showarrow=False,
        font=dict(size=10)
    )
    
    fig.add_annotation(
        x=-2,
        y=median_margin * 0.5,
        text="Low Margin, High Elasticity<br>Improve Cost Structure",
        showarrow=False,
        font=dict(size=10)
    )
    
    # Set chart title and labels
    fig.update_layout(
        title="Price Elasticity vs Margin",
        xaxis_title="Elasticity",
        yaxis_title="Margin (%)",
        xaxis=dict(range=[min(elasticity_df['Elasticity']) * 1.1, 0])
    )
    
    # Enhance with interactive features
    try:
        if fig is not None:
            fig = enhance_elasticity_vs_margin_plot(fig, elasticity_df)
    except Exception as e:
        logger.warning(f"Could not enhance elasticity vs margin plot: {str(e)}")
    
    # Add export capabilities
    title = "Price_Elasticity_vs_Margin"
    if fig is not None:
        fig = enhance_figure_with_exports(fig, title)
    
    return fig