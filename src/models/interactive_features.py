"""
Interactive features for Plotly visualizations in the Pizza Predictive Ordering System.

This module provides functions to enhance Plotly visualizations with interactive features:
1. Enhanced hover tooltips with detailed information
2. Click events for drilling down into data
3. Zoom and pan capabilities for time series charts
4. Crossfiltering between charts
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('interactive_features')


def add_hover_tooltip(fig, data=None, hover_template=None, hover_data=None, custom_data=None):
    """
    Add enhanced hover tooltip to a Plotly figure.
    
    Args:
        fig: Plotly figure to enhance
        data: Optional DataFrame with data for hover information
        hover_template: Custom hover template string
        hover_data: List of columns to include in hover data
        custom_data: Custom data array for hover
        
    Returns:
        go.Figure: Enhanced figure with hover tooltips
    """
    try:
        # Safety check for figure
        if fig is None or not hasattr(fig, 'data') or fig.data is None:
            logger.warning("Cannot add hover tooltip to None figure or figure without data")
            return fig
            
        # If no specific template is provided, use a general enhancement
        if hover_template is None and data is not None and hover_data is not None:
            # For each trace in the figure
            for i, trace in enumerate(fig.data):
                if hasattr(trace, 'hovertemplate') and trace.hovertemplate is not None:
                    # Skip traces that already have custom hover templates
                    continue
                
                # Create a custom hover template based on hover_data
                template_parts = []
                for col in hover_data:
                    if col in data.columns:
                        # Format based on data type
                        if pd.api.types.is_numeric_dtype(data[col]):
                            # Format numbers with commas and appropriate decimals
                            if data[col].dtype == 'float64':
                                template_parts.append(f"<b>{col}</b>: %{{customdata[{hover_data.index(col)}]:.2f}}")
                            else:
                                template_parts.append(f"<b>{col}</b>: %{{customdata[{hover_data.index(col)}]:,}}")
                        else:
                            template_parts.append(f"<b>{col}</b>: %{{customdata[{hover_data.index(col)}]}}")
                
                # Add x and y values if they're not already in hover_data
                x_label = getattr(trace, 'name', 'x')
                y_label = 'Value'
                
                if hasattr(trace, 'x') and hasattr(trace, 'y'):
                    if x_label not in hover_data:
                        template_parts.insert(0, f"<b>{x_label}</b>: %{{x}}")
                    if y_label not in hover_data:
                        template_parts.insert(1, f"<b>{y_label}</b>: %{{y:.2f}}")
                
                # Create the complete template
                hover_template = "<br>".join(template_parts) + "<extra></extra>"
                
                # Create custom_data array if it doesn't exist
                if custom_data is None and data is not None:
                    # Extract the subset of data for this trace
                    if hasattr(trace, 'x'):
                        x_values = trace.x
                        if isinstance(x_values, (list, np.ndarray)) and len(x_values) > 0:
                            trace_data = data[data.index.isin(x_values) if isinstance(data.index, pd.DatetimeIndex) else data[data.columns[0]].isin(x_values)]
                            custom_data = trace_data[hover_data].values
                
                # Update the trace with custom hover template and data
                if custom_data is not None:
                    fig.update_traces(
                        hovertemplate=hover_template,
                        customdata=custom_data,
                        selector=dict(type=trace.type, name=trace.name)
                    )
        
        # If a specific hover_template is provided, use it directly
        elif hover_template is not None:
            fig.update_traces(hovertemplate=hover_template)
        
        # Set a nice hover mode that works well for multi-trace charts
        fig.update_layout(hovermode="closest")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error adding hover tooltip: {str(e)}")
        return fig


def add_drill_down_click(fig, callback_data=None, target_property=None):
    """
    Add click event capability to a Plotly figure for drilling down into data.
    
    Args:
        fig: Plotly figure to enhance
        callback_data: JSON-serializable data to include with click events
        target_property: Property name where to store target info for callbacks
        
    Returns:
        go.Figure: Enhanced figure with click capability
    """
    try:
        # Make the chart clickable
        for i, trace in enumerate(fig.data):
            # Store information in customdata if not present
            if not hasattr(trace, 'customdata') or trace.customdata is None:
                # Create default customdata based on x values
                if hasattr(trace, 'x'):
                    customdata = np.array([[x] for x in trace.x])
                    fig.data[i].customdata = customdata
            
            # Update the trace with attributes to enable click events
            trace_type = trace.type if hasattr(trace, 'type') else 'scatter'
            if trace_type in ['scatter', 'bar', 'line']:
                fig.data[i].clickmode = 'event'
        
        # Add callback information to the figure's metadata
        if callback_data is not None and fig is not None and hasattr(fig, 'layout'):
            # Store callback data in layout.metadata
            metadata = {'callback_data': callback_data}
            if target_property is not None:
                metadata['target_property'] = target_property
                
            # Store in figure layout meta
            if not hasattr(fig.layout, 'meta'):
                fig.layout.meta = {}
            elif fig.layout.meta is None:  # Ensure meta is not None
                fig.layout.meta = {}
                
            # Store drill_down metadata directly without JSON serialization
            # This allows the client-side JavaScript to access it directly
            fig.layout.meta['drill_down'] = metadata
        
        return fig
        
    except Exception as e:
        logger.error(f"Error adding drill down click capability: {str(e)}")
        return fig


def add_zoom_pan(fig, connect_axes=False, range_slider=False, range_buttons=True):
    """
    Add zoom and pan capabilities to a Plotly figure.
    
    Args:
        fig: Plotly figure to enhance
        connect_axes: Whether to connect axes in multi-subplot figures
        range_slider: Whether to add a range slider
        range_buttons: Whether to add zoom range buttons
        
    Returns:
        go.Figure: Enhanced figure with zoom and pan capabilities
    """
    try:
        # Safety check - make sure fig is not None and is a valid figure
        if fig is None:
            logger.warning("Cannot add zoom and pan to None figure")
            return fig
            
        # Check if figure has layout
        if not hasattr(fig, 'layout') or fig.layout is None:
            logger.warning("Figure has no layout attribute or layout is None")
            # Initialize layout if needed
            fig.layout = go.Layout()
            
        # Enable zoom and pan
        try:
            fig.update_layout(
                dragmode='zoom',
                selectdirection='h'
            )
        except Exception as e:
            logger.warning(f"Could not update layout for zoom and pan: {str(e)}")
            # Continue with the function
        
        # Safety check - make sure fig.data exists and is not None
        if not hasattr(fig, 'data') or fig.data is None:
            return fig
            
        # Add range slider if requested (for time series)
        if range_slider and hasattr(fig, 'layout') and fig.layout is not None and hasattr(fig.layout, 'xaxis'):
            has_dates = False
            
            # Check if any trace has date values on x-axis
            for trace in fig.data:
                if hasattr(trace, 'x') and trace.x is not None and len(trace.x) > 0:
                    if isinstance(trace.x[0], (pd.Timestamp, np.datetime64)) or (
                            isinstance(trace.x, list) and isinstance(trace.x[0], str) and len(trace.x[0]) > 8):
                        has_dates = True
                        break
            
            # Only add range slider for date axes
            if has_dates:
                fig.update_layout(
                    xaxis=dict(
                        rangeslider=dict(visible=True, thickness=0.05),
                        type='date'
                    )
                )
        
        # Add range buttons for time series
        if range_buttons and hasattr(fig, 'layout') and fig.layout is not None and hasattr(fig.layout, 'xaxis'):
            has_dates = False
            for trace in fig.data:
                if hasattr(trace, 'x') and trace.x is not None and len(trace.x) > 0:
                    if isinstance(trace.x[0], (pd.Timestamp, np.datetime64)) or (
                            isinstance(trace.x, list) and isinstance(trace.x[0], str) and len(trace.x[0]) > 8):
                        has_dates = True
                        break
            
            if has_dates:
                # Add range selector with time frame buttons
                fig.update_layout(
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=7, label="7d", step="day", stepmode="backward"),
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(step="all", label="All")
                            ]),
                            font=dict(size=10),
                            y=-0.12 if range_slider else 0.99,
                            x=0.01,
                            yanchor='top' if range_slider else 'bottom'
                        )
                    )
                )
        
        # Connect axes for multi-subplot figures with safety check
        if connect_axes and isinstance(fig, go.Figure) and hasattr(fig, '_grid_ref') and fig._grid_ref is not None and hasattr(fig._grid_ref, 'shape') and len(fig._grid_ref) > 1:
            # This is a subplot figure
            rows = max(fig._grid_ref.shape[0], 1)
            cols = max(fig._grid_ref.shape[1], 1)
            
            # Connect x-axes for all subplots in the same column
            for col in range(1, cols + 1):
                for row in range(2, rows + 1):
                    fig.update_layout(**{f'xaxis{(row-1)*cols+col}': {'matches': f'x{col}'}})
        
        return fig
        
    except Exception as e:
        logger.error(f"Error adding zoom and pan capabilities: {str(e)}")
        return fig


def setup_crossfilter(figures, filter_dimensions=None):
    """
    Set up crossfiltering between multiple charts.
    
    Args:
        figures: List of Plotly figures to connect
        filter_dimensions: Dictionary mapping dimension names to chart indices
        
    Returns:
        list: List of figures configured for crossfiltering
    """
    try:
        # If no specific filter dimensions provided, create a default mapping
        if filter_dimensions is None:
            filter_dimensions = {}
            
            # Identify common dimensions across charts
            all_dimensions = set()
            for fig in figures:
                for trace in fig.data:
                    if hasattr(trace, 'x') and len(trace.x) > 0:
                        # Use x-axis values as potential dimension
                        dimension = str(type(trace.x[0]).__name__)
                        all_dimensions.add(dimension)
            
            # Create mapping for common dimensions
            for dim in all_dimensions:
                filter_dimensions[dim] = [i for i, fig in enumerate(figures)
                                        if any(hasattr(trace, 'x') and len(trace.x) > 0 and
                                              str(type(trace.x[0]).__name__) == dim
                                              for trace in fig.data)]
        
        # Configure each figure for crossfiltering
        for i, fig in enumerate(figures):
            # Set up unique chart IDs for linking
            chart_id = f"chart_{i}"
            
            # Add shared meta for crossfiltering
            if not hasattr(fig.layout, 'meta'):
                fig.layout.meta = {}
            
            # Store crossfilter configuration
            # Ensure format matches what the JavaScript expects
            fig.layout.meta['crossfilter'] = {
                'chart_id': chart_id,
                'linked_charts': [f"chart_{j}" for j in range(len(figures)) if j != i],
                'filter_dimensions': {dim: charts for dim, charts in filter_dimensions.items()
                                    if i in charts}
            }
            
            # Enable selection tools appropriate for crossfiltering
            fig.update_layout(
                dragmode='select',
                clickmode='event+select'
            )
        
        return figures
        
    except Exception as e:
        logger.error(f"Error setting up crossfiltering: {str(e)}")
        return figures


def enhance_chart(fig, data=None, hover_data=None, enable_drill_down=True, 
                 enable_zoom_pan=True, connect_axes=False):
    """
    Apply all interactive enhancements to a Plotly figure.
    
    Args:
        fig: Plotly figure to enhance
        data: DataFrame with data for hover information
        hover_data: List of columns to include in hover data
        enable_drill_down: Whether to enable drill down functionality
        enable_zoom_pan: Whether to enable zoom and pan capabilities
        connect_axes: Whether to connect axes in multi-subplot figures
        
    Returns:
        go.Figure: Enhanced figure with all interactive features
    """
    try:
        # Apply hover tooltips
        if hover_data is not None:
            fig = add_hover_tooltip(fig, data=data, hover_data=hover_data)
        
        # Apply drill down capability if enabled
        if enable_drill_down:
            callback_data = {'chart_type': getattr(fig.layout, 'title', {}).get('text', 'Unknown Chart')}
            fig = add_drill_down_click(fig, callback_data=callback_data)
        
        # Apply zoom and pan capabilities if enabled
        if enable_zoom_pan:
            range_slider = isinstance(fig, go.Figure) and len(fig.data) == 1  # Only use range slider for single-trace charts
            fig = add_zoom_pan(fig, connect_axes=connect_axes, range_slider=range_slider)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error enhancing chart: {str(e)}")
        return fig


def enhance_elasticity_distribution(fig, elasticity_df=None):
    """
    Enhance elasticity distribution plot with interactive features.
    
    Args:
        fig: Elasticity distribution figure to enhance
        elasticity_df: DataFrame with elasticity data
        
    Returns:
        go.Figure: Enhanced figure
    """
    try:
        if fig is None:
            # Return an empty figure if input is None
            fig = go.Figure()
            fig.add_annotation(
                text="No figure to enhance",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
            
        # Check if figure has data attribute
        if not hasattr(fig, 'data') or fig.data is None:
            logger.warning("Figure has no data attribute")
            return fig
            
        if elasticity_df is None:
            return fig
            
        hover_data = ['Product', 'Elasticity', 'Avg_Price', 'Cost', 'Margin_Pct']
        hover_template = (
            "<b>Elasticity</b>: %{x:.2f}<br>"
            "<b>Count</b>: %{y}<br>"
            "<extra></extra>"
        )
        
        # Only update traces if the figure actually has traces
        if hasattr(fig, 'data') and fig.data is not None and len(fig.data) > 0:
            # Apply trace by trace to avoid issues with table traces
            for i, trace in enumerate(fig.data):
                if trace.type != 'table':  # Skip table traces which don't support hovertemplate
                    fig.data[i].hoverinfo = 'x+y+text'
                    fig.data[i].hovertemplate = hover_template
            
        # Apply zoom and pan functionality with safety checks
        try:
            fig = add_zoom_pan(fig, range_slider=False)
        except Exception as e:
            logger.warning(f"Could not apply zoom and pan to elasticity distribution: {str(e)}")
        
        return fig
    except Exception as e:
        logger.error(f"Error enhancing elasticity distribution: {str(e)}")
        return fig


def enhance_price_sensitivity_curve(fig, price_data=None):
    """
    Enhance price sensitivity curve with interactive features.
    
    Args:
        fig: Price sensitivity curve figure to enhance
        price_data: DataFrame with price data
        
    Returns:
        go.Figure: Enhanced figure
    """
    try:
        if fig is None:
            # Return an empty figure if input is None
            fig = go.Figure()
            fig.add_annotation(
                text="No figure to enhance",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
            
        hover_templates = {
            'Quantity': (
                "<b>Price</b>: $%{x:.2f}<br>"
                "<b>Quantity Ratio</b>: %{y:.2f}<br>"
                "<extra>Quantity</extra>"
            ),
            'Revenue': (
                "<b>Price</b>: $%{x:.2f}<br>"
                "<b>Revenue Ratio</b>: %{y:.2f}<br>"
                "<extra>Revenue</extra>"
            ),
            'Profit': (
                "<b>Price</b>: $%{x:.2f}<br>"
                "<b>Profit Ratio</b>: %{y:.2f}<br>"
                "<extra>Profit</extra>"
            )
        }
        
        if not hasattr(fig, 'data') or fig.data is None or len(fig.data) == 0:
            logger.warning("Invalid figure provided to enhance_price_sensitivity_curve")
            return fig
            
        # Apply hover templates to traces if they exist
        if hasattr(fig, 'data') and fig.data is not None:
            for trace in fig.data:
                if trace is None:
                    continue
                if hasattr(trace, 'name') and trace.name is not None:
                    if trace.name == 'Quantity':
                        trace.hovertemplate = hover_templates['Quantity']
                    elif trace.name == 'Revenue':
                        trace.hovertemplate = hover_templates['Revenue']
                    elif trace.name == 'Profit':
                        trace.hovertemplate = hover_templates['Profit']
        
        # Apply zoom and pan functionality with safety checks
        try:
            fig = add_zoom_pan(fig)
        except Exception as e:
            logger.warning(f"Could not apply zoom and pan to price sensitivity curve: {str(e)}")
        
        return fig
    except Exception as e:
        logger.error(f"Error enhancing price sensitivity curve: {str(e)}")
        return fig


def enhance_profit_impact_waterfall(fig, impact_df=None):
    """
    Enhance profit impact waterfall chart with interactive features.
    
    Args:
        fig: Profit impact waterfall figure to enhance
        impact_df: DataFrame with impact data
        
    Returns:
        go.Figure: Enhanced figure
    """
    try:
        if fig is None:
            # Return an empty figure if input is None
            fig = go.Figure()
            fig.add_annotation(
                text="No figure to enhance",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
            
        if not hasattr(fig, 'data') or fig.data is None or len(fig.data) == 0:
            logger.warning("Invalid figure provided to enhance_profit_impact_waterfall")
            return fig
            
        hover_template = (
            "<b>%{x}</b><br>"
            "<b>Impact</b>: $%{y:.2f}<br>"
            "<extra></extra>"
        )
        
        # Only apply hover template if there are actual traces in the figure and they support the hovertemplate property
        try:
            if len(fig.data) > 0:
                # Apply trace by trace to avoid issues with table traces
                for i, trace in enumerate(fig.data):
                    if trace.type != 'table':  # Skip table traces which don't support hovertemplate
                        fig.data[i].hoverinfo = 'x+y+text'
                        fig.data[i].hovertemplate = hover_template
        except Exception as e:
            logger.warning(f"Could not apply hover template to profit impact waterfall: {str(e)}")
        
        return fig
    except Exception as e:
        logger.error(f"Error enhancing profit impact waterfall: {str(e)}")
        return fig


def enhance_elasticity_vs_margin_plot(fig, elasticity_df=None):
    """
    Enhance elasticity vs margin scatter plot with interactive features.
    
    Args:
        fig: Elasticity vs margin figure to enhance
        elasticity_df: DataFrame with elasticity data
        
    Returns:
        go.Figure: Enhanced figure
    """
    try:
        # Safety checks
        if fig is None:
            logger.warning("None figure provided to enhance_elasticity_vs_margin_plot")
            return fig
            
        if not hasattr(fig, 'data') or fig.data is None:
            logger.warning("Figure has no data attribute")
            return fig
            
        if len(fig.data) == 0:
            logger.warning("Figure has no traces")
            return fig
            
        if elasticity_df is None:
            logger.warning("No elasticity data provided to enhance_elasticity_vs_margin_plot")
            return fig
            
        hover_data = ['Product', 'Elasticity', 'Margin_Pct', 'Avg_Price', 'Cost', 'Store_Id', 'Item']
        
        # Check that all required columns exist in the dataframe
        missing_cols = [col for col in hover_data if col not in elasticity_df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in elasticity data: {missing_cols}")
            # Use only available columns
            hover_data = [col for col in hover_data if col in elasticity_df.columns]
            
        if not hover_data:
            # If no hover data columns are available, just return the figure
            return fig
            
        hover_template = (
            "<b>%{text}</b><br>"
            "<b>Elasticity</b>: %{x:.2f}<br>"
            "<b>Margin</b>: %{y:.1f}%<br>"
        )
        
        # Add optional fields to hover template if available
        if 'Avg_Price' in hover_data and 'Cost' in hover_data:
            hover_template += (
                "<b>Price</b>: $%{customdata[" + str(hover_data.index('Avg_Price')) + "]:.2f}<br>"
                "<b>Cost</b>: $%{customdata[" + str(hover_data.index('Cost')) + "]:.2f}<br>"
            )
            
        hover_template += "<extra></extra>"
        
        try:
            custom_data = elasticity_df[hover_data].values
            
            # Make sure all traces exist and are valid
            for trace in fig.data:
                if trace is None:
                    continue
                    
                # Apply trace by trace to avoid issues with table traces
                for i, trace in enumerate(fig.data):
                    if trace.type != 'table':  # Skip table traces which don't support hovertemplate
                        fig.data[i].hoverinfo = 'x+y+text'
                        fig.data[i].hovertemplate = hover_template
                        fig.data[i].customdata = custom_data
        except Exception as e:
            logger.warning(f"Could not update trace data: {str(e)}")
            # Continue with the function even if customdata update fails
        
        fig = add_zoom_pan(fig)
        
        # Make points larger and add hover effect
        fig.update_traces(
            marker=dict(
                size=12,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            selector=dict(mode='markers')
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error enhancing elasticity vs margin plot: {str(e)}")
        return fig


def enhance_integrated_chart(fig, model=None, store_id=None, item_id=None):
    """
    Enhance integrated chart with interactive features.
    
    Args:
        fig: Integrated chart figure to enhance
        model: UnifiedDataModel instance containing all data
        store_id: Store ID
        item_id: Product ID
        
    Returns:
        go.Figure: Enhanced figure
    """
    # Safety check for figure
    if fig is None or not hasattr(fig, 'data') or fig.data is None:
        logger.warning("Cannot enhance None figure or figure without data")
        return fig
        
    # First, set hover templates for each subplot
    for trace in fig.data:
        if 'Historical Sales' in trace.name:
            trace.hovertemplate = (
                "<b>Date</b>: %{x|%b %d, %Y}<br>"
                "<b>Sales</b>: %{y:.1f} units<br>"
                "<extra>Historical Sales</extra>"
            )
        elif 'Demand Forecast' in trace.name:
            trace.hovertemplate = (
                "<b>Date</b>: %{x|%b %d, %Y}<br>"
                "<b>Forecast</b>: %{y:.1f} units<br>"
                "<extra>Forecast</extra>"
            )
        elif 'Inventory Level' in trace.name:
            trace.hovertemplate = (
                "<b>Date</b>: %{x|%b %d, %Y}<br>"
                "<b>Stock</b>: %{y:.0f} units<br>"
                "<extra>Inventory</extra>"
            )
        elif 'Price Elasticity' in trace.name:
            trace.hovertemplate = (
                "<b>Price</b>: $%{x:.2f}<br>"
                "<b>Demand Ratio</b>: %{y:.2f}<br>"
                "<extra>Price Elasticity</extra>"
            )
        elif 'Original Profit' in trace.name or 'New Profit' in trace.name:
            trace.hovertemplate = (
                "<b>Date</b>: %{x|%b %d, %Y}<br>"
                "<b>Profit</b>: $%{y:.2f}<br>"
                "<extra>" + trace.name + "</extra>"
            )
        elif 'Profit Difference' in trace.name:
            trace.hovertemplate = (
                "<b>Date</b>: %{x|%b %d, %Y}<br>"
                "<b>Difference</b>: $%{y:.2f}<br>"
                "<extra>Profit Difference</extra>"
            )
    
    # Enable synchronized zooming for date axes
    fig = add_zoom_pan(fig, connect_axes=True)
    
    # Make the chart and its elements responsive
    fig.update_layout(
        hoverdistance=100,
        spikedistance=1000,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    # Add spikes (vertical and horizontal lines on hover) to each x-axis
    for i in range(1, 5):  # Assuming 4 subplots
        fig.update_layout(**{
            f'xaxis{i}': {
                'showspikes': True,
                'spikethickness': 1,
                'spikedash': 'dot',
                'spikecolor': 'grey',
                'spikemode': 'across'
            }
        })
    
    return fig


def enhance_impact_heatmap(fig, model=None, store_id=None, item_id=None):
    """
    Enhance impact heatmap with interactive features.
    
    Args:
        fig: Impact heatmap figure to enhance
        model: UnifiedDataModel instance containing all data
        store_id: Store ID
        item_id: Product ID
        
    Returns:
        go.Figure: Enhanced figure
    """
    # Safety check for figure
    if fig is None or not hasattr(fig, 'data') or fig.data is None:
        logger.warning("Cannot enhance None figure or figure without data")
        return fig
    
    # Create hover template for the heatmap
    hover_template = (
        "<b>Price Adjustment</b>: %{x}<br>"
        "<b>Inventory Adjustment</b>: %{y}<br>"
        "<b>Impact</b>: %{z:.1f}%<br>"
        "<extra></extra>"
    )
    
    try:
        fig.update_traces(hovertemplate=hover_template)
    except Exception as e:
        logger.warning(f"Could not update traces with hover template: {str(e)}")
    
    # Add slider for colorscale
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Impact (%)",
            tickformat=".1f",
            len=0.6,
            thickness=15,
            outlinewidth=1,
            outlinecolor='grey'
        )
    )
    
    return fig


def enhance_kpi_indicators(figures):
    """
    Enhance KPI indicators with interactive features.
    
    Args:
        figures: List of KPI indicator figures
        
    Returns:
        list: List of enhanced figures
    """
    # Safety check for figures
    if figures is None or not isinstance(figures, list):
        logger.warning("Cannot enhance None or non-list figures")
        return figures or []
        
    enhanced_figures = []
    
    for fig in figures:
        # Skip None figures
        if fig is None or not hasattr(fig, 'data') or fig.data is None:
            enhanced_figures.append(fig)  # Keep original
            continue
        # KPI indicators are usually simple, just add basic hover info
        for trace in fig.data:
            if trace.type == 'indicator':
                hover_info = {
                    'value': trace.value,
                    'reference': trace.delta.reference if hasattr(trace, 'delta') and hasattr(trace.delta, 'reference') else None,
                    'title': trace.title.text if hasattr(trace, 'title') and hasattr(trace.title, 'text') else 'Value'
                }
                
                # Store hover info in customdata
                trace.customdata = [hover_info]
                
                # Update layout to make indicator more responsive
                fig.update_layout(
                    margin=dict(l=10, r=10, t=30, b=10),
                    height=150
                )
        
        enhanced_figures.append(fig)
    
    return enhanced_figures


def enhance_recommendations_table(fig):
    """
    Enhance recommendations table with interactive features.
    
    Args:
        fig: Recommendations table figure to enhance
        
    Returns:
        go.Figure: Enhanced figure
    """
    # Safety check for figure
    if fig is None or not hasattr(fig, 'data') or fig.data is None:
        logger.warning("Cannot enhance None figure or figure without data")
        return fig
        
    # For table, we can enable sorting and filtering
    if len(fig.data) > 0 and hasattr(fig.data[0], 'type') and fig.data[0].type == 'table':
        # Enable column sorting
        fig.update_traces(
            columnwidth=[80, 300, 80, 80],
            header=dict(
                align='left',
                font=dict(size=14, color='white'),
                fill=dict(color='#2c3e50')
            ),
            cells=dict(
                align='left',
                font=dict(size=13),
                height=30
            )
        )
    
    return fig
