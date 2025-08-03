"""
Interactive callback functions for the Pizza Predictive Ordering Dashboard.

This module implements the client-side callbacks required for the interactive features:
1. Hover tooltip handling
2. Click events for drill-down
3. Zoom and pan event handling
4. Crossfiltering between charts
"""
from dash import html, dcc, Input, Output, State, ClientsideFunction
import dash_bootstrap_components as dbc
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('interactive_callbacks')


def register_interactive_callbacks(app):
    """
    Register all interactive callbacks for the dashboard.
    
    Args:
        app: Dash app instance
    """
    # Register clientside callbacks for interactive features
    register_tooltip_callback(app)
    register_drill_down_callback(app)
    register_zoom_sync_callback(app)
    register_crossfilter_callback(app)


def register_tooltip_callback(app):
    """
    Register enhanced tooltip callback.
    
    Args:
        app: Dash app instance
    """
    # Create client-side callback for hover tooltip enhancement
    app.clientside_callback(
        ClientsideFunction(
            namespace='interactivity',
            function_name='handleTooltip'
        ),
        Output('tooltip-container', 'children'),
        [Input('tooltip-data', 'data')],
    )


def register_drill_down_callback(app):
    """
    Register drill down callback for chart clicks.
    
    Args:
        app: Dash app instance
    """
    # Server-side callback for handling drill-down clicks
    @app.callback(
        [Output('drill-down-modal', 'is_open'),
         Output('drill-down-modal-title', 'children'),
         Output('drill-down-modal-body', 'children')],
        [Input('drill-down-trigger', 'n_clicks')],
        [State('drill-down-data', 'data'),
         State('drill-down-modal', 'is_open')]
    )
    def handle_drill_down(n_clicks, click_data, is_open):
        """Handle drill-down click events from charts"""
        if n_clicks is None:
            return is_open, "", ""
        """Handle drill-down click events from charts"""
        if click_data is None:
            return is_open, "", ""
        
        try:
            # Parse click data
            data = json.loads(click_data)
            
            # Get info for modal
            chart_type = data.get('chart_type', 'Chart')
            point_data = data.get('point_data', {})
            
            # Create title based on chart type
            if 'Price Sensitivity' in chart_type:
                title = f"Price Point Analysis: ${point_data.get('x', 0):.2f}"
            elif 'Elasticity' in chart_type:
                title = f"Elasticity Analysis: {point_data.get('x', 0):.2f}"
            elif 'Forecast' in chart_type:
                title = f"Date Analysis: {point_data.get('x', '')}"
            else:
                title = f"Drill Down: {chart_type}"
            
            # Create content based on data
            content = []
            
            # Add general info section
            content.append(html.H5("Point Information"))
            
            # Create table with point data
            rows = []
            for key, value in point_data.items():
                if key not in ['pointIndex', 'pointNumber', 'curveNumber']:
                    # Format value based on type
                    if isinstance(value, float):
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    
                    rows.append(
                        html.Tr([
                            html.Td(key.capitalize().replace('_', ' ')),
                            html.Td(formatted_value)
                        ])
                    )
            
            if rows:
                content.append(
                    dbc.Table(
                        [html.Tbody(rows)],
                        bordered=True,
                        hover=True,
                        striped=True,
                        className="mb-4"
                    )
                )
            
            # Add recommendations section based on chart type
            content.append(html.H5("Recommendations"))
            
            if 'Price Sensitivity' in chart_type:
                price = point_data.get('x', 0)
                ratio = point_data.get('y', 0)
                
                content.append(html.P([
                    f"At price point ",
                    html.Strong(f"${price:.2f}"),
                    f", the predicted demand would be ",
                    html.Strong(f"{ratio:.1%}"),
                    " of the current demand."
                ]))
                
                # Add action recommendations based on position
                if ratio > 1.0:
                    content.append(html.P([
                        html.Strong("Recommendation: "),
                        "Consider price reduction to increase demand and potentially total revenue."
                    ]))
                else:
                    content.append(html.P([
                        html.Strong("Recommendation: "),
                        "Consider if the higher profit margin compensates for the reduced demand."
                    ]))
                    
            elif 'Forecast' in chart_type:
                content.append(html.P([
                    "View detailed metrics for this time period or compare with previous periods."
                ]))
                
                # Add buttons for actions
                content.append(
                    dbc.ButtonGroup(
                        [
                            dbc.Button("Compare Periods", color="primary", className="mr-2"),
                            dbc.Button("Detailed Analysis", color="secondary")
                        ],
                        className="mt-2"
                    )
                )
            
            return True, title, content
            
        except Exception as e:
            logger.error(f"Error processing drill-down data: {str(e)}")
            return is_open, "Error", html.P(f"An error occurred: {str(e)}")


def register_zoom_sync_callback(app):
    """
    Register zoom synchronization callback.
    
    Args:
        app: Dash app instance
    """
    # Create client-side callback for zoom synchronization
    app.clientside_callback(
        ClientsideFunction(
            namespace='interactivity',
            function_name='syncZoom'
        ),
        Output('zoom-sync-data', 'data'),
        [Input('zoom-event-data', 'data')],
        prevent_initial_call=True
    )


def register_crossfilter_callback(app):
    """
    Register crossfilter callback.
    
    Args:
        app: Dash app instance
    """
    # Create client-side callback for crossfiltering
    app.clientside_callback(
        ClientsideFunction(
            namespace='interactivity',
            function_name='applyCrossfilter'
        ),
        Output('crossfilter-data', 'data'),
        [Input('filter-event-data', 'data')],
        prevent_initial_call=True
    )


def create_interactive_components():
    """
    Create hidden components needed for interactive features.
    
    Returns:
        list: List of hidden components for interactivity
    """
    components = [
        # Hidden divs for storing interactive data
        dcc.Store(id='tooltip-data', data=None),
        dcc.Store(id='drill-down-data', data=None),
        dcc.Store(id='zoom-event-data', data=None),
        dcc.Store(id='zoom-sync-data', data=None),
        dcc.Store(id='filter-event-data', data=None),
        dcc.Store(id='crossfilter-data', data=None),
        
        # Hidden button to trigger drill-down modal
        html.Button(id='drill-down-trigger', style={'display': 'none'}),
        
        # Container for enhanced tooltips
        html.Div(id='tooltip-container', style={'position': 'relative', 'zIndex': 1000}),
        
        # Modal for drill-down content
        dbc.Modal(
            [
                dbc.ModalHeader(html.H4(id='drill-down-modal-title')),
                dbc.ModalBody(id='drill-down-modal-body'),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-drill-down", className="ml-auto")
                ),
            ],
            id="drill-down-modal",
            size="lg"
        )
    ]
    
    return components


def create_clientside_functions():
    """
    Create JavaScript functions for client-side interactivity.
    
    Returns:
        dict: Dictionary of clientside functions
    """
    # JavaScript code for enhanced tooltips
    tooltip_js = """
    function(tooltipData) {
        if (!tooltipData) return null;
        
        // Parse tooltip data
        const data = JSON.parse(tooltipData);
        
        // Create enhanced tooltip
        const tooltip = document.createElement('div');
        tooltip.className = 'enhanced-tooltip';
        tooltip.style = `
            position: absolute;
            left: ${data.x + 10}px;
            top: ${data.y + 10}px;
            background-color: white;
            border: 1px solid #ccc;
            border-radius: 3px;
            padding: 8px 12px;
            font-size: 12px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            z-index: 1000;
            pointer-events: none;
            max-width: 300px;
        `;
        
        // Add content
        if (data.title) {
            const title = document.createElement('div');
            title.style = 'font-weight: bold; margin-bottom: 5px; border-bottom: 1px solid #eee; padding-bottom: 3px;';
            title.innerHTML = data.title;
            tooltip.appendChild(title);
        }
        
        // Add data points
        if (data.points && data.points.length > 0) {
            const content = document.createElement('div');
            
            data.points.forEach(point => {
                const row = document.createElement('div');
                row.style = 'display: flex; justify-content: space-between; margin-bottom: 2px;';
                
                const label = document.createElement('span');
                label.style = 'font-weight: 600; margin-right: 10px;';
                label.innerHTML = point.label + ':';
                
                const value = document.createElement('span');
                value.innerHTML = point.value;
                
                row.appendChild(label);
                row.appendChild(value);
                content.appendChild(row);
            });
            
            tooltip.appendChild(content);
        }
        
        return tooltip;
    }
    """
    
    # JavaScript code for drill-down functionality
    drill_down_js = """
    function(chartClickData, elementId) {
        if (!chartClickData) return;
        
        // Get chart click data
        const clickData = JSON.parse(chartClickData);
        
        // Find the graph element
        const graphElement = document.getElementById(elementId);
        if (!graphElement) return;
        
        // Store data for server-side callback
        document.getElementById('drill-down-data').data = JSON.stringify({
            chart_type: clickData.chartType,
            point_data: clickData.points[0]
        });
        
        // Trigger server-side callback
        const drillDownTrigger = document.getElementById('drill-down-trigger');
        if (drillDownTrigger) {
            drillDownTrigger.click();
        }
    }
    """
    
    # JavaScript code for zoom synchronization
    zoom_sync_js = """
    function(zoomData) {
        if (!zoomData) return {};
        
        // Parse zoom event data
        const zoomEvent = JSON.parse(zoomData);
        const sourceId = zoomEvent.sourceId;
        const xRange = zoomEvent.xRange;
        const yRange = zoomEvent.yRange;
        
        // Find all connected charts
        if (zoomEvent.linkedCharts && zoomEvent.linkedCharts.length > 0) {
            zoomEvent.linkedCharts.forEach(chartId => {
                const chartElement = document.getElementById(chartId);
                if (!chartElement) return;
                
                // Apply zoom to connected chart
                if (chartElement._fullLayout) {
                    // Only sync x-axis for time series charts
                    if (zoomEvent.syncX) {
                        Plotly.relayout(chartId, {
                            'xaxis.range': xRange
                        });
                    }
                    
                    // Sync both axes for non-time series charts if specified
                    if (zoomEvent.syncY) {
                        Plotly.relayout(chartId, {
                            'yaxis.range': yRange
                        });
                    }
                }
            });
        }
        
        return {processed: true};
    }
    """
    
    # JavaScript code for crossfiltering
    crossfilter_js = """
    function(filterData) {
        if (!filterData) return {};
        
        // Parse filter event data
        const filterEvent = JSON.parse(filterData);
        const sourceId = filterEvent.sourceId;
        const selectedData = filterEvent.selectedData;
        
        // Apply filter to connected charts
        if (filterEvent.linkedCharts && filterEvent.linkedCharts.length > 0 && selectedData) {
            filterEvent.linkedCharts.forEach(chartId => {
                const chartElement = document.getElementById(chartId);
                if (!chartElement || !chartElement._fullData) return;
                
                // Extract dimension values from selected data
                let dimensionValues = [];
                if (filterEvent.dimension === 'x') {
                    dimensionValues = selectedData.points.map(p => p.x);
                } else if (filterEvent.dimension === 'y') {
                    dimensionValues = selectedData.points.map(p => p.y);
                }
                
                // Apply filter to this chart
                for (let i = 0; i < chartElement._fullData.length; i++) {
                    const trace = chartElement._fullData[i];
                    if (!trace.x || !trace.y) continue;
                    
                    // Create a mask of points to highlight
                    const mask = trace.x.map((x, idx) => {
                        const value = filterEvent.dimension === 'x' ? x : trace.y[idx];
                        return dimensionValues.includes(value);
                    });
                    
                    // Update trace opacity
                    Plotly.restyle(chartId, {
                        'opacity': mask.map(m => m ? 1.0 : 0.3),
                        'selectedpoints': mask.map((m, i) => m ? i : null).filter(i => i !== null)
                    }, [i]);
                }
            });
        }
        
        return {processed: true};
    }
    """
    
    # Return dictionary of functions
    return {
        'handleTooltip': tooltip_js,
        'handleDrillDown': drill_down_js,
        'syncZoom': zoom_sync_js,
        'applyCrossfilter': crossfilter_js
    }