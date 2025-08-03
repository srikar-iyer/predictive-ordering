"""
Export capabilities for Plotly visualizations.

This module provides functions to add export capabilities to Plotly charts,
allowing users to export charts as images (PNG, JPEG, SVG, PDF) and data (CSV, Excel).
"""
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import base64
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('chart_export_utils')


def configure_chart_export_buttons(config=None):
    """
    Configure chart export buttons for Plotly figures.
    
    Args:
        config: Existing config dictionary to update (optional)
        
    Returns:
        dict: Updated configuration dictionary with export options
    """
    # Start with default configuration if not provided
    if config is None:
        config = {}
    
    # Ensure basic config options are set
    default_config = {
        'displayModeBar': True,
        'responsive': True,
        'scrollZoom': True,
    }
    
    # Update with provided config
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
    
    # Configure image export options
    image_config = {
        'format': 'png',  # Default format
        'filename': 'chart_export',
        'scale': 2,  # High quality for printing
        'height': None,  # Maintain aspect ratio
        'width': None,
    }
    
    # Add image export configuration
    if 'toImageButtonOptions' not in config:
        config['toImageButtonOptions'] = image_config
    
    # Add additional buttons for different export formats
    export_buttons = [
        {
            'name': 'Download as PNG',
            'icon': 'camera',
            'method': 'toImage',
            'format': 'png',
            'filename': 'chart_export_png',
            'title': 'Download as PNG'
        },
        {
            'name': 'Download as SVG',
            'icon': 'camera',
            'method': 'toImage',
            'format': 'svg',
            'filename': 'chart_export_svg',
            'title': 'Download as SVG'
        },
        {
            'name': 'Download as PDF',
            'icon': 'camera',
            'method': 'toImage',
            'format': 'pdf',
            'filename': 'chart_export_pdf',
            'title': 'Download as PDF'
        },
        {
            'name': 'Download Data as CSV',
            'icon': 'download',
            'method': 'downloadImage',
            'format': 'csv',
            'filename': 'chart_data_csv',
            'title': 'Download data as CSV'
        },
        {
            'name': 'Download Data as Excel',
            'icon': 'download',
            'method': 'downloadImage',
            'format': 'xlsx',
            'filename': 'chart_data_xlsx',
            'title': 'Download data as Excel'
        }
    ]
    
    # Configure custom buttons
    if 'modeBarButtonsToAdd' not in config:
        config['modeBarButtonsToAdd'] = []
    
    # These buttons will be handled via custom JavaScript in the Dash app
    config['modeBarButtonsToAdd'].extend(['downloadasimage', 'downloaddata'])
    
    return config


def add_export_menu_to_figure(fig):
    """
    Add export menu to a Plotly figure.
    
    Args:
        fig: Plotly figure to enhance with export menu
        
    Returns:
        go.Figure: Enhanced figure with export menu
    """
    # Safety check for None figure
    if fig is None:
        logger.warning("Cannot add export menu to None figure")
        return fig
        
    # Check if figure has layout attribute
    if not hasattr(fig, 'layout') or fig.layout is None:
        logger.warning("Figure has no valid layout attribute")
        return fig
        
    # Add a button in the top right corner
    try:
        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    showactive=False,
                    buttons=[
                        dict(
                            args=[{"format": "png", "scale": 2, "filename": "chart_export"}],
                            label="Download as PNG",
                            method="update"
                        ),
                        dict(
                            args=[{"format": "svg", "scale": 2, "filename": "chart_export"}],
                            label="Download as SVG",
                            method="update"
                        ),
                        dict(
                            args=[{"format": "pdf", "scale": 2, "filename": "chart_export"}],
                            label="Download as PDF", 
                            method="update"
                        ),
                        dict(
                            args=[{"format": "csv", "filename": "chart_data"}],
                            label="Download data as CSV",
                            method="relayout"
                        ),
                        dict(
                            args=[{"format": "xlsx", "filename": "chart_data"}],
                            label="Download data as Excel",
                            method="relayout"
                        )
                    ],
                    x=1.0,
                    y=1.1,
                    xanchor='right',
                    yanchor='top',
                    bgcolor='#FFFFFF',
                    bordercolor='#CCCCCC',
                    borderwidth=1,
                    pad={"r": 10, "t": 10}
                )
            ]
        )
    except Exception as e:
        logger.warning(f"Could not update layout with export menu: {str(e)}")
    
    return fig


def extract_figure_data(fig):
    """
    Extract data from a Plotly figure for export.
    
    Args:
        fig: Plotly figure to extract data from
        
    Returns:
        pd.DataFrame: DataFrame containing trace data from the figure
    """
    try:
        # Initialize data dictionary
        data_dict = {}
        
        # Extract data from each trace
        for i, trace in enumerate(fig.data):
            # Get trace name
            trace_name = getattr(trace, 'name', f'Trace_{i+1}')
            
            # Extract x and y values
            if hasattr(trace, 'x') and hasattr(trace, 'y'):
                x_values = trace.x
                y_values = trace.y
                
                # Handle different trace types
                if trace.type in ['bar', 'scatter', 'line']:
                    data_dict[f"{trace_name}_x"] = x_values
                    data_dict[f"{trace_name}_y"] = y_values
                elif trace.type == 'histogram':
                    data_dict[f"{trace_name}"] = x_values
                elif trace.type == 'pie':
                    data_dict["Label"] = trace.labels
                    data_dict["Value"] = trace.values
                elif trace.type == 'heatmap':
                    data_dict["x"] = x_values
                    data_dict["y"] = y_values
                    data_dict["z"] = trace.z
                    
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        return df
    
    except Exception as e:
        logger.error(f"Error extracting figure data: {str(e)}")
        # Return empty DataFrame if extraction fails
        return pd.DataFrame()


def prepare_csv_export(fig, filename="chart_data"):
    """
    Prepare CSV export data for a Plotly figure.
    
    Args:
        fig: Plotly figure to export
        filename: Base filename for export (without extension)
        
    Returns:
        str: Base64 encoded CSV data for download
    """
    try:
        # Extract data
        df = extract_figure_data(fig)
        
        # Convert to CSV
        csv_data = df.to_csv(index=False)
        
        # Encode as base64
        b64_csv = base64.b64encode(csv_data.encode('utf-8')).decode('utf-8')
        
        # Return data URI
        return f"data:text/csv;base64,{b64_csv}"
    
    except Exception as e:
        logger.error(f"Error preparing CSV export: {str(e)}")
        return None


def prepare_excel_export(fig, filename="chart_data"):
    """
    Prepare Excel export data for a Plotly figure.
    
    Args:
        fig: Plotly figure to export
        filename: Base filename for export (without extension)
        
    Returns:
        str: Base64 encoded Excel data for download
    """
    try:
        # Extract data
        df = extract_figure_data(fig)
        
        # Create Excel file in memory
        import io
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        
        # Encode as base64
        b64_excel = base64.b64encode(excel_buffer.read()).decode('utf-8')
        
        # Return data URI
        return f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}"
    
    except Exception as e:
        logger.error(f"Error preparing Excel export: {str(e)}")
        return None


def configure_dash_app_for_exports(app):
    """
    Configure a Dash app for chart exports by adding necessary JavaScript.
    
    Args:
        app: Dash app to configure
        
    Returns:
        dash.Dash: Configured app
        
    Raises:
        ValueError: If app is None or doesn't have the clientside_callback method
    """
    # Validate app parameter
    if app is None:
        raise ValueError("App cannot be None")
    
    if not hasattr(app, 'clientside_callback'):
        raise ValueError("App must be a valid Dash application with a clientside_callback method")
        
    # Add custom JavaScript for data exports
    export_js = '''
    window.dash_clientside = Object.assign({}, window.dash_clientside, {
        export: {
            downloadData: function(clickData, graphId) {
                if (!clickData) return;
                
                // Get graph data and figure
                const graphDiv = document.getElementById(graphId);
                if (!graphDiv || !graphDiv.data || !Array.isArray(graphDiv.data)) return;
                
                // Extract data from the graph
                const data = [];
                const headers = ['x', 'y', 'name'];
                
                // Build CSV header
                let csv = headers.join(',') + '\\n';
                
                // Process each trace
                graphDiv.data.forEach(trace => {
                    // First check if trace.x and trace.y exist and are arrays with length property
                    if (trace.x && trace.y && Array.isArray(trace.x) && Array.isArray(trace.y) && 
                        typeof trace.x.length === 'number' && typeof trace.y.length === 'number') {
                        const name = trace.name || 'Trace';
                        for (let i = 0; i < Math.min(trace.x.length, trace.y.length); i++) {
                            const row = [trace.x[i], trace.y[i], name];
                            csv += row.join(',') + '\\n';
                        }
                    }
                });
                
                // Try to get a filename from meta
                let filename = 'chart_data';
                try {
                    if (graphDiv._fullLayout && graphDiv._fullLayout.meta && 
                        graphDiv._fullLayout.meta.export_filename) {
                        filename = graphDiv._fullLayout.meta.export_filename;
                    } else if (graphDiv.layout && graphDiv.layout.meta && 
                               graphDiv.layout.meta.export_filename) {
                        filename = graphDiv.layout.meta.export_filename;
                    }
                } catch (e) {
                    console.warn("Could not extract export filename from meta", e);
                }
                
                // Create download link
                const blob = new Blob([csv], { type: 'text/csv' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.setAttribute('hidden', '');
                a.setAttribute('href', url);
                a.setAttribute('download', `${filename}.csv`);
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                
                return;
            },
            
            // Additional function to check and apply meta-based export settings
            setupExportButtons: function() {
                // Find all charts with export options meta
                document.querySelectorAll('.js-plotly-plot').forEach(chart => {
                    try {
                        const chartDiv = chart;
                        let meta = null;
                        
                        // Try different ways to access meta
                        if (chartDiv._fullLayout && chartDiv._fullLayout.meta) {
                            meta = chartDiv._fullLayout.meta;
                        } else if (chartDiv.layout && chartDiv.layout.meta) {
                            meta = chartDiv.layout.meta;
                        }
                        
                        if (meta && meta.export_options) {
                            // Customize export options based on meta
                            const exportOptions = meta.export_options;
                            
                            // Update toImageButtonOptions if available
                            if (chartDiv._context && chartDiv._context.toImageButtonOptions) {
                                chartDiv._context.toImageButtonOptions.filename = exportOptions.filename || 'chart_export';
                            }
                        }
                    } catch (e) {
                        console.warn("Error setting up export buttons:", e);
                    }
                });
            }
        }
    });
    '''
    
    # Add the JavaScript to the Dash app with proper output/input configuration
    try:
        # Create valid output and input values
        from dash.dependencies import Output, Input
        app.clientside_callback(
            export_js,
            Output('dummy-output', 'children'),
            Input('dummy-input', 'children')
        )
        
        # Add setup call to document.onload
        app.clientside_callback(
            "function() { setTimeout(() => { if (window.dash_clientside && window.dash_clientside.export) window.dash_clientside.export.setupExportButtons(); }, 1000); return null; }",
            Output('dummy-output', 'data'),
            Input('dummy-input', 'data')
        )
    except Exception as e:
        logger.error(f"Error configuring clientside callback: {str(e)}")
        # Fallback to basic configuration
        app.clientside_callback(
            export_js,
            output='dummy-output.children',
            inputs='dummy-input.children'
        )
    
    return app


def enhance_figure_with_exports(fig, title=None):
    """
    Enhance a Plotly figure with export capabilities.
    
    Args:
        fig: Plotly figure to enhance
        title: Title to use for export filenames (optional)
        
    Returns:
        go.Figure: Enhanced figure with export menu
    """
    # Safety check for None figure
    if fig is None:
        logger.warning("Cannot enhance None figure with exports")
        return fig
        
    # Add export menu
    fig = add_export_menu_to_figure(fig)
    
    # Clean title for filenames
    clean_title = None
    if title:
        # Remove special characters and spaces
        clean_title = ''.join(c if c.isalnum() else '_' for c in title)
    
    # Add filename to figure meta (not metadata, which is not supported)
    if not hasattr(fig, 'layout'):
        logger.warning("Figure has no layout attribute")
        return fig
        
    if fig.layout is None:
        fig.layout = go.Layout()
        
    if not hasattr(fig.layout, 'meta'):
        fig.layout.meta = {}
    elif fig.layout.meta is None:
        fig.layout.meta = {}
        
    fig.layout.meta['export_filename'] = clean_title or 'chart_export'
    
    # Add export button data for client-side access
    export_data = {
        'filename': clean_title or 'chart_export',
        'formats': ['png', 'svg', 'pdf', 'csv', 'xlsx']
    }
    fig.layout.meta['export_options'] = export_data
    
    return fig