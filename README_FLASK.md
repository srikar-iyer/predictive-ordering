# Flask-based Predictive Ordering System

This is the Flask implementation of the Predictive Ordering System visualization UI. The application provides interactive charts and visualizations for sales forecasting, inventory management, and price optimization using Flask and Plotly.

## Features

- Sales forecast visualization with confidence intervals
- Price elasticity analysis and price optimization tools
- Inventory management with safety stock tracking
- Integrated business view showing combined impacts
- Interactive filters and toggles for data exploration
- Export capabilities for charts (PNG, SVG) and data (CSV)

## Requirements

- Python 3.7+
- Flask
- Plotly
- Pandas
- NumPy
- Additional requirements listed in requirements.txt

## Installation

1. Make sure you have all the required packages installed:

```bash
pip install flask plotly pandas numpy
```

Or use the requirements.txt file:

```bash
pip install -r requirements.txt
```

## Running the Application

To start the Flask application:

```bash
python app.py
```

The application will be available at: http://127.0.0.1:8050/

## Project Structure

```
├── app.py                  # Main Flask application
├── static/                 # Static assets
│   ├── css/                # CSS stylesheets
│   │   └── style.css       # Custom styles
│   └── js/                 # JavaScript files
│       └── main.js         # Custom JavaScript
├── templates/              # HTML templates
│   ├── base.html           # Base template with common elements
│   ├── index.html          # Home page
│   ├── dashboard.html      # Main dashboard
│   ├── forecast.html       # Sales forecast page
│   ├── pricing.html        # Price optimization page
│   ├── inventory.html      # Inventory management page
│   └── integrated.html     # Integrated view page
└── src/                    # Source code (existing structure)
    └── models/             # Data models and visualization components
```

## API Endpoints

The application provides several API endpoints for chart data:

- `/api/forecast-data`: Provides sales forecast data with confidence intervals
- `/api/elasticity-data`: Returns price elasticity distribution data
- `/api/price-sensitivity`: Returns price sensitivity curve data
- `/api/profit-impact`: Returns profit impact of price changes data

## Customization

You can customize the application by:

1. Modifying templates in the `templates` directory
2. Adding new API endpoints in `app.py`
3. Updating styles in `static/css/style.css`
4. Enhancing client-side functionality in `static/js/main.js`

## Integration with Existing Codebase

This Flask implementation integrates with the existing visualization components from:
- `src/models/plotly_visualizations.py`
- `src/models/integrated_visualizations.py`

It's designed to work with the existing data models and provide a web-based interface for the visualization system.