# Add these callback registrations to plotly_dashboard.py

# Callback to update the inventory chart
@app.callback(
    Output("inventory-chart", "figure"),
    [Input("apply-filters", "n_clicks"),
     Input("apply-stock-adjustment", "n_clicks")],
    [State("store-dropdown", "value"),
     State("product-dropdown", "value"),
     State("weather-toggle", "value"),
     State("location-input", "value"),
     State("stock-adjustment-input", "value")]
)
def inventory_chart_callback(n_clicks, adjust_clicks, store, product, apply_weather, location, stock_adjustment):
    return update_inventory_chart(n_clicks, adjust_clicks, store, product, apply_weather, 
                                location, stock_adjustment, combined_data, weather_service, app)

# Callback to update stock recommendations
@app.callback(
    Output("stock-recommendations", "children"),
    [Input("apply-filters", "n_clicks"),
     Input("apply-stock-adjustment", "n_clicks")],
    [State("store-dropdown", "value"),
     State("product-dropdown", "value"),
     State("stock-adjustment-input", "value")]
)
def stock_recommendations_callback(n_clicks, adjust_clicks, store, product, stock_adjustment):
    return update_stock_recommendations(n_clicks, adjust_clicks, store, product, 
                                      stock_adjustment, combined_data, html, dbc, app)

# Callback to update stock velocity chart
@app.callback(
    Output("stock-velocity-chart", "figure"),
    [Input("apply-filters", "n_clicks")],
    [State("store-dropdown", "value"),
     State("product-dropdown", "value")]
)
def stock_velocity_callback(n_clicks, store, product):
    return update_stock_velocity_chart(n_clicks, store, product, combined_data, go, np)

# Callback to update stock penalty chart
@app.callback(
    Output("stock-penalty-chart", "figure"),
    [Input("apply-filters", "n_clicks")],
    [State("store-dropdown", "value"),
     State("product-dropdown", "value")]
)
def stock_penalty_callback(n_clicks, store, product):
    return update_stock_penalty_chart(n_clicks, store, product, combined_data, go, np)

# Callback to update inventory summary statistics
@app.callback(
    Output("inventory-summary-stats", "children"),
    [Input("apply-filters", "n_clicks")],
    [State("store-dropdown", "value")]
)
def inventory_summary_callback(n_clicks, store):
    return update_inventory_summary_stats(n_clicks, store, combined_data, html, dbc)