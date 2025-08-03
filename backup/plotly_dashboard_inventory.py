# This file contains the inventory management functionality for the plotly dashboard
# It will be imported in the main plotly_dashboard.py file

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Callback to update the inventory chart
def update_inventory_chart(n_clicks, adjust_clicks, store, product, apply_weather, location, stock_adjustment, adjustment_date=None):
    try:
        # Import required data and modules from parent module
        from plotly_dashboard import combined_data, weather_service, app
        
        # Input validation
        if combined_data is None:
            # Handle case where data is missing
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No Data Available",
                annotations=[{
                    'text': "Error: Required data is missing. Please check that data files are loaded correctly.",
                    'showarrow': False,
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'font': {'size': 16, 'color': 'red'}
                }]
            )
            return empty_fig
            
        if store is None or product is None:
            # Handle case where selection is missing
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No Selection Made",
                annotations=[{
                    'text': "Please select a store and product to view inventory data.",
                    'showarrow': False,
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'font': {'size': 16}
                }]
            )
            return empty_fig
        
        # Filter data for selected store and product with improved error handling
        try:
            filtered_data = combined_data[(combined_data['Store_Id'] == store) & 
                                        (combined_data['Item'] == product)].copy()
                                        
            # Validate that we have data after filtering
            if len(filtered_data) == 0:
                # No data found for this selection
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    title=f"No Data Available for Selected Product",
                    annotations=[{
                        'text': f"No inventory data found for Store {store}, Product {product}.",
                        'showarrow': False,
                        'xref': 'paper',
                        'yref': 'paper',
                        'x': 0.5,
                        'y': 0.5,
                        'font': {'size': 14}
                    }]
                )
                return empty_fig
        except KeyError as ke:
            # Handle the case where column names don't match
            print(f"KeyError while filtering data: {ke}")
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="Data Structure Error",
                annotations=[{
                    'text': f"Error: Could not filter data. Required column missing: {str(ke)}",
                    'showarrow': False,
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'font': {'size': 14, 'color': 'red'}
                }]
            )
            return empty_fig
                                    
        # Add derived metrics for inventory analysis if they don't already exist
        if 'Inventory_Value' not in filtered_data.columns:
            filtered_data['Inventory_Value'] = filtered_data['Stock_Level'] * filtered_data['Price']
            
        # Use the correct columns for average movement
        if 'Avg_Weekly_Sales_4W' in filtered_data.columns:
            filtered_data['Week_4_Avg_Movement'] = filtered_data['Avg_Weekly_Sales_4W']
        else:
            filtered_data['Week_4_Avg_Movement'] = filtered_data['Recent_Daily_Sales'] * 7  # estimate weekly
            
        if 'Avg_Weekly_Sales_13W' in filtered_data.columns:
            filtered_data['Week_13_Avg_Movement'] = filtered_data['Avg_Weekly_Sales_13W']
        else:
            filtered_data['Week_13_Avg_Movement'] = filtered_data['Week_4_Avg_Movement']  # fallback
            
        # Use the correct column for weeks of supply
        if 'Stock_Coverage_Weeks' in filtered_data.columns:
            filtered_data['Weeks_of_Supply'] = filtered_data['Stock_Coverage_Weeks']
        elif 'Weeks_Of_Stock' in filtered_data.columns:
            filtered_data['Weeks_of_Supply'] = filtered_data['Weeks_Of_Stock']
        else:
            # Calculate weeks of supply based on average weekly movement
            week_4_movement = filtered_data['Week_4_Avg_Movement'].replace(0, 0.01)  # Replace zeros with small value
            filtered_data['Weeks_of_Supply'] = filtered_data['Stock_Level'] / week_4_movement
            
        # Calculate turnover rate
        filtered_data['Turnover_Rate'] = filtered_data['Week_4_Avg_Movement'] / filtered_data['Stock_Level'].replace(0, 0.01)
        filtered_data['Turnover_Rate'] = filtered_data['Turnover_Rate'].fillna(0.01)  # Ensure nonzero values
        
        # Apply manual stock adjustment if provided
        key = f"{store}_{product}"
        date_key = None
        
        if adjust_clicks and stock_adjustment is not None:
            # Store the adjustment in app's state
            app.manual_stock_adjustments[key] = stock_adjustment
            
            # Store with date if provided
            if adjustment_date is not None:
                try:
                    # Convert string date to datetime if needed
                    if isinstance(adjustment_date, str):
                        adj_date = pd.to_datetime(adjustment_date).date()
                    else:
                        adj_date = adjustment_date
                        
                    date_key = f"{store}_{product}_{adj_date}"
                    if not hasattr(app, 'manual_stock_adjustments_with_dates'):
                        app.manual_stock_adjustments_with_dates = {}
                    app.manual_stock_adjustments_with_dates[date_key] = stock_adjustment
                except:
                    # If date conversion fails, fall back to just using the regular key
                    pass
            
        # Apply any stored adjustment for this store/product combo
        update_applied = False
        
        # First try to apply date-specific adjustment if available
        if adjustment_date is not None:
            try:
                adj_date = pd.to_datetime(adjustment_date).date() if isinstance(adjustment_date, str) else adjustment_date
                date_key = f"{store}_{product}_{adj_date}"
                
                if hasattr(app, 'manual_stock_adjustments_with_dates') and date_key in app.manual_stock_adjustments_with_dates:
                    # Find the specific date in the data
                    match_date = pd.Timestamp(adj_date)
                    date_idx = filtered_data[filtered_data['Date'] == match_date].index
                    
                    if len(date_idx) > 0:
                        # Update the stock level for the specific date
                        filtered_data.loc[date_idx, 'Stock_Level'] = app.manual_stock_adjustments_with_dates[date_key]
                        update_applied = True
                        
                        # Recalculate weeks of stock based on new stock level
                        avg_daily_sales = filtered_data.loc[date_idx, 'Recent_Daily_Sales'].values[0]
                        if avg_daily_sales > 0:
                            weeks_of_stock = app.manual_stock_adjustments_with_dates[date_key] / (avg_daily_sales * 7)
                        else:
                            weeks_of_stock = 4.0
                            
                        filtered_data.loc[date_idx, 'Weeks_Of_Stock'] = weeks_of_stock
                        filtered_data.loc[date_idx, 'Weeks_of_Supply'] = weeks_of_stock
                        
                        # Update stock status
                        if weeks_of_stock < 1:
                            stock_status = 'Low'
                        elif weeks_of_stock <= 3:
                            stock_status = 'Adequate'
                        else:
                            stock_status = 'Excess'
                            
                        filtered_data.loc[date_idx, 'Stock_Status'] = stock_status
            except:
                # If date-specific application fails, continue to try the regular key
                pass
        
        # If no date-specific update was applied, try the regular key (for backward compatibility)
        if not update_applied and key in app.manual_stock_adjustments:
            # Find the latest date in the data
            latest_date = filtered_data['Date'].max()
            latest_idx = filtered_data[filtered_data['Date'] == latest_date].index
            
            if len(latest_idx) > 0:
                # Update the stock level for the latest date
                filtered_data.loc[latest_idx, 'Stock_Level'] = app.manual_stock_adjustments[key]
                
                # Recalculate weeks of stock based on new stock level
                avg_daily_sales = filtered_data.loc[latest_idx, 'Recent_Daily_Sales'].values[0]
                if avg_daily_sales > 0:
                    weeks_of_stock = app.manual_stock_adjustments[key] / (avg_daily_sales * 7)
                else:
                    weeks_of_stock = 4.0
                    
                filtered_data.loc[latest_idx, 'Weeks_Of_Stock'] = weeks_of_stock
                filtered_data.loc[latest_idx, 'Weeks_of_Supply'] = weeks_of_stock
                
                # Update stock status
                if weeks_of_stock < 1:
                    stock_status = 'Low'
                elif weeks_of_stock <= 3:
                    stock_status = 'Adequate'
                else:
                    stock_status = 'Excess'
                    
                filtered_data.loc[latest_idx, 'Stock_Status'] = stock_status
        
        # Create figure with separate rows for each subplot - each subplot gets its own row
        # Further increased vertical spacing and adjusted heights for better visualization
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.15,  # Reduced spacing for better proportions
            subplot_titles=("Stock Level", "Daily Demand"),
            row_heights=[0.65, 0.35]  # Adjusted to give more space to stock level
        )
        
        # Add stock level to top subplot
        fig.add_trace(
            go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data['Stock_Level'],
                mode='lines',
                name='Stock Level',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add safety stock and target stock lines if available
        if len(filtered_data) > 0:
            # Get latest safety stock and target stock values
            latest_date = filtered_data['Date'].max()
            latest_data = filtered_data[filtered_data['Date'] == latest_date]
            
            if 'Safety_Stock' in latest_data.columns:
                safety_stock = latest_data['Safety_Stock'].values[0]
                
                fig.add_trace(
                    go.Scatter(
                        x=filtered_data['Date'],
                        y=[safety_stock] * len(filtered_data),
                        mode='lines',
                        name='Safety Stock',
                        line=dict(color='red', dash='dot')
                    ),
                    row=1, col=1
                )
            
            if 'Target_Stock' in latest_data.columns:
                target_stock = latest_data['Target_Stock'].values[0]
                
                fig.add_trace(
                    go.Scatter(
                        x=filtered_data['Date'],
                        y=[target_stock] * len(filtered_data),
                        mode='lines',
                        name='Target Stock',
                        line=dict(color='green', dash='dot')
                    ),
                    row=1, col=1
                )
        
        # Add both 4-week and 13-week average movement lines to bottom subplot
        fig.add_trace(
            go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data['Week_4_Avg_Movement'] / 7,  # Convert to daily for comparison
                mode='lines',
                name='4-Week Avg Daily Movement',
                line=dict(color='purple', dash='dot')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data['Week_13_Avg_Movement'] / 7,  # Convert to daily for comparison
                mode='lines',
                name='13-Week Avg Daily Movement',
                line=dict(color='darkblue', dash='dot')
            ),
            row=2, col=1
        )
        
        # Add predicted demand on secondary y-axis
        if apply_weather and location:
            try:
                # Get weather forecast with proper validation
                if not isinstance(location, str) or len(location.strip()) == 0:
                    raise ValueError("Invalid location provided")
                    
                weather_forecast = weather_service.get_weather_forecast(location)
                if not weather_forecast or not isinstance(weather_forecast, list):
                    raise ValueError("Could not retrieve weather forecast")
                
                # Get product information for product-specific adjustments
                product_name = filtered_data['Product'].iloc[0] if len(filtered_data) > 0 else None
                
                # Get the base demand values - use average weekly movement divided by 7 for daily
                if 'Week_4_Avg_Movement' not in filtered_data.columns:
                    raise KeyError("Required column 'Week_4_Avg_Movement' not found in data")
                    
                base_demand = filtered_data['Week_4_Avg_Movement'].values / 7
                
                # Validate base_demand
                if len(base_demand) == 0 or np.all(base_demand == 0):
                    raise ValueError("No demand data available for weather adjustment")
                
                # Adjust demand based on weather forecast
                adjusted_demand, adjustment_factors = weather_service.get_weather_adjusted_demand(
                    base_demand, weather_forecast, product_type=product_name
                )
                
                # Add weather-adjusted demand to bottom subplot
                fig.add_trace(
                    go.Scatter(
                        x=filtered_data['Date'],
                        y=adjusted_demand,
                        mode='lines',
                        name='Weather-Adjusted Demand',
                        line=dict(color='orange')
                    ),
                    row=2, col=1
                )
                
                # Add annotation explaining weather impact
                avg_factor = sum(adjustment_factors) / len(adjustment_factors) if adjustment_factors else 1.0
                impact_direction = "increase" if avg_factor > 1.0 else "decrease" if avg_factor < 1.0 else "no change"
                impact_pct = abs(round((avg_factor - 1.0) * 100, 1))
                
                if impact_pct > 1.0:  # Only add annotation if there's a meaningful impact
                    fig.add_annotation(
                        text=f"Weather impact: {impact_pct}% {impact_direction} in demand",
                        xref="paper", yref="paper",
                        x=0.98, y=0.35,
                        showarrow=False,
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="orange",
                        borderwidth=1,
                        font=dict(color="darkblue", size=10)
                    )
            except ValueError as ve:
                print(f"Weather data error: {ve}")
                # Add an annotation about the weather error instead of failing
                fig.add_annotation(
                    text=f"Weather data unavailable: {str(ve)}",
                    xref="paper", yref="paper",
                    x=0.98, y=0.35,
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="red",
                    borderwidth=1,
                    font=dict(color="red", size=10)
                )
            except Exception as e:
                print(f"Error applying weather adjustment: {e}")
        
        # Add recent daily sales for comparison to bottom subplot
        fig.add_trace(
            go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data['Recent_Daily_Sales'],
                mode='lines',
                name='Recent Daily Sales',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        # Add order points with labels for key inventory metrics
        latest_idx = filtered_data['Date'].idxmax()
        if latest_idx is not None:
            latest_data = filtered_data.loc[latest_idx]
            
            # Add annotation for current inventory metrics
            fig.add_annotation(
                x=latest_data['Date'],
                y=latest_data['Stock_Level'],
                text=f"On Hand: {int(latest_data['Stock_Level'])}<br>Value: ${latest_data['Inventory_Value']:.2f}<br>Weeks Supply: {latest_data['Weeks_of_Supply']:.1f}<br>Turnover: {latest_data['Turnover_Rate']:.2f}",
                showarrow=True,
                arrowhead=1,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                row=1, col=1
            )
        
        # Add order points if available
        if 'Orders_Received' in filtered_data.columns:
            orders = filtered_data[filtered_data['Orders_Received'] > 0]
            if len(orders) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=orders['Date'],
                        y=orders['Stock_Level'],
                        mode='markers',
                        name='Order Received',
                        marker=dict(color='green', size=10, symbol='triangle-up')
                    ),
                    row=1, col=1
                )
        
        # Update layout with increased spacing and separated titles
        product_name = filtered_data['Product'].iloc[0] if len(filtered_data) > 0 else "Product"
        
        fig.update_layout(
            title={
                'text': f"Inventory Projection for {product_name}",
                'y': 0.95,  # Moved title higher for better spacing
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 22}  # Increased font size
            },
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02,  # Adjusted legend position
                xanchor="right", 
                x=1,
                font=dict(size=12),  # Increased font size
                itemsizing='constant'
            ),
            template="plotly_white",
            height=1200,  # Significantly increased height for better visibility
            margin=dict(t=180, b=100, l=100, r=100),  # Adjusted margins
            hovermode='closest'  # Better hover interaction
        )
        
        # Update axis titles for both subplots with improved positioning
        fig.update_yaxes(
            title_text="Stock Level (units)",
            title_standoff=10,  # More space for y-axis title
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Daily Demand (units)",
            title_standoff=10,  # More space for y-axis title
            row=2, col=1
        )
        fig.update_xaxes(
            title_text="Date",
            title_standoff=20,  # More space for x-axis title
            row=2, col=1
        )
        
        # Update subplot titles position for better spacing with full-width layout
        fig.update_annotations(font_size=20, y=0.95, selector=dict(text="Stock Level"))
        fig.update_annotations(font_size=20, y=0.32, selector=dict(text="Daily Demand"))
        
        return fig
        
    except ImportError as ie:
        # Specifically handle import errors
        print(f"Import error in update_inventory_chart: {ie}")
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Module Import Error",
            annotations=[{
                'text': "Error: Could not import required modules. Please check your installation.",
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'font': {'size': 16, 'color': 'red'}
            }]
        )
        return empty_fig
    except ValueError as ve:
        # Handle value errors (data conversion issues)
        print(f"Value error in update_inventory_chart: {ve}")
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Data Error",
            annotations=[{
                'text': f"Error processing data values: {str(ve)}",
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'font': {'size': 16, 'color': 'red'}
            }]
        )
        return empty_fig
    except KeyError as ke:
        # Handle missing key errors (column not found)
        print(f"Key error in update_inventory_chart: {ke}")
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Missing Data Column",
            annotations=[{
                'text': f"Error: Required data column not found. Please check your data.",
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'font': {'size': 16, 'color': 'red'}
            }]
        )
        return empty_fig
    except Exception as e:
        # General error handler for unexpected errors
        print(f"Error updating inventory chart: {e}")
        # Return an empty figure with error message
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error Loading Inventory Data",
            annotations=[{
                'text': f"Error: {str(e)}",
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'font': {'size': 16, 'color': 'red'}
            }]
        )
        return empty_fig

# Function to update stock recommendations
def update_stock_recommendations(n_clicks, adjust_clicks, store, product, stock_adjustment, combined_data, html, dbc, app, adjustment_date=None):
    if combined_data is None or store is None or product is None:
        return html.P("No data available for stock recommendations.")
    
    try:
        # Get the latest data for the selected store and product
        latest_date = combined_data['Date'].max()
        latest_data = combined_data[(combined_data['Store_Id'] == store) & 
                                 (combined_data['Item'] == product) & 
                                 (combined_data['Date'] == latest_date)]
        
        if len(latest_data) == 0:
            return html.P("No data available for the selected product.")
        
        # Create a copy of the data to add derived metrics
        latest_data = latest_data.copy()
        
        # Apply stock adjustment if provided
        key = f"{store}_{product}"
        current_stock = latest_data['Stock_Level'].values[0]
        
        # First check for date-specific adjustment if available
        if adjustment_date is not None:
            try:
                adj_date = pd.to_datetime(adjustment_date).date() if isinstance(adjustment_date, str) else adjustment_date
                date_key = f"{store}_{product}_{adj_date}"
                
                # Store the adjustment in app's state if we're adjusting stock now
                if adjust_clicks and stock_adjustment is not None:
                    if not hasattr(app, 'manual_stock_adjustments_with_dates'):
                        app.manual_stock_adjustments_with_dates = {}
                    app.manual_stock_adjustments_with_dates[date_key] = stock_adjustment
                
                # Use the stored date-specific adjustment if it exists
                if hasattr(app, 'manual_stock_adjustments_with_dates') and date_key in app.manual_stock_adjustments_with_dates:
                    current_stock = app.manual_stock_adjustments_with_dates[date_key]
                    latest_data['Stock_Level'] = current_stock
            except:
                # If date-specific application fails, continue to try the regular key
                pass
        
        # Fall back to the standard key if no date-specific adjustment was found
        if adjust_clicks and stock_adjustment is not None:
            app.manual_stock_adjustments[key] = stock_adjustment
            current_stock = stock_adjustment
            latest_data['Stock_Level'] = current_stock
        elif key in app.manual_stock_adjustments:
            current_stock = app.manual_stock_adjustments[key]
            latest_data['Stock_Level'] = current_stock
        
        # Map or calculate required metrics
        # Week 4 Average Movement
        if 'Avg_Weekly_Sales_4W' in latest_data.columns and not pd.isna(latest_data['Avg_Weekly_Sales_4W'].values[0]):
            week_4_avg_movement = latest_data['Avg_Weekly_Sales_4W'].values[0]
        else:
            week_4_avg_movement = latest_data['Recent_Daily_Sales'].values[0] * 7 if 'Recent_Daily_Sales' in latest_data.columns else 0
        
        # Week 13 Average Movement
        if 'Avg_Weekly_Sales_13W' in latest_data.columns and not pd.isna(latest_data['Avg_Weekly_Sales_13W'].values[0]):
            week_13_avg_movement = latest_data['Avg_Weekly_Sales_13W'].values[0]
        else:
            week_13_avg_movement = week_4_avg_movement
        
        # Weeks of Supply
        if 'Stock_Coverage_Weeks' in latest_data.columns and not pd.isna(latest_data['Stock_Coverage_Weeks'].values[0]):
            weeks_of_supply = latest_data['Stock_Coverage_Weeks'].values[0]
        elif 'Weeks_Of_Stock' in latest_data.columns and not pd.isna(latest_data['Weeks_Of_Stock'].values[0]):
            weeks_of_supply = latest_data['Weeks_Of_Stock'].values[0]
        else:
            weeks_of_supply = current_stock / week_4_avg_movement if week_4_avg_movement > 0 else 4.0
        
        # Calculate inventory value
        price = latest_data['Price'].values[0]
        inventory_value = current_stock * price
        
        # Calculate turnover rate
        turnover_rate = week_4_avg_movement / current_stock if current_stock > 0 else 0
        
        # Calculate safety stock based on average movement and lead time
        lead_time_weeks = 1.0  # Assume 1 week lead time
        safety_factor = 2.0  # Provide a safety buffer
        safety_stock = (week_4_avg_movement * lead_time_weeks) * safety_factor
        
        # Calculate target stock
        target_weeks = latest_data['Target_Stock_Weeks'].values[0] if 'Target_Stock_Weeks' in latest_data.columns else 2.0
        target_stock = week_4_avg_movement * target_weeks
        
        # Get forecasted demand for the next 2 weeks if available
        forecast_available = False
        total_forecasted_demand = 0
        daily_forecast = []
        
        if 'Daily_Sales_Forecast' in latest_data.columns:
            # Try to get forecasts from the data for the next 14 days
            forecast_data = combined_data[(combined_data['Store_Id'] == store) & 
                                      (combined_data['Item'] == product) &
                                      (combined_data['Date'] > latest_data['Date'].values[0])]
            
            if len(forecast_data) > 0:
                # Take up to 14 days of forecast data
                forecast_data = forecast_data.sort_values('Date').head(14)
                forecast_available = True
                total_forecasted_demand = forecast_data['Daily_Sales_Forecast'].sum()
                daily_forecast = list(zip(forecast_data['Date'].dt.strftime('%Y-%m-%d'), 
                                         forecast_data['Daily_Sales_Forecast']))
        
        # If no forecast data available, use average weekly movement
        if not forecast_available or total_forecasted_demand == 0:
            total_forecasted_demand = week_4_avg_movement * 2  # 2 weeks of demand
            
        # Calculate order quantity based on forecast or safety stock
        if current_stock < safety_stock:
            # Immediate reordering needed due to below safety stock
            order_quantity = max(0, int(target_stock - current_stock))
        else:
            # Check if current stock will be sufficient for the next 2 weeks
            projected_stock = current_stock - total_forecasted_demand
            if projected_stock < safety_stock:
                # Order enough to reach target stock level considering upcoming demand
                order_quantity = max(0, int(target_stock - projected_stock))
            else:
                order_quantity = 0
        
        # Determine inventory status based on safety stock and weeks of supply
        # Safety stock is generally 1 week of supply, target is 2 weeks
        if weeks_of_supply < 1:
            stock_status = 'Needs Reordering'
            status_color = 'danger'
        elif weeks_of_supply >= 1 and weeks_of_supply <= 2:
            stock_status = 'Good'
            status_color = 'success'
        else:  # weeks_of_supply > 2
            stock_status = 'Overstocked'
            status_color = 'warning'
        
        # Get recent daily sales from data (fixing undefined variable issue)
        recent_daily_sales = 0  # Default to 0 if not available
        if 'Recent_Daily_Sales' in latest_data.columns and not pd.isna(latest_data['Recent_Daily_Sales'].values[0]):
            recent_daily_sales = latest_data['Recent_Daily_Sales'].values[0]
        
        # Calculate inventory turnover rate
        if current_stock > 0 and recent_daily_sales > 0:
            # Use stock velocity from data if available
            if 'Stock_Velocity' in latest_data.columns and not pd.isna(latest_data['Stock_Velocity'].values[0]):
                turnover_rate = latest_data['Stock_Velocity'].values[0]
            else:
                turnover_rate = recent_daily_sales * 7 / current_stock  # Weekly turnover rate
                
            days_of_supply = current_stock / recent_daily_sales if recent_daily_sales > 0 else float('inf')
            turnover_text = f"Inventory sells {turnover_rate:.1f}x per week ({days_of_supply:.1f} days of supply)"
            
            # Add additional context based on turnover rate
            if turnover_rate > 2:
                turnover_text += " (fast-moving item)"
            elif turnover_rate < 0.5:
                turnover_text += " (slow-moving item)"
        else:
            turnover_text = "Cannot calculate turnover rate (insufficient data)"
        
        # Create recommendation card
        recommendation_card = dbc.Card(
            [
                dbc.CardHeader(html.H5("Inventory Metrics Dashboard", className="text-center")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Current Status", className="text-center mb-3"),
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col(html.H6("Status:"), width=6),
                                        dbc.Col(html.H6(stock_status, className=f"text-{status_color} fw-bold"), width=6)
                                    ]),
                                    html.Hr(),
                                    dbc.Row([
                                        dbc.Col(html.P("On Hand:"), width=6),
                                        dbc.Col(html.P(f"{int(current_stock)} units", className="fw-bold"), width=6)
                                    ]),
                                    dbc.Row([
                                        dbc.Col(html.P("Inventory Value:"), width=6),
                                        dbc.Col(html.P(f"${inventory_value:.2f}", className="fw-bold"), width=6)
                                    ])
                                ])
                            ])
                        ], width=6),
                        dbc.Col([
                            html.H6("Movement Analysis", className="text-center mb-3"),
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col(html.P("4-Week Avg Movement:"), width=6),
                                        dbc.Col(html.P(f"{week_4_avg_movement:.1f} units/week", className="fw-bold"), width=6)
                                    ]),
                                    dbc.Row([
                                        dbc.Col(html.P("13-Week Avg Movement:"), width=6),
                                        dbc.Col(html.P(f"{week_13_avg_movement:.1f} units/week", className="fw-bold"), width=6)
                                    ]),
                                    dbc.Row([
                                        dbc.Col(html.P("Turnover Rate:"), width=6),
                                        dbc.Col(html.P(f"{turnover_rate:.2f}x per week", className="fw-bold"), width=6)
                                    ])
                                ])
                            ])
                        ], width=6)
                    ]),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.H6("Supply Planning", className="text-center mb-3"),
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col(html.P("Weeks of Supply:"), width=6),
                                        dbc.Col(html.P(f"{weeks_of_supply:.1f} weeks", 
                                                    className=f"fw-bold {'text-success' if 1 <= weeks_of_supply <= 4 else 'text-danger'}"), width=6)
                                    ]),
                                    dbc.Row([
                                        dbc.Col(html.P("Safety Stock:"), width=6),
                                        dbc.Col(html.P(f"{int(safety_stock)} units", className="fw-bold"), width=6)
                                    ]),
                                    dbc.Row([
                                        dbc.Col(html.P("Target Stock:"), width=6),
                                        dbc.Col(html.P(f"{int(target_stock)} units", className="fw-bold"), width=6)
                                    ])
                                ])
                            ])
                        ], width=6),
                        dbc.Col([
                            html.H6("2-Week Forecast", className="text-center mb-3"),
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5(
                                        f"Order {order_quantity} units now" if order_quantity > 0 else "No reordering needed",
                                        className=f"text-center {'text-danger' if order_quantity > 0 else 'text-success'} fw-bold"
                                    ),
                                    html.Hr(),
                                    html.Div([
                                        html.P("2-Week Forecast Summary:", className="fw-bold"),
                                        html.P(f"Projected Demand: {total_forecasted_demand:.1f} units"),
                                        html.P(f"Current Stock: {current_stock} units"),
                                        html.P(f"Projected Stock after 2 weeks: {max(0, current_stock - total_forecasted_demand):.1f} units"),
                                        html.P(f"Safety Stock Level: {safety_stock:.1f} units"),
                                        html.P(
                                            f"Status in 2 weeks: {'Below Safety Stock' if current_stock - total_forecasted_demand < safety_stock else 'Adequate'}",
                                            className=f"{'text-danger fw-bold' if current_stock - total_forecasted_demand < safety_stock else 'text-success'}"
                                        )
                                    ]),
                                    html.Hr() if forecast_available and len(daily_forecast) > 0 else None,
                                    html.P(
                                        "Daily Forecast Breakdown:",
                                        className="fw-bold"
                                    ) if forecast_available and len(daily_forecast) > 0 else None,
                                    html.Div([
                                        html.Small(
                                            [f"{date}: {demand:.1f} units", html.Br()] 
                                            for date, demand in daily_forecast[:7]  # Show first 7 days
                                        ) if forecast_available and len(daily_forecast) > 0 else None
                                    ])
                                ])
                            ])
                        ], width=6)
                    ]),
                    html.Div([
                        # Use stock status from data if available
                        dbc.Badge("Needs Reordering", color="danger", className="me-1") if stock_status == 'Needs Reordering' or weeks_of_supply < 1 else None,
                        
                        # Determine if it's a fast seller based on turnover rate
                        # First check if we have Stock_Velocity in the data
                        dbc.Badge("Fast Seller", color="info", className="me-1")
                            if ('Stock_Velocity' in latest_data.columns and 
                                not pd.isna(latest_data['Stock_Velocity'].values[0]) and 
                                latest_data['Stock_Velocity'].values[0] > 1.5) or 
                               (current_stock > 0 and recent_daily_sales > 0 and (recent_daily_sales * 7 / current_stock) > 1.5) 
                            else None,
                        
                        # Determine if it's a slow mover
                        dbc.Badge("Slow Mover", color="warning")
                            if ('Stock_Velocity' in latest_data.columns and 
                                not pd.isna(latest_data['Stock_Velocity'].values[0]) and 
                                latest_data['Stock_Velocity'].values[0] < 0.5) or 
                               (current_stock > 0 and recent_daily_sales > 0 and (recent_daily_sales * 7 / current_stock) < 0.5) 
                            else None,
                    ], className="mt-2")
                ])
            ],
            className="shadow-sm"
        )
        
        return recommendation_card
    except Exception as e:
        print(f"Error updating stock recommendations: {e}")
        return html.P(f"Error generating stock recommendations: {str(e)}")

# Function to update inventory turnover chart
def update_stock_velocity_chart(n_clicks, adjust_clicks, store, product, combined_data, go, np, stock_adjustment=None, adjustment_date=None):
    if combined_data is None or store is None or product is None:
        return go.Figure()
    
    try:
        # Get data for this product
        product_data = combined_data[(combined_data['Store_Id'] == store) & 
                                   (combined_data['Item'] == product)].copy()
                                   
        # Apply stock adjustment if provided
        key = f"{store}_{product}"
        date_key = None
        
        try:
            from plotly_dashboard import app
            
            if adjust_clicks and stock_adjustment is not None:
                # Store the adjustment in app's state
                app.manual_stock_adjustments[key] = stock_adjustment
                
                # Store with date if provided
                if adjustment_date is not None:
                    try:
                        # Convert string date to datetime if needed
                        if isinstance(adjustment_date, str):
                            adj_date = pd.to_datetime(adjustment_date).date()
                        else:
                            adj_date = adjustment_date
                            
                        date_key = f"{store}_{product}_{adj_date}"
                        if not hasattr(app, 'manual_stock_adjustments_with_dates'):
                            app.manual_stock_adjustments_with_dates = {}
                        app.manual_stock_adjustments_with_dates[date_key] = stock_adjustment
                    except:
                        # If date conversion fails, fall back to just using the regular key
                        pass
                
            # Apply any stored adjustment for this store/product combo
            update_applied = False
            
            # First try to apply date-specific adjustment if available
            if adjustment_date is not None:
                try:
                    adj_date = pd.to_datetime(adjustment_date).date() if isinstance(adjustment_date, str) else adjustment_date
                    date_key = f"{store}_{product}_{adj_date}"
                    
                    if hasattr(app, 'manual_stock_adjustments_with_dates') and date_key in app.manual_stock_adjustments_with_dates:
                        # Find the specific date in the data
                        match_date = pd.Timestamp(adj_date)
                        date_idx = product_data[product_data['Date'] == match_date].index
                        
                        if len(date_idx) > 0:
                            # Update the stock level for the specific date
                            product_data.loc[date_idx, 'Stock_Level'] = app.manual_stock_adjustments_with_dates[date_key]
                            product_data.loc[date_idx, 'On_Hand'] = app.manual_stock_adjustments_with_dates[date_key]
                            update_applied = True
                except:
                    # If date-specific application fails, continue to try the regular key
                    pass
            
            # If no date-specific update was applied, try the regular key (for backward compatibility)
            if not update_applied and hasattr(app, 'manual_stock_adjustments') and key in app.manual_stock_adjustments:
                # Find the latest date in the data
                latest_date = product_data['Date'].max()
                latest_idx = product_data[product_data['Date'] == latest_date].index
                
                if len(latest_idx) > 0:
                    # Update the stock level for the latest date
                    product_data.loc[latest_idx, 'Stock_Level'] = app.manual_stock_adjustments[key]
                    product_data.loc[latest_idx, 'On_Hand'] = app.manual_stock_adjustments[key]
        except (ImportError, AttributeError):
            # Handle case where app is not accessible
            pass
        
        # Map existing columns to standardized names
        if 'Stock_Level' in product_data.columns:
            product_data['On_Hand'] = product_data['Stock_Level']
        else:
            product_data['On_Hand'] = 0
            
        if 'Avg_Weekly_Sales_4W' in product_data.columns:
            product_data['Week_4_Avg_Movement'] = product_data['Avg_Weekly_Sales_4W']
        else:
            product_data['Week_4_Avg_Movement'] = product_data['Recent_Daily_Sales'] * 7
            
        if 'Avg_Weekly_Sales_13W' in product_data.columns:
            product_data['Week_13_Avg_Movement'] = product_data['Avg_Weekly_Sales_13W']
        else:
            product_data['Week_13_Avg_Movement'] = product_data['Week_4_Avg_Movement']
        
        # Calculate turnover rates using different time windows
        # 4-week turnover rate
        product_data['Turnover_Rate_4W'] = product_data['Week_4_Avg_Movement'] / product_data['On_Hand'].replace(0, 0.01)
        product_data['Turnover_Rate_4W'] = product_data['Turnover_Rate_4W'].fillna(0.01)  # Use 0.01 instead of 0
        product_data['Turnover_Rate_4W'] = product_data['Turnover_Rate_4W'].replace([np.inf, -np.inf], 0.01)
        
        # 13-week turnover rate for longer-term trend
        product_data['Turnover_Rate_13W'] = product_data['Week_13_Avg_Movement'] / product_data['On_Hand'].replace(0, 0.01)
        product_data['Turnover_Rate_13W'] = product_data['Turnover_Rate_13W'].fillna(0.01)  # Use 0.01 instead of 0
        product_data['Turnover_Rate_13W'] = product_data['Turnover_Rate_13W'].replace([np.inf, -np.inf], 0.01)
        
        # Calculate historical turnover using rolling windows
        if 'Sales' in product_data.columns:
            product_data['Rolling_Sales_28d'] = product_data['Sales'].rolling(window=28).sum()
            product_data['Stock_Velocity_28d'] = product_data['Rolling_Sales_28d'] / product_data['On_Hand'].replace(0, 0.01)
            product_data['Stock_Velocity_28d'] = product_data['Stock_Velocity_28d'].replace([np.inf, -np.inf], 0.01)
            product_data['Stock_Velocity_28d'] = product_data['Stock_Velocity_28d'].fillna(0.01)  # Use 0.01 instead of 0
        else:
            # Fall back to calculated turnover rates
            product_data['Stock_Velocity_28d'] = product_data['Turnover_Rate_4W']
        
        # Create figure with separate rows for each subplot - each subplot gets its own row
        # Adjusted spacing for better visualization
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.15,  # Reduced spacing for better proportions
            subplot_titles=("Inventory Turnover Rates", "On-Hand Inventory"),
            row_heights=[0.65, 0.35]  # Adjusted to give more space to turnover rates
        )
        
        # Add turnover rate lines to top subplot
        fig.add_trace(
            go.Scatter(
                x=product_data['Date'],
                y=product_data['Turnover_Rate_4W'],
                mode='lines',
                name='4-Week Turnover Rate',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=product_data['Date'],
                y=product_data['Turnover_Rate_13W'],
                mode='lines',
                name='13-Week Turnover Rate',
                line=dict(color='purple', width=2, dash='dot')
            ),
            row=1, col=1
        )
        
        # Add on-hand inventory on bottom subplot
        fig.add_trace(
            go.Scatter(
                x=product_data['Date'],
                y=product_data['On_Hand'],
                mode='lines',
                name='On-Hand Inventory',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        # Calculate target turnover rates based on industry standards
        # Industry benchmarks: Fast-moving = 2+, Average = 1-2, Slow-moving < 1
        fast_turnover = 2.0
        average_turnover = 1.0
        
        # Add target turnover reference lines to top subplot
        fig.add_shape(
            type="line",
            x0=product_data['Date'].min(),
            x1=product_data['Date'].max(),
            y0=fast_turnover,
            y1=fast_turnover,
            line=dict(color="green", dash="dash"),
            row=1, col=1
        )
        
        fig.add_annotation(
            x=product_data['Date'].max(),
            y=fast_turnover,
            text=f"Fast Turnover ({fast_turnover})",
            showarrow=False,
            xshift=10,
            row=1, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=product_data['Date'].min(),
            x1=product_data['Date'].max(),
            y0=average_turnover,
            y1=average_turnover,
            line=dict(color="orange", dash="dash"),
            row=1, col=1
        )
        
        fig.add_annotation(
            x=product_data['Date'].max(),
            y=average_turnover,
            text=f"Average Turnover ({average_turnover})",
            showarrow=False,
            xshift=10,
            row=1, col=1
        )
        
        # Calculate current and average turnover rates for the title
        current_turnover = product_data['Turnover_Rate_4W'].iloc[-1] if len(product_data) > 0 else 0
        avg_turnover = product_data['Turnover_Rate_4W'].mean()
        
        # Update layout with additional metrics and more spacing
        product_name = product_data['Product'].iloc[0] if len(product_data) > 0 else "Product"
        fig.update_layout(
            title={
                'text': f"Inventory Turnover Analysis for {product_name}<br><sup>Current: {current_turnover:.2f}x/week, Average: {avg_turnover:.2f}x/week</sup>",
                'y': 0.95,  # Adjusted title position
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 22}  # Increased font size for better readability
            },
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02,  # Adjusted legend position
                xanchor="right", 
                x=1,
                font=dict(size=12),
                itemsizing='constant'
            ),
            template="plotly_white",
            height=1200,  # Increased height for better spacing
            margin=dict(t=180, b=100, l=100, r=100),  # Adjusted margins
            hovermode='closest'  # Better hover interaction
        )
        
        # Update axis titles with improved positioning
        fig.update_yaxes(
            title_text="Weekly Turnover Rate",
            title_standoff=10,  # More space for y-axis title
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="On-Hand Inventory (units)",
            title_standoff=10,  # More space for y-axis title
            row=2, col=1
        )
        fig.update_xaxes(
            title_text="Date",
            title_standoff=20,  # More space for x-axis title
            row=2, col=1
        )
        
        # Update subplot titles position for better spacing with full-width layout
        fig.update_annotations(font_size=20, y=0.95, selector=dict(text="Inventory Turnover Rates"))
        fig.update_annotations(font_size=20, y=0.32, selector=dict(text="On-Hand Inventory"))
        
        return fig
    except Exception as e:
        print(f"Error updating stock velocity chart: {e}")
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error Loading Inventory Turnover Data",
            annotations=[{
                'text': f"Error: {str(e)}",
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5
            }]
        )
        return empty_fig

# Function to update inventory cost impact chart
def update_stock_penalty_chart(n_clicks, adjust_clicks, store, product, combined_data, go, np, stock_adjustment=None, adjustment_date=None):
    if combined_data is None or store is None or product is None:
        return go.Figure()
    
    try:
        # Get data for this product
        product_data = combined_data[(combined_data['Store_Id'] == store) & 
                                   (combined_data['Item'] == product)].copy()
                                   
        # Apply stock adjustment if provided
        key = f"{store}_{product}"
        date_key = None
        
        try:
            from plotly_dashboard import app
            
            if adjust_clicks and stock_adjustment is not None:
                # Store the adjustment in app's state
                app.manual_stock_adjustments[key] = stock_adjustment
                
                # Store with date if provided
                if adjustment_date is not None:
                    try:
                        # Convert string date to datetime if needed
                        if isinstance(adjustment_date, str):
                            adj_date = pd.to_datetime(adjustment_date).date()
                        else:
                            adj_date = adjustment_date
                            
                        date_key = f"{store}_{product}_{adj_date}"
                        if not hasattr(app, 'manual_stock_adjustments_with_dates'):
                            app.manual_stock_adjustments_with_dates = {}
                        app.manual_stock_adjustments_with_dates[date_key] = stock_adjustment
                    except:
                        # If date conversion fails, fall back to just using the regular key
                        pass
                
            # Apply any stored adjustment for this store/product combo
            update_applied = False
            
            # First try to apply date-specific adjustment if available
            if adjustment_date is not None:
                try:
                    adj_date = pd.to_datetime(adjustment_date).date() if isinstance(adjustment_date, str) else adjustment_date
                    date_key = f"{store}_{product}_{adj_date}"
                    
                    if hasattr(app, 'manual_stock_adjustments_with_dates') and date_key in app.manual_stock_adjustments_with_dates:
                        # Find the specific date in the data
                        match_date = pd.Timestamp(adj_date)
                        date_idx = product_data[product_data['Date'] == match_date].index
                        
                        if len(date_idx) > 0:
                            # Update the stock level for the specific date
                            product_data.loc[date_idx, 'Stock_Level'] = app.manual_stock_adjustments_with_dates[date_key]
                            product_data.loc[date_idx, 'On_Hand'] = app.manual_stock_adjustments_with_dates[date_key]
                            update_applied = True
                except:
                    # If date-specific application fails, continue to try the regular key
                    pass
            
            # If no date-specific update was applied, try the regular key (for backward compatibility)
            if not update_applied and hasattr(app, 'manual_stock_adjustments') and key in app.manual_stock_adjustments:
                # Find the latest date in the data
                latest_date = product_data['Date'].max()
                latest_idx = product_data[product_data['Date'] == latest_date].index
                
                if len(latest_idx) > 0:
                    # Update the stock level for the latest date
                    product_data.loc[latest_idx, 'Stock_Level'] = app.manual_stock_adjustments[key]
                    product_data.loc[latest_idx, 'On_Hand'] = app.manual_stock_adjustments[key]
        except (ImportError, AttributeError):
            # Handle case where app is not accessible
            pass
        
        # Add derived metrics for inventory analysis
        
        # Map existing columns to standardized names
        if 'Stock_Level' in product_data.columns:
            product_data['On_Hand'] = product_data['Stock_Level']
        else:
            product_data['On_Hand'] = 0
            
        if 'Avg_Weekly_Sales_4W' in product_data.columns:
            product_data['Week_4_Avg_Movement'] = product_data['Avg_Weekly_Sales_4W']
        else:
            product_data['Week_4_Avg_Movement'] = product_data['Recent_Daily_Sales'] * 7
            
        if 'Avg_Weekly_Sales_13W' in product_data.columns:
            product_data['Week_13_Avg_Movement'] = product_data['Avg_Weekly_Sales_13W']
        else:
            product_data['Week_13_Avg_Movement'] = product_data['Week_4_Avg_Movement']
            
        if 'Stock_Coverage_Weeks' in product_data.columns:
            product_data['Weeks_of_Supply'] = product_data['Stock_Coverage_Weeks']
        elif 'Weeks_Of_Stock' in product_data.columns:
            product_data['Weeks_of_Supply'] = product_data['Weeks_Of_Stock']
        else:
            week_4_movement = product_data['Week_4_Avg_Movement'].replace(0, np.nan)
            product_data['Weeks_of_Supply'] = product_data['On_Hand'] / week_4_movement.fillna(1)
            
        # Calculate carrying cost (excess stock penalty)
        # Get price and calculate cost based on adaptive inventory values
        avg_price = product_data['Price'].mean() if 'Price' in product_data.columns else 0
        
        # Generate valid cost data if it's missing or all zeros
        if 'Cost' not in product_data.columns or product_data['Cost'].sum() == 0:
            # Use price and standard margin to estimate cost
            # Different products have different margins, so vary slightly per product
            item_id = product_data['Item'].iloc[0] if len(product_data) > 0 else 0
            seed_val = int(str(item_id)[-2:]) if len(str(item_id)) > 1 else item_id
            np.random.seed(seed_val)
            margin_pct = np.random.uniform(0.25, 0.4)  # Random margin between 25-40%
            avg_cost = avg_price * (1 - margin_pct)
            
            # Add the cost column to the dataframe
            product_data['Cost'] = avg_price * (1 - margin_pct)
        else:
            avg_cost = product_data['Cost'].mean()
            
        # Ensure cost is valid
        avg_cost = max(avg_cost, avg_price * 0.6)  # Cost should be at least 60% of price
        
        carrying_cost_factor = 0.15  # 15% annual carrying cost
        weekly_carrying_cost = carrying_cost_factor / 52  # weekly rate
        
        # Use weeks of supply to calculate excess stock penalty
        # Only penalize for stock beyond 2 weeks of supply (updated from 3 weeks to match new inventory status logic)
        product_data['Excess_Stock_Penalty'] = product_data.apply(
            lambda row: max(0.01, (row['Weeks_of_Supply'] - 2) * row['Week_4_Avg_Movement'] * max(avg_cost, 1.0) * weekly_carrying_cost)
            if row['Weeks_of_Supply'] > 2 else 0.01, axis=1  # Ensure minimum nonzero value
        )
        
        # Calculate lost sales cost (stockout penalty)
        margin = avg_price - avg_cost if avg_price > 0 and avg_cost > 0 else max(avg_price * 0.3, 1.0)  # Default to 30% margin with minimum value
        
        # Use weeks of supply to calculate stockout penalty
        # Only penalize when weeks of supply is less than 1 week
        product_data['Stockout_Penalty'] = product_data.apply(
            lambda row: max(0.01, (1 - row['Weeks_of_Supply']) * row['Week_4_Avg_Movement'] * margin)
            if row['Weeks_of_Supply'] < 1 else 0.01, axis=1  # Ensure minimum nonzero value
        )
        
        # Create figure with separate rows for each subplot - each subplot gets its own row
        # Adjusted spacing for better visualization
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.15,  # Reduced spacing for better proportions
            subplot_titles=("Inventory Cost Impact", "Weeks of Supply"),
            row_heights=[0.65, 0.35]  # Adjusted to give more space to cost impact
        )
        
        # Add excess stock penalty to top subplot
        fig.add_trace(
            go.Bar(
                x=product_data['Date'],
                y=product_data['Excess_Stock_Penalty'],
                name='Carrying Cost',
                marker_color='orange',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Add stockout penalty to top subplot
        fig.add_trace(
            go.Bar(
                x=product_data['Date'],
                y=product_data['Stockout_Penalty'],
                name='Potential Lost Sales',
                marker_color='red',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Add weeks of supply line to bottom subplot
        fig.add_trace(
            go.Scatter(
                x=product_data['Date'],
                y=product_data['Weeks_of_Supply'],
                name='Weeks of Supply',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        # Add optimal range for weeks of supply (1-3 weeks) to bottom subplot
        fig.add_trace(
            go.Scatter(
                x=[product_data['Date'].min(), product_data['Date'].max()],
                y=[1, 1],
                mode='lines',
                name='Min Supply Target',
                line=dict(color='red', dash='dot')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[product_data['Date'].min(), product_data['Date'].max()],
                y=[3, 3],
                mode='lines',
                name='Max Supply Target',
                line=dict(color='red', dash='dot'),
                fill='tonexty',
                fillcolor='rgba(0, 255, 0, 0.1)'
            ),
            row=2, col=1
        )
        
        # Calculate and show total cost impact
        total_excess_cost = product_data['Excess_Stock_Penalty'].sum()
        total_stockout_cost = product_data['Stockout_Penalty'].sum()
        total_cost = total_excess_cost + total_stockout_cost
        
        # Update layout with more spacing and better separation
        product_name = product_data['Product'].iloc[0] if len(product_data) > 0 else "Product"
        fig.update_layout(
            title={
                'text': f"Inventory Cost Impact for {product_name}<br><sup>Total Cost Impact: ${total_cost:.2f} (Excess: ${total_excess_cost:.2f}, Stockout Risk: ${total_stockout_cost:.2f})</sup>",
                'y': 0.95,  # Adjusted title position
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 22}  # Increased font size for better readability
            },
            barmode='stack',
            template="plotly_white",
            height=1200,  # Increased height for better spacing
            margin=dict(t=180, b=100, l=100, r=100),  # Adjusted margins
            hovermode='closest',  # Better hover interaction
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,  # Adjusted legend position
                xanchor="right",
                x=1,
                font=dict(size=12),
                itemsizing='constant'
            )
        )
        
        # Update axis titles with improved positioning
        fig.update_yaxes(
            title_text="Cost Impact ($)",
            title_standoff=10,  # More space for y-axis title
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Weeks of Supply",
            title_standoff=10,  # More space for y-axis title
            row=2, col=1
        )
        fig.update_xaxes(
            title_text="Date",
            title_standoff=20,  # More space for x-axis title
            row=2, col=1
        )
        
        # Update subplot titles position for better spacing with full-width layout
        fig.update_annotations(font_size=20, y=0.95, selector=dict(text="Inventory Cost Impact"))
        fig.update_annotations(font_size=20, y=0.32, selector=dict(text="Weeks of Supply"))
        
        return fig
    except Exception as e:
        print(f"Error updating stock penalty chart: {e}")
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error Loading Inventory Cost Impact Data",
            annotations=[{
                'text': f"Error: {str(e)}",
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5
            }]
        )
        return empty_fig

# Function to update inventory summary statistics
def update_inventory_summary_stats(n_clicks, adjust_clicks, store, product, stock_adjustment, combined_data, html, dbc, app, adjustment_date=None):
    if combined_data is None or store is None:
        return html.P("No inventory data available.")
        
    # Apply stock adjustment if provided
    if product is not None and adjust_clicks and stock_adjustment is not None:
        key = f"{store}_{product}"
        # Store the adjustment in app's state
        app.manual_stock_adjustments[key] = stock_adjustment
        
        # Store with date if provided
        if adjustment_date is not None:
            try:
                # Convert string date to datetime if needed
                if isinstance(adjustment_date, str):
                    adj_date = pd.to_datetime(adjustment_date).date()
                else:
                    adj_date = adjustment_date
                    
                date_key = f"{store}_{product}_{adj_date}"
                if not hasattr(app, 'manual_stock_adjustments_with_dates'):
                    app.manual_stock_adjustments_with_dates = {}
                app.manual_stock_adjustments_with_dates[date_key] = stock_adjustment
            except:
                # If date conversion fails, fall back to just using the regular key
                pass
    
    try:
        # Get the latest date
        latest_date = combined_data['Date'].max()
        
        # Get all products for this store at latest date
        store_data = combined_data[(combined_data['Store_Id'] == store) & 
                                 (combined_data['Date'] == latest_date)].copy()
        
        if len(store_data) == 0:
            return html.P("No inventory data available for selected store.")
        
        # Apply any stored adjustments to this store data
        # This allows the summary to reflect all product adjustments
        try:
            if hasattr(app, 'manual_stock_adjustments'):
                for key, value in app.manual_stock_adjustments.items():
                    if key.startswith(f"{store}_"):
                        product_id = int(key.split('_')[1])
                        product_idx = store_data[store_data['Item'] == product_id].index
                        if len(product_idx) > 0:
                            store_data.loc[product_idx, 'Stock_Level'] = value
                            
            if hasattr(app, 'manual_stock_adjustments_with_dates'):
                # Only apply date-specific adjustments for the current date (latest date)
                latest_date_str = latest_date.strftime('%Y-%m-%d')
                for key, value in app.manual_stock_adjustments_with_dates.items():
                    if key.startswith(f"{store}_") and key.endswith(f"_{latest_date_str}"):
                        parts = key.split('_')
                        if len(parts) >= 3:
                            try:
                                product_id = int(parts[1])
                                product_idx = store_data[store_data['Item'] == product_id].index
                                if len(product_idx) > 0:
                                    store_data.loc[product_idx, 'Stock_Level'] = value
                            except:
                                continue
        except:
            pass
        
        # Calculate summary statistics
        total_products = len(store_data)
        products_below_safety = len(store_data[store_data['Stock_Status'] == 'Low'])
        products_excess = len(store_data[store_data['Stock_Status'] == 'Excess'])
        products_adequate = total_products - products_below_safety - products_excess
        
        # Calculate percentages
        pct_below = products_below_safety / total_products * 100 if total_products > 0 else 0
        pct_adequate = products_adequate / total_products * 100 if total_products > 0 else 0
        pct_excess = products_excess / total_products * 100 if total_products > 0 else 0
        
        # Calculate total value of inventory and inventory costs
        total_inventory_value = (store_data['Stock_Level'] * store_data['Price']).sum()
        excess_cost = store_data['Excess_Stock_Penalty'].sum() if 'Excess_Stock_Penalty' in store_data.columns else 0
        stockout_risk = store_data['Stockout_Penalty'].sum() if 'Stockout_Penalty' in store_data.columns else 0
        
        # Create summary cards with full-width layout and increased spacing
        summary_row = html.Div([
            html.H4("Inventory Performance Summary", className="text-center mb-4 mt-2"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Inventory Status", className="text-center m-0")),
                        dbc.CardBody([
                            html.Div([
                                html.Div(f"{products_below_safety} Products Need Reordering ({pct_below:.1f}%)", 
                                        className="p-3 mb-2 bg-danger text-white rounded"),
                                html.Div(f"{products_adequate} Products at Good Levels ({pct_adequate:.1f}%)", 
                                        className="p-3 mb-2 bg-success text-white rounded"),
                                html.Div(f"{products_excess} Products Overstocked ({pct_excess:.1f}%)", 
                                        className="p-3 mb-2 bg-warning text-dark rounded")
                            ])
                        ])
                    ], className="h-100 shadow-sm")
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Inventory Value", className="text-center m-0")),
                        dbc.CardBody([
                            html.H2(f"${total_inventory_value:.2f}", className="text-center mb-3 display-6"),
                            html.P(f"Total value across {total_products} products", className="text-center fs-5")
                        ], className="d-flex flex-column justify-content-center")
                    ], className="h-100 shadow-sm")
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Inventory Cost Impact", className="text-center m-0")),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col(["Carrying Cost (Excess):"], width=8, className="fs-5"),
                                dbc.Col([f"${excess_cost:.2f}"], width=4, className="fs-5")
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col(["Potential Lost Sales:"], width=8, className="fs-5"),
                                dbc.Col([f"${stockout_risk:.2f}"], width=4, className="fs-5")
                            ], className="mb-2"),
                            html.Hr(),
                            dbc.Row([
                                dbc.Col(["Total Cost Impact:"], width=8, className="fs-5"),
                                dbc.Col([f"${(excess_cost + stockout_risk):.2f}"], width=4, className="fs-5 fw-bold")
                            ])
                        ])
                    ], className="h-100 shadow-sm")
                ], width=4)
            ], className="mb-4")
        ], className="mt-4 p-3 border rounded bg-light")
        
        return summary_row
    except Exception as e:
        print(f"Error updating inventory summary stats: {e}")
        return html.P(f"Error generating inventory summary: {str(e)}")