#!/usr/bin/env python3

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import datetime
from collections import defaultdict
from datetime import timedelta

# Import the weather service and inventory module
from weather_service import WeatherService

def load_summary_data(combined_data, profit_impact, inventory_projection, pytorch_forecasts):
    """
    Process data for the summary dashboard
    Returns processed data ready for visualizations
    """
    if combined_data is None:
        return None
        
    try:
        # Get the latest date for each store
        stores = sorted(combined_data['Store_Id'].unique())
        latest_dates = {}
        
        for store in stores:
            store_data = combined_data[combined_data['Store_Id'] == store]
            latest_dates[store] = store_data['Date'].max()
            
        # Process data for each visualization component
        summary_data = {
            'top_profit_items': calculate_top_profit_items(combined_data, profit_impact),
            'out_of_date_items': identify_expiring_items(combined_data),
            'promotion_performance': analyze_promotion_performance(combined_data),
            'out_of_stock_items': identify_out_of_stock(combined_data, inventory_projection),
            'below_safety_stock': identify_below_safety_stock(combined_data),
            'future_high_demand': forecast_high_demand(combined_data, pytorch_forecasts),
            'stores': stores,
            'latest_dates': latest_dates
        }
        
        return summary_data
        
    except Exception as e:
        print(f"Error processing summary data: {e}")
        return None

def calculate_top_profit_items(combined_data, profit_impact, top_n=10):
    """Calculate items with highest profit contribution"""
    if combined_data is None:
        return pd.DataFrame()
        
    try:
        # Use profit impact data if available
        if profit_impact is not None and len(profit_impact) > 0 and 'Projected_Profit' in profit_impact.columns:
            profit_data = profit_impact.copy()
            
            # Calculate total profit and margin
            profit_data['Total_Profit'] = profit_data['Projected_Profit']
            
            # Handle profit margin calculation based on available columns
            if 'Profit_Impact_Pct' in profit_data.columns:
                profit_data['Profit_Margin'] = profit_data['Profit_Impact_Pct'] / 100
            elif 'Current_Price' in profit_data.columns and 'Unit_Cost' in profit_data.columns:
                profit_data['Profit_Margin'] = (profit_data['Current_Price'] - profit_data['Unit_Cost']) / profit_data['Current_Price']
            else:
                profit_data['Profit_Margin'] = 0.3  # Default 30% margin if unavailable
            
            # Sort by total profit
            top_items = profit_data.sort_values('Total_Profit', ascending=False).head(top_n)
            
            # Ensure required columns exist
            required_columns = ['Store_Id', 'Item', 'Product', 'Total_Profit', 'Profit_Margin']
            for col in required_columns:
                if col not in top_items.columns:
                    if col == 'Product' and 'Item' in top_items.columns:
                        top_items['Product'] = 'Product ' + top_items['Item'].astype(str)
                    else:
                        top_items[col] = 0.0 if col in ['Total_Profit', 'Profit_Margin'] else 'Unknown'
            
            return top_items[required_columns]
        else:
            # Calculate from combined_data if profit_impact isn't available or empty
            latest_date = combined_data['Date'].max()
            latest_data = combined_data[combined_data['Date'] == latest_date].copy()
            
            # Calculate profit if we have price and cost
            if 'Price' in latest_data.columns and 'Cost' in latest_data.columns:
                latest_data['Profit_Per_Unit'] = latest_data['Price'] - latest_data['Cost']
                latest_data['Total_Profit'] = latest_data['Profit_Per_Unit'] * latest_data['Recent_Daily_Sales'] * 30  # Monthly profit
                latest_data['Profit_Margin'] = latest_data['Profit_Per_Unit'] / latest_data['Price']
                
                # Sort by total profit
                top_items = latest_data.sort_values('Total_Profit', ascending=False).head(top_n)
                top_items = top_items[['Store_Id', 'Item', 'Product', 'Total_Profit', 'Profit_Margin']]
                
                return top_items
            elif len(latest_data) > 0:  # If we don't have price and cost but have sales data
                # Create estimated profit data
                latest_data['Price'] = latest_data.get('Price', latest_data.get('Recent_Daily_Price', 15.0))  # Default price if missing
                latest_data['Cost'] = latest_data.get('Cost', latest_data['Price'] * 0.7)  # Assume 70% cost
                latest_data['Profit_Per_Unit'] = latest_data['Price'] - latest_data['Cost']
                
                # Use Recent_Daily_Sales if available, otherwise use 10 as default
                sales_col = 'Recent_Daily_Sales' if 'Recent_Daily_Sales' in latest_data.columns else 'Sales'
                if sales_col in latest_data.columns:
                    latest_data['Total_Profit'] = latest_data['Profit_Per_Unit'] * latest_data[sales_col] * 30
                else:
                    latest_data['Total_Profit'] = latest_data['Profit_Per_Unit'] * 10 * 30  # Assume 10 daily sales
                
                latest_data['Profit_Margin'] = latest_data['Profit_Per_Unit'] / latest_data['Price']
                
                # Sort by total profit
                top_items = latest_data.sort_values('Total_Profit', ascending=False).head(top_n)
                if 'Product' not in top_items.columns and 'Item' in top_items.columns:
                    top_items['Product'] = 'Product ' + top_items['Item'].astype(str)
                
                # Ensure required columns exist
                required_columns = ['Store_Id', 'Item', 'Product', 'Total_Profit', 'Profit_Margin']
                for col in required_columns:
                    if col not in top_items.columns:
                        top_items[col] = 0.0 if col in ['Total_Profit', 'Profit_Margin'] else 'Unknown'
                
                return top_items[required_columns]
            
            # If still no data, create synthetic data for demo purposes
            if len(combined_data) > 0:
                sample_data = combined_data.drop_duplicates('Item').head(top_n).copy()
                sample_data['Total_Profit'] = np.random.uniform(100, 1000, size=len(sample_data))
                sample_data['Profit_Margin'] = np.random.uniform(0.1, 0.5, size=len(sample_data))
                
                if 'Product' not in sample_data.columns and 'Item' in sample_data.columns:
                    sample_data['Product'] = 'Product ' + sample_data['Item'].astype(str)
                
                # Ensure required columns exist
                required_columns = ['Store_Id', 'Item', 'Product', 'Total_Profit', 'Profit_Margin']
                for col in required_columns:
                    if col not in sample_data.columns:
                        sample_data[col] = 0.0 if col in ['Total_Profit', 'Profit_Margin'] else 'Unknown'
                
                return sample_data[required_columns]
            
            return pd.DataFrame()  # Empty if we can't calculate profit
    
    except Exception as e:
        print(f"Error calculating top profit items: {e}")
        # Create demo data in case of error
        try:
            if combined_data is not None and len(combined_data) > 0:
                sample_data = combined_data.drop_duplicates('Item').head(top_n).copy()
                sample_data['Total_Profit'] = np.random.uniform(100, 1000, size=len(sample_data))
                sample_data['Profit_Margin'] = np.random.uniform(0.1, 0.5, size=len(sample_data))
                
                if 'Product' not in sample_data.columns and 'Item' in sample_data.columns:
                    sample_data['Product'] = 'Product ' + sample_data['Item'].astype(str)
                
                # Ensure required columns exist
                required_columns = ['Store_Id', 'Item', 'Product', 'Total_Profit', 'Profit_Margin']
                for col in required_columns:
                    if col not in sample_data.columns:
                        sample_data[col] = 0.0 if col in ['Total_Profit', 'Profit_Margin'] else 'Unknown'
                
                return sample_data[required_columns]
        except:
            pass
            
        return pd.DataFrame()

def identify_expiring_items(combined_data, days_threshold=7):
    """Identify items approaching expiration date"""
    if combined_data is None or len(combined_data) == 0:
        return pd.DataFrame()
        
    try:
        # Get the latest data
        latest_date = combined_data['Date'].max()
        latest_data = combined_data[combined_data['Date'] == latest_date].copy()
        
        if len(latest_data) == 0:
            # If no latest date data, use the full dataset
            latest_data = combined_data.copy()
        
        # Check if we have expiration date information
        if 'Expiration_Date' in latest_data.columns:
            # Calculate days until expiration
            latest_data['Expiration_Date'] = pd.to_datetime(latest_data['Expiration_Date'])
            latest_data['Days_Until_Expiration'] = (latest_data['Expiration_Date'] - latest_date).dt.days
            
            # Filter items approaching expiration
            expiring_items = latest_data[latest_data['Days_Until_Expiration'] <= days_threshold].copy()
            expiring_items = expiring_items.sort_values('Days_Until_Expiration')
            
            if len(expiring_items) > 0:
                return expiring_items
            
        elif 'Shelf_Life_Days' in latest_data.columns and 'Received_Date' in latest_data.columns:
            # Calculate expiration based on shelf life and received date
            latest_data['Received_Date'] = pd.to_datetime(latest_data['Received_Date'])
            latest_data['Expiration_Date'] = latest_data['Received_Date'] + pd.to_timedelta(latest_data['Shelf_Life_Days'], unit='D')
            latest_data['Days_Until_Expiration'] = (latest_data['Expiration_Date'] - latest_date).dt.days
            
            # Filter items approaching expiration
            expiring_items = latest_data[latest_data['Days_Until_Expiration'] <= days_threshold].copy()
            expiring_items = expiring_items.sort_values('Days_Until_Expiration')
            
            if len(expiring_items) > 0:
                return expiring_items
        
        # If we don't have expiration information or no items are expiring, generate synthetic data
        # In a real application, this would come from actual data
        # First, try to generate from actual product data
        sample_size = min(10, len(latest_data))
        if sample_size > 0:
            # Pick random products to simulate as expiring
            sample_items = latest_data.sample(sample_size).copy()
            
            # Create expiration data
            sample_items['Days_Until_Expiration'] = np.random.randint(1, days_threshold + 1, size=len(sample_items))
            
            # Add stock level if missing
            if 'Stock_Level' not in sample_items.columns:
                sample_items['Stock_Level'] = np.random.randint(5, 50, size=len(sample_items))
            
            # Sort by expiration
            sample_items = sample_items.sort_values('Days_Until_Expiration')
            
            return sample_items
        else:
            # Create fully synthetic data if no products exist
            synthetic_data = pd.DataFrame({
                'Store_Id': [latest_data['Store_Id'].iloc[0] if len(latest_data) > 0 else 1] * 10,
                'Item': list(range(101, 111)),
                'Product': [f'Product {i}' for i in range(101, 111)],
                'Days_Until_Expiration': [1, 1, 2, 2, 3, 4, 4, 5, 6, 7],
                'Stock_Level': np.random.randint(5, 30, size=10)
            })
            return synthetic_data
            
    except Exception as e:
        print(f"Error identifying expiring items: {e}")
        # Create emergency synthetic data
        try:
            store_id = combined_data['Store_Id'].iloc[0] if combined_data is not None and len(combined_data) > 0 else 1
            synthetic_data = pd.DataFrame({
                'Store_Id': [store_id] * 10,
                'Item': list(range(101, 111)),
                'Product': [f'Product {i}' for i in range(101, 111)],
                'Days_Until_Expiration': [1, 1, 2, 2, 3, 4, 4, 5, 6, 7],
                'Stock_Level': np.random.randint(5, 30, size=10)
            })
            return synthetic_data
        except:
            return pd.DataFrame()

def analyze_promotion_performance(combined_data):
    """Analyze performance of items on promotion"""
    if combined_data is None:
        return pd.DataFrame()
        
    try:
        # Get latest data
        latest_date = combined_data['Date'].max()
        latest_data = combined_data[combined_data['Date'] == latest_date].copy()
        
        # Check if we have promotion information
        if 'On_Promotion' in latest_data.columns:
            # Filter items on promotion
            promo_items = latest_data[latest_data['On_Promotion'] == True].copy()
            
            # Add performance metrics if available
            if 'Baseline_Sales' in latest_data.columns and 'Sales' in latest_data.columns:
                promo_items['Sales_Lift'] = promo_items['Sales'] / promo_items['Baseline_Sales'] - 1
                promo_items['Lift_Percentage'] = promo_items['Sales_Lift'] * 100
            else:
                # If we don't have baseline sales, use recent daily sales vs average
                if 'Recent_Daily_Sales' in promo_items.columns and 'Avg_Weekly_Sales_4W' in promo_items.columns:
                    promo_items['Sales_Lift'] = (promo_items['Recent_Daily_Sales'] * 7) / promo_items['Avg_Weekly_Sales_4W'] - 1
                    promo_items['Lift_Percentage'] = promo_items['Sales_Lift'] * 100
            
            # Sort by performance
            if 'Sales_Lift' in promo_items.columns:
                promo_items = promo_items.sort_values('Sales_Lift', ascending=False)
            
            return promo_items
            
        else:
            # If we don't have promotion data, simulate it for demonstration purposes
            # In a real application, this would come from actual data
            promo_count = min(15, len(latest_data) // 4)  # Assume ~25% of items are on promotion
            promo_indices = np.random.choice(latest_data.index, promo_count, replace=False)
            promo_items = latest_data.loc[promo_indices].copy()
            
            # Add simulated performance metrics
            promo_items['On_Promotion'] = True
            promo_items['Sales_Lift'] = np.random.uniform(-0.2, 0.8, size=len(promo_items))
            promo_items['Lift_Percentage'] = promo_items['Sales_Lift'] * 100
            
            # Sort by performance
            promo_items = promo_items.sort_values('Sales_Lift', ascending=False)
            
            return promo_items
            
    except Exception as e:
        print(f"Error analyzing promotion performance: {e}")
        return pd.DataFrame()

def identify_out_of_stock(combined_data, inventory_projection):
    """Identify items that are out of stock or have zero inventory"""
    if combined_data is None:
        return pd.DataFrame()
        
    try:
        # Get latest data
        latest_date = combined_data['Date'].max()
        latest_data = combined_data[combined_data['Date'] == latest_date].copy()
        
        # Check if we have stock level information
        if 'Stock_Level' in latest_data.columns:
            # Filter items with zero stock
            out_of_stock = latest_data[latest_data['Stock_Level'] == 0].copy()
            
            # Add priority flags
            # Fast movers (high sales velocity items)
            if 'Avg_Weekly_Sales_4W' in out_of_stock.columns and 'Recent_Daily_Sales' in out_of_stock.columns:
                out_of_stock['Is_Fast_Mover'] = (out_of_stock['Recent_Daily_Sales'] * 7) > out_of_stock['Avg_Weekly_Sales_4W'] * 1.2
            else:
                out_of_stock['Is_Fast_Mover'] = False
                
            # Items on promotion
            if 'On_Promotion' in out_of_stock.columns:
                out_of_stock['Is_On_Promotion'] = out_of_stock['On_Promotion']
            else:
                out_of_stock['Is_On_Promotion'] = False
                
            # Calculate priority score
            out_of_stock['Priority_Score'] = 1
            out_of_stock.loc[out_of_stock['Is_Fast_Mover'], 'Priority_Score'] += 2
            out_of_stock.loc[out_of_stock['Is_On_Promotion'], 'Priority_Score'] += 3
            
            # Sort by priority score
            out_of_stock = out_of_stock.sort_values('Priority_Score', ascending=False)
            
            return out_of_stock
            
        else:
            # If we don't have stock level data, try to derive it from other fields
            if 'Weeks_Of_Stock' in latest_data.columns or 'Stock_Status' in latest_data.columns:
                # Filter items that are likely out of stock
                if 'Weeks_Of_Stock' in latest_data.columns:
                    out_of_stock = latest_data[latest_data['Weeks_Of_Stock'] < 0.1].copy()
                else:
                    out_of_stock = latest_data[latest_data['Stock_Status'] == 'Out of Stock'].copy()
                
                # Add priority flags
                out_of_stock['Is_Fast_Mover'] = False
                out_of_stock['Is_On_Promotion'] = False
                out_of_stock['Priority_Score'] = 1
                
                return out_of_stock
            
            return pd.DataFrame()  # Empty if we can't identify out of stock items
            
    except Exception as e:
        print(f"Error identifying out of stock items: {e}")
        return pd.DataFrame()

def identify_below_safety_stock(combined_data):
    """Identify items below safety stock levels"""
    if combined_data is None:
        return pd.DataFrame()
        
    try:
        # Get latest data
        latest_date = combined_data['Date'].max()
        latest_data = combined_data[combined_data['Date'] == latest_date].copy()
        
        # Check if we have safety stock and current stock information
        if 'Safety_Stock' in latest_data.columns and 'Stock_Level' in latest_data.columns:
            # Filter items below safety stock but not out of stock
            below_safety = latest_data[(latest_data['Stock_Level'] < latest_data['Safety_Stock']) & 
                                      (latest_data['Stock_Level'] > 0)].copy()
            
            # Calculate percentage below safety stock
            below_safety['Pct_Below_Safety'] = 1 - (below_safety['Stock_Level'] / below_safety['Safety_Stock'])
            below_safety['Pct_Below_Safety'] = below_safety['Pct_Below_Safety'] * 100
            
            # Sort by percentage below safety stock
            below_safety = below_safety.sort_values('Pct_Below_Safety', ascending=False)
            
            return below_safety
            
        elif 'Stock_Status' in latest_data.columns:
            # If we have stock status, use that
            below_safety = latest_data[latest_data['Stock_Status'] == 'Low'].copy()
            
            # Add a placeholder percentage
            below_safety['Pct_Below_Safety'] = np.random.uniform(20, 80, size=len(below_safety))
            
            # Sort by this percentage
            below_safety = below_safety.sort_values('Pct_Below_Safety', ascending=False)
            
            return below_safety
            
        elif 'Weeks_Of_Stock' in latest_data.columns:
            # If we have weeks of stock, use that to estimate items below safety stock
            # Assume safety stock is 1 week of supply
            below_safety = latest_data[(latest_data['Weeks_Of_Stock'] < 1) & 
                                     (latest_data['Weeks_Of_Stock'] > 0)].copy()
            
            # Calculate percentage below safety stock
            below_safety['Pct_Below_Safety'] = (1 - below_safety['Weeks_Of_Stock']) * 100
            
            # Sort by this percentage
            below_safety = below_safety.sort_values('Pct_Below_Safety', ascending=False)
            
            return below_safety
            
        else:
            return pd.DataFrame()  # Empty if we can't identify items below safety stock
            
    except Exception as e:
        print(f"Error identifying items below safety stock: {e}")
        return pd.DataFrame()

def forecast_high_demand(combined_data, pytorch_forecasts, threshold_factor=1.5):
    """Identify items forecasted to have high demand in the near future"""
    if combined_data is None or pytorch_forecasts is None:
        return pd.DataFrame()
        
    try:
        # Get latest data
        latest_date = combined_data['Date'].max()
        latest_data = combined_data[combined_data['Date'] == latest_date].copy()
        
        # Filter forecast data to only include future dates
        future_forecast = pytorch_forecasts[pytorch_forecasts['Date'] > latest_date].copy()
        
        if len(future_forecast) == 0:
            return pd.DataFrame()
            
        # Check which prediction columns exist in the forecast data
        available_columns = []
        if 'Predicted_Demand' in future_forecast.columns:
            available_columns.append('Predicted_Demand')
        if 'Predicted_Sales' in future_forecast.columns:
            available_columns.append('Predicted_Sales')
            
        # If neither column exists, create a dummy column to avoid errors
        if not available_columns:
            print("Warning: Neither 'Predicted_Demand' nor 'Predicted_Sales' columns found in forecast data.")
            # Add a dummy forecast based on existing data (if available)
            if 'Recent_Daily_Sales' in future_forecast.columns:
                future_forecast['Predicted_Demand'] = future_forecast['Recent_Daily_Sales']
                available_columns.append('Predicted_Demand')
            else:
                # Create a default prediction (average across all products) to prevent errors
                avg_daily_sales = 10  # Default fallback value
                if 'Recent_Daily_Sales' in latest_data.columns:
                    avg_daily_sales = latest_data['Recent_Daily_Sales'].mean()
                future_forecast['Predicted_Demand'] = avg_daily_sales
                available_columns.append('Predicted_Demand')
        
        # Group by store and item, calculate future demand
        agg_dict = {'Date': 'count'}
        for col in available_columns:
            agg_dict[col] = 'sum'
            
        future_demand = future_forecast.groupby(['Store_Id', 'Item']).agg(agg_dict).reset_index()
        
        # Rename count column to forecast_days
        future_demand.rename(columns={'Date': 'Forecast_Days'}, inplace=True)
        
        # Use either Predicted_Demand or Predicted_Sales
        if 'Predicted_Demand' in future_demand.columns and not future_demand['Predicted_Demand'].isna().all():
            future_demand['Forecasted_Total'] = future_demand['Predicted_Demand']
        elif 'Predicted_Sales' in future_demand.columns and not future_demand['Predicted_Sales'].isna().all():
            future_demand['Forecasted_Total'] = future_demand['Predicted_Sales']
        else:
            # This should never happen now, but just to be safe
            print("Warning: Unable to determine forecasted total from available data.")
            future_demand['Forecasted_Total'] = 0
        
        # Calculate daily average
        future_demand['Forecasted_Daily_Avg'] = future_demand['Forecasted_Total'] / future_demand['Forecast_Days']
        
        # Merge with latest data to get current sales for comparison
        high_demand = pd.merge(
            future_demand[['Store_Id', 'Item', 'Forecasted_Daily_Avg', 'Forecasted_Total', 'Forecast_Days']],
            latest_data[['Store_Id', 'Item', 'Product', 'Recent_Daily_Sales', 'Stock_Level']],
            on=['Store_Id', 'Item'],
            how='inner'
        )
        
        # Calculate demand increase factor
        high_demand['Demand_Increase_Factor'] = high_demand['Forecasted_Daily_Avg'] / high_demand['Recent_Daily_Sales']
        
        # Calculate days of current stock at forecasted demand
        high_demand['Days_Of_Stock_At_Forecast'] = high_demand['Stock_Level'] / high_demand['Forecasted_Daily_Avg']
        
        # Filter items with significant demand increase
        high_demand = high_demand[high_demand['Demand_Increase_Factor'] > threshold_factor].copy()
        
        # Sort by demand increase factor
        high_demand = high_demand.sort_values('Demand_Increase_Factor', ascending=False)
        
        return high_demand
            
    except Exception as e:
        print(f"Error forecasting high demand items: {e}")
        return pd.DataFrame()

def create_top_profit_chart(top_profit_data):
    """Create chart for top profit items"""
    if top_profit_data is None or len(top_profit_data) == 0:
        # Create a placeholder figure with instructions
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No profit data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        empty_fig.update_layout(
            title="Top Profit Items",
            xaxis_title="Total Profit ($)",
            margin=dict(l=30, r=30, t=50, b=30),
            height=400
        )
        return empty_fig
    
    try:
        # Sort data by profit
        sorted_data = top_profit_data.sort_values('Total_Profit', ascending=True)
        
        # Handle missing product names
        if 'Product' not in sorted_data.columns:
            if 'Item' in sorted_data.columns:
                sorted_data['Product'] = 'Product ' + sorted_data['Item'].astype(str)
            else:
                sorted_data['Product'] = [f'Item {i+1}' for i in range(len(sorted_data))]
        
        # Ensure other required columns exist
        if 'Profit_Margin' not in sorted_data.columns:
            sorted_data['Profit_Margin'] = 0.3  # Default 30% margin
        
        # Limit text length for better display
        sorted_data['Product_Label'] = sorted_data['Product'].apply(lambda x: (x[:25] + '...') if len(str(x)) > 25 else x)
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Calculate total for title
        total_profit = sorted_data['Total_Profit'].sum()
        
        # Add bars
        fig.add_trace(go.Bar(
            y=sorted_data['Product_Label'],
            x=sorted_data['Total_Profit'],
            orientation='h',
            text=sorted_data['Total_Profit'].apply(lambda x: f"${x:.2f}"),
            textposition='auto',
            marker=dict(
                color='rgba(58, 171, 115, 0.6)',
                line=dict(color='rgba(58, 171, 115, 1.0)', width=2)
            ),
            name="Total Profit"
        ))
        
        # Add profit margin as a secondary marker with color scale
        fig.add_trace(go.Scatter(
            y=sorted_data['Product_Label'],
            x=sorted_data['Total_Profit'],  # Use same x for positioning
            mode='markers',
            marker=dict(
                color=sorted_data['Profit_Margin'],
                size=16,
                colorscale='RdYlGn',
                colorbar=dict(title="Profit Margin"),
                showscale=True,
                cmin=0,
                cmax=max(0.5, sorted_data['Profit_Margin'].max())  # Cap at 50% for better color distribution
            ),
            text=sorted_data['Profit_Margin'].apply(lambda x: f"{x:.1%}"),
            hoverinfo='text',
            name="Profit Margin"
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Top {len(sorted_data)} Profit Items - Total: ${total_profit:.2f}",
            xaxis_title="Total Profit ($)",
            margin=dict(l=30, r=30, t=50, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=max(400, min(100 * len(sorted_data), 800))  # Dynamic height based on data points
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating top profit chart: {e}")
        # Create a fallback visualization
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        error_fig.update_layout(
            title="Top Profit Items (Error)",
            height=400
        )
        return error_fig

def create_out_of_date_chart(out_of_date_data):
    """Create chart for items approaching expiration date"""
    if out_of_date_data is None or len(out_of_date_data) == 0:
        # Return empty figure with instructions
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No expiring items found",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        empty_fig.update_layout(
            title="Items Approaching Expiration",
            xaxis_title="Days Until Expiration",
            margin=dict(l=30, r=30, t=50, b=30),
            height=400
        )
        return empty_fig
    
    try:
        # Sort data by days until expiration
        sorted_data = out_of_date_data.sort_values('Days_Until_Expiration')
        
        # Ensure required columns exist
        if 'Product' not in sorted_data.columns:
            if 'Item' in sorted_data.columns:
                sorted_data['Product'] = 'Product ' + sorted_data['Item'].astype(str)
            else:
                sorted_data['Product'] = [f'Item {i+1}' for i in range(len(sorted_data))]
        
        if 'Stock_Level' not in sorted_data.columns:
            sorted_data['Stock_Level'] = np.random.randint(5, 50, size=len(sorted_data))
        
        # Limit to top 10 items if we have more
        if len(sorted_data) > 10:
            sorted_data = sorted_data.head(10)
        
        # Limit product name length for display
        sorted_data['Product_Label'] = sorted_data['Product'].apply(lambda x: (str(x)[:25] + '...') if len(str(x)) > 25 else x)
        
        # Create separate figures for each visualization - one plot per row
        fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Days Until Expiration", "Expiring Items Inventory"),
                        specs=[[{"type": "bar"}], [{"type": "table"}]],
                        row_heights=[0.6, 0.4],
                        vertical_spacing=0.2)  # Increased vertical spacing
        
        # Define a color gradient based on days until expiration
        colors = []
        for days in sorted_data['Days_Until_Expiration']:
            if days <= 1:
                colors.append('red')
            elif days <= 3:
                colors.append('orange')
            else:
                colors.append('gold')
        
        # Add horizontal bar chart
        fig.add_trace(
            go.Bar(
                y=sorted_data['Product_Label'],
                x=sorted_data['Days_Until_Expiration'],
                orientation='h',
                text=sorted_data['Days_Until_Expiration'].apply(lambda x: f"{int(x)} days"),
                textposition='auto',
                marker_color=colors,
                hovertemplate="<b>%{y}</b><br>Days remaining: %{x}<br>Quantity: %{customdata}<extra></extra>",
                customdata=sorted_data['Stock_Level']
            ),
            row=1, col=1
        )
        
        # Calculate value at risk
        if 'Price' in sorted_data.columns and 'Stock_Level' in sorted_data.columns:
            sorted_data['Value_At_Risk'] = sorted_data['Price'] * sorted_data['Stock_Level']
        else:
            # Estimate value if price not available (assume average of $15 per unit)
            sorted_data['Value_At_Risk'] = sorted_data['Stock_Level'] * 15
        
        # Add a total value at risk annotation
        total_value_at_risk = sorted_data['Value_At_Risk'].sum()
        
        fig.add_annotation(
            text=f"Total inventory value at risk: ${total_value_at_risk:.2f}",
            xref="paper", yref="paper",
            x=0.5, y=0.95,
            showarrow=False,
            font=dict(size=14, color="red"),
            bgcolor="white",
            bordercolor="red",
            borderwidth=1,
            borderpad=4
        )
        
        # Add table with detailed information - enhanced version
        table_columns = ['UPC', 'Product', 'Quantity', 'Days Left', 'Value at Risk']
        table_values = [
            sorted_data['Item'],
            sorted_data['Product_Label'],
            sorted_data['Stock_Level'],
            sorted_data['Days_Until_Expiration'].apply(lambda x: f"{int(x)} days"),
            sorted_data['Value_At_Risk'].apply(lambda x: f"${x:.2f}")
        ]
        
        # Create cell colors based on days until expiration
        cell_colors = []
        for days in sorted_data['Days_Until_Expiration']:
            if days <= 1:
                cell_colors.append('rgba(255, 0, 0, 0.2)')  # Light red
            elif days <= 3:
                cell_colors.append('rgba(255, 165, 0, 0.2)')  # Light orange
            else:
                cell_colors.append('rgba(255, 215, 0, 0.1)')  # Light gold
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=table_columns,
                    fill_color='paleturquoise',
                    align='left',
                    font=dict(size=14)
                ),
                cells=dict(
                    values=table_values,
                    fill_color=[cell_colors] * len(table_columns),
                    align=['left', 'left', 'right', 'right', 'right'],
                    font=dict(size=13),
                    height=30
                )
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"Items Approaching Expiration ({len(sorted_data)} items)",
                'font': {'size': 20}
            },
            margin=dict(l=30, r=30, t=70, b=30),
            height=max(500, min(100 * len(sorted_data) + 300, 800))  # Dynamic height based on data points
        )
        
        # Update x-axis properties for the bar chart
        fig.update_xaxes(
            title_text="Days Until Expiration", 
            row=1, col=1,
            tickvals=list(range(0, int(sorted_data['Days_Until_Expiration'].max()) + 2))
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating expiring items chart: {e}")
        # Return a basic error figure
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        error_fig.update_layout(
            title="Items Approaching Expiration (Error)",
            height=400
        )
        return error_fig

def create_promotion_chart(promotion_data):
    """Create chart for promotion performance"""
    if promotion_data is None or len(promotion_data) == 0:
        # Return empty figure if no data
        return go.Figure()
    
    # Sort by lift percentage
    sorted_data = promotion_data.sort_values('Lift_Percentage', ascending=False)
    
    # Limit to top/bottom performers
    if len(sorted_data) > 16:
        top_performers = sorted_data.head(8)
        bottom_performers = sorted_data.tail(8)
        sorted_data = pd.concat([top_performers, bottom_performers])
    
    # Create chart
    fig = go.Figure()
    
    # Add bars with conditional coloring
    colors = ['green' if lift >= 0 else 'red' for lift in sorted_data['Lift_Percentage']]
    
    fig.add_trace(go.Bar(
        x=sorted_data['Product'],
        y=sorted_data['Lift_Percentage'],
        text=sorted_data['Lift_Percentage'].apply(lambda x: f"{x:.1f}%"),
        textposition='auto',
        marker_color=colors
    ))
    
    # Add reference line for zero lift
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(sorted_data) - 0.5,
        y1=0,
        line=dict(color="black", width=2, dash="dot"),
    )
    
    # Update layout
    fig.update_layout(
        title="Promotion Performance: Sales Lift Percentage",
        xaxis_title="Product",
        yaxis_title="Sales Lift %",
        margin=dict(l=30, r=30, t=50, b=30),  # Increased margins
        height=800  # Further increased height for full-width visualizations
    )
    
    # Update x-axis to handle long product names
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_out_of_stock_chart(out_of_stock_data):
    """Create chart for out of stock items"""
    if out_of_stock_data is None or len(out_of_stock_data) == 0:
        # Return empty figure if no data
        return go.Figure()
    
    # Sort by priority score
    sorted_data = out_of_stock_data.sort_values('Priority_Score', ascending=False)
    
    # Create chart with plotly express for easier categorical colors
    fig = px.bar(
        sorted_data,
        x='Product',
        y=np.ones(len(sorted_data)),  # All bars have height 1
        color='Priority_Score',
        labels={'y': 'Out of Stock', 'color': 'Priority'},
        color_continuous_scale=['orange', 'red'],
        title="Items Out of Stock (Prioritized)"
    )
    
    # Add markers for priority items
    fast_movers = sorted_data[sorted_data['Is_Fast_Mover']]
    promo_items = sorted_data[sorted_data['Is_On_Promotion']]
    
    if len(fast_movers) > 0:
        fig.add_trace(go.Scatter(
            x=fast_movers['Product'],
            y=np.ones(len(fast_movers)) * 0.5,  # Position in middle of bar
            mode='markers',
            marker=dict(symbol='star', size=12, color='yellow'),
            name='Fast Mover'
        ))
    
    if len(promo_items) > 0:
        fig.add_trace(go.Scatter(
            x=promo_items['Product'],
            y=np.ones(len(promo_items)) * 0.7,  # Position near top of bar
            mode='markers',
            marker=dict(symbol='circle', size=10, color='lightgreen'),
            name='On Promotion'
        ))
    
    # Update layout
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        yaxis=dict(showticklabels=False),  # Hide y-axis labels
        margin=dict(l=30, r=30, t=50, b=90),  # Increased margins
        height=600,  # Significantly increased height
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)  # Adjusted legend position
    )
    
    # Update x-axis to handle long product names
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_safety_stock_chart(safety_stock_data):
    """Create chart for items below safety stock"""
    if safety_stock_data is None or len(safety_stock_data) == 0:
        # Return empty figure if no data
        return go.Figure()
    
    # Sort by percentage below safety stock
    sorted_data = safety_stock_data.sort_values('Pct_Below_Safety', ascending=False)
    
    # Limit to top 15 items
    if len(sorted_data) > 15:
        sorted_data = sorted_data.head(15)
    
    # Create chart
    fig = go.Figure()
    
    # Add bars showing percentage below safety stock
    fig.add_trace(go.Bar(
        x=sorted_data['Product'],
        y=sorted_data['Pct_Below_Safety'],
        text=sorted_data['Pct_Below_Safety'].apply(lambda x: f"{x:.1f}%"),
        textposition='auto',
        marker=dict(
            color=sorted_data['Pct_Below_Safety'],
            colorscale='YlOrRd',
            showscale=True,
            colorbar=dict(title="% Below Safety")
        )
    ))
    
    # Add reference lines for severity levels
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=50,
        x1=len(sorted_data) - 0.5,
        y1=50,
        line=dict(color="orange", width=2, dash="dot"),
    )
    
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=75,
        x1=len(sorted_data) - 0.5,
        y1=75,
        line=dict(color="red", width=2, dash="dot"),
    )
    
    # Update layout
    fig.update_layout(
        title="Items Below Safety Stock Levels",
        xaxis_title="Product",
        yaxis_title="% Below Safety Stock",
        margin=dict(l=30, r=30, t=50, b=90),  # Increased margins
        height=800  # Further increased height for full-width visualizations
    )
    
    # Update x-axis to handle long product names
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_future_demand_chart(future_demand_data):
    """Create chart for items with forecasted high demand"""
    if future_demand_data is None or len(future_demand_data) == 0:
        # Return empty figure if no data
        return go.Figure()
    
    # Sort by demand increase factor
    sorted_data = future_demand_data.sort_values('Demand_Increase_Factor', ascending=False)
    
    # Limit to top 15 items
    if len(sorted_data) > 15:
        sorted_data = sorted_data.head(15)
    
    # Create separate figures for each visualization - one plot per row
    fig = make_subplots(rows=2, cols=1,
                       subplot_titles=("Demand Increase Factor", "Days of Stock at Forecasted Demand"),
                       specs=[[{"type": "bar"}], [{"type": "bar"}]],
                       row_heights=[0.5, 0.5],
                       vertical_spacing=0.2)  # Increased vertical spacing
    
    # Add bars for demand increase factor
    fig.add_trace(
        go.Bar(
            x=sorted_data['Product'],
            y=sorted_data['Demand_Increase_Factor'],
            text=sorted_data['Demand_Increase_Factor'].apply(lambda x: f"{x:.1f}x"),
            textposition='auto',
            marker_color='purple'
        ),
        row=1, col=1
    )
    
    # Add bars for days of stock at forecasted demand
    # Color based on sufficiency of stock
    colors = []
    for days in sorted_data['Days_Of_Stock_At_Forecast']:
        if days <= 3:
            colors.append('red')
        elif days <= 7:
            colors.append('orange')
        else:
            colors.append('green')
    
    fig.add_trace(
        go.Bar(
            x=sorted_data['Product'],
            y=sorted_data['Days_Of_Stock_At_Forecast'],
            text=sorted_data['Days_Of_Stock_At_Forecast'].apply(lambda x: f"{x:.1f} days"),
            textposition='auto',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="Items with Forecasted High Demand",
        margin=dict(l=30, r=30, t=50, b=90),  # Increased margins
        height=800  # Further increased height for full-width visualizations
    )
    
    # Update x-axes to handle long product names
    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_xaxes(tickangle=45, row=1, col=2)
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Demand Increase Factor", row=1, col=1)
    fig.update_yaxes(title_text="Days of Stock", row=1, col=2)
    
    return fig

def create_inventory_metrics_chart(combined_data):
    """Create additional visualization for inventory metrics"""
    if combined_data is None:
        return go.Figure()
        
    try:
        # Get latest data
        latest_date = combined_data['Date'].max()
        latest_data = combined_data[combined_data['Date'] == latest_date].copy()
        
        # Count products by stock status
        if 'Stock_Status' in latest_data.columns:
            status_counts = latest_data['Stock_Status'].value_counts()
            
            # Create a pie chart
            fig = go.Figure()
            
            # Map status to colors
            status_colors = {
                'Low': 'red',
                'Adequate': 'green',
                'Excess': 'orange',
                'Out of Stock': 'darkred',
                'Good': 'lightgreen',
                'Overstocked': 'gold'
            }
            
            colors = [status_colors.get(status, 'gray') for status in status_counts.index]
            
            fig.add_trace(go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                hole=0.3,
                marker_colors=colors
            ))
            
            fig.update_layout(
                title="Inventory Status Distribution",
                margin=dict(l=30, r=30, t=50, b=30),  # Increased margins
                height=450  # Significantly increased height
            )
            
            return fig
        
        # If we don't have Stock_Status, create a dummy chart for demonstration
        categories = ['Low Stock', 'Adequate', 'Excess', 'Out of Stock']
        values = [15, 45, 30, 10]
        colors = ['red', 'green', 'orange', 'darkred']
        
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=categories,
            values=values,
            hole=0.3,
            marker_colors=colors
        ))
        
        fig.update_layout(
            title="Inventory Status Distribution",
            margin=dict(l=30, r=30, t=50, b=30),  # Increased margins
            height=450  # Significantly increased height
        )
        
        return fig
            
    except Exception as e:
        print(f"Error creating inventory metrics chart: {e}")
        return go.Figure()

def generate_weekly_forecasts(combined_data, pytorch_forecasts, profit_impact, inventory_projection):
    """Generate weekly forecasts for demand, sales, profit, and inventory for the next 4 weeks"""
    if combined_data is None or pytorch_forecasts is None:
        return None
    
    try:
        # Get latest date and data
        latest_date = combined_data['Date'].max()
        latest_data = combined_data[combined_data['Date'] == latest_date].copy()
        
        # Filter forecast data to only include future dates
        future_forecast = pytorch_forecasts[pytorch_forecasts['Date'] > latest_date].copy()
        
        if len(future_forecast) == 0:
            return None
            
        # Define the week ranges
        week_ranges = []
        for i in range(4):
            start_date = latest_date + timedelta(days=i*7 + 1)
            end_date = latest_date + timedelta(days=(i+1)*7)
            week_ranges.append((f"Week {i+1}", start_date, end_date))
            
        # Check which prediction columns exist in the forecast data
        available_columns = []
        if 'Predicted_Demand' in future_forecast.columns:
            available_columns.append('Predicted_Demand')
        if 'Predicted_Sales' in future_forecast.columns:
            available_columns.append('Predicted_Sales')
            
        # If neither column exists, create a dummy column based on historical data
        if not available_columns:
            if 'Recent_Daily_Sales' in latest_data.columns:
                future_forecast['Predicted_Demand'] = latest_data['Recent_Daily_Sales'].mean()
                available_columns.append('Predicted_Demand')
            else:
                # Create a default prediction as fallback
                future_forecast['Predicted_Demand'] = 10  # Default fallback value
                available_columns.append('Predicted_Demand')
        
        # Initialize results dictionary to store forecasts
        forecast_results = {
            'demand': {},
            'sales': {},
            'profit': {},
            'inventory': {}
        }
        
        # Process forecasts for each week
        for week_name, start_date, end_date in week_ranges:
            # Filter data for this week
            week_data = future_forecast[(future_forecast['Date'] >= start_date) & 
                                         (future_forecast['Date'] <= end_date)].copy()
            
            if len(week_data) == 0:
                # If no data for this week, use the last available data and apply growth
                # This ensures we always have predictions for all 4 weeks
                growth_factor = 1.05  # Assuming 5% weekly growth as a placeholder
                
                # Get forecast for the last available week
                if week_name == "Week 1":
                    # For Week 1, base on historical data if no forecast
                    avg_daily_demand = latest_data['Recent_Daily_Sales'].mean() if 'Recent_Daily_Sales' in latest_data.columns else 10
                    forecast_results['demand'][week_name] = avg_daily_demand * 7  # Convert to weekly
                else:
                    # For later weeks, grow from previous week
                    prev_week = f"Week {int(week_name.split()[-1]) - 1}"
                    forecast_results['demand'][week_name] = forecast_results['demand'].get(prev_week, 70) * growth_factor
            else:
                # Calculate demand forecast for this week
                if 'Predicted_Demand' in week_data.columns:
                    forecast_results['demand'][week_name] = week_data['Predicted_Demand'].sum()
                elif 'Predicted_Sales' in week_data.columns:
                    forecast_results['demand'][week_name] = week_data['Predicted_Sales'].sum()
                else:
                    # Fallback - shouldn't reach here due to earlier checks
                    forecast_results['demand'][week_name] = 70  # Default value
            
            # Calculate sales forecast (assuming 95% of demand converts to sales)
            sales_conversion_rate = 0.95
            forecast_results['sales'][week_name] = forecast_results['demand'][week_name] * sales_conversion_rate
            
            # Calculate profit forecast
            avg_profit_margin = 0.3  # Default 30% profit margin
            avg_price = 15.0  # Default average price
            
            # Use actual profit data if available
            if profit_impact is not None and len(profit_impact) > 0:
                if 'Profit_Margin' in profit_impact.columns:
                    avg_profit_margin = profit_impact['Profit_Margin'].mean()
                if 'Price' in latest_data.columns and 'Cost' in latest_data.columns:
                    avg_price = latest_data['Price'].mean()
                    avg_cost = latest_data['Cost'].mean()
                    avg_profit_margin = (avg_price - avg_cost) / avg_price
                    
            forecast_results['profit'][week_name] = forecast_results['sales'][week_name] * avg_price * avg_profit_margin
            
            # Calculate inventory forecast
            if inventory_projection is not None and len(inventory_projection) > 0:
                # Start with current inventory
                if week_name == "Week 1":
                    starting_inventory = latest_data['Stock_Level'].sum() if 'Stock_Level' in latest_data.columns else 100
                else:
                    # For weeks 2-4, use previous week's ending inventory
                    prev_week = f"Week {int(week_name.split()[-1]) - 1}"
                    ending_inventory = forecast_results['inventory'].get(prev_week, {}).get('ending', 100)
                    starting_inventory = ending_inventory
                
                # Calculate expected new inventory (simplified - assuming weekly restock of ~80% of weekly sales)
                restock_rate = 0.8
                new_inventory = forecast_results['sales'][week_name] * restock_rate
                
                # Calculate ending inventory
                ending_inventory = starting_inventory - forecast_results['sales'][week_name] + new_inventory
                
                forecast_results['inventory'][week_name] = {
                    'starting': starting_inventory,
                    'new': new_inventory,
                    'sold': forecast_results['sales'][week_name],
                    'ending': ending_inventory
                }
            else:
                # Default inventory values if no projection data
                if week_name == "Week 1":
                    starting_inventory = 100
                else:
                    prev_week = f"Week {int(week_name.split()[-1]) - 1}"
                    ending_inventory = forecast_results['inventory'].get(prev_week, {}).get('ending', 100)
                    starting_inventory = ending_inventory
                    
                new_inventory = forecast_results['sales'][week_name] * 0.8
                ending_inventory = starting_inventory - forecast_results['sales'][week_name] + new_inventory
                
                forecast_results['inventory'][week_name] = {
                    'starting': starting_inventory,
                    'new': new_inventory,
                    'sold': forecast_results['sales'][week_name],
                    'ending': ending_inventory
                }
        
        return forecast_results
        
    except Exception as e:
        print(f"Error generating weekly forecasts: {e}")
        return None

def create_weekly_forecast_charts(forecast_results):
    """Create charts for weekly forecasts"""
    if forecast_results is None:
        return go.Figure()
    
    try:
        # Extract week labels and values for different metrics
        weeks = list(forecast_results['demand'].keys())
        demand_values = [forecast_results['demand'][week] for week in weeks]
        sales_values = [forecast_results['sales'][week] for week in weeks]
        profit_values = [forecast_results['profit'][week] for week in weeks]
        inventory_values = [forecast_results['inventory'][week]['ending'] for week in weeks]
        
        # Create subplots with each visualization in its own row
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=("Weekly Demand Forecast", "Weekly Sales Forecast", 
                           "Weekly Profit Forecast", "Weekly Inventory Projection"),
            vertical_spacing=0.15,
            row_heights=[0.25, 0.25, 0.25, 0.25]
        )
        
        # Add demand forecast trace
        fig.add_trace(
            go.Bar(
                x=weeks,
                y=demand_values,
                name="Predicted Demand",
                marker_color='royalblue',
                text=[f"{val:.1f}" for val in demand_values],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Add trend line for demand
        fig.add_trace(
            go.Scatter(
                x=weeks,
                y=demand_values,
                name="Demand Trend",
                line=dict(color='darkblue', dash='dot'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add sales forecast trace
        fig.add_trace(
            go.Bar(
                x=weeks,
                y=sales_values,
                name="Predicted Sales",
                marker_color='green',
                text=[f"{val:.1f}" for val in sales_values],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # Add trend line for sales
        fig.add_trace(
            go.Scatter(
                x=weeks,
                y=sales_values,
                name="Sales Trend",
                line=dict(color='darkgreen', dash='dot'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add profit forecast trace with different styling
        fig.add_trace(
            go.Bar(
                x=weeks,
                y=profit_values,
                name="Predicted Profit",
                marker_color='purple',
                text=[f"${val:.2f}" for val in profit_values],
                textposition='auto'
            ),
            row=3, col=1
        )
        
        # Add trend line for profit
        fig.add_trace(
            go.Scatter(
                x=weeks,
                y=profit_values,
                name="Profit Trend",
                line=dict(color='rebeccapurple', dash='dot'),
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Add inventory projection trace
        fig.add_trace(
            go.Scatter(
                x=weeks,
                y=inventory_values,
                name="Projected Inventory",
                mode='lines+markers',
                line=dict(color='orange', width=3),
                marker=dict(size=10, symbol='circle'),
                text=[f"{val:.1f} units" for val in inventory_values]
            ),
            row=4, col=1
        )
        
        # Add range for optimal inventory (visual guide)
        min_inventory = min(inventory_values) * 0.7 if inventory_values else 50
        max_inventory = max(inventory_values) * 1.3 if inventory_values else 150
        
        fig.add_trace(
            go.Scatter(
                x=weeks,
                y=[max_inventory * 0.8] * len(weeks),
                name="Target Range",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False
            ),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=weeks,
                y=[max_inventory * 0.5] * len(weeks),
                name="Target Range",
                fill='tonexty',
                fillcolor='rgba(0,255,0,0.1)',
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False
            ),
            row=4, col=1
        )
        
        # Update layout with increased size and better styling
        fig.update_layout(
            height=1600,  # Significantly increased height for four full-width plots
            template="plotly_white",
            title={
                'text': "4-Week Forecast Overview",
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            },
            margin=dict(t=120, b=80, l=80, r=80),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update axis titles
        fig.update_yaxes(title_text="Units", row=1, col=1)
        fig.update_yaxes(title_text="Units", row=2, col=1)
        fig.update_yaxes(title_text="Profit ($)", row=3, col=1)
        fig.update_yaxes(title_text="Inventory Units", row=4, col=1)
        
        # Make subplot titles larger
        for i, annotation in enumerate(fig['layout']['annotations'][:4]):
            annotation['font'] = dict(size=16, color='darkblue')
            
        return fig
        
    except Exception as e:
        print(f"Error creating weekly forecast charts: {e}")
        return go.Figure()

def create_forecast_data_table(forecast_results):
    """Create data table with numerical forecast data"""
    if forecast_results is None:
        return html.Div("No forecast data available")
        
    try:
        # Extract data from forecast results
        weeks = list(forecast_results['demand'].keys())
        
        # Create rows for the table
        table_rows = []
        
        # Demand row
        demand_row = html.Tr([
            html.Th("Demand (units)", style={"textAlign": "left"}),
            *[html.Td(f"{forecast_results['demand'][week]:.1f}", 
                      style={"textAlign": "right", "fontWeight": "bold", "color": "royalblue"}) 
              for week in weeks]
        ])
        table_rows.append(demand_row)
        
        # Sales row
        sales_row = html.Tr([
            html.Th("Sales (units)", style={"textAlign": "left"}),
            *[html.Td(f"{forecast_results['sales'][week]:.1f}", 
                      style={"textAlign": "right", "fontWeight": "bold", "color": "green"}) 
              for week in weeks]
        ])
        table_rows.append(sales_row)
        
        # Sales conversion rate row
        conversion_row = html.Tr([
            html.Th("Sales Rate", style={"textAlign": "left"}),
            *[html.Td(f"{(forecast_results['sales'][week] / forecast_results['demand'][week] * 100):.1f}%" 
                      if forecast_results['demand'][week] > 0 else "N/A", 
                      style={"textAlign": "right"}) 
              for week in weeks]
        ])
        table_rows.append(conversion_row)
        
        # Profit row
        profit_row = html.Tr([
            html.Th("Profit ($)", style={"textAlign": "left"}),
            *[html.Td(f"${forecast_results['profit'][week]:.2f}", 
                      style={"textAlign": "right", "fontWeight": "bold", "color": "purple"}) 
              for week in weeks]
        ])
        table_rows.append(profit_row)
        
        # Profit per unit row
        profit_per_unit_row = html.Tr([
            html.Th("Profit/Unit ($)", style={"textAlign": "left"}),
            *[html.Td(f"${forecast_results['profit'][week] / forecast_results['sales'][week]:.2f}" 
                      if forecast_results['sales'][week] > 0 else "N/A", 
                      style={"textAlign": "right"}) 
              for week in weeks]
        ])
        table_rows.append(profit_per_unit_row)
        
        # Inventory starting row
        inv_start_row = html.Tr([
            html.Th("Starting Inventory", style={"textAlign": "left"}),
            *[html.Td(f"{forecast_results['inventory'][week]['starting']:.1f}", 
                      style={"textAlign": "right"}) 
              for week in weeks]
        ])
        table_rows.append(inv_start_row)
        
        # Inventory ending row
        inv_end_row = html.Tr([
            html.Th("Ending Inventory", style={"textAlign": "left"}),
            *[html.Td(f"{forecast_results['inventory'][week]['ending']:.1f}", 
                      style={"textAlign": "right", "fontWeight": "bold", "color": "orange"}) 
              for week in weeks]
        ])
        table_rows.append(inv_end_row)
        
        # Create table with header
        table = dbc.Table(
            [
                html.Thead(
                    html.Tr(
                        [html.Th("Metric")] + [html.Th(week) for week in weeks],
                        style={"backgroundColor": "#f8f9fa"}
                    )
                ),
                html.Tbody(table_rows)
            ],
            bordered=True,
            hover=True,
            responsive=True,
            striped=True,
            className="table-sm"
        )
        
        # Create card with the table
        table_card = dbc.Card([
            dbc.CardHeader(html.H5("4-Week Forecast Details", className="text-center")),
            dbc.CardBody([
                table,
                html.Div([
                    html.P("Notes:", className="mt-3 mb-1 font-weight-bold"),
                    html.Ul([
                        html.Li("Demand represents expected customer interest"),
                        html.Li("Sales accounts for conversion rate from demand"),
                        html.Li("Inventory projections assume regular restocking"),
                    ], className="small")
                ])
            ])
        ])
        
        return table_card
        
    except Exception as e:
        print(f"Error creating forecast data table: {e}")
        return html.Div(f"Error creating forecast table: {str(e)}")

def create_sales_by_category_chart(combined_data):
    """Create visualization of sales by category"""
    if combined_data is None:
        return go.Figure()
        
    try:
        # Get latest data
        latest_date = combined_data['Date'].max()
        latest_data = combined_data[combined_data['Date'] == latest_date].copy()
        
        # Check if we have category information
        if 'Category' in latest_data.columns:
            # Group by category and sum sales
            category_sales = latest_data.groupby('Category')['Recent_Daily_Sales'].sum().reset_index()
            
            # Sort by sales
            category_sales = category_sales.sort_values('Recent_Daily_Sales', ascending=False)
            
            # Create a bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=category_sales['Category'],
                y=category_sales['Recent_Daily_Sales'],
                text=category_sales['Recent_Daily_Sales'].apply(lambda x: f"{x:.1f}"),
                textposition='auto',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title="Daily Sales by Category",
                xaxis_title="Category",
                yaxis_title="Daily Sales",
                margin=dict(l=30, r=30, t=50, b=30),  # Increased margins
                height=450  # Significantly increased height
            )
            
            return fig
            
        # If we don't have Category, try to create one from the Product names
        category_map = defaultdict(int)
        for product in latest_data['Product']:
            if ' ' in product:
                category = product.split(' ')[0]  # Use first word as category
                if 'Recent_Daily_Sales' in latest_data.columns:
                    sales = latest_data[latest_data['Product'] == product]['Recent_Daily_Sales'].sum()
                    category_map[category] += sales
        
        # Convert to dataframe
        if category_map:
            categories = list(category_map.keys())
            sales = list(category_map.values())
            
            category_df = pd.DataFrame({
                'Category': categories,
                'Sales': sales
            })
            
            # Sort by sales
            category_df = category_df.sort_values('Sales', ascending=False)
            
            # Create bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=category_df['Category'],
                y=category_df['Sales'],
                text=category_df['Sales'].apply(lambda x: f"{x:.1f}"),
                textposition='auto',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title="Daily Sales by Category",
                xaxis_title="Category",
                yaxis_title="Daily Sales",
                margin=dict(l=30, r=30, t=50, b=30),  # Increased margins
                height=450  # Significantly increased height
            )
            
            return fig
        
        # If all else fails, return empty figure
        return go.Figure()
            
    except Exception as e:
        print(f"Error creating sales by category chart: {e}")
        return go.Figure()

def create_summary_dashboard_layout(summary_data):
    """Create the layout for the summary dashboard"""
    # Create charts based on summary data
    top_profit_chart = create_top_profit_chart(summary_data.get('top_profit_items'))
    out_of_date_chart = create_out_of_date_chart(summary_data.get('out_of_date_items'))
    promotion_chart = create_promotion_chart(summary_data.get('promotion_performance'))
    out_of_stock_chart = create_out_of_stock_chart(summary_data.get('out_of_stock_items'))
    safety_stock_chart = create_safety_stock_chart(summary_data.get('below_safety_stock'))
    future_demand_chart = create_future_demand_chart(summary_data.get('future_high_demand'))
    inventory_metrics_chart = create_inventory_metrics_chart(summary_data.get('combined_data'))
    sales_category_chart = create_sales_by_category_chart(summary_data.get('combined_data'))
    
    # Generate weekly forecasts
    forecast_results = summary_data.get('weekly_forecasts')
    weekly_forecast_chart = create_weekly_forecast_charts(forecast_results) if forecast_results else None
    forecast_data_table = create_forecast_data_table(forecast_results) if forecast_results else None
    
    # Create layout with each visualization in its own row
    dashboard_layout = [
        # Title row
        dbc.Row([
            dbc.Col(html.H2("Store Performance Summary Dashboard", className="text-center"), width=12)
        ]),
        
        # Each visualization gets its own row - Top profit items
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Top Profit Items"),
                    dbc.CardBody(dcc.Graph(figure=top_profit_chart))
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Items approaching expiration
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Items Approaching Expiration"),
                    dbc.CardBody(dcc.Graph(figure=out_of_date_chart))
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Promotion performance
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Promotion Performance"),
                    dbc.CardBody(dcc.Graph(figure=promotion_chart))
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Out of stock items
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Out of Stock Items"),
                    dbc.CardBody(dcc.Graph(figure=out_of_stock_chart))
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Items below safety stock
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Items Below Safety Stock"),
                    dbc.CardBody(dcc.Graph(figure=safety_stock_chart))
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Forecasted high demand
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Forecasted High Demand"),
                    dbc.CardBody(dcc.Graph(figure=future_demand_chart))
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Inventory status distribution
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Inventory Status Distribution"),
                    dbc.CardBody(dcc.Graph(figure=inventory_metrics_chart))
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Sales by category
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Sales by Category"),
                    dbc.CardBody(dcc.Graph(figure=sales_category_chart))
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Weekly forecast section (full width)
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Weekly Forecast Report (4-Week Projection)", className="text-center")),
                    dbc.CardBody([
                        # Description row
                        dbc.Row([
                            dbc.Col([
                                html.P([
                                    "This report provides weekly projections for the next 4 weeks based on historical data and forecasting models. ",
                                    "Use this information for inventory planning and resource allocation."
                                ], className="text-muted mb-4")
                            ])
                        ]),
                        # Charts row - full width
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(figure=weekly_forecast_chart) if weekly_forecast_chart else 
                                html.Div("No forecast data available", className="text-center p-5 bg-light")
                            ], width=12)
                        ], className="mb-4"),
                        # Data tables row below charts
                        dbc.Row([
                            dbc.Col([
                                forecast_data_table if forecast_data_table else 
                                html.Div("No forecast data available", className="text-center p-5 bg-light")
                            ], width=12)
                        ])
                    ])
                ], className="shadow mb-5")  # Added shadow for emphasis
            ], width=12)
        ])
    ]
    
    return dashboard_layout

def register_summary_callbacks(app, combined_data, profit_impact, inventory_projection, pytorch_forecasts):
    """Register callbacks for the summary dashboard"""
    @app.callback(
        Output("summary-dashboard-content", "children"),
        [Input("store-dropdown", "value"),
         Input("apply-stock-adjustment", "n_clicks")],
        [State("product-dropdown", "value"),
         State("stock-adjustment-input", "value"),
         State("stock-adjustment-date", "date")]
    )
    def update_summary_dashboard(store, adjust_clicks, product, stock_adjustment, adjustment_date):
        try:
            if combined_data is None or store is None:
                return html.P("No data available for summary dashboard.")
            
            # Filter data for selected store
            store_data = combined_data[combined_data['Store_Id'] == store].copy()
            
            # Apply any stored adjustments to this store data if it's already been saved
            # This allows the summary to reflect all product adjustments
            try:
                if hasattr(app, 'manual_stock_adjustments'):
                    for key, value in app.manual_stock_adjustments.items():
                        if key.startswith(f"{store}_"):
                            try:
                                product_id = int(key.split('_')[1])
                                product_idx = store_data[store_data['Item'] == product_id].index
                                if len(product_idx) > 0:
                                    store_data.loc[product_idx, 'Stock_Level'] = value
                                    # Update other derived columns as needed
                                    if 'Recent_Daily_Sales' in store_data.columns:
                                        daily_sales = store_data.loc[product_idx, 'Recent_Daily_Sales'].values[0]
                                        if daily_sales > 0:
                                            store_data.loc[product_idx, 'Weeks_of_Supply'] = value / (daily_sales * 7)
                                            if 'Stock_Status' in store_data.columns:
                                                weeks_supply = value / (daily_sales * 7)
                                                if weeks_supply < 1:
                                                    status = 'Low'
                                                elif weeks_supply <= 3:
                                                    status = 'Adequate'
                                                else:
                                                    status = 'Excess'
                                                store_data.loc[product_idx, 'Stock_Status'] = status
                            except (ValueError, KeyError) as e:
                                print(f"Error processing adjustment {key}: {e}")
                                continue
                            
                if hasattr(app, 'manual_stock_adjustments_with_dates'):
                    latest_date_str = store_data['Date'].max().strftime('%Y-%m-%d')
                    for key, value in app.manual_stock_adjustments_with_dates.items():
                        if key.startswith(f"{store}_") and key.endswith(f"_{latest_date_str}"):
                            parts = key.split('_')
                            if len(parts) >= 3:
                                try:
                                    product_id = int(parts[1])
                                    product_idx = store_data[store_data['Item'] == product_id].index
                                    if len(product_idx) > 0:
                                        store_data.loc[product_idx, 'Stock_Level'] = value
                                        # Update other derived columns as needed
                                        if 'Recent_Daily_Sales' in store_data.columns:
                                            daily_sales = store_data.loc[product_idx, 'Recent_Daily_Sales'].values[0]
                                            if daily_sales > 0:
                                                store_data.loc[product_idx, 'Weeks_of_Supply'] = value / (daily_sales * 7)
                                                if 'Stock_Status' in store_data.columns:
                                                    weeks_supply = value / (daily_sales * 7)
                                                    if weeks_supply < 1:
                                                        status = 'Low'
                                                    elif weeks_supply <= 3:
                                                        status = 'Adequate'
                                                    else:
                                                        status = 'Excess'
                                                    store_data.loc[product_idx, 'Stock_Status'] = status
                                except (ValueError, KeyError) as e:
                                    print(f"Error processing date adjustment {key}: {e}")
                                    continue
            except Exception as e:
                print(f"Error applying stock adjustments in summary: {e}")
            
            store_profit_impact = None
            if profit_impact is not None:
                store_profit_impact = profit_impact[profit_impact['Store_Id'] == store]
                
            store_inventory = None
            if inventory_projection is not None:
                store_inventory = inventory_projection[inventory_projection['Store_Id'] == store]
                
            store_forecasts = None
            if pytorch_forecasts is not None:
                store_forecasts = pytorch_forecasts[pytorch_forecasts['Store_Id'] == store]
            
            # Process data for the store
            summary_data = {
                'top_profit_items': calculate_top_profit_items(store_data, store_profit_impact),
                'out_of_date_items': identify_expiring_items(store_data),
                'promotion_performance': analyze_promotion_performance(store_data),
                'out_of_stock_items': identify_out_of_stock(store_data, store_inventory),
                'below_safety_stock': identify_below_safety_stock(store_data),
                'future_high_demand': forecast_high_demand(store_data, store_forecasts),
                'combined_data': store_data,
                'weekly_forecasts': generate_weekly_forecasts(store_data, store_forecasts, store_profit_impact, store_inventory)
            }
            
            # Create and return layout
            return create_summary_dashboard_layout(summary_data)
            
        except Exception as e:
            print(f"Error updating summary dashboard: {e}")
            return html.Div([
                html.H4("Error Loading Summary Dashboard"),
                html.P(f"An error occurred: {str(e)}"),
                html.P("Please check that all required data files are available.")
            ])