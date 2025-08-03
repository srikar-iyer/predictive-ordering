"""
Profit optimization module for the Pizza Predictive Ordering System.
This module calculates price elasticities and recommends optimal prices.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('profit_optimizer')

# Import settings if available
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from config.settings import (
        COMBINED_DATA_FILE, PYTORCH_FORECASTS_FILE, RF_FORECASTS_FILE, 
        PRICE_ELASTICITIES_FILE, PRICE_RECOMMENDATIONS_FILE, PROFIT_IMPACT_FILE,
        PRODUCT_MIX_FILE, STATIC_DIR, MAX_PRICE_INCREASE, MAX_PRICE_DECREASE, MIN_MARGIN,
        ELASTICITY_CONSTRAINT
    )
except ImportError:
    # Default paths for backward compatibility
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    COMBINED_DATA_FILE = os.path.join(ROOT_DIR, "data", "processed", "combined_pizza_data.csv")
    PYTORCH_FORECASTS_FILE = os.path.join(ROOT_DIR, "data", "processed", "pytorch_forecasts.csv")
    RF_FORECASTS_FILE = os.path.join(ROOT_DIR, "data", "processed", "rf_forecasts.csv")
    PRICE_ELASTICITIES_FILE = os.path.join(ROOT_DIR, "data", "processed", "price_elasticities.csv")
    PRICE_RECOMMENDATIONS_FILE = os.path.join(ROOT_DIR, "data", "processed", "price_recommendations.csv")
    PROFIT_IMPACT_FILE = os.path.join(ROOT_DIR, "data", "processed", "profit_impact.csv")
    PRODUCT_MIX_FILE = os.path.join(ROOT_DIR, "data", "processed", "product_mix_optimization.csv")
    STATIC_DIR = os.path.join(ROOT_DIR, "static")
    MAX_PRICE_INCREASE = 20  # percentage
    MAX_PRICE_DECREASE = 15  # percentage
    MIN_MARGIN = 25  # percentage
    ELASTICITY_CONSTRAINT = 10  # percentage


class ProfitOptimizer:
    """
    Class for optimizing prices and calculating profit impact
    """
    def __init__(self, df=None, forecast_df=None):
        """
        Initialize the profit optimizer with data.
        
        Args:
            df: DataFrame with historical sales data
            forecast_df: DataFrame with sales forecasts
        """
        self.df = df
        self.forecast_df = forecast_df
        self.elasticity_models = {}
        
    def load_data(self, df_path=COMBINED_DATA_FILE, forecast_path=None):
        """
        Load data from CSV files.
        
        Args:
            df_path: Path to the historical data CSV
            forecast_path: Path to the forecast data CSV (optional)
            
        Returns:
            self: ProfitOptimizer instance for method chaining
        """
        logger.info(f"Loading data from {df_path}")
        self.df = pd.read_csv(df_path)
        
        # Convert date column to datetime
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Load forecasts if provided
        if forecast_path is not None:
            logger.info(f"Loading forecasts from {forecast_path}")
            self.forecast_df = pd.read_csv(forecast_path)
            if 'Date' in self.forecast_df.columns:
                self.forecast_df['Date'] = pd.to_datetime(self.forecast_df['Date'])
        
        return self
    
    def calculate_elasticities(self):
        """
        Calculate price elasticity for each product based on historical data.
        
        Returns:
            DataFrame: DataFrame with price elasticity data
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Calculating price elasticities")
        elasticity_results = []
        
        # Group by store and item
        for (store_id, item), group in self.df.groupby(['Store_Id', 'Item']):
            # Need sufficient price variation to calculate elasticity
            unique_prices = group['Price'].dropna().unique()
            
            if len(unique_prices) <= 1:
                # Not enough price variation
                elasticity = 0.0
                r_squared = 0.0
                coef = 0.0
                intercept = 0.0
                is_significant = False
                status = "Insufficient price variation"
            else:
                try:
                    # Log-log model for elasticity
                    # ln(Q) = a + b*ln(P)
                    # Elasticity = b
                    
                    # Filter for valid price and sales data
                    valid_data = group[(group['Price'] > 0) & (group['Sales'] > 0)].copy()
                    
                    if len(valid_data) < 5:
                        elasticity = -1.0  # Default elasticity
                        r_squared = 0.0
                        coef = 0.0
                        intercept = 0.0
                        is_significant = False
                        status = "Insufficient data points"
                    else:
                        # Calculate log values
                        valid_data['Log_Price'] = np.log(valid_data['Price'])
                        valid_data['Log_Sales'] = np.log(valid_data['Sales'])
                        
                        # Create model
                        X = valid_data['Log_Price'].values.reshape(-1, 1)
                        y = valid_data['Log_Sales'].values
                        
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        # Calculate elasticity (coefficient of log-log model)
                        elasticity = model.coef_[0]
                        intercept = model.intercept_
                        
                        # Calculate R-squared
                        y_pred = model.predict(X)
                        ss_total = np.sum((y - np.mean(y))**2)
                        ss_residual = np.sum((y - y_pred)**2)
                        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
                        
                        # Check if elasticity is significant
                        is_significant = (r_squared >= 0.3) and (elasticity < 0)
                        status = "Valid elasticity" if is_significant else "Poor fit"
                        
                        # Store the model for later use
                        self.elasticity_models[(store_id, item)] = {
                            'intercept': intercept,
                            'elasticity': elasticity,
                            'r_squared': r_squared
                        }
                except Exception as e:
                    elasticity = -1.0  # Default elasticity on error
                    r_squared = 0.0
                    coef = 0.0
                    intercept = 0.0
                    is_significant = False
                    status = f"Error: {str(e)}"
            
            # Get product information
            product = group['Product'].iloc[0] if 'Product' in group.columns else 'Unknown'
            
            # Get cost information (use most recent or average)
            if 'Cost' in group.columns:
                cost = group['Cost'].iloc[-1] / max(1, group['Sales'].iloc[-1])  # Per unit cost
            else:
                cost = 0.0
            
            # Get current price (most recent)
            current_price = group['Price'].iloc[-1] if 'Price' in group.columns else 0.0
            
            # Calculate margin
            margin_pct = ((current_price - cost) / current_price * 100) if current_price > 0 else 0.0
            
            # Add to results
            elasticity_results.append({
                'Store_Id': store_id,
                'Item': item,
                'Product': product,
                'Elasticity': elasticity,
                'R_Squared': r_squared,
                'Is_Significant': is_significant,
                'Status': status,
                'Current_Price': current_price,
                'Cost': cost,
                'Margin_Pct': margin_pct,
                'Avg_Daily_Sales': group['Sales'].mean(),
                'Intercept': intercept
            })
        
        # Convert to DataFrame
        elasticity_df = pd.DataFrame(elasticity_results)
        
        # For products with invalid elasticities, use default values
        elasticity_df.loc[~elasticity_df['Is_Significant'], 'Elasticity'] = -1.0  # Default elasticity
        
        # Apply constraints
        elasticity_df['Elasticity'] = elasticity_df['Elasticity'].clip(upper=-0.1, lower=-3.0)
        
        logger.info(f"Calculated elasticities for {len(elasticity_df)} products")
        return elasticity_df
    
    def optimize_prices(self, elasticity_df, constraints=None):
        """
        Optimize prices for each product based on elasticities.
        
        Args:
            elasticity_df: DataFrame with elasticity data
            constraints: Dictionary of optimization constraints
            
        Returns:
            DataFrame: DataFrame with price recommendations
        """
        if constraints is None:
            constraints = {
                'max_price_increase': MAX_PRICE_INCREASE,  # Maximum 20% price increase
                'max_price_decrease': MAX_PRICE_DECREASE,  # Maximum 15% price decrease
                'min_margin': MIN_MARGIN,          # Minimum 25% margin
                'elasticity_constraint': ELASTICITY_CONSTRAINT,  # Max price change for elastic products
            }
            
        logger.info("Optimizing prices")
        
        optimization_results = []
        
        for _, row in elasticity_df.iterrows():
            # Skip products with zero or positive elasticity
            if row['Elasticity'] >= 0:
                optimization_results.append({
                    'Store_Id': row['Store_Id'],
                    'Item': row['Item'],
                    'Product': row['Product'],
                    'Current_Price': row['Current_Price'],
                    'Optimal_Price': row['Current_Price'],
                    'Price_Change_Pct': 0.0,
                    'Expected_Sales_Change_Pct': 0.0,
                    'Expected_Profit_Change_Pct': 0.0,
                    'New_Margin_Pct': row['Margin_Pct'],
                    'Recommendation': 'No change (invalid elasticity)'
                })
                continue
                
            # Get elasticity and current values
            elasticity = row['Elasticity']
            current_price = row['Current_Price']
            cost = row['Cost']
            
            # Skip if current price is zero or invalid
            if current_price <= 0:
                optimization_results.append({
                    'Store_Id': row['Store_Id'],
                    'Item': row['Item'],
                    'Product': row['Product'],
                    'Current_Price': current_price,
                    'Optimal_Price': current_price,
                    'Price_Change_Pct': 0.0,
                    'Expected_Sales_Change_Pct': 0.0,
                    'Expected_Profit_Change_Pct': 0.0,
                    'New_Margin_Pct': row['Margin_Pct'],
                    'Recommendation': 'No change (invalid price)'
                })
                continue
            
            # Define profit function to maximize
            # For elasticity, Q2 = Q1 * (P2/P1)^e
            def profit_function(price):
                # Calculate new quantity based on elasticity
                price_ratio = price / current_price
                quantity_ratio = price_ratio ** elasticity
                new_profit = (price - cost) * quantity_ratio
                current_profit = (current_price - cost)
                
                # Return negative profit for minimization
                return -new_profit / current_profit
            
            # Set standard price bounds based on constraints
            standard_lower_bound = current_price * (1 - constraints['max_price_decrease'] / 100)
            standard_upper_bound = current_price * (1 + constraints['max_price_increase'] / 100)
            
            # Calculate elasticity-based constraints
            # Higher absolute elasticity values (more elastic) get tighter constraints
            elasticity_abs = abs(elasticity)
            
            # Apply tighter constraints for more elastic products
            if elasticity_abs >= 1.5:  # Highly elastic
                max_change_pct = constraints.get('elasticity_constraint', 10.0)
                elasticity_lower_bound = current_price * (1 - max_change_pct / 100)
                elasticity_upper_bound = current_price * (1 + max_change_pct / 100)
                
                # Use the more restrictive bounds (tighter constraints)
                lower_bound = max(standard_lower_bound, elasticity_lower_bound)
                upper_bound = min(standard_upper_bound, elasticity_upper_bound)
                constraint_applied = "elasticity"
            else:
                # Use standard bounds for less elastic products
                lower_bound = standard_lower_bound
                upper_bound = standard_upper_bound
                constraint_applied = "standard"
            
            # Ensure minimum margin is respected
            min_price = cost / (1 - constraints['min_margin'] / 100)
            if min_price > lower_bound:
                lower_bound = min_price
                constraint_applied = "margin"
            
            # If bounds are invalid, skip optimization
            if lower_bound >= upper_bound:
                optimal_price = current_price
                price_change_pct = 0.0
                expected_sales_change_pct = 0.0
                expected_profit_change_pct = 0.0
                recommendation = 'No change (constraints conflict)'
                constraint_applied = "conflict"
            else:
                # Optimize price within bounds
                result = minimize(
                    profit_function,
                    current_price,
                    bounds=[(lower_bound, upper_bound)],
                    method='L-BFGS-B'
                )
                
                if result.success:
                    optimal_price = result.x[0]
                    
                    # Calculate changes
                    price_change_pct = (optimal_price / current_price - 1) * 100
                    price_ratio = optimal_price / current_price
                    expected_sales_change_pct = (price_ratio ** elasticity - 1) * 100
                    
                    # Calculate profit change
                    current_profit = (current_price - cost)
                    new_profit = (optimal_price - cost) * (price_ratio ** elasticity)
                    expected_profit_change_pct = (new_profit / current_profit - 1) * 100
                    
                    # Check if bound constraints were hit
                    if abs(optimal_price - lower_bound) < 0.01:
                        constraint_applied = f"{constraint_applied}_lower_bound"
                    elif abs(optimal_price - upper_bound) < 0.01:
                        constraint_applied = f"{constraint_applied}_upper_bound"
                    
                    # Determine recommendation
                    if abs(price_change_pct) < 1.0:
                        recommendation = 'Maintain current price'
                    elif price_change_pct > 0:
                        recommendation = f'Increase price by {price_change_pct:.1f}%'
                    else:
                        recommendation = f'Decrease price by {abs(price_change_pct):.1f}%'
                else:
                    # Optimization failed
                    optimal_price = current_price
                    price_change_pct = 0.0
                    expected_sales_change_pct = 0.0
                    expected_profit_change_pct = 0.0
                    recommendation = 'No change (optimization failed)'
                    constraint_applied = "failed"
            
            # Calculate new margin
            new_margin_pct = ((optimal_price - cost) / optimal_price * 100) if optimal_price > 0 else 0.0
            
            # Add to results
            optimization_results.append({
                'Store_Id': row['Store_Id'],
                'Item': row['Item'],
                'Product': row['Product'],
                'Elasticity': elasticity,
                'Current_Price': current_price,
                'Optimal_Price': optimal_price,
                'Price_Change_Pct': price_change_pct,
                'Expected_Sales_Change_Pct': expected_sales_change_pct,
                'Expected_Profit_Change_Pct': expected_profit_change_pct,
                'Cost': cost,
                'Current_Margin_Pct': row['Margin_Pct'],
                'New_Margin_Pct': new_margin_pct,
                'Constraint_Applied': constraint_applied,
                'Recommendation': recommendation
            })
        
        # Convert to DataFrame
        optimization_df = pd.DataFrame(optimization_results)
        
        logger.info(f"Generated price recommendations for {len(optimization_df)} products")
        return optimization_df
    
    def calculate_profit_impact(self, price_recommendations, forecasts=None):
        """
        Calculate the expected profit impact of price recommendations.
        
        Args:
            price_recommendations: DataFrame with price recommendations
            forecasts: DataFrame with sales forecasts (optional)
            
        Returns:
            DataFrame: DataFrame with profit impact data
        """
        if forecasts is None:
            forecasts = self.forecast_df
            
        logger.info("Calculating profit impact")
        
        if forecasts is None:
            logger.warning("No forecasts available for profit impact calculation")
            return None
        
        # Initialize impact results
        impact_results = []
        
        # Process each product
        for _, row in price_recommendations.iterrows():
            store_id = row['Store_Id']
            item = row['Item']
            
            # Get forecast for this product
            product_forecast = forecasts[
                (forecasts['Store_Id'] == store_id) & 
                (forecasts['Item'] == item)
            ].copy() if forecasts is not None else None
            
            # Skip if no forecast available
            if product_forecast is None or len(product_forecast) == 0:
                logger.warning(f"No forecast available for Store {store_id}, Item {item}")
                continue
                
            # Get product and price information
            product = row['Product']
            current_price = row['Current_Price']
            optimal_price = row['Optimal_Price']
            elasticity = row['Elasticity']
            cost = row['Cost']
            
            # Calculate price ratio and expected quantity change
            price_ratio = optimal_price / current_price if current_price > 0 else 1.0
            quantity_multiplier = price_ratio ** elasticity if elasticity < 0 else 1.0
            
            # Apply forecast adjustments
            # Check if 'Forecast' column exists, otherwise use 'Sales' column
            forecast_col = 'Forecast' if 'Forecast' in product_forecast.columns else 'Sales'
            logger.info(f"Using {forecast_col} column for profit impact calculations")
            
            product_forecast['Original_Forecast'] = product_forecast[forecast_col]
            product_forecast['Adjusted_Forecast'] = product_forecast[forecast_col] * quantity_multiplier
            
            # Calculate financial metrics
            product_forecast['Current_Revenue'] = product_forecast['Original_Forecast'] * current_price
            product_forecast['Current_Cost'] = product_forecast['Original_Forecast'] * cost
            product_forecast['Current_Profit'] = product_forecast['Current_Revenue'] - product_forecast['Current_Cost']
            
            product_forecast['New_Revenue'] = product_forecast['Adjusted_Forecast'] * optimal_price
            product_forecast['New_Cost'] = product_forecast['Adjusted_Forecast'] * cost
            product_forecast['New_Profit'] = product_forecast['New_Revenue'] - product_forecast['New_Cost']
            
            product_forecast['Profit_Difference'] = product_forecast['New_Profit'] - product_forecast['Current_Profit']
            
            # Calculate summary metrics
            total_current_profit = product_forecast['Current_Profit'].sum()
            total_new_profit = product_forecast['New_Profit'].sum()
            total_profit_difference = product_forecast['Profit_Difference'].sum()
            profit_change_pct = (total_new_profit / total_current_profit - 1) * 100 if total_current_profit > 0 else 0
            
            # Add to results
            impact_results.append({
                'Store_Id': store_id,
                'Item': item,
                'Product': product,
                'Current_Price': current_price,
                'Optimal_Price': optimal_price,
                'Price_Change_Pct': (optimal_price / current_price - 1) * 100 if current_price > 0 else 0,
                'Elasticity': elasticity,
                'Total_Current_Profit': total_current_profit,
                'Total_New_Profit': total_new_profit,
                'Total_Profit_Difference': total_profit_difference,
                'Profit_Change_Pct': profit_change_pct,
                'Forecast_Days': len(product_forecast),
                'Daily_Profit_Impact': total_profit_difference / len(product_forecast) if len(product_forecast) > 0 else 0
            })
        
        # Convert to DataFrame
        impact_df = pd.DataFrame(impact_results)
        
        # Sort by profit impact (check if column exists first)
        if 'Total_Profit_Difference' in impact_df.columns and not impact_df.empty:
            impact_df = impact_df.sort_values('Total_Profit_Difference', ascending=False)
        else:
            logger.warning("Column 'Total_Profit_Difference' not found in impact_df or DataFrame is empty")
        
        logger.info(f"Calculated profit impact for {len(impact_df)} products")
        return impact_df
    
    def optimize_product_mix(self, elasticity_df, forecasts=None):
        """
        Optimize the product mix considering cross-elasticities.
        
        Args:
            elasticity_df: DataFrame with elasticity data
            forecasts: DataFrame with sales forecasts
            
        Returns:
            DataFrame: DataFrame with product mix recommendations
        """
        logger.info("Optimizing product mix")
        
        # Group by store to optimize product mix for each store
        stores = elasticity_df['Store_Id'].unique()
        
        mix_results = []
        
        for store_id in stores:
            store_elasticities = elasticity_df[elasticity_df['Store_Id'] == store_id].copy()
            
            # Prioritize items based on profit margin and sales volume
            store_elasticities['Profit_Per_Unit'] = store_elasticities['Current_Price'] - store_elasticities['Cost']
            
            # Get average daily sales for this store and items
            store_items = store_elasticities['Item'].unique()
            
            # Calculate importance score (profit per unit * sales volume)
            for _, row in store_elasticities.iterrows():
                item = row['Item']
                profit_per_unit = row['Profit_Per_Unit']
                avg_sales = row['Avg_Daily_Sales'] if 'Avg_Daily_Sales' in row else 1.0
                
                # Calculate importance score
                importance_score = profit_per_unit * avg_sales
                
                # Determine category
                if row['Elasticity'] <= -1.5:
                    category = 'High price sensitivity'
                elif row['Elasticity'] <= -0.5:
                    category = 'Moderate price sensitivity'
                else:
                    category = 'Low price sensitivity'
                    
                # Determine strategic recommendation
                if importance_score > 0 and row['Margin_Pct'] > 30 and row['Elasticity'] > -1.0:
                    strategy = 'Premium pricing (high margin focus)'
                elif importance_score > 0 and row['Elasticity'] <= -1.5:
                    strategy = 'Competitive pricing (volume focus)'
                else:
                    strategy = 'Balanced pricing (standard approach)'
                
                mix_results.append({
                    'Store_Id': store_id,
                    'Item': item,
                    'Product': row['Product'],
                    'Current_Price': row['Current_Price'],
                    'Profit_Per_Unit': profit_per_unit,
                    'Avg_Daily_Sales': avg_sales,
                    'Importance_Score': importance_score,
                    'Elasticity': row['Elasticity'],
                    'Category': category,
                    'Strategy': strategy
                })
        
        # Convert to DataFrame
        mix_df = pd.DataFrame(mix_results)
        
        # Sort by importance score
        mix_df = mix_df.sort_values(['Store_Id', 'Importance_Score'], ascending=[True, False])
        
        logger.info(f"Generated product mix recommendations for {len(stores)} stores")
        return mix_df

    def plot_elasticity_distribution(self, elasticity_df, output_file=None):
        """
        Create a histogram of elasticity values.
        
        Args:
            elasticity_df: DataFrame with elasticity data
            output_file: Path to save the plot (optional)
            
        Returns:
            None
        """
        logger.info("Creating elasticity distribution plot")
        
        plt.figure(figsize=(10, 6))
        plt.hist(elasticity_df['Elasticity'], bins=20, alpha=0.7, color='blue')
        plt.axvline(elasticity_df['Elasticity'].mean(), color='red', linestyle='dashed', linewidth=2)
        plt.title('Distribution of Price Elasticities')
        plt.xlabel('Elasticity Value')
        plt.ylabel('Count')
        plt.grid(alpha=0.3)
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file, bbox_inches='tight')
            logger.info(f"Saved elasticity distribution plot to {output_file}")
            plt.close()
        else:
            plt.show()
    
    def plot_price_sensitivity_curves(self, elasticity_df, output_file=None):
        """
        Create price sensitivity curves for selected products.
        
        Args:
            elasticity_df: DataFrame with elasticity data
            output_file: Path to save the plot (optional)
            
        Returns:
            None
        """
        logger.info("Creating price sensitivity curves")
        
        # Select a few representative products
        if len(elasticity_df) > 5:
            sample = elasticity_df.sample(min(5, len(elasticity_df)))
        else:
            sample = elasticity_df
        
        plt.figure(figsize=(12, 6))
        
        for _, row in sample.iterrows():
            # Skip invalid elasticities
            if row['Elasticity'] >= 0:
                continue
                
            # Get values
            current_price = row['Current_Price']
            elasticity = row['Elasticity']
            product = row['Product']
            
            # Create price range (80% to 120% of current price)
            price_range = np.linspace(current_price * 0.8, current_price * 1.2, 100)
            
            # Calculate quantity at each price point
            quantity = []
            for price in price_range:
                price_ratio = price / current_price
                quantity_ratio = price_ratio ** elasticity
                quantity.append(quantity_ratio)
            
            # Plot the curve
            plt.plot(price_range, quantity, label=f"{product} (e={elasticity:.2f})")
            
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.3)
        plt.axvline(x=current_price, color='black', linestyle='--', alpha=0.3)
        plt.title('Price Sensitivity Curves')
        plt.xlabel('Price ($)')
        plt.ylabel('Relative Quantity (Current = 1.0)')
        plt.grid(alpha=0.3)
        plt.legend()
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file, bbox_inches='tight')
            logger.info(f"Saved price sensitivity curves to {output_file}")
            plt.close()
        else:
            plt.show()
    
    def plot_elasticity_margin(self, elasticity_df, output_file=None):
        """
        Create a scatter plot of elasticity vs margin.
        
        Args:
            elasticity_df: DataFrame with elasticity data
            output_file: Path to save the plot (optional)
            
        Returns:
            None
        """
        logger.info("Creating elasticity vs margin plot")
        
        plt.figure(figsize=(10, 6))
        plt.scatter(elasticity_df['Elasticity'], elasticity_df['Margin_Pct'], alpha=0.7)
        plt.title('Price Elasticity vs Margin')
        plt.xlabel('Elasticity')
        plt.ylabel('Margin (%)')
        plt.grid(alpha=0.3)
        
        # Add trend line
        z = np.polyfit(elasticity_df['Elasticity'], elasticity_df['Margin_Pct'], 1)
        p = np.poly1d(z)
        plt.plot(elasticity_df['Elasticity'], p(elasticity_df['Elasticity']), "r--")
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file, bbox_inches='tight')
            logger.info(f"Saved elasticity vs margin plot to {output_file}")
            plt.close()
        else:
            plt.show()
    
    def plot_profit_impact_waterfall(self, impact_df, output_file=None):
        """
        Create a waterfall chart showing profit impact of price changes.
        
        Args:
            impact_df: DataFrame with profit impact data
            output_file: Path to save the plot (optional)
            
        Returns:
            None
        """
        logger.info("Creating profit impact waterfall chart")
        
        # Check if impact_df is valid and has the required column
        if impact_df is None or impact_df.empty or 'Total_Profit_Difference' not in impact_df.columns:
            logger.warning("Cannot create profit impact waterfall chart: impact_df is empty or missing required columns")
            return
            
        # Select top items by profit impact
        top_items = impact_df.sort_values('Total_Profit_Difference', ascending=False).head(10)
        
        plt.figure(figsize=(12, 8))
        
        # Set up initial values
        bottom = 0
        current_total = 0
        x_labels = []
        colors = []
        
        # Start with baseline profit
        total_current_profit = top_items['Total_Current_Profit'].sum()
        heights = [total_current_profit]
        x_labels.append('Current Profit')
        colors.append('blue')
        
        # Add each product's contribution
        for _, row in top_items.iterrows():
            product = row['Product']
            profit_diff = row['Total_Profit_Difference']
            
            # Only include significant impacts
            if abs(profit_diff) > total_current_profit * 0.01:
                heights.append(profit_diff)
                x_labels.append(f"{product}\n({row['Price_Change_Pct']:.1f}%)")
                colors.append('green' if profit_diff > 0 else 'red')
                current_total += profit_diff
        
        # Add final value
        heights.append(total_current_profit + current_total)
        x_labels.append('New Profit')
        colors.append('blue')
        
        # Create the plot
        plt.bar(range(len(heights)), heights, bottom=bottom, color=colors)
        plt.xticks(range(len(heights)), x_labels, rotation=45, ha='right')
        plt.title('Profit Impact of Price Optimization')
        plt.ylabel('Profit ($)')
        plt.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file, bbox_inches='tight')
            logger.info(f"Saved profit impact waterfall chart to {output_file}")
            plt.close()
        else:
            plt.show()

    def run_optimization(self, create_plots=True):
        """
        Run the complete profit optimization process.
        
        Args:
            create_plots: Whether to create visualization plots
            
        Returns:
            tuple: (elasticity_df, price_recommendations, profit_impact, product_mix)
        """
        logger.info("Running complete profit optimization process")
        
        # Calculate elasticities
        elasticity_df = self.calculate_elasticities()
        
        # Save elasticities
        # Ensure directory exists
        os.makedirs(os.path.dirname(PRICE_ELASTICITIES_FILE), exist_ok=True)
        
        # Save to processed directory
        elasticity_df.to_csv(PRICE_ELASTICITIES_FILE, index=False)
        logger.info(f"Saved elasticities to {PRICE_ELASTICITIES_FILE}")
        
        # Also save to root directory for backward compatibility
        root_file = os.path.join(ROOT_DIR, "price_elasticities.csv")
        elasticity_df.to_csv(root_file, index=False)
        logger.info(f"Saved elasticities to {root_file} for backward compatibility")
        
        # Optimize prices
        price_recommendations = self.optimize_prices(elasticity_df)
        
        # Save price recommendations
        # Ensure directory exists
        os.makedirs(os.path.dirname(PRICE_RECOMMENDATIONS_FILE), exist_ok=True)
        
        # Save to processed directory
        price_recommendations.to_csv(PRICE_RECOMMENDATIONS_FILE, index=False)
        logger.info(f"Saved price recommendations to {PRICE_RECOMMENDATIONS_FILE}")
        
        # Also save to root directory for backward compatibility
        root_file = os.path.join(ROOT_DIR, "price_recommendations.csv")
        price_recommendations.to_csv(root_file, index=False)
        logger.info(f"Saved price recommendations to {root_file} for backward compatibility")
        
        # Calculate profit impact
        profit_impact = None
        if self.forecast_df is not None:
            profit_impact = self.calculate_profit_impact(price_recommendations)
            
            if profit_impact is not None:
                # Ensure directory exists
                os.makedirs(os.path.dirname(PROFIT_IMPACT_FILE), exist_ok=True)
                
                # Save to processed directory
                profit_impact.to_csv(PROFIT_IMPACT_FILE, index=False)
                logger.info(f"Saved profit impact to {PROFIT_IMPACT_FILE}")
                
                # Also save to root directory for backward compatibility
                root_file = os.path.join(ROOT_DIR, "profit_impact.csv")
                profit_impact.to_csv(root_file, index=False)
                logger.info(f"Saved profit impact to {root_file} for backward compatibility")
        
        # Optimize product mix
        product_mix = self.optimize_product_mix(elasticity_df)
        # Ensure directory exists
        os.makedirs(os.path.dirname(PRODUCT_MIX_FILE), exist_ok=True)
        
        # Save to processed directory
        product_mix.to_csv(PRODUCT_MIX_FILE, index=False)
        logger.info(f"Saved product mix to {PRODUCT_MIX_FILE}")
        
        # Also save to root directory for backward compatibility
        root_file = os.path.join(ROOT_DIR, "product_mix_optimization.csv")
        product_mix.to_csv(root_file, index=False)
        logger.info(f"Saved product mix to {root_file} for backward compatibility")
        
        # Create visualization plots
        if create_plots:
            # Create profit directory if needed
            profit_dir = os.path.join(STATIC_DIR, 'images', 'profit')
            os.makedirs(profit_dir, exist_ok=True)
            
            # Create plots
            self.plot_elasticity_distribution(
                elasticity_df, 
                os.path.join(profit_dir, 'elasticity_distribution.png')
            )
            
            self.plot_price_sensitivity_curves(
                elasticity_df, 
                os.path.join(profit_dir, 'price_sensitivity_curves.png')
            )
            
            self.plot_elasticity_margin(
                elasticity_df, 
                os.path.join(profit_dir, 'elasticity_vs_margin.png')
            )
            
            if profit_impact is not None and not profit_impact.empty and 'Total_Profit_Difference' in profit_impact.columns:
                self.plot_profit_impact_waterfall(
                    profit_impact, 
                    os.path.join(profit_dir, 'profit_impact_waterfall.png')
                )
            else:
                logger.warning("Skipping profit impact waterfall chart: profit_impact data not suitable for plotting")
            
            logger.info("Created all visualization plots")
        
        return elasticity_df, price_recommendations, profit_impact, product_mix


def run_profit_optimization(data_file=COMBINED_DATA_FILE, 
                           rf_forecast_file=RF_FORECASTS_FILE,
                           pytorch_forecast_file=PYTORCH_FORECASTS_FILE):
    """
    Run the profit optimization process with the specified data files.
    
    Args:
        data_file: Path to the historical data file
        rf_forecast_file: Path to the RF forecast file
        pytorch_forecast_file: Path to the PyTorch forecast file
        
    Returns:
        ProfitOptimizer: The optimizer instance
    """
    logger.info("Starting profit optimization process")
    
    try:
        # Check if data file exists
        if not os.path.exists(data_file):
            # Try in root directory
            alt_path = os.path.join(ROOT_DIR, "combined_pizza_data.csv")
            if os.path.exists(alt_path):
                logger.info(f"Using alternative data path: {alt_path}")
                data_file = alt_path
            else:
                logger.error(f"Data file not found: {data_file}")
                raise FileNotFoundError(f"Data file not found: {data_file}")
                
        # Initialize optimizer
        optimizer = ProfitOptimizer()
        
        # Load historical data
        optimizer.load_data(df_path=data_file)
        
        # Load forecast data (try PyTorch forecasts first, then RF forecasts)
        forecasts = None
        
        if os.path.exists(pytorch_forecast_file):
            try:
                forecasts = pd.read_csv(pytorch_forecast_file)
                if 'Date' in forecasts.columns:
                    forecasts['Date'] = pd.to_datetime(forecasts['Date'])
                logger.info(f"Loaded PyTorch forecasts from {pytorch_forecast_file}")
            except Exception as e:
                logger.error(f"Error loading PyTorch forecasts: {str(e)}")
                forecasts = None
                
        if forecasts is None and os.path.exists(rf_forecast_file):
            try:
                forecasts = pd.read_csv(rf_forecast_file)
                if 'Date' in forecasts.columns:
                    forecasts['Date'] = pd.to_datetime(forecasts['Date'])
                logger.info(f"Loaded RF forecasts from {rf_forecast_file}")
            except Exception as e:
                logger.error(f"Error loading RF forecasts: {str(e)}")
                forecasts = None
        
        optimizer.forecast_df = forecasts
        
        # Run optimization
        elasticity_df, price_recommendations, profit_impact, product_mix = optimizer.run_optimization()
        
        logger.info("Profit optimization completed successfully")
        return optimizer
        
    except Exception as e:
        logger.error(f"Error in profit optimization process: {str(e)}", exc_info=True)
        raise


def main():
    """
    Main function to run when script is called directly
    """
    import argparse
    parser = argparse.ArgumentParser(description='Profit optimization tool')
    parser.add_argument('--data', type=str, default=COMBINED_DATA_FILE, help='Path to historical data file')
    parser.add_argument('--rf-forecasts', type=str, default=RF_FORECASTS_FILE, help='Path to RF forecast file')
    parser.add_argument('--pytorch-forecasts', type=str, default=PYTORCH_FORECASTS_FILE, help='Path to PyTorch forecast file')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot creation')
    
    args = parser.parse_args()
    
    run_profit_optimization(
        data_file=args.data,
        rf_forecast_file=args.rf_forecasts,
        pytorch_forecast_file=args.pytorch_forecasts
    )


if __name__ == "__main__":
    main()