import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import time

class ProfitOptimizer:
    def __init__(self, df=None, forecast_df=None):
        """
        Initialize the profit optimizer with data.
        
        Parameters:
        - df: DataFrame with historical sales data
        - forecast_df: DataFrame with sales forecasts
        """
        self.df = df
        self.forecast_df = forecast_df
        self.elasticity_models = {}
        
    def load_data(self, df_path='combined_pizza_data.csv', forecast_path=None):
        """
        Load data from CSV files.
        
        Parameters:
        - df_path: Path to the historical data CSV
        - forecast_path: Path to the forecast data CSV (optional)
        """
        self.df = pd.read_csv(df_path)
        
        # Convert date column to datetime
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Load forecasts if provided
        if forecast_path is not None:
            self.forecast_df = pd.read_csv(forecast_path)
            if 'Date' in self.forecast_df.columns:
                self.forecast_df['Date'] = pd.to_datetime(self.forecast_df['Date'])
        
        return self
    
    def calculate_elasticities(self):
        """
        Calculate price elasticity for each product based on historical data.
        
        Returns:
        - elasticity_df: DataFrame with price elasticity data
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
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
                        r_squared = model.score(X, y)
                        
                        # Determine if elasticity is significant
                        is_significant = r_squared > 0.2 and len(valid_data) >= 10
                        
                        # If elasticity is positive (unusual), set to default
                        if elasticity > 0:
                            elasticity = -1.0
                            status = "Positive elasticity corrected"
                        else:
                            status = "Valid elasticity"
                        
                        # Store the model
                        self.elasticity_models[(store_id, item)] = {
                            'model': model,
                            'elasticity': elasticity,
                            'r_squared': r_squared,
                            'intercept': intercept
                        }
                except Exception as e:
                    elasticity = -1.0  # Default elasticity
                    r_squared = 0.0
                    coef = 0.0
                    intercept = 0.0
                    is_significant = False
                    status = f"Error: {str(e)}"
            
            # Calculate price range
            min_price = group['Price'].min()
            max_price = group['Price'].max()
            avg_price = group['Price'].mean()
            
            # Calculate unit cost if possible
            if 'Cost' in group.columns and 'Sales' in group.columns:
                total_cost = group['Cost'].sum()
                total_sales = group['Sales'].sum()
                unit_cost = total_cost / total_sales if total_sales > 0 else avg_price * 0.7
            else:
                unit_cost = avg_price * 0.7  # Default 70% cost
            
            # Calculate average profit margin
            profit_margin = (avg_price - unit_cost) / avg_price if avg_price > 0 else 0.3
            
            # Get product details
            product = group['Product'].iloc[0] if 'Product' in group.columns else f"Item {item}"
            
            elasticity_results.append({
                'Store_Id': store_id,
                'Item': item,
                'Product': product,
                'Elasticity': elasticity,
                'R_Squared': r_squared,
                'Significant': is_significant,
                'Status': status,
                'Min_Price': min_price,
                'Max_Price': max_price,
                'Avg_Price': avg_price,
                'Unit_Cost': unit_cost,
                'Profit_Margin': profit_margin,
                'Transactions': len(group)
            })
        
        elasticity_df = pd.DataFrame(elasticity_results)
        
        return elasticity_df
    
    def optimize_price(self, elasticity_df, target_margin=None):
        """
        Optimize prices for profit maximization based on elasticities.
        
        Parameters:
        - elasticity_df: DataFrame with elasticity data
        - target_margin: Target profit margin constraint (optional)
        
        Returns:
        - price_recommendations: DataFrame with optimized prices
        """
        price_recommendations = []
        
        for _, row in elasticity_df.iterrows():
            store_id = row['Store_Id']
            item = row['Item']
            elasticity = row['Elasticity']
            unit_cost = row['Unit_Cost']
            avg_price = row['Avg_Price']
            min_price = row['Min_Price']
            max_price = row['Max_Price']
            
            # Get elasticity model if available
            model_info = self.elasticity_models.get((store_id, item))
            
            if model_info is not None and abs(elasticity) > 0.1:
                # Use elasticity model for optimization
                model = model_info['model']
                intercept = model_info['intercept']
                
                # Profit function to maximize
                def negative_profit(price):
                    log_price = np.log(price[0])
                    log_quantity = intercept + elasticity * log_price
                    quantity = np.exp(log_quantity)
                    profit = (price[0] - unit_cost) * quantity
                    return -profit
                
                # Constraint: price must be within 20% of current price
                price_lower = max(unit_cost * 1.05, avg_price * 0.8)
                price_upper = avg_price * 1.2
                
                # If target margin is specified, add constraint
                if target_margin is not None:
                    price_lower = max(price_lower, unit_cost / (1 - target_margin))
                
                # Optimize price
                result = minimize(
                    negative_profit, 
                    [avg_price],
                    bounds=[(price_lower, price_upper)]
                )
                
                optimal_price = result.x[0]
                
                # Calculate projected sales at optimal price
                log_optimal_price = np.log(optimal_price)
                log_optimal_quantity = intercept + elasticity * log_optimal_price
                optimal_quantity = np.exp(log_optimal_quantity)
                
                # Calculate projected profit
                optimal_profit = (optimal_price - unit_cost) * optimal_quantity
                
                # Calculate baseline profit
                log_baseline_price = np.log(avg_price)
                log_baseline_quantity = intercept + elasticity * log_baseline_price
                baseline_quantity = np.exp(log_baseline_quantity)
                baseline_profit = (avg_price - unit_cost) * baseline_quantity
                
                # Calculate profit improvement
                profit_improvement = optimal_profit - baseline_profit
                profit_improvement_pct = (profit_improvement / baseline_profit * 100) if baseline_profit > 0 else 0
                
                # Calculate price difference
                price_diff_pct = (optimal_price - avg_price) / avg_price * 100
                
                # Calculate new margin
                optimal_margin = (optimal_price - unit_cost) / optimal_price
                
                recommendation_reason = (
                    f"Based on price elasticity of {elasticity:.2f}, "
                    f"{'increasing' if price_diff_pct > 0 else 'decreasing'} price by {abs(price_diff_pct):.1f}% "
                    f"would maximize profit. Projected profit increase: {profit_improvement_pct:.1f}%"
                )
            else:
                # Use standard markup if no valid elasticity
                if abs(elasticity) <= 0.1:
                    # Inelastic - price can be higher
                    optimal_margin = 0.35  # 35% margin
                else:
                    # Use default margin
                    optimal_margin = 0.3  # 30% margin
                
                optimal_price = unit_cost / (1 - optimal_margin)
                
                # Ensure price is within 20% bounds of current price
                optimal_price = max(avg_price * 0.8, min(optimal_price, avg_price * 1.2))
                
                # Calculate price difference
                price_diff_pct = (optimal_price - avg_price) / avg_price * 100
                
                # Use default values for other metrics
                optimal_quantity = 0
                optimal_profit = 0
                baseline_profit = 0
                profit_improvement = 0
                profit_improvement_pct = 0
                
                recommendation_reason = (
                    f"Insufficient elasticity data. Using standard markup pricing "
                    f"with {optimal_margin:.0%} margin."
                )
            
            # Add to recommendations
            price_recommendations.append({
                'Store_Id': store_id,
                'Item': item,
                'Product': row['Product'],
                'Current_Price': avg_price,
                'Optimal_Price': optimal_price,
                'Price_Change_Pct': price_diff_pct,
                'Unit_Cost': unit_cost,
                'Current_Margin': row['Profit_Margin'],
                'Optimal_Margin': optimal_margin,
                'Elasticity': elasticity,
                'R_Squared': row['R_Squared'],
                'Profit_Improvement_Pct': profit_improvement_pct,
                'Recommendation': recommendation_reason
            })
        
        return pd.DataFrame(price_recommendations)
    
    def optimize_product_mix(self, price_recommendations, inventory_recs, constraints=None):
        """
        Optimize product mix considering price elasticities, inventory constraints,
        and cross-product effects.
        
        Parameters:
        - price_recommendations: DataFrame with price optimization results
        - inventory_recs: DataFrame with inventory recommendations
        - constraints: Dictionary of constraints
        
        Returns:
        - product_mix: DataFrame with optimized product mix
        """
        if constraints is None:
            constraints = {
                'max_price_increase': 20,  # Max price increase percentage
                'max_price_decrease': 20,  # Max price decrease percentage (updated to 20%)
                'min_margin': 25,  # Minimum margin percentage
                'max_stock_weeks': 1,  # Maximum weeks of stock to maintain (reduced from 3 to 1)
            }
        
        # Combine price and inventory recommendations
        combined_recs = pd.merge(
            price_recommendations,
            inventory_recs[['Store_Id', 'Item', 'Current_Stock', 'Avg_Weekly_Demand', 'Weeks_Of_Supply']],
            on=['Store_Id', 'Item'],
            how='left'
        )
        
        # Fill missing inventory data
        combined_recs['Current_Stock'] = combined_recs['Current_Stock'].fillna(0)
        combined_recs['Avg_Weekly_Demand'] = combined_recs['Avg_Weekly_Demand'].fillna(0)
        combined_recs['Weeks_Of_Supply'] = combined_recs['Weeks_Of_Supply'].fillna(0)
        
        # Apply constraints
        adjusted_recs = []
        
        for _, row in combined_recs.iterrows():
            # Check price increase constraint
            if row['Price_Change_Pct'] > constraints['max_price_increase']:
                adjusted_price = row['Current_Price'] * (1 + constraints['max_price_increase'] / 100)
                price_change_pct = constraints['max_price_increase']
                price_reason = f"Limited to {constraints['max_price_increase']}% increase"
            # Check price decrease constraint
            elif row['Price_Change_Pct'] < -constraints['max_price_decrease']:
                adjusted_price = row['Current_Price'] * (1 - constraints['max_price_decrease'] / 100)
                price_change_pct = -constraints['max_price_decrease']
                price_reason = f"Limited to {constraints['max_price_decrease']}% decrease"
            else:
                adjusted_price = row['Optimal_Price']
                price_change_pct = row['Price_Change_Pct']
                price_reason = "No adjustment needed"
            
            # Check margin constraint
            adjusted_margin = (adjusted_price - row['Unit_Cost']) / adjusted_price
            if adjusted_margin < constraints['min_margin'] / 100:
                adjusted_price = row['Unit_Cost'] / (1 - constraints['min_margin'] / 100)
                adjusted_margin = constraints['min_margin'] / 100
                price_change_pct = (adjusted_price - row['Current_Price']) / row['Current_Price'] * 100
                price_reason = f"Adjusted to maintain {constraints['min_margin']}% minimum margin"
            
            # Inventory strategy based on stock level
            if row['Weeks_Of_Supply'] < 1:
                inventory_strategy = "Increase stock (below safety level)"
                price_strategy = "Consider higher price to manage demand"
            elif row['Weeks_Of_Supply'] > constraints['max_stock_weeks']:
                inventory_strategy = "Reduce stock (excess inventory)"
                price_strategy = "Consider promotion to reduce stock"
            else:
                inventory_strategy = "Maintain stock (adequate level)"
                price_strategy = "Optimize for profit"
            
            # Calculate estimated impact
            try:
                elasticity = row['Elasticity']
                price_ratio = adjusted_price / row['Current_Price']
                quantity_ratio = price_ratio ** elasticity
                demand_impact_pct = (quantity_ratio - 1) * 100
            except:
                demand_impact_pct = 0
            
            adjusted_recs.append({
                'Store_Id': row['Store_Id'],
                'Item': row['Item'],
                'Product': row['Product'],
                'Current_Price': row['Current_Price'],
                'Adjusted_Price': adjusted_price,
                'Price_Change_Pct': price_change_pct,
                'Adjusted_Margin': adjusted_margin,
                'Unit_Cost': row['Unit_Cost'],  # Ensure Unit_Cost is included
                'Current_Stock': row['Current_Stock'],
                'Avg_Weekly_Demand': row['Avg_Weekly_Demand'],
                'Weeks_Of_Supply': row['Weeks_Of_Supply'],
                'Demand_Impact_Pct': demand_impact_pct,
                'Price_Adjustment_Reason': price_reason,
                'Inventory_Strategy': inventory_strategy,
                'Price_Strategy': price_strategy
            })
        
        product_mix = pd.DataFrame(adjusted_recs)
        
        return product_mix
    
    def calculate_profit_impact(self, product_mix, forecast_days=30):
        """
        Calculate projected profit impact of optimized product mix.
        
        Parameters:
        - product_mix: DataFrame with optimized product mix
        - forecast_days: Number of days in the forecast period
        
        Returns:
        - profit_impact: DataFrame with profit impact analysis
        """
        impact_results = []
        
        # Use purchase and sales data to calculate demand for profit impact
        # First try to load the purchase and sales data
        try:
            sales_data = pd.read_csv('FrozenPizzaSales.csv')
            purchase_data = pd.read_csv('FrozenPizzaPurchases.csv')
            
            # Convert column names to match our expected format
            sales_data = sales_data.rename(columns={
                'store_id': 'Store_Id',
                'item': 'Item',
                'Total_units': 'Sales',
                'Total_Retail_$': 'Revenue',
                'Total_Cost_$': 'Cost',
                'Proc_date': 'Date'
            })
            
            purchase_data = purchase_data.rename(columns={
                'store_id': 'Store_Id',
                'item': 'Item',
                'Total_units': 'Units_Purchased',
                'Total_Retail_$': 'Purchase_Retail_Value',
                'Total_Cost_$': 'Purchase_Cost',
                'Proc_date': 'Date'
            })
            
            # Convert to proper data types
            sales_data['Store_Id'] = sales_data['Store_Id'].astype(float)
            sales_data['Item'] = sales_data['Item'].astype(float)
            purchase_data['Store_Id'] = purchase_data['Store_Id'].astype(float)
            purchase_data['Item'] = purchase_data['Item'].astype(float)
            
            # Group by Store_Id and Item to get total sales and purchases
            sales_summary = sales_data.groupby(['Store_Id', 'Item']).agg({
                'Sales': 'sum', 
                'Revenue': 'sum', 
                'Cost': 'sum'
            }).reset_index()
            
            purchase_summary = purchase_data.groupby(['Store_Id', 'Item']).agg({
                'Units_Purchased': 'sum', 
                'Purchase_Cost': 'sum'
            }).reset_index()
            
            # Merge with our product mix
            demand_data = pd.merge(sales_summary, product_mix[['Store_Id', 'Item', 'Product']], 
                                   on=['Store_Id', 'Item'], how='right')
            
            # If we have purchase data, merge that too
            if not purchase_summary.empty:
                demand_data = pd.merge(demand_data, purchase_summary, 
                                      on=['Store_Id', 'Item'], how='left')
            
            # Fill missing values with 0
            for col in ['Sales', 'Revenue', 'Cost', 'Units_Purchased', 'Purchase_Cost']:
                if col in demand_data.columns:
                    demand_data[col] = demand_data[col].fillna(0)
            
            # Use sales data as demand if available, otherwise fall back to forecast
            forecast_demand = demand_data[['Store_Id', 'Item']].copy()
            forecast_demand['Forecast_Period_Demand'] = demand_data['Sales'] * (forecast_days / 30.0)  # Scale to forecast period
            
            print(f"Using actual sales data for profit calculations")
            
        except Exception as e:
            print(f"Error using sales data: {e}. Falling back to forecast data.")
            
            # Fall back to forecast data if available, otherwise estimate from Avg_Weekly_Demand
            if self.forecast_df is not None:
                # Group forecast by store and item
                forecast_demand = self.forecast_df.groupby(['Store_Id', 'Item'])['Predicted_Demand'].sum().reset_index()
                forecast_demand.rename(columns={'Predicted_Demand': 'Forecast_Period_Demand'}, inplace=True)
            else:
                # Create forecast estimate from avg weekly demand
                forecast_demand = product_mix[['Store_Id', 'Item', 'Avg_Weekly_Demand']].copy()
                forecast_demand['Forecast_Period_Demand'] = forecast_demand['Avg_Weekly_Demand'] / 7 * forecast_days
        
        # Merge forecast with product mix
        profit_data = pd.merge(product_mix, forecast_demand, on=['Store_Id', 'Item'], how='left')
        
        # Fill missing forecast data
        profit_data['Forecast_Period_Demand'] = profit_data['Forecast_Period_Demand'].fillna(0)
        
        for _, row in profit_data.iterrows():
            # Calculate baseline
            baseline_revenue = row['Current_Price'] * row['Forecast_Period_Demand']
            baseline_cost = row['Unit_Cost'] * row['Forecast_Period_Demand']
            baseline_profit = baseline_revenue - baseline_cost
            
            # Calculate projected with adjusted price
            # Apply elasticity to demand
            elasticity = row.get('Elasticity', -1.0)  # Default to -1.0 if not available
            try:
                price_ratio = row['Adjusted_Price'] / row['Current_Price']
                quantity_ratio = price_ratio ** elasticity
                projected_demand = row['Forecast_Period_Demand'] * quantity_ratio
            except:
                projected_demand = row['Forecast_Period_Demand']
            
            projected_revenue = row['Adjusted_Price'] * projected_demand
            projected_cost = row['Unit_Cost'] * projected_demand
            projected_profit = projected_revenue - projected_cost
            
            # Calculate impact
            revenue_impact = projected_revenue - baseline_revenue
            profit_impact = projected_profit - baseline_profit
            profit_impact_pct = (profit_impact / baseline_profit * 100) if baseline_profit > 0 else 0
            
            impact_results.append({
                'Store_Id': row['Store_Id'],
                'Item': row['Item'],
                'Product': row['Product'],
                'Current_Price': row['Current_Price'],
                'Adjusted_Price': row['Adjusted_Price'],
                'Price_Change_Pct': row['Price_Change_Pct'],  # Add this from product_mix
                'Unit_Cost': row['Unit_Cost'],
                'Forecast_Period_Demand': row['Forecast_Period_Demand'],
                'Projected_Demand': projected_demand,
                'Baseline_Revenue': baseline_revenue,
                'Projected_Revenue': projected_revenue,
                'Revenue_Impact': revenue_impact,
                'Baseline_Profit': baseline_profit,
                'Projected_Profit': projected_profit,
                'Profit_Impact': profit_impact,
                'Profit_Impact_Pct': profit_impact_pct
            })
        
        impact_df = pd.DataFrame(impact_results)
        
        return impact_df
    
    def generate_elasticity_charts(self, elasticity_df, output_dir='static/images/profit'):
        """
        Generate elasticity visualization charts.
        
        Parameters:
        - elasticity_df: DataFrame with elasticity data
        - output_dir: Directory to save charts
        
        Returns:
        - chart_paths: List of chart file paths
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        chart_paths = []
        
        # 1. Elasticity distribution
        plt.figure(figsize=(12, 6))
        # Filter out extreme values
        filtered_elasticity = elasticity_df[
            (elasticity_df['Elasticity'] > -10) & 
            (elasticity_df['Elasticity'] < 0)
        ]
        
        plt.hist(filtered_elasticity['Elasticity'], bins=20, alpha=0.7)
        plt.axvline(x=-1, color='red', linestyle='--', 
                   label='Unitary Elasticity')
        plt.xlabel('Price Elasticity')
        plt.ylabel('Number of Products')
        plt.title('Distribution of Price Elasticity Across Products')
        plt.legend()
        plt.grid(alpha=0.3)
        
        elasticity_dist_chart = f"{output_dir}/elasticity_distribution.png"
        plt.savefig(elasticity_dist_chart)
        plt.close()
        chart_paths.append(elasticity_dist_chart)
        
        # 2. Elasticity vs. Profit Margin
        plt.figure(figsize=(12, 6))
        plt.scatter(
            filtered_elasticity['Elasticity'], 
            filtered_elasticity['Profit_Margin'] * 100,
            alpha=0.6
        )
        plt.xlabel('Price Elasticity')
        plt.ylabel('Profit Margin (%)')
        plt.title('Price Elasticity vs. Current Profit Margin')
        plt.grid(alpha=0.3)
        
        elasticity_margin_chart = f"{output_dir}/elasticity_vs_margin.png"
        plt.savefig(elasticity_margin_chart)
        plt.close()
        chart_paths.append(elasticity_margin_chart)
        
        # 3. Create price sensitivity curves for top products
        top_products = elasticity_df.sort_values('Transactions', ascending=False).head(9)
        
        plt.figure(figsize=(16, 10))
        
        for i, (_, row) in enumerate(top_products.iterrows()):
            if i < 9 and row['R_Squared'] > 0.1:  # Only include if R-squared is reasonable
                plt.subplot(3, 3, i+1)
                
                store_id = row['Store_Id']
                item = row['Item']
                elasticity = row['Elasticity']
                avg_price = row['Avg_Price']
                unit_cost = row['Unit_Cost']
                
                # Get model parameters
                model_info = self.elasticity_models.get((store_id, item))
                
                if model_info is not None:
                    intercept = model_info['intercept']
                    
                    # Create price range for curve
                    price_range = np.linspace(
                        avg_price * 0.7, 
                        avg_price * 1.3, 
                        100
                    )
                    
                    # Calculate quantity for each price
                    log_price = np.log(price_range)
                    log_quantity = intercept + elasticity * log_price
                    quantity = np.exp(log_quantity)
                    
                    # Calculate revenue and profit
                    revenue = price_range * quantity
                    profit = (price_range - unit_cost) * quantity
                    
                    # Find optimal price for profit
                    optimal_idx = np.argmax(profit)
                    optimal_price = price_range[optimal_idx]
                    
                    # Plot price sensitivity curve
                    plt.plot(price_range, quantity, 'b-', label='Demand')
                    
                    # Add optimal price point
                    plt.axvline(x=optimal_price, color='g', linestyle='--',
                               label=f'Optimal: ${optimal_price:.2f}')
                    
                    # Add current price point
                    plt.axvline(x=avg_price, color='r', linestyle=':',
                               label=f'Current: ${avg_price:.2f}')
                    
                    # Add cost line
                    plt.axvline(x=unit_cost, color='k', linestyle='-.',
                               label=f'Cost: ${unit_cost:.2f}')
                    
                    plt.title(f"{row['Product'][:20]}...")
                    plt.xlabel('Price ($)')
                    plt.ylabel('Demand (units)')
                    plt.legend(fontsize=8)
                    plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Price Sensitivity Curves by Product', y=1.05)
        
        price_sensitivity_chart = f"{output_dir}/price_sensitivity_curves.png"
        plt.savefig(price_sensitivity_chart)
        plt.close()
        chart_paths.append(price_sensitivity_chart)
        
        return chart_paths
    
    def generate_profit_impact_charts(self, profit_impact, output_dir='static/images/profit'):
        """
        Generate profit impact visualization charts.
        
        Parameters:
        - profit_impact: DataFrame with profit impact data
        - output_dir: Directory to save charts
        
        Returns:
        - chart_paths: List of chart file paths
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        chart_paths = []
        
        # 1. Waterfall chart of top profit impacts
        top_impact = profit_impact.sort_values('Profit_Impact', ascending=False).head(10)
        
        plt.figure(figsize=(14, 8))
        
        # Calculate total impact
        total_impact = top_impact['Profit_Impact'].sum()
        
        # Create waterfall chart data
        products = []
        values = []
        
        # Add each product's impact
        for _, row in top_impact.iterrows():
            products.append(f"{row['Product'][:15]}... (${row['Profit_Impact']:.0f})")
            values.append(row['Profit_Impact'])
        
        # Add total
        products.append('Total Impact')
        values.append(0)  # Placeholder, will be calculated as sum
        
        # Plot bars
        plt.bar(products[:-1], values[:-1], color='green' if total_impact > 0 else 'red')
        plt.bar(products[-1:], [total_impact], color='blue')
        
        # Add value labels
        for i, v in enumerate(values[:-1]):
            plt.text(i, v + (total_impact * 0.02), f"${v:.0f}", 
                    ha='center', va='bottom' if v > 0 else 'top')
        
        plt.text(len(values) - 1, total_impact + (total_impact * 0.02), f"${total_impact:.0f}", 
                ha='center', va='bottom' if total_impact > 0 else 'top', fontweight='bold')
        
        plt.ylabel('Profit Impact ($)')
        plt.title('Profit Impact by Product (30-Day Forecast Period)')
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        waterfall_chart = f"{output_dir}/profit_impact_waterfall.png"
        plt.savefig(waterfall_chart)
        plt.close()
        chart_paths.append(waterfall_chart)
        
        # 2. Price change vs. profit impact scatter
        plt.figure(figsize=(12, 6))
        
        # Use safer column access - check if column exists first
        if 'Price_Change_Pct' in profit_impact.columns:
            x_values = profit_impact['Price_Change_Pct']
        else:
            # Calculate from the price difference if not available
            x_values = ((profit_impact['Adjusted_Price'] - profit_impact['Current_Price']) / 
                        profit_impact['Current_Price'] * 100)
            
        plt.scatter(
            x_values,
            profit_impact['Profit_Impact_Pct'],
            alpha=0.6
        )
        
        plt.xlabel('Price Change (%)')
        plt.ylabel('Profit Impact (%)')
        plt.title('Price Change vs. Profit Impact')
        plt.grid(alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--')
        plt.axvline(x=0, color='k', linestyle='--')
        
        price_profit_chart = f"{output_dir}/price_vs_profit_impact.png"
        plt.savefig(price_profit_chart)
        plt.close()
        chart_paths.append(price_profit_chart)
        
        # 3. Price optimization recommendations
        top_recommendations = profit_impact.sort_values('Profit_Impact', ascending=False).head(20)
        
        plt.figure(figsize=(14, 8))
        x = range(len(top_recommendations))
        
        # Plot current and adjusted prices
        width = 0.35
        plt.bar([i - width/2 for i in x], top_recommendations['Current_Price'], width, 
               label='Current Price', color='blue', alpha=0.7)
        plt.bar([i + width/2 for i in x], top_recommendations['Adjusted_Price'], width,
               label='Optimized Price', color='green', alpha=0.7)
        
        # Add product labels
        plt.xticks(x, [f"{row['Product'][:15]}..." for _, row in top_recommendations.iterrows()], 
                  rotation=45, ha='right')
        
        plt.ylabel('Price ($)')
        plt.title('Current vs. Optimized Prices for Top Products')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        price_rec_chart = f"{output_dir}/price_recommendations.png"
        plt.savefig(price_rec_chart)
        plt.close()
        chart_paths.append(price_rec_chart)
        
        return chart_paths

def main():
    """Main function to demonstrate profit optimization"""
    start_time = time.time()
    print("Starting profit optimization process...")
    
    # Initialize optimizer
    optimizer = ProfitOptimizer()
    
    # Load data
    optimizer.load_data(df_path='combined_pizza_data.csv')
    
    # Try to load forecasts if available
    try:
        optimizer.forecast_df = pd.read_csv('rf_forecasts.csv')
        optimizer.forecast_df['Date'] = pd.to_datetime(optimizer.forecast_df['Date'])
        print("Using Random Forest forecasts for profit optimization")
    except:
        try:
            optimizer.forecast_df = pd.read_csv('pytorch_forecasts.csv')
            optimizer.forecast_df['Date'] = pd.to_datetime(optimizer.forecast_df['Date'])
            print("Using PyTorch forecasts for profit optimization")
        except:
            print("No forecast data found, will estimate from historical data")
    
    print("Calculating price elasticities...")
    elasticity_df = optimizer.calculate_elasticities()
    elasticity_df.to_csv('price_elasticities.csv', index=False)
    
    print("Optimizing prices for profit maximization...")
    price_recommendations = optimizer.optimize_price(elasticity_df)
    price_recommendations.to_csv('price_recommendations.csv', index=False)
    
    # Load inventory recommendations if available
    try:
        inventory_recs = pd.read_csv('inventory_recommendations.csv')
        print("Using inventory recommendations for product mix optimization")
    except:
        print("No inventory recommendations found, creating placeholder")
        # Create placeholder inventory data
        inventory_recs = pd.DataFrame({
            'Store_Id': elasticity_df['Store_Id'],
            'Item': elasticity_df['Item'],
            'Current_Stock': 0,
            'Avg_Weekly_Demand': 0,
            'Weeks_Of_Supply': 0
        })
    
    print("Optimizing product mix...")
    product_mix = optimizer.optimize_product_mix(price_recommendations, inventory_recs)
    product_mix.to_csv('product_mix_optimization.csv', index=False)
    
    print("Calculating profit impact...")
    profit_impact = optimizer.calculate_profit_impact(product_mix)
    profit_impact.to_csv('profit_impact.csv', index=False)
    
    # Calculate total projected profit impact
    total_impact = profit_impact['Profit_Impact'].sum()
    baseline_profit = profit_impact['Baseline_Profit'].sum()
    impact_pct = (total_impact / baseline_profit * 100) if baseline_profit > 0 else 0
    
    print(f"Total projected profit improvement: ${total_impact:.2f} ({impact_pct:.2f}%)")
    
    print("Generating visualization charts...")
    elasticity_charts = optimizer.generate_elasticity_charts(elasticity_df)
    profit_charts = optimizer.generate_profit_impact_charts(profit_impact)
    
    print(f"Profit optimization process completed in {time.time() - start_time:.2f} seconds!")

if __name__ == '__main__':
    main()