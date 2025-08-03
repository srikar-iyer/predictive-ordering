"""
Integrated forecasting module that connects ARIMA forecasting, inventory optimization, 
and pricing optimization for a complete predictive ordering system.
"""
import pandas as pd
import numpy as np
import logging
import os
import sys
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.arima_model import ARIMAForecaster, run_arima_forecasting
from src.optimization.profit_optimizer import ProfitOptimizer, run_profit_optimization

# Import settings
try:
    from config.settings import (
        COMBINED_DATA_FILE, ARIMA_FORECASTS_FILE, MODELS_DIR, STATIC_DIR,
        PRICE_ELASTICITIES_FILE, PRICE_RECOMMENDATIONS_FILE, PROFIT_IMPACT_FILE,
        INVENTORY_RECOMMENDATIONS_FILE, MIN_STOCK_WEEKS, TARGET_STOCK_WEEKS
    )
except ImportError:
    # Default paths for backward compatibility
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    COMBINED_DATA_FILE = os.path.join(ROOT_DIR, "combined_pizza_data.csv")
    ARIMA_FORECASTS_FILE = os.path.join(ROOT_DIR, "arima_forecasts.csv")
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    STATIC_DIR = os.path.join(ROOT_DIR, "static")
    PRICE_ELASTICITIES_FILE = os.path.join(ROOT_DIR, "price_elasticities.csv")
    PRICE_RECOMMENDATIONS_FILE = os.path.join(ROOT_DIR, "price_recommendations.csv")
    PROFIT_IMPACT_FILE = os.path.join(ROOT_DIR, "profit_impact.csv")
    INVENTORY_RECOMMENDATIONS_FILE = os.path.join(ROOT_DIR, "inventory_recommendations.csv")
    MIN_STOCK_WEEKS = 1
    TARGET_STOCK_WEEKS = 2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('integrated_forecasting')

class IntegratedForecaster:
    """
    Class that integrates ARIMA forecasting with pricing and inventory optimization
    """
    def __init__(self, data_path=COMBINED_DATA_FILE):
        """
        Initialize the integrated forecaster
        
        Args:
            data_path: Path to the data file
        """
        self.data_path = data_path
        self.df = None
        self.forecasts = None
        self.elasticities = None
        self.price_recommendations = None
        self.inventory_recommendations = None
        self.profit_impact = None
    
    def load_data(self):
        """
        Load data from CSV
        
        Returns:
            DataFrame: The loaded data
        """
        logger.info(f"Loading data from {self.data_path}")
        try:
            self.df = pd.read_csv(self.data_path)
            
            # Convert date to datetime
            if 'Date' in self.df.columns:
                self.df['Date'] = pd.to_datetime(self.df['Date'])
            
            logger.info(f"Loaded {len(self.df)} records from {self.data_path}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def generate_forecasts(self, days_to_forecast=30, use_existing=True):
        """
        Generate ARIMA forecasts
        
        Args:
            days_to_forecast: Number of days to forecast
            use_existing: Whether to use existing models if available
            
        Returns:
            DataFrame: The generated forecasts
        """
        logger.info(f"Generating ARIMA forecasts for {days_to_forecast} days")
        
        try:
            # Ensure data is loaded
            if self.df is None:
                self.load_data()
            
            # Run ARIMA forecasting
            self.forecasts = run_arima_forecasting(
                data_file=self.data_path,
                days_to_forecast=days_to_forecast,
                use_existing=use_existing
            )
            
            logger.info(f"Generated forecasts for {days_to_forecast} days")
            return self.forecasts
        except Exception as e:
            logger.error(f"Error generating forecasts: {str(e)}")
            raise
    
    def optimize_prices(self):
        """
        Optimize prices based on forecasts
        
        Returns:
            tuple: (price elasticities, price recommendations, profit impact)
        """
        logger.info("Optimizing prices")
        
        try:
            # Ensure data and forecasts are available
            if self.df is None:
                self.load_data()
            
            if self.forecasts is None:
                raise ValueError("No forecasts available. Run generate_forecasts() first.")
            
            # Initialize profit optimizer
            optimizer = ProfitOptimizer(df=self.df, forecast_df=self.forecasts)
            
            # Calculate elasticities
            self.elasticities = optimizer.calculate_elasticities()
            
            # Optimize prices
            self.price_recommendations = optimizer.optimize_prices(self.elasticities)
            
            # Calculate profit impact
            self.profit_impact = optimizer.calculate_profit_impact(self.price_recommendations)
            
            # Save results
            if self.elasticities is not None:
                self.elasticities.to_csv(PRICE_ELASTICITIES_FILE, index=False)
                logger.info(f"Saved elasticities to {PRICE_ELASTICITIES_FILE}")
            
            if self.price_recommendations is not None:
                self.price_recommendations.to_csv(PRICE_RECOMMENDATIONS_FILE, index=False)
                logger.info(f"Saved price recommendations to {PRICE_RECOMMENDATIONS_FILE}")
            
            if self.profit_impact is not None:
                self.profit_impact.to_csv(PROFIT_IMPACT_FILE, index=False)
                logger.info(f"Saved profit impact to {PROFIT_IMPACT_FILE}")
            
            logger.info("Price optimization complete")
            return self.elasticities, self.price_recommendations, self.profit_impact
        except Exception as e:
            logger.error(f"Error optimizing prices: {str(e)}")
            raise
    
    def optimize_inventory(self):
        """
        Optimize inventory based on forecasts
        
        Returns:
            DataFrame: The inventory recommendations
        """
        logger.info("Optimizing inventory")
        
        try:
            # Ensure data and forecasts are available
            if self.df is None:
                self.load_data()
            
            if self.forecasts is None:
                raise ValueError("No forecasts available. Run generate_forecasts() first.")
            
            # Create inventory recommendations DataFrame
            inventory_recommendations = []
            
            # Process each store-item combination
            for (store_id, item), group in self.forecasts.groupby(['Store_Id', 'Item']):
                # Get current stock level from most recent historical data
                current_stock = 0
                item_data = self.df[(self.df['Store_Id'] == store_id) & (self.df['Item'] == item)]
                if len(item_data) > 0 and 'Stock_Level' in item_data.columns:
                    current_stock = item_data.sort_values('Date', ascending=False)['Stock_Level'].iloc[0]
                
                # Get forecast column name
                forecast_col = 'Forecast'
                if 'Predicted_Demand' in group.columns:
                    forecast_col = 'Predicted_Demand'
                elif 'Forecast' in group.columns:
                    forecast_col = 'Forecast'
                elif 'Predicted_Sales' in group.columns:
                    forecast_col = 'Predicted_Sales'
                    
                # Log the columns available in group
                logger.info(f"Group columns for store {store_id}, item {item}: {group.columns.tolist()}")
                
                # Check if forecast column exists in the dataframe
                if forecast_col not in group.columns:
                    # Try to find any column that might contain forecast values
                    numeric_cols = group.select_dtypes(include=[np.number]).columns.tolist()
                    potential_forecast_cols = [col for col in numeric_cols 
                                              if col not in ['Store_Id', 'Item', 'Days_In_Future'] 
                                              and not col.startswith('Lower_') 
                                              and not col.startswith('Upper_')]
                    
                    if potential_forecast_cols:
                        forecast_col = potential_forecast_cols[0]
                        logger.info(f"Using '{forecast_col}' as fallback forecast column for store {store_id}, item {item}")
                    else:
                        logger.warning(f"Forecast column '{forecast_col}' not found for store {store_id}, item {item}. Available columns: {group.columns}")
                        continue
                
                # Get forecast for next 7, 14, 30 days
                forecast_7d = group[group['Days_In_Future'] <= 7][forecast_col].sum()
                forecast_14d = group[group['Days_In_Future'] <= 14][forecast_col].sum()
                forecast_30d = group[group['Days_In_Future'] <= 30][forecast_col].sum()
                
                # Calculate daily average sales
                avg_daily_sales_7d = forecast_7d / 7
                avg_daily_sales_14d = forecast_14d / 14
                avg_daily_sales_30d = forecast_30d / 30
                
                # Calculate weeks of stock
                weeks_of_stock = current_stock / (avg_daily_sales_7d * 7) if avg_daily_sales_7d > 0 else 4
                
                # Calculate order point
                order_point = avg_daily_sales_7d * 7 * MIN_STOCK_WEEKS
                
                # Calculate order quantity
                target_stock = avg_daily_sales_7d * 7 * TARGET_STOCK_WEEKS
                order_quantity = max(0, target_stock - current_stock) if target_stock > current_stock else 0
                order_quantity = round(order_quantity)
                
                # Determine stock status
                if weeks_of_stock < MIN_STOCK_WEEKS:
                    stock_status = 'Low'
                elif weeks_of_stock <= TARGET_STOCK_WEEKS:
                    stock_status = 'Adequate'
                else:
                    stock_status = 'Excess'
                
                # Add to recommendations
                product = group['Product'].iloc[0] if 'Product' in group.columns else 'Unknown'
                
                inventory_recommendations.append({
                    'Store_Id': store_id,
                    'Item': item,
                    'Product': product,
                    'Current_Stock': current_stock,
                    'Forecast_7d': forecast_7d,
                    'Forecast_14d': forecast_14d,
                    'Forecast_30d': forecast_30d,
                    'Avg_Daily_Sales': avg_daily_sales_7d,
                    'Weeks_Of_Stock': weeks_of_stock,
                    'Stock_Status': stock_status,
                    'Order_Point': order_point,
                    'Order_Quantity': order_quantity,
                    'Recommended_Date': datetime.now()
                })
            
            # Create DataFrame
            self.inventory_recommendations = pd.DataFrame(inventory_recommendations)
            
            # Save recommendations
            self.inventory_recommendations.to_csv(INVENTORY_RECOMMENDATIONS_FILE, index=False)
            logger.info(f"Saved inventory recommendations to {INVENTORY_RECOMMENDATIONS_FILE}")
            
            logger.info("Inventory optimization complete")
            return self.inventory_recommendations
        except Exception as e:
            logger.error(f"Error optimizing inventory: {str(e)}")
            raise
    
    def run_integrated_optimization(self, days_to_forecast=30, use_existing=True, create_visuals=True):
        """
        Run the complete integrated optimization process
        
        Args:
            days_to_forecast: Number of days to forecast
            use_existing: Whether to use existing models
            create_visuals: Whether to create visualization charts
            
        Returns:
            tuple: (forecasts, price_recommendations, inventory_recommendations)
        """
        logger.info("Starting integrated optimization process")
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Generate forecasts
            self.generate_forecasts(days_to_forecast, use_existing)
            
            # Step 3: Optimize prices
            self.optimize_prices()
            
            # Step 4: Optimize inventory
            self.optimize_inventory()
            
            # Step 5: Create visualizations
            if create_visuals:
                self.create_visualizations()
            
            logger.info("Integrated optimization process complete")
            
            return self.forecasts, self.price_recommendations, self.inventory_recommendations
        
        except Exception as e:
            logger.error(f"Error in integrated optimization process: {str(e)}")
            raise
    
    def create_visualizations(self):
        """
        Create visualization charts for forecasts, pricing, and inventory
        
        Returns:
            None
        """
        logger.info("Creating visualization charts")
        
        try:
            # Create directory if it doesn't exist
            charts_dir = os.path.join(STATIC_DIR, 'images', 'integrated')
            os.makedirs(charts_dir, exist_ok=True)
            
            # Chart 1: Forecasts vs Actual Sales
            if self.forecasts is not None and self.df is not None:
                self._create_forecast_comparison_chart(charts_dir)
            
            # Chart 2: Price Elasticity Distribution
            if self.elasticities is not None:
                self._create_elasticity_chart(charts_dir)
            
            # Chart 3: Inventory Projection
            if self.forecasts is not None and self.inventory_recommendations is not None:
                self._create_inventory_projection_chart(charts_dir)
            
            # Chart 4: Price Optimization Impact
            if self.profit_impact is not None:
                self._create_price_impact_chart(charts_dir)
            
            # Chart 5: Integrated Dashboard
            if self.forecasts is not None and self.price_recommendations is not None and self.inventory_recommendations is not None:
                self._create_integrated_dashboard(charts_dir)
            
            logger.info(f"All visualization charts created in {charts_dir}")
        
        except Exception as e:
            logger.error(f"Error creating visualization charts: {str(e)}")
    
    def _create_forecast_comparison_chart(self, output_dir):
        """
        Create forecast comparison chart
        
        Args:
            output_dir: Output directory for the chart
            
        Returns:
            None
        """
        # Determine forecast column name
        forecast_col = 'Forecast'
        if 'Predicted_Demand' in self.forecasts.columns:
            forecast_col = 'Predicted_Demand'
        elif 'Forecast' in self.forecasts.columns:
            forecast_col = 'Forecast'
        elif 'Predicted_Sales' in self.forecasts.columns:
            forecast_col = 'Predicted_Sales'
            
        # Log available columns
        logger.info(f"Forecast columns: {self.forecasts.columns.tolist()}")
            
        if forecast_col not in self.forecasts.columns:
            logger.warning(f"Forecast column '{forecast_col}' not found in forecasts. Using first numeric column as fallback.")
            # Try to find a suitable numeric column as fallback
            numeric_cols = self.forecasts.select_dtypes(include=[np.number]).columns
            potential_forecast_cols = [col for col in numeric_cols 
                                     if col not in ['Store_Id', 'Item', 'Days_In_Future'] 
                                     and not col.startswith('Lower_') 
                                     and not col.startswith('Upper_')]
            
            if potential_forecast_cols:
                forecast_col = potential_forecast_cols[0]
                logger.info(f"Using '{forecast_col}' as fallback forecast column")
            else:
                logger.error("No suitable forecast column found")
                return
                
        # Aggregate forecasts by date
        agg_forecasts = self.forecasts.groupby('Date')[forecast_col].sum().reset_index()
        
        # Get historical data for the last 30 days
        last_date = self.df['Date'].max()
        start_date = last_date - timedelta(days=30)
        
        hist_data = self.df[self.df['Date'] >= start_date].groupby('Date')['Sales'].sum().reset_index()
        
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(hist_data['Date'], hist_data['Sales'], 'b-', label='Historical Sales')
        
        # Plot forecast
        plt.plot(agg_forecasts['Date'], agg_forecasts[forecast_col], 'r-', label='ARIMA Forecast')
        
        # Add labels and legend
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title('Sales Forecast vs Historical Sales')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Save the chart
        plt.savefig(os.path.join(output_dir, 'forecast_comparison.png'), bbox_inches='tight')
        plt.close()
    
    def _create_elasticity_chart(self, output_dir):
        """
        Create price elasticity chart
        
        Args:
            output_dir: Output directory for the chart
            
        Returns:
            None
        """
        plt.figure(figsize=(12, 6))
        
        # Filter to significant elasticities
        valid_elasticities = self.elasticities[self.elasticities['Is_Significant']]['Elasticity']
        
        # Create histogram
        sns.histplot(valid_elasticities, bins=15, kde=True)
        
        # Add vertical line for average
        plt.axvline(x=valid_elasticities.mean(), color='red', linestyle='--', 
                   label=f'Mean Elasticity: {valid_elasticities.mean():.2f}')
        
        # Add labels and legend
        plt.xlabel('Price Elasticity')
        plt.ylabel('Count')
        plt.title('Distribution of Price Elasticities')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Save the chart
        plt.savefig(os.path.join(output_dir, 'price_elasticity.png'), bbox_inches='tight')
        plt.close()
    
    def _create_inventory_projection_chart(self, output_dir):
        """
        Create inventory projection chart
        
        Args:
            output_dir: Output directory for the chart
            
        Returns:
            None
        """
        plt.figure(figsize=(12, 6))
        
        # Sort by order quantity
        top_orders = self.inventory_recommendations.sort_values('Order_Quantity', ascending=False).head(10)
        
        # Create bar chart
        bars = plt.bar(top_orders['Product'], top_orders['Order_Quantity'], color='skyblue')
        
        # Add current stock as red bars
        plt.bar(top_orders['Product'], top_orders['Current_Stock'], color='salmon', 
               label='Current Stock')
        
        # Add labels
        plt.xlabel('Product')
        plt.ylabel('Units')
        plt.title('Top 10 Products by Recommended Order Quantity')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save the chart
        plt.savefig(os.path.join(output_dir, 'inventory_projection.png'), bbox_inches='tight')
        plt.close()
    
    def _create_price_impact_chart(self, output_dir):
        """
        Create price impact waterfall chart
        
        Args:
            output_dir: Output directory for the chart
            
        Returns:
            None
        """
        plt.figure(figsize=(12, 6))
        
        # Sort by profit impact
        top_impact = self.profit_impact.sort_values('Total_Profit_Difference', ascending=False).head(10)
        
        # Create bar chart
        plt.bar(top_impact['Product'], top_impact['Total_Profit_Difference'], color='green')
        
        # Add price change as text
        for i, (_, row) in enumerate(top_impact.iterrows()):
            plt.text(i, row['Total_Profit_Difference'], 
                    f"{row['Price_Change_Pct']:.1f}%", 
                    ha='center', va='bottom' if row['Total_Profit_Difference'] > 0 else 'top')
        
        # Add labels
        plt.xlabel('Product')
        plt.ylabel('Profit Impact ($)')
        plt.title('Top 10 Products by Profit Impact from Price Optimization')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save the chart
        plt.savefig(os.path.join(output_dir, 'price_impact.png'), bbox_inches='tight')
        plt.close()
    
    def _create_integrated_dashboard(self, output_dir):
        """
        Create integrated dashboard with forecasts, pricing, and inventory
        
        Args:
            output_dir: Output directory for the chart
            
        Returns:
            None
        """
        plt.figure(figsize=(15, 10))
        
        # Create 2x2 subplot layout
        plt.subplot(2, 2, 1)
        
        # Determine forecast column name
        forecast_col = 'Forecast'
        if 'Predicted_Demand' in self.forecasts.columns:
            forecast_col = 'Predicted_Demand'
        elif 'Forecast' in self.forecasts.columns:
            forecast_col = 'Forecast'
            
        # Top left: Total sales forecast
        agg_dict = {forecast_col: 'sum'}
        if 'Lower_Bound' in self.forecasts.columns and 'Upper_Bound' in self.forecasts.columns:
            agg_dict['Lower_Bound'] = 'sum'
            agg_dict['Upper_Bound'] = 'sum'
            
        agg_forecasts = self.forecasts.groupby('Date').agg(agg_dict).reset_index()
        
        plt.plot(agg_forecasts['Date'], agg_forecasts[forecast_col], 'r-', label='Forecast')
        
        # Only draw confidence interval if bounds are available
        if 'Lower_Bound' in agg_forecasts.columns and 'Upper_Bound' in agg_forecasts.columns:
            plt.fill_between(
                agg_forecasts['Date'],
                agg_forecasts['Lower_Bound'],
                agg_forecasts['Upper_Bound'],
                color='r', alpha=0.2
            )
        plt.title('Total Sales Forecast')
        plt.grid(alpha=0.3)
        
        # Top right: Price optimization summary
        plt.subplot(2, 2, 2)
        
        price_impact_by_status = self.price_recommendations.groupby('Recommendation').size()
        plt.pie(price_impact_by_status, labels=price_impact_by_status.index, autopct='%1.1f%%',
               colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
        plt.title('Price Recommendation Distribution')
        
        # Bottom left: Inventory status summary
        plt.subplot(2, 2, 3)
        
        inv_status = self.inventory_recommendations.groupby('Stock_Status').size()
        plt.bar(inv_status.index, inv_status, color=['#ff9999','#66b3ff','#99ff99'])
        plt.title('Inventory Status Distribution')
        plt.grid(axis='y', alpha=0.3)
        
        # Bottom right: Combined profit impact
        plt.subplot(2, 2, 4)
        
        if len(self.profit_impact) > 0:
            total_impact = self.profit_impact['Total_Profit_Difference'].sum()
            total_percent = self.profit_impact['Profit_Change_Pct'].mean()
            
            plt.text(0.5, 0.5, f"Total Profit Impact:\n${total_impact:.2f}\n\nAverage Change:\n{total_percent:.2f}%",
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
            plt.title('Profit Optimization Summary')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'integrated_dashboard.png'), bbox_inches='tight')
        plt.close()


def main():
    """
    Main function to run the integrated forecasting process
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Integrated forecasting and optimization')
    parser.add_argument('--days', type=int, default=30, help='Number of days to forecast')
    parser.add_argument('--use-existing', action='store_true', help='Use existing models if available')
    parser.add_argument('--no-visuals', dest='create_visuals', action='store_false', 
                      help='Disable visualization creation')
    parser.add_argument('--data', type=str, default=COMBINED_DATA_FILE, help='Path to data file')
    parser.set_defaults(create_visuals=True)
    
    args = parser.parse_args()
    
    # Create integrated forecaster
    forecaster = IntegratedForecaster(data_path=args.data)
    
    # Run integrated optimization
    forecaster.run_integrated_optimization(
        days_to_forecast=args.days,
        use_existing=args.use_existing,
        create_visuals=args.create_visuals
    )
    
    logger.info("Integrated forecasting and optimization completed successfully")


if __name__ == "__main__":
    main()