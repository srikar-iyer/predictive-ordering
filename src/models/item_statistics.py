"""
Item-based statistics with extended forecasting module.
This module provides enhanced item-level statistics and extended forecasting capabilities
that work with the existing ARIMA models and time series forecasting components.
"""
import pandas as pd
import numpy as np
import os
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import (
    COMBINED_DATA_FILE, MODELS_DIR, STATIC_DIR, 
    WEIGHTED_ARIMA_FORECASTS_FILE, ARIMA_FORECASTS_FILE,
    ITEM_STATISTICS_FILE, EXTENDED_FORECASTS_FILE, ITEM_STATISTICS_DIR
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('item_statistics')

class ItemStatisticsAnalyzer:
    """
    Class for analyzing detailed item-level statistics and extended forecasting
    """
    def __init__(self, data_path=COMBINED_DATA_FILE):
        """
        Initialize the item statistics analyzer
        
        Args:
            data_path: Path to the combined data file
        """
        self.data_path = data_path
        self.df = None
        self.forecasts = None
        self.item_stats = None
        self.extended_forecasts = None
        # Setup path for Plotly visualizations
        self.plotly_dir = os.path.join(STATIC_DIR, 'plotly_visualizations')
        
    def load_data(self):
        """
        Load data from CSV file
        
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
            
    def load_forecasts(self, forecast_file=None):
        """
        Load forecasts from CSV file
        
        Args:
            forecast_file: Path to the forecasts file (optional)
            
        Returns:
            DataFrame: The loaded forecasts
        """
        if forecast_file is None:
            # Try weighted ARIMA forecasts first, then regular ARIMA
            if os.path.exists(WEIGHTED_ARIMA_FORECASTS_FILE):
                forecast_file = WEIGHTED_ARIMA_FORECASTS_FILE
            elif os.path.exists(ARIMA_FORECASTS_FILE):
                forecast_file = ARIMA_FORECASTS_FILE
            else:
                logger.warning("No forecast files found")
                return None
                
        logger.info(f"Loading forecasts from {forecast_file}")
        try:
            self.forecasts = pd.read_csv(forecast_file)
            
            # Convert date to datetime
            if 'Date' in self.forecasts.columns:
                self.forecasts['Date'] = pd.to_datetime(self.forecasts['Date'])
            
            logger.info(f"Loaded {len(self.forecasts)} forecast records from {forecast_file}")
            return self.forecasts
        except Exception as e:
            logger.error(f"Error loading forecasts: {str(e)}")
            return None
    
    def calculate_item_statistics(self):
        """
        Calculate comprehensive item-level statistics
        
        Returns:
            DataFrame: Item-level statistics
        """
        logger.info("Calculating comprehensive item-level statistics")
        
        # Make sure data is loaded
        if self.df is None:
            self.load_data()
            
        # Initialize list for storing item statistics
        item_stats_list = []
        
        # Process each store-item combination
        store_items = self.df[['Store_Id', 'Item']].drop_duplicates()
        
        for _, row in store_items.iterrows():
            store_id = row['Store_Id']
            item = row['Item']
            
            logger.info(f"Calculating statistics for Store {store_id}, Item {item}")
            
            # Get data for this store-item
            item_df = self.df[(self.df['Store_Id'] == store_id) & (self.df['Item'] == item)]
            
            # Skip if not enough data
            if len(item_df) < 30:
                logger.warning(f"Insufficient data for Store {store_id}, Item {item}")
                continue
                
            # Basic information
            product_name = item_df['Product'].iloc[0]
            last_date = item_df['Date'].max()
            first_date = item_df['Date'].min()
            days_of_data = (last_date - first_date).days + 1
            
            # Sales statistics
            total_sales = item_df['Sales'].sum()
            avg_daily_sales = item_df['Sales'].mean()
            sales_std = item_df['Sales'].std()
            sales_cv = sales_std / avg_daily_sales if avg_daily_sales > 0 else np.nan
            sales_min = item_df['Sales'].min()
            sales_max = item_df['Sales'].max()
            sales_median = item_df['Sales'].median()
            
            # Calculate percentiles
            sales_p25 = item_df['Sales'].quantile(0.25)
            sales_p75 = item_df['Sales'].quantile(0.75)
            sales_p90 = item_df['Sales'].quantile(0.90)
            sales_p95 = item_df['Sales'].quantile(0.95)
            
            # Calculate zero sales days
            zero_sales_days = (item_df['Sales'] == 0).sum()
            zero_sales_pct = zero_sales_days / len(item_df) * 100
            
            # Time-based sales patterns
            item_df['Day_Name'] = item_df['Date'].dt.day_name()
            daily_avg = item_df.groupby('Day_Of_Week')['Sales'].mean().to_dict()
            weekly_pattern = {idx: daily_avg.get(idx, 0) for idx in range(7)}
            
            # Monthly pattern
            item_df['Month_Name'] = item_df['Date'].dt.month_name()
            monthly_avg = item_df.groupby('Month')['Sales'].mean().to_dict()
            monthly_pattern = {month: monthly_avg.get(month, 0) for month in range(1, 13)}
            
            # Sales trend calculation
            item_df = item_df.sort_values('Date')
            dates_num = np.arange(len(item_df))
            
            # Simple linear trend
            if len(dates_num) > 1:
                trend_model = sm.OLS(item_df['Sales'], sm.add_constant(dates_num)).fit()
                trend_coefficient = trend_model.params[1]
                trend_pvalue = trend_model.pvalues[1]
                trend_r2 = trend_model.rsquared
                
                # Interpret trend
                if trend_pvalue < 0.05:
                    if trend_coefficient > 0:
                        trend_interpretation = "Significantly increasing"
                    else:
                        trend_interpretation = "Significantly decreasing"
                else:
                    trend_interpretation = "No significant trend"
            else:
                trend_coefficient = np.nan
                trend_pvalue = np.nan
                trend_r2 = np.nan
                trend_interpretation = "Insufficient data"
            
            # Detect seasonality
            seasonality_detected = False
            seasonal_period = None
            seasonal_strength = None
            
            try:
                if len(item_df) >= 28:  # Need at least 4 weeks of data
                    for period in [7, 14, 30]:
                        if len(item_df) >= period * 2:  # Need at least 2 full periods
                            seas_result = seasonal_decompose(
                                item_df['Sales'], 
                                model='additive', 
                                period=period,
                                extrapolate_trend='freq'
                            )
                            strength = np.std(seas_result.seasonal) / np.std(item_df['Sales'] - seas_result.trend)
                            
                            if strength > 0.1:  # Significant seasonality
                                seasonality_detected = True
                                seasonal_period = period
                                seasonal_strength = strength
                                break
            except Exception as e:
                logger.warning(f"Error detecting seasonality for Store {store_id}, Item {item}: {e}")
            
            # Price statistics
            avg_price = item_df['Price'].mean()
            price_std = item_df['Price'].std()
            price_cv = price_std / avg_price if avg_price > 0 else np.nan
            price_min = item_df['Price'].min()
            price_max = item_df['Price'].max()
            
            # Calculate promotion frequency
            promotion_days = item_df['Promotion'].sum()
            promotion_pct = promotion_days / len(item_df) * 100
            
            # Stock metrics
            if 'Stock_Level' in item_df.columns:
                avg_stock = item_df['Stock_Level'].mean()
                current_stock = item_df.iloc[-1]['Stock_Level']
                stock_coverage = item_df.iloc[-1]['Weeks_Of_Stock'] if 'Weeks_Of_Stock' in item_df.columns else np.nan
            else:
                avg_stock = np.nan
                current_stock = np.nan
                stock_coverage = np.nan
            
            # Profit metrics
            if 'Profit' in item_df.columns:
                total_profit = item_df['Profit'].sum()
                avg_profit = item_df['Profit'].mean()
                profit_margin = total_profit / item_df['Retail_Revenue'].sum() if item_df['Retail_Revenue'].sum() > 0 else np.nan
            else:
                total_profit = np.nan
                avg_profit = np.nan
                profit_margin = np.nan
            
            # Weather impact analysis
            weather_impact = {}
            if 'Weather' in item_df.columns:
                weather_groups = item_df.groupby('Weather')
                baseline_sales = item_df[item_df['Weather'] == 'Normal']['Sales'].mean()
                
                for weather, group in weather_groups:
                    weather_sales = group['Sales'].mean()
                    weather_impact[weather] = weather_sales / baseline_sales if baseline_sales > 0 else 1.0
            
            # Holiday impact analysis
            holiday_impact = None
            if 'Is_Holiday' in item_df.columns and 'Holiday_Name' in item_df.columns:
                non_holiday_sales = item_df[item_df['Is_Holiday'] == 0]['Sales'].mean()
                holiday_sales = item_df[item_df['Is_Holiday'] == 1]['Sales'].mean()
                
                if non_holiday_sales > 0:
                    holiday_impact = holiday_sales / non_holiday_sales
            
            # Calculate day-of-week effect
            dow_effect = {}
            for day in range(7):
                day_avg = item_df[item_df['Day_Of_Week'] == day]['Sales'].mean()
                dow_effect[day] = day_avg / avg_daily_sales if avg_daily_sales > 0 else 1.0
            
            # Correlation with other products
            # This would require a separate analysis across products
            
            # Shelf-life statistics (if available)
            shelf_life = None
            spoilage_rate = None
            
            # Create stat dictionary
            item_stats = {
                'Store_Id': store_id,
                'Item': item,
                'Product': product_name,
                'First_Date': first_date,
                'Last_Date': last_date,
                'Days_Of_Data': days_of_data,
                'Total_Sales': total_sales,
                'Avg_Daily_Sales': avg_daily_sales,
                'Sales_StdDev': sales_std,
                'Sales_CV': sales_cv,  # Coefficient of variation
                'Sales_Min': sales_min,
                'Sales_Max': sales_max,
                'Sales_Median': sales_median,
                'Sales_P25': sales_p25,
                'Sales_P75': sales_p75,
                'Sales_P90': sales_p90,
                'Sales_P95': sales_p95,
                'Zero_Sales_Days': zero_sales_days,
                'Zero_Sales_Pct': zero_sales_pct,
                'Weekly_Pattern': weekly_pattern,
                'Monthly_Pattern': monthly_pattern,
                'Trend_Coefficient': trend_coefficient,
                'Trend_P_Value': trend_pvalue,
                'Trend_R2': trend_r2,
                'Trend_Interpretation': trend_interpretation,
                'Seasonality_Detected': seasonality_detected,
                'Seasonal_Period': seasonal_period,
                'Seasonal_Strength': seasonal_strength,
                'Avg_Price': avg_price,
                'Price_StdDev': price_std,
                'Price_CV': price_cv,
                'Price_Min': price_min,
                'Price_Max': price_max,
                'Promotion_Days': promotion_days,
                'Promotion_Pct': promotion_pct,
                'Avg_Stock': avg_stock,
                'Current_Stock': current_stock,
                'Stock_Coverage_Weeks': stock_coverage,
                'Total_Profit': total_profit,
                'Avg_Profit': avg_profit,
                'Profit_Margin': profit_margin,
                'Weather_Impact': weather_impact,
                'Holiday_Impact': holiday_impact,
                'Day_Of_Week_Effect': dow_effect,
                'Shelf_Life': shelf_life,
                'Spoilage_Rate': spoilage_rate,
            }
            
            # Add to list
            item_stats_list.append(item_stats)
        
        # Create DataFrame
        self.item_stats = pd.DataFrame(item_stats_list)
        
        logger.info(f"Calculated statistics for {len(self.item_stats)} store-item combinations")
        return self.item_stats
    
    def extend_forecast(self, days_to_extend=60):
        """
        Extend forecasts beyond the current range using time series models
        
        Args:
            days_to_extend: Number of additional days to forecast
            
        Returns:
            DataFrame: Extended forecasts
        """
        logger.info(f"Extending forecasts by {days_to_extend} days")
        
        # Make sure data and forecasts are loaded
        if self.df is None:
            self.load_data()
            
        if self.forecasts is None:
            self.load_forecasts()
            if self.forecasts is None:
                logger.error("No forecast data available for extension")
                return None
        
        # Calculate item statistics if not already done
        if self.item_stats is None:
            self.calculate_item_statistics()
        
        # Initialize list for extended forecasts
        extended_forecasts_list = []
        
        # Get the maximum forecast date from existing forecasts
        max_forecast_date = self.forecasts['Date'].max()
        
        # Create date range for extended forecasts
        extended_dates = pd.date_range(
            start=max_forecast_date + timedelta(days=1),
            periods=days_to_extend
        )
        
        # Process each store-item combination
        for _, stats in self.item_stats.iterrows():
            store_id = stats['Store_Id']
            item = stats['Item']
            
            logger.info(f"Extending forecasts for Store {store_id}, Item {item}")
            
            # Get existing forecasts for this store-item
            item_forecasts = self.forecasts[
                (self.forecasts['Store_Id'] == store_id) &
                (self.forecasts['Item'] == item)
            ].sort_values('Date')
            
            # Skip if no existing forecasts
            if len(item_forecasts) == 0:
                logger.warning(f"No existing forecasts for Store {store_id}, Item {item}")
                continue
            
            # Get forecast statistics
            forecast_mean = item_forecasts['Forecast'].mean()
            forecast_std = item_forecasts['Forecast'].std()
            has_bounds = 'Lower_Bound' in item_forecasts.columns and 'Upper_Bound' in item_forecasts.columns
            
            # Apply extension method based on item characteristics
            
            # Method 1: Use trend and seasonality from item stats
            if stats['Seasonality_Detected'] and stats['Seasonal_Period'] is not None:
                # Apply seasonal pattern with trend
                seasonal_period = stats['Seasonal_Period']
                seasonal_indices = self._calculate_seasonal_indices(
                    self.df[(self.df['Store_Id'] == store_id) & (self.df['Item'] == item)],
                    seasonal_period
                )
                
                # Apply trend adjustment
                trend_coef = stats['Trend_Coefficient']
                
                for i, date in enumerate(extended_dates):
                    # Calculate the seasonal index for this date
                    day_of_period = i % seasonal_period
                    seasonal_factor = seasonal_indices.get(day_of_period, 1.0)
                    
                    # Apply trend component (linear extrapolation)
                    days_from_last = i + 1  # Days from the last known forecast
                    trend_component = trend_coef * days_from_last
                    
                    # Calculate forecast
                    base_forecast = forecast_mean * seasonal_factor
                    adjusted_forecast = max(0, base_forecast + trend_component)
                    
                    # Calculate confidence bounds
                    if has_bounds:
                        # Increase uncertainty with time
                        uncertainty_factor = 1.0 + (days_from_last / 100)  # 1% increase per day
                        bound_width = (item_forecasts['Upper_Bound'] - item_forecasts['Lower_Bound']).mean()
                        scaled_bound_width = bound_width * uncertainty_factor
                        
                        lower_bound = max(0, adjusted_forecast - (scaled_bound_width / 2))
                        upper_bound = adjusted_forecast + (scaled_bound_width / 2)
                    else:
                        # Use standard deviation to approximate bounds
                        uncertainty_factor = 1.0 + (days_from_last / 100)
                        scaled_std = forecast_std * uncertainty_factor
                        
                        lower_bound = max(0, adjusted_forecast - (1.96 * scaled_std))
                        upper_bound = adjusted_forecast + (1.96 * scaled_std)
                    
                    # Create forecast record
                    forecast_record = {
                        'Store_Id': store_id,
                        'Item': item,
                        'Product': stats['Product'],
                        'Date': date,
                        'Forecast': adjusted_forecast,
                        'Lower_Bound': lower_bound,
                        'Upper_Bound': upper_bound,
                        'Forecast_Generated': datetime.now(),
                        'Days_In_Future': len(item_forecasts) + i + 1,
                        'Is_Extended': True
                    }
                    
                    extended_forecasts_list.append(forecast_record)
                
            # Method 2: Use day-of-week patterns
            elif stats['Day_Of_Week_Effect']:
                dow_effect = stats['Day_Of_Week_Effect']
                
                for i, date in enumerate(extended_dates):
                    # Get day of week and apply effect
                    day_of_week = date.dayofweek
                    day_factor = dow_effect.get(day_of_week, 1.0)
                    
                    # Calculate forecast
                    adjusted_forecast = forecast_mean * day_factor
                    
                    # Calculate confidence bounds with increasing uncertainty
                    days_from_last = i + 1
                    uncertainty_factor = 1.0 + (days_from_last / 50)  # 2% increase per day
                    
                    if has_bounds:
                        bound_width = (item_forecasts['Upper_Bound'] - item_forecasts['Lower_Bound']).mean()
                        scaled_bound_width = bound_width * uncertainty_factor
                        
                        lower_bound = max(0, adjusted_forecast - (scaled_bound_width / 2))
                        upper_bound = adjusted_forecast + (scaled_bound_width / 2)
                    else:
                        scaled_std = forecast_std * uncertainty_factor
                        
                        lower_bound = max(0, adjusted_forecast - (1.96 * scaled_std))
                        upper_bound = adjusted_forecast + (1.96 * scaled_std)
                    
                    # Create forecast record
                    forecast_record = {
                        'Store_Id': store_id,
                        'Item': item,
                        'Product': stats['Product'],
                        'Date': date,
                        'Forecast': adjusted_forecast,
                        'Lower_Bound': lower_bound,
                        'Upper_Bound': upper_bound,
                        'Forecast_Generated': datetime.now(),
                        'Days_In_Future': len(item_forecasts) + i + 1,
                        'Is_Extended': True
                    }
                    
                    extended_forecasts_list.append(forecast_record)
                    
            # Method 3: Simple trend extrapolation
            else:
                # Apply simple trend with exponentially increasing uncertainty
                trend_coef = stats['Trend_Coefficient'] if not pd.isna(stats['Trend_Coefficient']) else 0
                
                for i, date in enumerate(extended_dates):
                    # Apply trend component (linear extrapolation)
                    days_from_last = i + 1
                    trend_component = trend_coef * days_from_last
                    
                    # Calculate forecast
                    adjusted_forecast = max(0, forecast_mean + trend_component)
                    
                    # Calculate confidence bounds with exponentially increasing uncertainty
                    uncertainty_factor = 1.0 + (0.01 * days_from_last)**1.5
                    
                    if has_bounds:
                        bound_width = (item_forecasts['Upper_Bound'] - item_forecasts['Lower_Bound']).mean()
                        scaled_bound_width = bound_width * uncertainty_factor
                        
                        lower_bound = max(0, adjusted_forecast - (scaled_bound_width / 2))
                        upper_bound = adjusted_forecast + (scaled_bound_width / 2)
                    else:
                        scaled_std = forecast_std * uncertainty_factor
                        
                        lower_bound = max(0, adjusted_forecast - (1.96 * scaled_std))
                        upper_bound = adjusted_forecast + (1.96 * scaled_std)
                    
                    # Create forecast record
                    forecast_record = {
                        'Store_Id': store_id,
                        'Item': item,
                        'Product': stats['Product'],
                        'Date': date,
                        'Forecast': adjusted_forecast,
                        'Lower_Bound': lower_bound,
                        'Upper_Bound': upper_bound,
                        'Forecast_Generated': datetime.now(),
                        'Days_In_Future': len(item_forecasts) + i + 1,
                        'Is_Extended': True
                    }
                    
                    extended_forecasts_list.append(forecast_record)
        
        # Create DataFrame with extended forecasts
        extended_forecasts_df = pd.DataFrame(extended_forecasts_list)
        
        # Combine with original forecasts
        self.extended_forecasts = pd.concat([self.forecasts, extended_forecasts_df], ignore_index=True)
        
        logger.info(f"Extended forecasts for {len(self.item_stats)} store-item combinations by {days_to_extend} days")
        return self.extended_forecasts
    
    def _calculate_seasonal_indices(self, item_df, period):
        """
        Calculate seasonal indices for a time series
        
        Args:
            item_df: DataFrame with time series data
            period: Seasonal period
            
        Returns:
            dict: Seasonal indices by position in period
        """
        # Sort by date
        item_df = item_df.sort_values('Date')
        
        # Calculate seasonal indices
        indices = {}
        sales_data = item_df['Sales'].values
        
        # For each position in period
        for i in range(period):
            # Get values for this position
            position_values = sales_data[i::period]
            
            # Calculate mean if there are values
            if len(position_values) > 0:
                overall_mean = sales_data.mean()
                position_mean = position_values.mean()
                
                if overall_mean > 0:
                    indices[i] = position_mean / overall_mean
                else:
                    indices[i] = 1.0
            else:
                indices[i] = 1.0
                
        return indices
    
    def create_item_statistics_visualizations(self, output_dir=None):
        """
        Create visualizations for item statistics
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            list: Paths to created visualization files
        """
        if self.item_stats is None:
            logger.warning("No item statistics available for visualization")
            return []
            
        if output_dir is None:
            output_dir = os.path.join(STATIC_DIR, 'images', 'item_statistics')
            
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Creating item statistics visualizations in {output_dir}")
        
        plot_paths = []
        
        # Create aggregate statistics plots
        
        # 1. Sales distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(self.item_stats['Avg_Daily_Sales'], kde=True)
        plt.title('Distribution of Average Daily Sales Across Items')
        plt.xlabel('Average Daily Sales')
        plt.ylabel('Count')
        plt.grid(alpha=0.3)
        
        sales_dist_path = os.path.join(output_dir, 'sales_distribution.png')
        plt.savefig(sales_dist_path, bbox_inches='tight')
        plt.close()
        plot_paths.append(sales_dist_path)
        
        # 2. Price distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(self.item_stats['Avg_Price'], kde=True)
        plt.title('Distribution of Average Prices Across Items')
        plt.xlabel('Average Price ($)')
        plt.ylabel('Count')
        plt.grid(alpha=0.3)
        
        price_dist_path = os.path.join(output_dir, 'price_distribution.png')
        plt.savefig(price_dist_path, bbox_inches='tight')
        plt.close()
        plot_paths.append(price_dist_path)
        
        # 3. Coefficient of Variation vs. Average Sales
        plt.figure(figsize=(12, 6))
        plt.scatter(self.item_stats['Avg_Daily_Sales'], self.item_stats['Sales_CV'])
        plt.title('Sales Variability vs. Average Daily Sales')
        plt.xlabel('Average Daily Sales')
        plt.ylabel('Coefficient of Variation')
        plt.grid(alpha=0.3)
        
        cv_path = os.path.join(output_dir, 'sales_variability.png')
        plt.savefig(cv_path, bbox_inches='tight')
        plt.close()
        plot_paths.append(cv_path)
        
        # 4. Zero Sales Percentage Distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(self.item_stats['Zero_Sales_Pct'], kde=True)
        plt.title('Distribution of Zero Sales Day Percentage')
        plt.xlabel('Zero Sales Days (%)')
        plt.ylabel('Count')
        plt.grid(alpha=0.3)
        
        zero_sales_path = os.path.join(output_dir, 'zero_sales_distribution.png')
        plt.savefig(zero_sales_path, bbox_inches='tight')
        plt.close()
        plot_paths.append(zero_sales_path)
        
        # 5. Stock Coverage Distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(self.item_stats['Stock_Coverage_Weeks'], kde=True)
        plt.title('Distribution of Stock Coverage (Weeks)')
        plt.xlabel('Stock Coverage (Weeks)')
        plt.ylabel('Count')
        plt.grid(alpha=0.3)
        
        stock_coverage_path = os.path.join(output_dir, 'stock_coverage_distribution.png')
        plt.savefig(stock_coverage_path, bbox_inches='tight')
        plt.close()
        plot_paths.append(stock_coverage_path)
        
        # 6. Top 10 items by sales
        plt.figure(figsize=(14, 8))
        top_items = self.item_stats.sort_values('Total_Sales', ascending=False).head(10)
        sns.barplot(x='Total_Sales', y='Product', data=top_items)
        plt.title('Top 10 Items by Total Sales')
        plt.xlabel('Total Sales')
        plt.ylabel('Product')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        top_items_path = os.path.join(output_dir, 'top_items_by_sales.png')
        plt.savefig(top_items_path, bbox_inches='tight')
        plt.close()
        plot_paths.append(top_items_path)
        
        # 7. Top 10 items by profit
        plt.figure(figsize=(14, 8))
        top_profit_items = self.item_stats.sort_values('Total_Profit', ascending=False).head(10)
        sns.barplot(x='Total_Profit', y='Product', data=top_profit_items)
        plt.title('Top 10 Items by Total Profit')
        plt.xlabel('Total Profit')
        plt.ylabel('Product')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        top_profit_path = os.path.join(output_dir, 'top_items_by_profit.png')
        plt.savefig(top_profit_path, bbox_inches='tight')
        plt.close()
        plot_paths.append(top_profit_path)
        
        # 8. Day of week effect heatmap
        plt.figure(figsize=(14, 8))
        
        # Extract day of week effects into a DataFrame
        dow_effects = []
        
        for _, row in self.item_stats.iterrows():
            if row['Day_Of_Week_Effect'] is not None:
                item_effects = row['Day_Of_Week_Effect']
                if isinstance(item_effects, dict):
                    item_data = {
                        'Store_Id': row['Store_Id'],
                        'Item': row['Item'],
                        'Product': row['Product']
                    }
                    
                    for day, effect in item_effects.items():
                        day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][int(day)]
                        item_data[day_name] = effect
                        
                    dow_effects.append(item_data)
        
        if dow_effects:
            dow_df = pd.DataFrame(dow_effects)
            dow_df = dow_df.set_index('Product')
            
            day_cols = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_data = dow_df[day_cols].head(15)  # Top 15 products
            
            sns.heatmap(day_data, cmap="YlGnBu", annot=True, fmt=".2f")
            plt.title('Day of Week Effect by Product')
            plt.tight_layout()
            
            dow_path = os.path.join(output_dir, 'day_of_week_effect.png')
            plt.savefig(dow_path, bbox_inches='tight')
            plt.close()
            plot_paths.append(dow_path)
        
        # 9. Weather impact analysis
        plt.figure(figsize=(14, 8))
        
        # Extract weather effects into a DataFrame
        weather_effects = []
        
        for _, row in self.item_stats.iterrows():
            if row['Weather_Impact'] is not None:
                item_effects = row['Weather_Impact']
                if isinstance(item_effects, dict):
                    item_data = {
                        'Store_Id': row['Store_Id'],
                        'Item': row['Item'],
                        'Product': row['Product']
                    }
                    
                    for weather, effect in item_effects.items():
                        item_data[weather] = effect
                        
                    weather_effects.append(item_data)
        
        if weather_effects:
            weather_df = pd.DataFrame(weather_effects)
            weather_df = weather_df.set_index('Product')
            
            weather_cols = ['Normal', 'Heavy Rain', 'Snow', 'Storm']
            weather_cols = [col for col in weather_cols if col in weather_df.columns]
            
            if weather_cols:
                weather_data = weather_df[weather_cols].head(15)  # Top 15 products
                
                sns.heatmap(weather_data, cmap="coolwarm", annot=True, fmt=".2f", center=1.0)
                plt.title('Weather Impact by Product')
                plt.tight_layout()
                
                weather_path = os.path.join(output_dir, 'weather_impact.png')
                plt.savefig(weather_path, bbox_inches='tight')
                plt.close()
                plot_paths.append(weather_path)
        
        logger.info(f"Created {len(plot_paths)} item statistics visualizations")
        return plot_paths
    
    def create_extended_forecast_visualizations(self, output_dir=None):
        """
        Create visualizations for extended forecasts
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            list: Paths to created visualization files
        """
        if self.extended_forecasts is None:
            logger.warning("No extended forecasts available for visualization")
            return []
            
        if output_dir is None:
            output_dir = os.path.join(STATIC_DIR, 'images', 'extended_forecasts')
            
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Creating extended forecast visualizations in {output_dir}")
        
        plot_paths = []
        
        # 1. Total sales forecast (original + extended)
        plt.figure(figsize=(14, 7))
        
        # Aggregate by date
        agg_forecasts = self.extended_forecasts.groupby('Date').agg({
            'Forecast': 'sum',
            'Lower_Bound': 'sum',
            'Upper_Bound': 'sum',
            'Is_Extended': 'any'  # Will be True if any forecast on that date is extended
        }).reset_index()
        
        # Split into original and extended forecasts
        original_forecasts = agg_forecasts[~agg_forecasts['Is_Extended']]
        extended_forecasts = agg_forecasts[agg_forecasts['Is_Extended']]
        
        # Plot original forecasts
        plt.plot(original_forecasts['Date'], original_forecasts['Forecast'], 'b-', label='Original Forecast')
        plt.fill_between(
            original_forecasts['Date'],
            original_forecasts['Lower_Bound'],
            original_forecasts['Upper_Bound'],
            color='b', alpha=0.2
        )
        
        # Plot extended forecasts
        plt.plot(extended_forecasts['Date'], extended_forecasts['Forecast'], 'r-', label='Extended Forecast')
        plt.fill_between(
            extended_forecasts['Date'],
            extended_forecasts['Lower_Bound'],
            extended_forecasts['Upper_Bound'],
            color='r', alpha=0.2
        )
        
        plt.title('Total Sales Forecast (Original + Extended)')
        plt.xlabel('Date')
        plt.ylabel('Total Sales')
        plt.grid(alpha=0.3)
        plt.legend()
        
        total_forecast_path = os.path.join(output_dir, 'total_extended_forecast.png')
        plt.savefig(total_forecast_path, bbox_inches='tight')
        plt.close()
        plot_paths.append(total_forecast_path)
        
        # 2. Forecasts for top 5 items by sales
        if self.item_stats is not None:
            top_items = self.item_stats.sort_values('Total_Sales', ascending=False).head(5)
            
            for _, item_row in top_items.iterrows():
                store_id = item_row['Store_Id']
                item = item_row['Item']
                product = item_row['Product']
                
                plt.figure(figsize=(14, 7))
                
                # Get forecasts for this item
                item_forecasts = self.extended_forecasts[
                    (self.extended_forecasts['Store_Id'] == store_id) &
                    (self.extended_forecasts['Item'] == item)
                ].sort_values('Date')
                
                # Split into original and extended
                original = item_forecasts[~item_forecasts['Is_Extended']]
                extended = item_forecasts[item_forecasts['Is_Extended']]
                
                # Plot original forecasts
                plt.plot(original['Date'], original['Forecast'], 'b-', label='Original Forecast')
                plt.fill_between(
                    original['Date'],
                    original['Lower_Bound'],
                    original['Upper_Bound'],
                    color='b', alpha=0.2
                )
                
                # Plot extended forecasts
                plt.plot(extended['Date'], extended['Forecast'], 'r-', label='Extended Forecast')
                plt.fill_between(
                    extended['Date'],
                    extended['Lower_Bound'],
                    extended['Upper_Bound'],
                    color='r', alpha=0.2
                )
                
                plt.title(f'Extended Forecast for {product} (Store {store_id}, Item {item})')
                plt.xlabel('Date')
                plt.ylabel('Sales')
                plt.grid(alpha=0.3)
                plt.legend()
                
                item_forecast_path = os.path.join(output_dir, f'extended_forecast_{store_id}_{item}.png')
                plt.savefig(item_forecast_path, bbox_inches='tight')
                plt.close()
                plot_paths.append(item_forecast_path)
        
        # 3. Uncertainty growth with forecast horizon
        plt.figure(figsize=(12, 6))
        
        # Calculate relative confidence interval width for each forecast day
        self.extended_forecasts['CI_Width'] = (
            self.extended_forecasts['Upper_Bound'] - self.extended_forecasts['Lower_Bound']
        )
        self.extended_forecasts['CI_Width_Relative'] = self.extended_forecasts['CI_Width'] / self.extended_forecasts['Forecast']
        
        # Group by Days_In_Future and calculate average
        ci_by_horizon = self.extended_forecasts.groupby('Days_In_Future')['CI_Width_Relative'].mean().reset_index()
        
        plt.plot(ci_by_horizon['Days_In_Future'], ci_by_horizon['CI_Width_Relative'], 'g-', marker='o')
        plt.title('Forecast Uncertainty vs. Time Horizon')
        plt.xlabel('Days in Future')
        plt.ylabel('Relative Confidence Interval Width')
        plt.grid(alpha=0.3)
        
        uncertainty_path = os.path.join(output_dir, 'forecast_uncertainty.png')
        plt.savefig(uncertainty_path, bbox_inches='tight')
        plt.close()
        plot_paths.append(uncertainty_path)
        
        # 4. Daily vs. Weekly vs. Monthly views
        plt.figure(figsize=(15, 10))
        
        # Daily view (last 30 days)
        plt.subplot(3, 1, 1)
        daily_forecasts = agg_forecasts.tail(30)
        plt.plot(daily_forecasts['Date'], daily_forecasts['Forecast'], 'b-')
        plt.title('Daily Forecast (Last 30 Days)')
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45)
        
        # Weekly view
        plt.subplot(3, 1, 2)
        agg_forecasts['Week'] = agg_forecasts['Date'].dt.to_period('W').astype(str)
        weekly_forecasts = agg_forecasts.groupby('Week').agg({'Forecast': 'sum'}).reset_index()
        plt.bar(weekly_forecasts['Week'], weekly_forecasts['Forecast'], color='skyblue')
        plt.title('Weekly Forecast')
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        
        # Monthly view
        plt.subplot(3, 1, 3)
        agg_forecasts['Month'] = agg_forecasts['Date'].dt.to_period('M').astype(str)
        monthly_forecasts = agg_forecasts.groupby('Month').agg({'Forecast': 'sum'}).reset_index()
        plt.bar(monthly_forecasts['Month'], monthly_forecasts['Forecast'], color='orange')
        plt.title('Monthly Forecast')
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        time_scales_path = os.path.join(output_dir, 'forecast_time_scales.png')
        plt.savefig(time_scales_path, bbox_inches='tight')
        plt.close()
        plot_paths.append(time_scales_path)
        
        logger.info(f"Created {len(plot_paths)} extended forecast visualizations")
        return plot_paths
    
    def save_item_statistics(self, output_file=None):
        """
        Save item statistics to a CSV file
        
        Args:
            output_file: Path to save the CSV file
            
        Returns:
            str: Path to saved file
        """
        if self.item_stats is None:
            logger.warning("No item statistics to save")
            return None
            
        if output_file is None:
            output_file = ITEM_STATISTICS_FILE
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert complex dictionary columns to strings
        stats_df = self.item_stats.copy()
        
        # Convert dictionary columns to strings
        for col in ['Weekly_Pattern', 'Monthly_Pattern', 'Weather_Impact', 'Day_Of_Week_Effect']:
            if col in stats_df.columns:
                stats_df[col] = stats_df[col].apply(lambda x: str(x) if x is not None else None)
        
        # Save to CSV
        stats_df.to_csv(output_file, index=False)
        logger.info(f"Saved item statistics to {output_file}")
        return output_file
    
    def save_extended_forecasts(self, output_file=None):
        """
        Save extended forecasts to a CSV file
        
        Args:
            output_file: Path to save the CSV file
            
        Returns:
            str: Path to saved file
        """
        if self.extended_forecasts is None:
            logger.warning("No extended forecasts to save")
            return None
            
        if output_file is None:
            output_file = EXTENDED_FORECASTS_FILE
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to CSV
        self.extended_forecasts.to_csv(output_file, index=False)
        logger.info(f"Saved extended forecasts to {output_file}")
        return output_file

    def create_plotly_forecast_plot(self, store_id, item_id, output_dir=None):
        """
        Create an interactive Plotly forecast plot
        
        Args:
            store_id: Store ID
            item_id: Item ID
            output_dir: Directory to save the plot
            
        Returns:
            str: Path to the saved plot
        """
        if self.df is None or self.extended_forecasts is None:
            return None
            
        if output_dir is None:
            output_dir = self.plotly_dir
            
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get item data
        item_data = self.df[(self.df['Store_Id'] == store_id) & (self.df['Item'] == item_id)]
        if len(item_data) == 0:
            return None
            
        # Get forecast data
        forecasts = self.extended_forecasts[
            (self.extended_forecasts['Store_Id'] == store_id) & 
            (self.extended_forecasts['Item'] == item_id)
        ]
        
        if len(forecasts) == 0:
            return None
            
        # Get product name
        product_name = item_data['Product'].iloc[0]
        
        # Split into original and extended
        if 'Is_Extended' in forecasts.columns:
            original = forecasts[~forecasts['Is_Extended']].sort_values('Date')
            extended = forecasts[forecasts['Is_Extended']].sort_values('Date')
        else:
            original = forecasts.sort_values('Date')
            extended = pd.DataFrame()  # Empty DataFrame
        
        # Create figure
        fig = go.Figure()
        
        # Add historical sales
        fig.add_trace(go.Scatter(
            x=item_data['Date'],
            y=item_data['Sales'],
            mode='lines',
            name='Historical Sales',
            line=dict(color='gray'),
            hovertemplate='%{x}<br>Sales: %{y:.1f}<extra></extra>'
        ))
        
        # Add original forecast
        if len(original) > 0:
            fig.add_trace(go.Scatter(
                x=original['Date'],
                y=original['Forecast'],
                mode='lines',
                name='Original Forecast',
                line=dict(color='blue'),
                hovertemplate='%{x}<br>Forecast: %{y:.1f}<extra></extra>'
            ))
            
            # Add confidence intervals
            if 'Lower_Bound' in original.columns and 'Upper_Bound' in original.columns:
                fig.add_trace(go.Scatter(
                    x=original['Date'],
                    y=original['Upper_Bound'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=original['Date'],
                    y=original['Lower_Bound'],
                    mode='lines',
                    name='Lower Bound',
                    fill='tonexty',
                    fillcolor='rgba(0, 0, 255, 0.2)',
                    line=dict(width=0),
                    showlegend=False
                ))
        
        # Add extended forecast
        if len(extended) > 0:
            fig.add_trace(go.Scatter(
                x=extended['Date'],
                y=extended['Forecast'],
                mode='lines',
                name='Extended Forecast',
                line=dict(color='red'),
                hovertemplate='%{x}<br>Forecast: %{y:.1f}<extra></extra>'
            ))
            
            # Add confidence intervals
            if 'Lower_Bound' in extended.columns and 'Upper_Bound' in extended.columns:
                fig.add_trace(go.Scatter(
                    x=extended['Date'],
                    y=extended['Upper_Bound'],
                    mode='lines',
                    name='Extended Upper Bound',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=extended['Date'],
                    y=extended['Lower_Bound'],
                    mode='lines',
                    name='Extended Lower Bound',
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    line=dict(width=0),
                    showlegend=False
                ))
        
        # Add transition point between original and extended
        if len(original) > 0 and len(extended) > 0:
            transition_date = original['Date'].max()
            
            fig.add_vline(
                x=transition_date,
                line_dash="dash",
                line_color="black",
                opacity=0.5,
                annotation_text="Forecast Extension Start",
                annotation_position="top right"
            )
        
        # Update layout
        fig.update_layout(
            title=f"Extended Forecast for {product_name} (Store {store_id}, Item {item_id})",
            xaxis_title="Date",
            yaxis_title="Sales",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified",
            template="plotly_white"
        )
        
        # Save as HTML file
        output_file = os.path.join(output_dir, f"forecast_{store_id}_{item_id}.html")
        fig.write_html(output_file, include_plotlyjs='cdn')
        logger.info(f"Created interactive forecast plot at {output_file}")
        
        return output_file
    
    def create_plotly_price_sensitivity(self, store_id, item_id, output_dir=None):
        """
        Create an interactive Plotly price sensitivity curve
        
        Args:
            store_id: Store ID
            item_id: Item ID
            output_dir: Directory to save the plot
            
        Returns:
            str: Path to the saved plot
        """
        if self.df is None:
            return None
            
        if output_dir is None:
            output_dir = self.plotly_dir
            
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get item data
        item_data = self.df[(self.df['Store_Id'] == store_id) & (self.df['Item'] == item_id)]
        if len(item_data) == 0 or 'Price' not in item_data.columns:
            return None
            
        # Get product name
        product_name = item_data['Product'].iloc[0]
        
        # Group by price and calculate average sales
        price_groups = item_data.groupby('Price')['Sales'].mean().reset_index()
        if len(price_groups) < 2:  # Need at least 2 price points
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot of actual data
        fig.add_trace(go.Scatter(
            x=price_groups['Price'],
            y=price_groups['Sales'],
            mode='markers',
            name='Observed Data',
            marker=dict(size=10, color='blue'),
            hovertemplate='Price: $%{x:.2f}<br>Avg Sales: %{y:.1f}<extra></extra>'
        ))
        
        # Try to fit elasticity curve if there are enough data points
        if len(price_groups) >= 3:
            try:
                # Log-log regression for elasticity
                log_price = np.log(price_groups['Price'])
                log_sales = np.log(price_groups['Sales'].replace(0, 0.01))  # Avoid log(0)
                
                # Simple linear regression
                coeffs = np.polyfit(log_price, log_sales, 1)
                elasticity = coeffs[0]
                intercept = np.exp(coeffs[1])
                
                # Generate points for the elasticity curve
                price_range = np.linspace(price_groups['Price'].min() * 0.9, 
                                         price_groups['Price'].max() * 1.1, 
                                         100)
                
                # Calculate curve points using elasticity
                curve_sales = intercept * (price_range ** elasticity)
                
                # Add the fitted curve
                fig.add_trace(go.Scatter(
                    x=price_range,
                    y=curve_sales,
                    mode='lines',
                    name=f'Elasticity Curve (e={elasticity:.2f})',
                    line=dict(color='red'),
                    hovertemplate='Price: $%{x:.2f}<br>Predicted Sales: %{y:.1f}<extra></extra>'
                ))
                
                # Calculate and add revenue curve
                revenue = price_range * curve_sales
                max_revenue_idx = np.argmax(revenue)
                optimal_price = price_range[max_revenue_idx]
                
                fig.add_trace(go.Scatter(
                    x=price_range,
                    y=revenue,
                    mode='lines',
                    name='Revenue',
                    line=dict(color='green', dash='dash'),
                    yaxis='y2',
                    hovertemplate='Price: $%{x:.2f}<br>Revenue: $%{y:.2f}<extra></extra>'
                ))
                
                # Add marker for optimal price
                fig.add_vline(
                    x=optimal_price,
                    line_dash="dot",
                    line_color="green",
                    annotation_text=f"Revenue-Optimal Price: ${optimal_price:.2f}",
                    annotation_position="top left"
                )
                
            except Exception as e:
                logger.warning(f"Error calculating elasticity curve: {e}")
        
        # Update layout
        fig.update_layout(
            title=f"Price Sensitivity for {product_name} (Store {store_id}, Item {item_id})",
            xaxis_title="Price ($)",
            yaxis_title="Average Sales",
            yaxis2=dict(
                title="Revenue ($)",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="closest",
            template="plotly_white"
        )
        
        # Save as HTML file
        output_file = os.path.join(output_dir, f"price_sensitivity_{store_id}_{item_id}.html")
        fig.write_html(output_file, include_plotlyjs='cdn')
        logger.info(f"Created interactive price sensitivity plot at {output_file}")
        
        return output_file
    
    def create_plotly_visualizations(self):
        """
        Create interactive Plotly visualizations for extended forecasts and item statistics
        
        Returns:
            list: Paths to created visualization files
        """
        if self.df is None or self.extended_forecasts is None or self.item_stats is None:
            return []
            
        # Create directory for Plotly visualizations
        os.makedirs(self.plotly_dir, exist_ok=True)
        
        logger.info("Creating interactive Plotly visualizations")
        plot_paths = []
        
        # Create forecast plots for top items by sales
        top_items = self.item_stats.sort_values('Total_Sales', ascending=False).head(5)
        for _, row in top_items.iterrows():
            store_id = row['Store_Id']
            item_id = row['Item']
            
            # Create forecast plot
            plot_path = self.create_plotly_forecast_plot(store_id, item_id)
            if plot_path:
                plot_paths.append(plot_path)
            
            # Create price sensitivity plot
            plot_path = self.create_plotly_price_sensitivity(store_id, item_id)
            if plot_path:
                plot_paths.append(plot_path)
            
        # Create aggregate forecast plot
        if len(self.extended_forecasts) > 0:
            # Aggregate forecasts by date
            agg_forecasts = self.extended_forecasts.groupby('Date').agg({
                'Forecast': 'sum',
                'Lower_Bound': 'sum',
                'Upper_Bound': 'sum',
                'Is_Extended': 'any'  # True if any forecast on this date is extended
            }).reset_index()
            
            # Create figure
            fig = go.Figure()
            
            # Split into original and extended
            if 'Is_Extended' in agg_forecasts.columns:
                original = agg_forecasts[~agg_forecasts['Is_Extended']]
                extended = agg_forecasts[agg_forecasts['Is_Extended']]
            else:
                original = agg_forecasts
                extended = pd.DataFrame()  # Empty DataFrame
            
            # Add original forecast
            fig.add_trace(go.Scatter(
                x=original['Date'],
                y=original['Forecast'],
                mode='lines',
                name='Original Forecast',
                line=dict(color='blue'),
                hovertemplate='%{x}<br>Forecast: %{y:.1f}<extra></extra>'
            ))
            
            # Add confidence intervals
            if 'Lower_Bound' in original.columns and 'Upper_Bound' in original.columns:
                fig.add_trace(go.Scatter(
                    x=original['Date'],
                    y=original['Upper_Bound'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=original['Date'],
                    y=original['Lower_Bound'],
                    mode='lines',
                    name='Lower Bound',
                    fill='tonexty',
                    fillcolor='rgba(0, 0, 255, 0.2)',
                    line=dict(width=0),
                    showlegend=False
                ))
            
            # Add extended forecast
            if len(extended) > 0:
                fig.add_trace(go.Scatter(
                    x=extended['Date'],
                    y=extended['Forecast'],
                    mode='lines',
                    name='Extended Forecast',
                    line=dict(color='red'),
                    hovertemplate='%{x}<br>Forecast: %{y:.1f}<extra></extra>'
                ))
                
                # Add confidence intervals
                if 'Lower_Bound' in extended.columns and 'Upper_Bound' in extended.columns:
                    fig.add_trace(go.Scatter(
                        x=extended['Date'],
                        y=extended['Upper_Bound'],
                        mode='lines',
                        name='Extended Upper Bound',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=extended['Date'],
                        y=extended['Lower_Bound'],
                        mode='lines',
                        name='Extended Lower Bound',
                        fill='tonexty',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        line=dict(width=0),
                        showlegend=False
                    ))
            
            # Add transition point
            if len(original) > 0 and len(extended) > 0:
                transition_date = original['Date'].max()
                
                fig.add_vline(
                    x=transition_date,
                    line_dash="dash",
                    line_color="black",
                    opacity=0.5,
                    annotation_text="Forecast Extension Start",
                    annotation_position="top right"
                )
            
            # Update layout
            fig.update_layout(
                title="Total Sales Forecast (Original + Extended)",
                xaxis_title="Date",
                yaxis_title="Total Sales",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode="x unified",
                template="plotly_white"
            )
            
            # Save as HTML file
            output_file = os.path.join(self.plotly_dir, "total_forecast.html")
            fig.write_html(output_file, include_plotlyjs='cdn')
            logger.info(f"Created interactive total forecast plot at {output_file}")
            plot_paths.append(output_file)
        
        return plot_paths

def run_item_statistics_analysis(data_file=COMBINED_DATA_FILE, forecast_file=None, days_to_extend=60, use_plotly=True):
    """
    Run the complete item statistics analysis and extended forecasting process
    
    Args:
        data_file: Path to the data file
        forecast_file: Path to the forecast file (optional)
        days_to_extend: Number of days to extend forecasts
        use_plotly: Whether to create interactive Plotly visualizations
        
    Returns:
        tuple: (item_stats, extended_forecasts)
    """
    logger.info("Starting item statistics analysis and extended forecasting")
    
    analyzer = ItemStatisticsAnalyzer(data_file)
    
    try:
        # Step 1: Load data
        analyzer.load_data()
        
        # Step 2: Load forecasts
        analyzer.load_forecasts(forecast_file)
        
        # Step 3: Calculate item statistics
        item_stats = analyzer.calculate_item_statistics()
        
        # Step 4: Create item statistics visualizations
        analyzer.create_item_statistics_visualizations()
        
        # Step 5: Extend forecasts
        extended_forecasts = analyzer.extend_forecast(days_to_extend)
        
        # Step 6: Create extended forecast visualizations
        analyzer.create_extended_forecast_visualizations()
        
        # Step 7: Create interactive Plotly visualizations if requested
        if use_plotly:
            analyzer.create_plotly_visualizations()
        
        # Step 7: Save results
        analyzer.save_item_statistics()
        analyzer.save_extended_forecasts()
        
        logger.info("Item statistics analysis and extended forecasting completed successfully")
        return item_stats, extended_forecasts
        
    except Exception as e:
        logger.error(f"Error in item statistics analysis: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Item Statistics Analysis with Extended Forecasting')
    parser.add_argument('--data', type=str, default=COMBINED_DATA_FILE, help='Path to data file')
    parser.add_argument('--forecast', type=str, help='Path to forecast file')
    parser.add_argument('--days', type=int, default=60, help='Number of days to extend forecasts')
    parser.add_argument('--no-plotly', action='store_true', help='Disable Plotly visualizations')
    
    args = parser.parse_args()
    
    run_item_statistics_analysis(
        data_file=args.data,
        forecast_file=args.forecast,
        days_to_extend=args.days,
        use_plotly=not args.no_plotly
    )