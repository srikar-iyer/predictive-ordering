"""
Category-based statistics with extended forecasting module.
This module provides category-level statistics and aggregation of item-level
statistics to provide comprehensive category analysis.
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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import (
    COMBINED_DATA_FILE, MODELS_DIR, STATIC_DIR, 
    WEIGHTED_ARIMA_FORECASTS_FILE, ARIMA_FORECASTS_FILE,
    ITEM_STATISTICS_FILE, EXTENDED_FORECASTS_FILE, CATEGORY_STATISTICS_FILE
)
from src.models.item_statistics import ItemStatisticsAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('category_statistics')

class CategoryStatisticsAnalyzer:
    """
    Class for analyzing detailed category-level statistics and extended forecasting
    """
    def __init__(self, data_path=COMBINED_DATA_FILE, item_stats_path=ITEM_STATISTICS_FILE):
        """
        Initialize the category statistics analyzer
        
        Args:
            data_path: Path to the combined data file
            item_stats_path: Path to the item statistics file
        """
        self.data_path = data_path
        self.item_stats_path = item_stats_path
        self.df = None
        self.item_stats_df = None
        self.category_stats = None
        self.extended_forecasts = None
        # Setup path for Plotly visualizations
        self.plotly_dir = os.path.join(STATIC_DIR, 'plotly_visualizations')
        self.item_analyzer = ItemStatisticsAnalyzer(data_path)
        
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
            
            # Make sure category information exists or try to add it
            if 'Category_Name' not in self.df.columns:
                logger.warning("Category_Name not found in data. Checking for product category mapping.")
                # Try to infer category from product name or other sources
                self._infer_categories()
                
            logger.info(f"Loaded {len(self.df)} records from {self.data_path}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def load_item_statistics(self):
        """
        Load item statistics from CSV file
        
        Returns:
            DataFrame: The loaded item statistics
        """
        logger.info(f"Loading item statistics from {self.item_stats_path}")
        try:
            if not os.path.exists(self.item_stats_path):
                logger.warning(f"Item statistics file not found: {self.item_stats_path}")
                logger.info("Calculating item statistics...")
                # Calculate item statistics using the item analyzer
                if self.df is None:
                    self.load_data()
                self.item_analyzer.df = self.df
                self.item_stats_df = self.item_analyzer.calculate_item_statistics()
                self.item_analyzer.save_item_statistics(self.item_stats_path)
            else:
                self.item_stats_df = pd.read_csv(self.item_stats_path)
                
                # Convert complex string columns back to dictionaries
                for col in ['Weekly_Pattern', 'Monthly_Pattern', 'Weather_Impact', 'Day_Of_Week_Effect']:
                    if col in self.item_stats_df.columns:
                        self.item_stats_df[col] = self.item_stats_df[col].apply(
                            lambda x: eval(x) if isinstance(x, str) else x
                        )
            
            logger.info(f"Loaded item statistics with {len(self.item_stats_df)} records")
            return self.item_stats_df
        except Exception as e:
            logger.error(f"Error loading item statistics: {str(e)}")
            raise
    
    def _infer_categories(self):
        """
        Infer category information if not available in the dataset
        """
        # Check if we already have a category field
        if 'Category_Name' in self.df.columns:
            return
        
        try:
            # Try to load from frozen_pizza_only.csv which has Category_Name
            category_file = os.path.join(os.path.dirname(self.data_path), "frozen_pizza_only.csv")
            if os.path.exists(category_file):
                category_df = pd.read_csv(category_file)
                if 'Category_Name' in category_df.columns:
                    # Create mapping of items to categories
                    item_to_category = {}
                    for _, row in category_df.iterrows():
                        item = row['Item']
                        category = row['Category_Name']
                        item_to_category[item] = category
                    
                    # Apply mapping to main dataframe
                    self.df['Category_Name'] = self.df['Item'].map(item_to_category)
                    logger.info(f"Added Category_Name from {category_file}")
                    return
            
            # If we couldn't load from a file, we need to create categories
            # For this example, we'll use "Frozen Pizza" as the default category
            logger.warning("Using default category 'Frozen Pizza' for all items")
            self.df['Category_Name'] = "Frozen Pizza"
            
        except Exception as e:
            logger.error(f"Error inferring categories: {str(e)}")
            # If all else fails, use a default category
            self.df['Category_Name'] = "Unknown"
    
    def calculate_category_statistics(self):
        """
        Calculate comprehensive category-level statistics
        
        Returns:
            DataFrame: Category-level statistics
        """
        logger.info("Calculating comprehensive category-level statistics")
        
        # Make sure data is loaded
        if self.df is None:
            self.load_data()
            
        # Make sure item statistics are loaded
        if self.item_stats_df is None:
            self.load_item_statistics()
            
        # Initialize list for storing category statistics
        category_stats_list = []
        
        # Process each store-category combination
        store_categories = self.df[['Store_Id', 'Category_Name']].drop_duplicates()
        
        for _, row in store_categories.iterrows():
            store_id = row['Store_Id']
            category_name = row['Category_Name']
            
            logger.info(f"Calculating statistics for Store {store_id}, Category {category_name}")
            
            # Get data for this store-category
            category_df = self.df[(self.df['Store_Id'] == store_id) & (self.df['Category_Name'] == category_name)]
            
            # Skip if not enough data
            if len(category_df) < 30:
                logger.warning(f"Insufficient data for Store {store_id}, Category {category_name}")
                continue
            
            # Get item statistics for this store-category
            items_in_category = category_df['Item'].unique()
            category_items_stats = self.item_stats_df[
                (self.item_stats_df['Store_Id'] == store_id) & 
                (self.item_stats_df['Item'].isin(items_in_category))
            ]
            
            # Skip if no item statistics available
            if len(category_items_stats) == 0:
                logger.warning(f"No item statistics available for Store {store_id}, Category {category_name}")
                continue
            
            # Basic information
            last_date = category_df['Date'].max()
            first_date = category_df['Date'].min()
            days_of_data = (last_date - first_date).days + 1
            item_count = len(items_in_category)
            
            # Create daily aggregated data for the category
            daily_agg = category_df.groupby(['Store_Id', 'Date']).agg({
                'Sales': 'sum',
                'Retail_Revenue': 'sum',
                'Cost': 'sum',
                'Profit': 'sum',
                'Stock_Level': 'sum',
                'Price': 'mean',
                'Promotion': 'max'  # If any product is on promotion
            }).reset_index()
            
            # Sales statistics
            total_sales = daily_agg['Sales'].sum()
            avg_daily_sales = daily_agg['Sales'].mean()
            sales_std = daily_agg['Sales'].std()
            sales_cv = sales_std / avg_daily_sales if avg_daily_sales > 0 else np.nan
            sales_min = daily_agg['Sales'].min()
            sales_max = daily_agg['Sales'].max()
            sales_median = daily_agg['Sales'].median()
            
            # Calculate percentiles
            sales_p25 = daily_agg['Sales'].quantile(0.25)
            sales_p75 = daily_agg['Sales'].quantile(0.75)
            sales_p90 = daily_agg['Sales'].quantile(0.90)
            sales_p95 = daily_agg['Sales'].quantile(0.95)
            
            # Calculate zero sales days
            zero_sales_days = (daily_agg['Sales'] == 0).sum()
            zero_sales_pct = zero_sales_days / len(daily_agg) * 100
            
            # Add day and month information
            daily_agg['Day_Of_Week'] = daily_agg['Date'].dt.dayofweek
            daily_agg['Month'] = daily_agg['Date'].dt.month
            
            # Time-based sales patterns
            daily_agg['Day_Name'] = daily_agg['Date'].dt.day_name()
            daily_avg = daily_agg.groupby('Day_Of_Week')['Sales'].mean().to_dict()
            weekly_pattern = {idx: daily_avg.get(idx, 0) for idx in range(7)}
            
            # Monthly pattern
            daily_agg['Month_Name'] = daily_agg['Date'].dt.month_name()
            monthly_avg = daily_agg.groupby('Month')['Sales'].mean().to_dict()
            monthly_pattern = {month: monthly_avg.get(month, 0) for month in range(1, 13)}
            
            # Sales trend calculation
            daily_agg = daily_agg.sort_values('Date')
            dates_num = np.arange(len(daily_agg))
            
            # Simple linear trend
            if len(dates_num) > 1:
                import statsmodels.api as sm
                trend_model = sm.OLS(daily_agg['Sales'], sm.add_constant(dates_num)).fit()
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
            
            # Price statistics
            avg_price = daily_agg['Price'].mean()
            price_std = daily_agg['Price'].std()
            price_cv = price_std / avg_price if avg_price > 0 else np.nan
            price_min = daily_agg['Price'].min()
            price_max = daily_agg['Price'].max()
            
            # Calculate promotion frequency
            promotion_days = daily_agg['Promotion'].sum()
            promotion_pct = promotion_days / len(daily_agg) * 100
            
            # Stock metrics
            avg_stock = daily_agg['Stock_Level'].mean()
            current_stock = daily_agg.iloc[-1]['Stock_Level']
            # Calculate stock coverage as weeks
            stock_coverage = current_stock / (avg_daily_sales * 7) if avg_daily_sales > 0 else np.nan
            
            # Profit metrics
            total_profit = daily_agg['Profit'].sum()
            avg_profit = daily_agg['Profit'].mean()
            total_revenue = daily_agg['Retail_Revenue'].sum()
            profit_margin = total_profit / total_revenue if total_revenue > 0 else np.nan
            
            # Weather impact analysis
            weather_impact = {}
            if 'Weather' in daily_agg.columns:
                weather_groups = daily_agg.groupby('Weather')
                baseline_sales = daily_agg[daily_agg['Weather'] == 'Normal']['Sales'].mean()
                
                for weather, group in weather_groups:
                    weather_sales = group['Sales'].mean()
                    weather_impact[weather] = weather_sales / baseline_sales if baseline_sales > 0 else 1.0
            
            # Holiday impact analysis
            holiday_impact = None
            if 'Is_Holiday' in daily_agg.columns:
                non_holiday_sales = daily_agg[daily_agg['Is_Holiday'] == 0]['Sales'].mean()
                holiday_sales = daily_agg[daily_agg['Is_Holiday'] == 1]['Sales'].mean()
                
                if non_holiday_sales > 0:
                    holiday_impact = holiday_sales / non_holiday_sales
            
            # Calculate day-of-week effect
            dow_effect = {}
            for day in range(7):
                day_avg = daily_agg[daily_agg['Day_Of_Week'] == day]['Sales'].mean()
                dow_effect[day] = day_avg / avg_daily_sales if avg_daily_sales > 0 else 1.0
            
            # Item-level metrics aggregated to category level
            top_selling_items = category_items_stats.sort_values('Total_Sales', ascending=False).head(5)['Item'].tolist()
            top_profit_items = category_items_stats.sort_values('Total_Profit', ascending=False).head(5)['Item'].tolist()
            top_margin_items = category_items_stats.sort_values('Profit_Margin', ascending=False).head(5)['Item'].tolist()
            
            # Calculate category-specific metrics
            item_sales_share = {}
            for _, item_row in category_items_stats.iterrows():
                item_id = item_row['Item']
                item_sales = item_row['Total_Sales']
                item_sales_share[item_id] = item_sales / total_sales if total_sales > 0 else 0
            
            # Category-level elasticity (weighted average of item elasticities)
            weighted_elasticity = 0
            elasticity_count = 0
            elasticity_items = {}
            
            # Check if Price_Elasticity data is available in the item stats
            if 'Elasticity' in category_items_stats.columns:
                for _, item_row in category_items_stats.iterrows():
                    if not pd.isna(item_row['Elasticity']):
                        item_id = item_row['Item']
                        item_elasticity = item_row['Elasticity']
                        item_weight = item_sales_share.get(item_id, 0)
                        
                        weighted_elasticity += item_elasticity * item_weight
                        elasticity_items[item_id] = item_elasticity
                        elasticity_count += 1
            
            if elasticity_count > 0:
                category_elasticity = weighted_elasticity
            else:
                category_elasticity = np.nan
            
            # Create category stats dictionary
            category_stats = {
                'Store_Id': store_id,
                'Category_Name': category_name,
                'First_Date': first_date,
                'Last_Date': last_date,
                'Days_Of_Data': days_of_data,
                'Item_Count': item_count,
                'Total_Sales': total_sales,
                'Avg_Daily_Sales': avg_daily_sales,
                'Sales_StdDev': sales_std,
                'Sales_CV': sales_cv,
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
                'Top_Selling_Items': top_selling_items,
                'Top_Profit_Items': top_profit_items,
                'Top_Margin_Items': top_margin_items,
                'Item_Sales_Share': item_sales_share,
                'Category_Elasticity': category_elasticity,
                'Elasticity_Items': elasticity_items
            }
            
            # Add to list
            category_stats_list.append(category_stats)
        
        # Create DataFrame
        self.category_stats = pd.DataFrame(category_stats_list)
        
        logger.info(f"Calculated statistics for {len(self.category_stats)} store-category combinations")
        return self.category_stats
    
    def create_category_statistics_visualizations(self, output_dir=None):
        """
        Create visualizations for category statistics
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            list: Paths to created visualization files
        """
        if self.category_stats is None:
            logger.warning("No category statistics available for visualization")
            return []
            
        if output_dir is None:
            output_dir = os.path.join(STATIC_DIR, 'images', 'category_statistics')
            
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Creating category statistics visualizations in {output_dir}")
        
        plot_paths = []
        
        # 1. Sales distribution across categories
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Category_Name', y='Total_Sales', data=self.category_stats)
        plt.title('Total Sales by Category')
        plt.xlabel('Category')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        sales_dist_path = os.path.join(output_dir, 'category_sales_distribution.png')
        plt.savefig(sales_dist_path, bbox_inches='tight')
        plt.close()
        plot_paths.append(sales_dist_path)
        
        # 2. Items per category
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Category_Name', y='Item_Count', data=self.category_stats)
        plt.title('Number of Items per Category')
        plt.xlabel('Category')
        plt.ylabel('Item Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        items_count_path = os.path.join(output_dir, 'category_item_counts.png')
        plt.savefig(items_count_path, bbox_inches='tight')
        plt.close()
        plot_paths.append(items_count_path)
        
        # 3. Profit margin by category
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Category_Name', y='Profit_Margin', data=self.category_stats)
        plt.title('Profit Margin by Category')
        plt.xlabel('Category')
        plt.ylabel('Profit Margin')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        profit_margin_path = os.path.join(output_dir, 'category_profit_margins.png')
        plt.savefig(profit_margin_path, bbox_inches='tight')
        plt.close()
        plot_paths.append(profit_margin_path)
        
        # 4. Category sales over time (if we have time series data)
        if self.df is not None:
            # For each category, get time series data
            for _, cat_row in self.category_stats.iterrows():
                store_id = cat_row['Store_Id']
                category_name = cat_row['Category_Name']
                
                # Filter data for this category
                cat_data = self.df[(self.df['Store_Id'] == store_id) & 
                                  (self.df['Category_Name'] == category_name)]
                
                if len(cat_data) == 0:
                    continue
                    
                # Aggregate by date
                cat_data = cat_data.groupby('Date')['Sales'].sum().reset_index()
                cat_data = cat_data.sort_values('Date')
                
                # Create time series plot
                plt.figure(figsize=(14, 7))
                plt.plot(cat_data['Date'], cat_data['Sales'])
                plt.title(f'Sales Trend for {category_name}')
                plt.xlabel('Date')
                plt.ylabel('Sales')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save plot
                time_series_path = os.path.join(output_dir, f'sales_trend_{store_id}_{category_name.replace(" ", "_")}.png')
                plt.savefig(time_series_path, bbox_inches='tight')
                plt.close()
                plot_paths.append(time_series_path)
        
        # 5. Category item share pie chart
        for _, cat_row in self.category_stats.iterrows():
            store_id = cat_row['Store_Id']
            category_name = cat_row['Category_Name']
            item_sales_share = cat_row['Item_Sales_Share']
            
            if not item_sales_share or len(item_sales_share) == 0:
                continue
                
            # Get top 10 items by sales share
            top_items = dict(sorted(item_sales_share.items(), key=lambda x: x[1], reverse=True)[:10])
            
            # If we have the item_stats_df, get product names
            if self.item_stats_df is not None:
                item_names = {}
                for item_id in top_items.keys():
                    item_row = self.item_stats_df[(self.item_stats_df['Store_Id'] == store_id) & 
                                                (self.item_stats_df['Item'] == float(item_id))]
                    if len(item_row) > 0 and 'Product' in item_row.columns:
                        product_name = item_row['Product'].iloc[0]
                        item_names[item_id] = product_name
                    else:
                        item_names[item_id] = f"Item {item_id}"
            else:
                item_names = {item_id: f"Item {item_id}" for item_id in top_items.keys()}
            
            # Create pie chart
            plt.figure(figsize=(12, 8))
            plt.pie(list(top_items.values()), labels=[item_names.get(item_id, item_id) for item_id in top_items.keys()], 
                   autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title(f'Sales Share by Item for {category_name}')
            plt.tight_layout()
            
            # Save pie chart
            pie_chart_path = os.path.join(output_dir, f'item_share_{store_id}_{category_name.replace(" ", "_")}.png')
            plt.savefig(pie_chart_path, bbox_inches='tight')
            plt.close()
            plot_paths.append(pie_chart_path)
        
        # 6. Day of week pattern comparison across categories
        if len(self.category_stats) > 1:
            plt.figure(figsize=(14, 8))
            
            # Extract day of week effects
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_data = []
            
            for _, row in self.category_stats.iterrows():
                category = row['Category_Name']
                dow_effect = row['Day_Of_Week_Effect']
                
                if dow_effect:
                    for day, effect in dow_effect.items():
                        dow_data.append({
                            'Category': category,
                            'Day': day_names[int(day)],
                            'Effect': effect
                        })
            
            if dow_data:
                dow_df = pd.DataFrame(dow_data)
                
                # Create plot
                sns.barplot(x='Day', y='Effect', hue='Category', data=dow_df)
                plt.title('Day of Week Effect by Category')
                plt.xlabel('Day of Week')
                plt.ylabel('Relative Sales (1.0 = Average)')
                plt.axhline(y=1.0, color='gray', linestyle='--')
                plt.legend(title='Category')
                plt.tight_layout()
                
                dow_path = os.path.join(output_dir, 'category_dow_comparison.png')
                plt.savefig(dow_path, bbox_inches='tight')
                plt.close()
                plot_paths.append(dow_path)
        
        # 7. Category elasticity comparison
        plt.figure(figsize=(12, 6))
        valid_elasticities = self.category_stats[~self.category_stats['Category_Elasticity'].isna()]
        
        if len(valid_elasticities) > 0:
            sns.barplot(x='Category_Name', y='Category_Elasticity', data=valid_elasticities)
            plt.title('Price Elasticity by Category')
            plt.xlabel('Category')
            plt.ylabel('Price Elasticity')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            elasticity_path = os.path.join(output_dir, 'category_elasticity.png')
            plt.savefig(elasticity_path, bbox_inches='tight')
            plt.close()
            plot_paths.append(elasticity_path)
        
        logger.info(f"Created {len(plot_paths)} category statistics visualizations")
        return plot_paths
    
    def create_category_forecast_aggregation(self, extended_forecasts_df=None):
        """
        Create category-level forecast by aggregating item-level forecasts
        
        Args:
            extended_forecasts_df: DataFrame with item-level forecasts
            
        Returns:
            DataFrame: Category-level forecasts
        """
        logger.info("Creating category-level forecast aggregation")
        
        # Load extended forecasts if not provided
        if extended_forecasts_df is None:
            if not os.path.exists(EXTENDED_FORECASTS_FILE):
                logger.warning(f"Extended forecasts file not found: {EXTENDED_FORECASTS_FILE}")
                return None
            
            try:
                extended_forecasts_df = pd.read_csv(EXTENDED_FORECASTS_FILE)
                
                # Convert date to datetime
                if 'Date' in extended_forecasts_df.columns:
                    extended_forecasts_df['Date'] = pd.to_datetime(extended_forecasts_df['Date'])
                    
                # Make sure Is_Extended column exists (for legacy compatibility)
                if 'Is_Extended' not in extended_forecasts_df.columns:
                    extended_forecasts_df['Is_Extended'] = False
            except Exception as e:
                logger.error(f"Error loading extended forecasts: {e}")
                return None
        
        # Make sure data is loaded
        if self.df is None:
            self.load_data()
        
        # Create item to category mapping
        item_to_category = {}
        for _, row in self.df[['Item', 'Category_Name']].drop_duplicates().iterrows():
            item_to_category[row['Item']] = row['Category_Name']
        
        # Add category to forecasts
        extended_forecasts_df['Category_Name'] = extended_forecasts_df['Item'].map(item_to_category)
        
        # Group by store, category, and date
        category_forecasts = extended_forecasts_df.groupby(
            ['Store_Id', 'Category_Name', 'Date', 'Is_Extended']
        ).agg({
            'Forecast': 'sum',
            'Lower_Bound': 'sum',
            'Upper_Bound': 'sum',
            'Days_In_Future': 'mean',
            'Forecast_Generated': 'max'
        }).reset_index()
        
        # Save as class member
        self.category_forecasts = category_forecasts
        
        logger.info(f"Created category forecasts with {len(category_forecasts)} records")
        return category_forecasts
    
    def create_plotly_forecast_plot(self, store_id, category_name, output_dir=None):
        """
        Create an interactive Plotly forecast plot for a category
        
        Args:
            store_id: Store ID
            category_name: Category name
            output_dir: Directory to save the plot
            
        Returns:
            str: Path to the saved plot
        """
        if self.category_forecasts is None:
            logger.warning("No category forecasts available")
            return None
            
        if output_dir is None:
            output_dir = self.plotly_dir
            
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get category forecast data
        forecast_data = self.category_forecasts[
            (self.category_forecasts['Store_Id'] == store_id) & 
            (self.category_forecasts['Category_Name'] == category_name)
        ]
        
        if len(forecast_data) == 0:
            logger.warning(f"No forecast data available for Store {store_id}, Category {category_name}")
            return None
        
        # Get historical data
        historical_data = None
        if self.df is not None:
            historical_data = self.df[
                (self.df['Store_Id'] == store_id) & 
                (self.df['Category_Name'] == category_name)
            ]
            
            if len(historical_data) > 0:
                # Aggregate by date
                historical_data = historical_data.groupby('Date')['Sales'].sum().reset_index()
        
        # Create figure
        fig = go.Figure()
        
        # Add historical sales to the plot if available
        if historical_data is not None and len(historical_data) > 0:
            fig.add_trace(go.Scatter(
                x=historical_data['Date'],
                y=historical_data['Sales'],
                mode='lines',
                name='Historical Sales',
                line=dict(color='gray')
            ))
        
        # Split into original and extended forecasts
        original = forecast_data[~forecast_data['Is_Extended']].sort_values('Date')
        extended = forecast_data[forecast_data['Is_Extended']].sort_values('Date')
        
        # Add original forecast
        if len(original) > 0:
            fig.add_trace(go.Scatter(
                x=original['Date'],
                y=original['Forecast'],
                mode='lines',
                name='Original Forecast',
                line=dict(color='blue')
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
                line=dict(color='red')
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
            title=f"Category Forecast: {category_name} (Store {store_id})",
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
        output_file = os.path.join(output_dir, f"category_forecast_{store_id}_{category_name.replace(' ', '_')}.html")
        fig.write_html(output_file, include_plotlyjs='cdn')
        logger.info(f"Created interactive category forecast plot at {output_file}")
        
        return output_file
    
    def create_plotly_category_comparison(self, store_id, output_dir=None):
        """
        Create an interactive Plotly plot comparing different categories
        
        Args:
            store_id: Store ID
            output_dir: Directory to save the plot
            
        Returns:
            str: Path to the saved plot
        """
        if self.category_stats is None:
            logger.warning("No category statistics available")
            return None
            
        if output_dir is None:
            output_dir = self.plotly_dir
            
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get store categories
        store_categories = self.category_stats[self.category_stats['Store_Id'] == store_id]
        
        if len(store_categories) == 0:
            logger.warning(f"No category statistics available for Store {store_id}")
            return None
        
        # Create subplot grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Total Sales by Category", "Profit Margin by Category", 
                          "Average Price by Category", "Stock Coverage by Category"),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                  [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Total sales by category
        fig.add_trace(
            go.Bar(
                x=store_categories['Category_Name'],
                y=store_categories['Total_Sales'],
                name='Total Sales',
                marker_color='blue'
            ),
            row=1, col=1
        )
        
        # 2. Profit margin by category
        fig.add_trace(
            go.Bar(
                x=store_categories['Category_Name'],
                y=store_categories['Profit_Margin'],
                name='Profit Margin',
                marker_color='green'
            ),
            row=1, col=2
        )
        
        # 3. Average price by category
        fig.add_trace(
            go.Bar(
                x=store_categories['Category_Name'],
                y=store_categories['Avg_Price'],
                name='Average Price',
                marker_color='orange'
            ),
            row=2, col=1
        )
        
        # 4. Stock coverage by category
        fig.add_trace(
            go.Bar(
                x=store_categories['Category_Name'],
                y=store_categories['Stock_Coverage_Weeks'],
                name='Stock Coverage (Weeks)',
                marker_color='red'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Category Comparison for Store {store_id}",
            height=800,
            showlegend=False,
            template="plotly_white"
        )
        
        # Save as HTML file
        output_file = os.path.join(output_dir, f"category_comparison_{store_id}.html")
        fig.write_html(output_file, include_plotlyjs='cdn')
        logger.info(f"Created interactive category comparison plot at {output_file}")
        
        return output_file
    
    def create_plotly_category_item_treemap(self, store_id, category_name, metric='sales', output_dir=None):
        """
        Create an interactive Plotly treemap showing item distribution within a category
        
        Args:
            store_id: Store ID
            category_name: Category name
            metric: Metric to visualize ('sales', 'profit', 'margin')
            output_dir: Directory to save the plot
            
        Returns:
            str: Path to the saved plot
        """
        if self.category_stats is None or self.item_stats_df is None:
            logger.warning("No category statistics or item statistics available")
            return None
            
        if output_dir is None:
            output_dir = self.plotly_dir
            
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get category data
        category_row = self.category_stats[
            (self.category_stats['Store_Id'] == store_id) & 
            (self.category_stats['Category_Name'] == category_name)
        ]
        
        if len(category_row) == 0:
            logger.warning(f"No data available for Store {store_id}, Category {category_name}")
            return None
            
        # Get item data for this category
        item_sales_share = category_row['Item_Sales_Share'].iloc[0]
        
        if not item_sales_share or len(item_sales_share) == 0:
            logger.warning(f"No item sales share data available for Store {store_id}, Category {category_name}")
            return None
            
        # Get item statistics for these items
        items_in_category = list(item_sales_share.keys())
        category_items_stats = self.item_stats_df[
            (self.item_stats_df['Store_Id'] == store_id) & 
            (self.item_stats_df['Item'].astype(str).isin([str(item) for item in items_in_category]))
        ]
        
        # Create data for treemap
        if len(category_items_stats) == 0:
            logger.warning(f"No item statistics available for Store {store_id}, Category {category_name}")
            return None
            
        # Prepare data based on selected metric
        if metric == 'sales':
            values = category_items_stats['Total_Sales'].values
            title = f"Sales Distribution in {category_name}"
            color_label = "Sales"
        elif metric == 'profit':
            values = category_items_stats['Total_Profit'].values
            title = f"Profit Distribution in {category_name}"
            color_label = "Profit"
        elif metric == 'margin':
            values = category_items_stats['Profit_Margin'].values
            title = f"Margin Distribution in {category_name}"
            color_label = "Margin"
        else:
            values = category_items_stats['Total_Sales'].values
            title = f"Sales Distribution in {category_name}"
            color_label = "Sales"
        
        # Create treemap
        fig = px.treemap(
            category_items_stats,
            names='Product',
            values=values,
            color=values,
            color_continuous_scale='Blues',
            title=title
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            template="plotly_white"
        )
        
        fig.update_coloraxes(colorbar_title=color_label)
        
        # Save as HTML file
        output_file = os.path.join(output_dir, f"category_treemap_{store_id}_{category_name.replace(' ', '_')}_{metric}.html")
        fig.write_html(output_file, include_plotlyjs='cdn')
        logger.info(f"Created interactive category treemap at {output_file}")
        
        return output_file
    
    def create_plotly_visualizations(self):
        """
        Create interactive Plotly visualizations for category statistics
        
        Returns:
            list: Paths to created visualization files
        """
        if self.category_stats is None or self.category_forecasts is None:
            logger.warning("No category statistics or forecasts available for visualization")
            return []
            
        # Create directory for Plotly visualizations
        os.makedirs(self.plotly_dir, exist_ok=True)
        
        logger.info("Creating interactive Plotly visualizations for categories")
        plot_paths = []
        
        # Create forecast plots for each category
        for _, row in self.category_stats.iterrows():
            store_id = row['Store_Id']
            category_name = row['Category_Name']
            
            # Create forecast plot
            forecast_path = self.create_plotly_forecast_plot(store_id, category_name)
            if forecast_path:
                plot_paths.append(forecast_path)
            
            # Create treemap for item distribution
            treemap_path = self.create_plotly_category_item_treemap(store_id, category_name)
            if treemap_path:
                plot_paths.append(treemap_path)
        
        # Create category comparison plots for each store
        for store_id in self.category_stats['Store_Id'].unique():
            comparison_path = self.create_plotly_category_comparison(store_id)
            if comparison_path:
                plot_paths.append(comparison_path)
        
        logger.info(f"Created {len(plot_paths)} interactive category visualizations")
        return plot_paths
    
    def save_category_statistics(self, output_file=None):
        """
        Save category statistics to a CSV file
        
        Args:
            output_file: Path to save the CSV file
            
        Returns:
            str: Path to saved file
        """
        if self.category_stats is None:
            logger.warning("No category statistics to save")
            return None
            
        if output_file is None:
            output_file = CATEGORY_STATISTICS_FILE
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert complex dictionary columns to strings
        stats_df = self.category_stats.copy()
        
        # Convert dictionary columns to strings
        for col in ['Weekly_Pattern', 'Monthly_Pattern', 'Weather_Impact', 'Day_Of_Week_Effect', 
                   'Top_Selling_Items', 'Top_Profit_Items', 'Top_Margin_Items', 
                   'Item_Sales_Share', 'Elasticity_Items']:
            if col in stats_df.columns:
                stats_df[col] = stats_df[col].apply(lambda x: str(x) if x is not None else None)
        
        # Save to CSV
        stats_df.to_csv(output_file, index=False)
        logger.info(f"Saved category statistics to {output_file}")
        return output_file
    
    def save_category_forecasts(self, output_file=None):
        """
        Save category forecasts to a CSV file
        
        Args:
            output_file: Path to save the CSV file
            
        Returns:
            str: Path to saved file
        """
        if self.category_forecasts is None:
            logger.warning("No category forecasts to save")
            return None
            
        if output_file is None:
            output_file = os.path.join(os.path.dirname(EXTENDED_FORECASTS_FILE), "category_forecasts.csv")
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to CSV
        self.category_forecasts.to_csv(output_file, index=False)
        logger.info(f"Saved category forecasts to {output_file}")
        return output_file

def run_category_statistics_analysis(data_file=COMBINED_DATA_FILE, item_stats_file=ITEM_STATISTICS_FILE, 
                                   save_output=True, create_visualizations=True):
    """
    Run the complete category statistics analysis
    
    Args:
        data_file: Path to the data file
        item_stats_file: Path to the item statistics file
        save_output: Whether to save output files
        create_visualizations: Whether to create visualizations
        
    Returns:
        tuple: (category_stats, category_forecasts)
    """
    logger.info("Starting category statistics analysis")
    
    analyzer = CategoryStatisticsAnalyzer(data_file, item_stats_file)
    
    try:
        # Step 1: Load data
        analyzer.load_data()
        
        # Step 2: Load item statistics
        analyzer.load_item_statistics()
        
        # Step 3: Calculate category statistics
        category_stats = analyzer.calculate_category_statistics()
        
        # Step 4: Save category statistics if requested
        if save_output and category_stats is not None:
            analyzer.save_category_statistics()
        
        # Step 5: Create category forecast aggregation
        category_forecasts = analyzer.create_category_forecast_aggregation()
        
        # Step 6: Save category forecasts if requested
        if save_output and category_forecasts is not None:
            analyzer.save_category_forecasts()
        
        # Step 7: Create visualizations if requested
        if create_visualizations:
            analyzer.create_category_statistics_visualizations()
            analyzer.create_plotly_visualizations()
        
        logger.info("Category statistics analysis completed successfully")
        return category_stats, category_forecasts
        
    except Exception as e:
        logger.error(f"Error in category statistics analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Category Statistics Analysis')
    parser.add_argument('--data', type=str, default=COMBINED_DATA_FILE, help='Path to data file')
    parser.add_argument('--item-stats', type=str, default=ITEM_STATISTICS_FILE, help='Path to item statistics file')
    parser.add_argument('--no-save', action='store_true', help='Do not save output files')
    parser.add_argument('--no-viz', action='store_true', help='Do not create visualizations')
    
    args = parser.parse_args()
    
    run_category_statistics_analysis(
        data_file=args.data,
        item_stats_file=args.item_stats,
        save_output=not args.no_save,
        create_visualizations=not args.no_viz
    )