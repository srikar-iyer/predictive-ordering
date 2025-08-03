import pandas as pd
import numpy as np

def integrate_on_hand_data(combined_file='combined_pizza_data.csv', 
                           on_hand_file='optional_on_hand_data.csv',
                           output_file='combined_pizza_data.csv'):
    """
    Integrates optional_on_hand_data.csv into the combined_pizza_data.csv,
    updating stock levels and related metrics based on the on-hand inventory data.
    """
    print(f"Integrating on-hand data from {on_hand_file} into {combined_file}...")
    
    try:
        # Load the datasets
        combined_df = pd.read_csv(combined_file)
        on_hand_df = pd.read_csv(on_hand_file)
        
        # Convert dates to datetime in combined_df
        if 'Date' in combined_df.columns:
            combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        
        # Process on_hand_df - ensure item column is string type for consistent merging
        on_hand_df['item'] = on_hand_df['item'].astype(str)
        
        # Rename columns to match the combined dataset structure
        on_hand_df.rename(columns={
            'On_Hand': 'Current_Stock',
            'Week_4_Avg_Movement': 'Avg_Weekly_Sales_4W',
            'Week_13_Avg_Movement': 'Avg_Weekly_Sales_13W',
            'Weeks_of_Supply': 'Stock_Coverage_Weeks'
        }, inplace=True)
        
        # Get the latest date in the combined dataset
        latest_date = combined_df['Date'].max()
        
        # Update stock level for the latest date in the combined dataset
        for _, on_hand_row in on_hand_df.iterrows():
            store_id = on_hand_row['Store_Id']
            item = on_hand_row['item']
            
            # Find the latest date rows for this store and item
            latest_rows = combined_df[(combined_df['Store_Id'] == store_id) & 
                                     (combined_df['Item'] == item) & 
                                     (combined_df['Date'] == latest_date)]
            
            if len(latest_rows) > 0:
                # Update the stock level
                combined_df.loc[latest_rows.index, 'Stock_Level'] = on_hand_row['Current_Stock']
                
                # Update the weekly sales averages and stock coverage
                combined_df.loc[latest_rows.index, 'Avg_Weekly_Sales_4W'] = on_hand_row['Avg_Weekly_Sales_4W']
                combined_df.loc[latest_rows.index, 'Avg_Weekly_Sales_13W'] = on_hand_row['Avg_Weekly_Sales_13W']
                combined_df.loc[latest_rows.index, 'Stock_Coverage_Weeks'] = on_hand_row['Stock_Coverage_Weeks']
                
                # Calculate stock velocity (higher is better - inventory turnover rate)
                if on_hand_row['Current_Stock'] > 0:
                    stock_velocity = on_hand_row['Avg_Weekly_Sales_4W'] / on_hand_row['Current_Stock']
                else:
                    stock_velocity = 0  # Can't calculate if no stock
                
                combined_df.loc[latest_rows.index, 'Stock_Velocity'] = stock_velocity
                
                # Recalculate the Weeks_Of_Stock metric
                daily_sales_estimate = on_hand_row['Avg_Weekly_Sales_4W'] / 7  # Convert weekly to daily
                if daily_sales_estimate > 0:
                    weeks_of_stock = on_hand_row['Current_Stock'] / (daily_sales_estimate * 7)
                else:
                    weeks_of_stock = 4.0  # Default to 4 weeks if no sales data
                
                combined_df.loc[latest_rows.index, 'Weeks_Of_Stock'] = weeks_of_stock
                
                # Calculate stockout risk
                if weeks_of_stock < 0.5:
                    stockout_risk = 'Very High'
                elif weeks_of_stock < 1:
                    stockout_risk = 'High'
                elif weeks_of_stock < 2:
                    stockout_risk = 'Moderate'
                else:
                    stockout_risk = 'Low'
                    
                combined_df.loc[latest_rows.index, 'Stockout_Risk'] = stockout_risk
                
                # Update inventory status based on weeks of stock
                if weeks_of_stock < 1:
                    stock_status = 'Needs Reordering'
                elif weeks_of_stock <= 3:
                    stock_status = 'Good'
                else:
                    stock_status = 'Overstocked'
                
                combined_df.loc[latest_rows.index, 'Stock_Status'] = stock_status
                
                # Get price and cost data for the product
                price_data = combined_df.loc[latest_rows.index, 'Price'].values
                cost_data = combined_df.loc[latest_rows.index, 'Cost'].values if 'Cost' in combined_df.columns else None
                
                if len(price_data) > 0:
                    avg_price = price_data[0]
                    avg_cost = cost_data[0] if cost_data is not None else avg_price * 0.7  # Assume 30% margin if no cost data
                    margin = avg_price - avg_cost
                else:
                    avg_price = 0
                    margin = 0
                
                # Calculate carrying cost (excess stock penalty)
                carrying_cost_factor = 0.15  # 15% annual carrying cost is industry standard
                weekly_carrying_cost_factor = carrying_cost_factor / 52  # Convert annual to weekly
                
                if weeks_of_stock > 3:
                    # Calculate excess weeks beyond optimal level
                    excess_weeks = weeks_of_stock - 3
                    # Calculate excess units
                    excess_units = excess_weeks * daily_sales_estimate * 7
                    # Calculate carrying cost based on product value
                    excess_penalty = excess_units * avg_price * weekly_carrying_cost_factor
                else:
                    excess_penalty = 0
                    
                combined_df.loc[latest_rows.index, 'Excess_Stock_Penalty'] = excess_penalty
                
                # Calculate stockout penalty (potential lost sales)
                if weeks_of_stock < 1:
                    # Calculate missing weeks of coverage
                    missing_weeks = 1 - weeks_of_stock
                    # Calculate potential lost units
                    lost_units = missing_weeks * daily_sales_estimate * 7
                    # Calculate lost profit based on margin
                    stockout_penalty = lost_units * margin
                else:
                    stockout_penalty = 0
                    
                combined_df.loc[latest_rows.index, 'Stockout_Penalty'] = stockout_penalty
                
                # Update order recommendations
                safety_stock = daily_sales_estimate * 7  # 1 week minimum stock
                target_stock = daily_sales_estimate * 7 * 2  # 2 weeks target stock
                
                # Order point is safety stock level
                order_point = safety_stock
                
                # Calculate order quantity to reach target stock
                if on_hand_row['Current_Stock'] < order_point:
                    order_qty = target_stock - on_hand_row['Current_Stock']
                    order_qty = max(0, round(order_qty))
                else:
                    order_qty = 0
                
                combined_df.loc[latest_rows.index, 'Safety_Stock'] = safety_stock
                combined_df.loc[latest_rows.index, 'Target_Stock'] = target_stock
                combined_df.loc[latest_rows.index, 'Order_Point'] = order_point
                combined_df.loc[latest_rows.index, 'Order_Quantity'] = order_qty
                
                # Add recommendation reason
                if order_qty > 0:
                    reason = f"Order {order_qty} units to reach target stock level of {int(target_stock)} units (currently {int(on_hand_row['Current_Stock'])} units, {weeks_of_stock:.1f} weeks of supply)"
                else:
                    reason = f"No order needed. Current stock of {int(on_hand_row['Current_Stock'])} units provides {weeks_of_stock:.1f} weeks of supply"
                
                combined_df.loc[latest_rows.index, 'Recommendation_Reason'] = reason
        
        # Save the updated combined dataset
        combined_df.to_csv(output_file, index=False)
        print(f"Successfully integrated on-hand data and saved to {output_file}")
        return combined_df
    
    except Exception as e:
        print(f"Error integrating on-hand data: {str(e)}")
        raise

if __name__ == "__main__":
    # Integrate on-hand data into the combined dataset
    updated_df = integrate_on_hand_data()