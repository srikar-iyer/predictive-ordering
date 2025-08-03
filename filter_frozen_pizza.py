#!/usr/bin/env python3
"""
Script to filter a CSV file to include only items of type 'Frozen Pizza'.
"""
import pandas as pd
import sys

def filter_frozen_pizza(input_file, output_file):
    """
    Filter CSV to include only frozen pizza items.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path where the filtered CSV will be saved
    """
    try:
        # Read the CSV file
        print(f"Reading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        # Get original row count
        original_count = len(df)
        print(f"Original CSV contains {original_count} rows")
        
        # Filter only for Frozen Pizza
        pizza_df = df[df['Category_Name'] == 'Frozen Pizza']
        
        # Get filtered row count
        filtered_count = len(pizza_df)
        print(f"Filtered CSV contains {filtered_count} rows")
        
        # Save to new file
        pizza_df.to_csv(output_file, index=False)
        print(f"Filtered data saved to {output_file}")
        
        # Return count of unique item IDs to help with reconfiguration
        unique_items = pizza_df['Item'].nunique()
        print(f"Found {unique_items} unique pizza item IDs")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_frozen_pizza.py <input_file> <output_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = filter_frozen_pizza(input_file, output_file)
    sys.exit(0 if success else 1)