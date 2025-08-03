import pandas as pd
import os

print("Checking forecast files...")

if os.path.exists("rf_forecasts.csv"):
    try:
        df = pd.read_csv("rf_forecasts.csv")
        print("RF Forecasts exists. Columns:", df.columns.tolist())
        print(f"Contains {len(df)} rows")
        print("First row sample:")
        print(df.head(1).to_string())
    except Exception as e:
        print(f"Error reading rf_forecasts.csv: {e}")
else:
    print("rf_forecasts.csv does not exist")

if os.path.exists("pytorch_forecasts.csv"):
    try:
        df = pd.read_csv("pytorch_forecasts.csv")
        print("\nPyTorch Forecasts exists. Columns:", df.columns.tolist())
        print(f"Contains {len(df)} rows")
        print("First row sample:")
        print(df.head(1).to_string())
    except Exception as e:
        print(f"\nError reading pytorch_forecasts.csv: {e}")
else:
    print("\npytorch_forecasts.csv does not exist")