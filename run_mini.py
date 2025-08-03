#!/usr/bin/env python3
import retail_forecast as rf
import time
import gradio as gr

def run_ui_test():
    # Load data with small sample size for quicker startup
    print("Loading data from CSV...")
    # First make sure we have the filtered pizza data
    import os
    if not os.path.exists('frozen_pizza_only.csv'):
        print("Filtered pizza data not found. Creating it now...")
        from filter_frozen_pizza import filter_frozen_pizza
        filter_frozen_pizza('Price_Elasticity_Frozen_Input.csv', 'frozen_pizza_only.csv')
    
    df = rf.load_data_from_csv(file_path='frozen_pizza_only.csv', sample_size=0.1)  # Use 10% of pizza data
    
    # Train a simpler model
    print("Training model...")
    model = rf.RandomForestRegressor(
        n_estimators=50,  # Reduced for speed
        max_depth=10,     # Reduced for speed
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    # Prepare features
    df_features, feature_cols = rf.prepare_features(df)
    
    # Define target
    target = 'Sales'
    
    # Split into features and target
    X = df_features[feature_cols].fillna(0)
    y = df_features[target]
    
    # Train model
    print("Training the model...")
    model.fit(X, y)
    
    # Create UI
    print("Creating UI...")
    demo = rf.create_gradio_ui(model, df, feature_cols)
    
    # Launch UI for a short time
    print("Launching UI for 5 seconds...")
    demo.launch(share=False, inbrowser=False, prevent_thread_lock=True)
    
    # Wait for a few seconds
    time.sleep(5)
    
    # Shutdown
    print("Test complete")
    return True

if __name__ == "__main__":
    run_ui_test()