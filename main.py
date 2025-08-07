#!/usr/bin/env python3
"""
Main entry point for the Pizza Predictive Ordering System.
This script orchestrates the complete workflow of the system.
"""
import argparse
import os
import subprocess
import time
import pandas as pd
import logging
import importlib.util
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('main')

# Import config if available, otherwise use defaults
try:
    from config.settings import (
        SALES_FILE, PURCHASES_FILE, STOCK_FILE, COMBINED_DATA_FILE,
        PYTORCH_FORECASTS_FILE, RF_FORECASTS_FILE, ARIMA_FORECASTS_FILE, 
        WEIGHTED_ARIMA_FORECASTS_FILE, STATIC_DIR, MODELS_DIR
    )
except ImportError:
    # Default paths for backward compatibility
    ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    SALES_FILE = ROOT_DIR / "FrozenPizzaSales.csv"
    PURCHASES_FILE = ROOT_DIR / "FrozenPizzaPurchases.csv"
    STOCK_FILE = ROOT_DIR / "FrozenPizzaStock.csv"
    COMBINED_DATA_FILE = ROOT_DIR / "combined_pizza_data.csv"
    PYTORCH_FORECASTS_FILE = ROOT_DIR / "pytorch_forecasts.csv"
    RF_FORECASTS_FILE = ROOT_DIR / "rf_forecasts.csv"
    ARIMA_FORECASTS_FILE = ROOT_DIR / "arima_forecasts.csv"
    WEIGHTED_ARIMA_FORECASTS_FILE = ROOT_DIR / "weighted_arima_forecasts.csv"
    STATIC_DIR = ROOT_DIR / "static"
    MODELS_DIR = ROOT_DIR / "models"

def run_command(command, description):
    """
    Run a shell command with error handling and timing
    
    Args:
        command (str): The shell command to run
        description (str): Description of the command for logging
        
    Returns:
        bool: True if command succeeded, False otherwise
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"RUNNING: {description}")
    logger.info(f"{'='*80}")
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"{'='*80}")
    
    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, check=True, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"COMPLETED in {elapsed_time:.2f} seconds: {description}")
        print(f"COMPLETED in {elapsed_time:.2f} seconds: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"ERROR: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        print(f"ERROR: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        
        elapsed_time = time.time() - start_time
        logger.error(f"FAILED in {elapsed_time:.2f} seconds: {description}")
        print(f"FAILED in {elapsed_time:.2f} seconds: {description}")
        return False

def verify_data_files():
    """
    Verify that necessary data files exist
    
    Returns:
        bool: True if all files exist, False otherwise
    """
    required_files = [
        SALES_FILE,
        PURCHASES_FILE,
        STOCK_FILE
    ]
    
    missing_files = []
    
    for file in required_files:
        if not os.path.isfile(file):
            missing_files.append(str(file))
    
    if missing_files:
        logger.error(f"ERROR: The following required data files are missing: {', '.join(missing_files)}")
        print(f"ERROR: The following required data files are missing: {', '.join(missing_files)}")
        return False
    
    logger.info(f"All required data files found.")
    print(f"All required data files found.")
    return True

def create_directories():
    """
    Create necessary directories for outputs
    
    Returns:
        bool: True if successful
    """
    dirs = [
        STATIC_DIR,
        os.path.join(STATIC_DIR, "images"),
        os.path.join(STATIC_DIR, "images/inventory"),
        os.path.join(STATIC_DIR, "images/profit"),
        os.path.join(STATIC_DIR, "images/time_series"),
        os.path.join(STATIC_DIR, "images/arima"),
        os.path.join(STATIC_DIR, "images/weighted_arima"),
        os.path.join(STATIC_DIR, "images/integrated_weighted_arima"),
        MODELS_DIR,
        os.path.join(MODELS_DIR, "time_series"),
        os.path.join(MODELS_DIR, "arima"),
        os.path.join(MODELS_DIR, "weighted_arima"),
        os.path.dirname(ARIMA_FORECASTS_FILE),
        os.path.dirname(WEIGHTED_ARIMA_FORECASTS_FILE)
    ]
    
    # Create data directories to ensure they exist
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dirs = [
        os.path.join(root_dir, "data"),
        os.path.join(root_dir, "data/processed"),
        os.path.join(root_dir, "data/processed/inventory"),
        os.path.join(root_dir, "data/raw")
    ]
    
    # Create all directories
    all_dirs = dirs + data_dirs
    
    for d in all_dirs:
        os.makedirs(d, exist_ok=True)
    
    logger.info(f"Created output directories.")
    print(f"Created output directories.")
    return True

def run_data_integration():
    """
    Run data integration process using either new or legacy module
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Ensure data files are in the right location
    ensure_data_files()
    
    # Try using the new module first
    if os.path.exists(os.path.join("src", "data", "data_loader.py")):
        try:
            logger.info("Using refactored data loader module")
            print("Using refactored data loader module")
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from src.data.data_loader import process_data
            
            # Get fallback directory for better error handling
            fallback_dir = None
            # Check if raw data exists in data/raw directory
            root_dir = os.path.dirname(os.path.abspath(__file__))
            raw_dir = os.path.join(root_dir, "data", "raw")
            if os.path.exists(raw_dir):
                fallback_dir = raw_dir
                logger.info(f"Using {raw_dir} as fallback directory for data files")
                print(f"Using {raw_dir} as fallback directory for data files")
            
            process_data(fallback_dir=fallback_dir, include_pre_reference_sales=True)
            return True
        except Exception as e:
            logger.error(f"Error using refactored data loader: {e}")
            print(f"Error using refactored data loader: {e}")
    
    # Fall back to the legacy script
    return run_command("python3 integrate_pizza_data.py", "Data Integration")

def ensure_data_files():
    """
    Ensure data files exist in both root and data directories
    
    Returns:
        None
    """
    # Get the root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Source files in root directory
    root_files = {
        'FrozenPizzaSales.csv': SALES_FILE,
        'FrozenPizzaPurchases.csv': PURCHASES_FILE,
        'FrozenPizzaStock.csv': STOCK_FILE,
        'combined_pizza_data.csv': COMBINED_DATA_FILE
    }
    
    # Processed files that may exist in both locations
    processed_files = {
        'combined_pizza_data.csv': COMBINED_DATA_FILE,
        'pytorch_forecasts.csv': PYTORCH_FORECASTS_FILE,
        'rf_forecasts.csv': RF_FORECASTS_FILE,
        'arima_forecasts.csv': ARIMA_FORECASTS_FILE,
        'weighted_arima_forecasts.csv': WEIGHTED_ARIMA_FORECASTS_FILE
    }
    
    # Ensure data/raw directory exists
    raw_dir = os.path.join(root_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # Ensure data/processed directory exists
    processed_dir = os.path.join(root_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Copy raw files to raw directory if they exist in root
    for filename, filepath in root_files.items():
        if os.path.exists(filepath):
            raw_path = os.path.join(raw_dir, filename)
            # Copy if raw file doesn't exist or is older
            if not os.path.exists(raw_path) or os.path.getmtime(filepath) > os.path.getmtime(raw_path):
                logger.info(f"Copying {filepath} to {raw_path}")
                shutil.copy2(filepath, raw_path)
        else:
            # Check if file exists in raw directory and copy back to root
            raw_path = os.path.join(raw_dir, filename)
            if os.path.exists(raw_path) and not os.path.exists(filepath):
                logger.info(f"Copying {raw_path} to {filepath}")
                shutil.copy2(raw_path, filepath)
    
    # Check processed files - these might be in root or in data/processed
    for filename, filepath in processed_files.items():
        root_path = os.path.join(root_dir, filename)
        processed_path = os.path.join(processed_dir, filename)
        
        # If file exists in root but not in processed, copy to processed
        if os.path.exists(root_path) and not os.path.exists(processed_path):
            logger.info(f"Copying {root_path} to {processed_path}")
            shutil.copy2(root_path, processed_path)
        
        # If file exists in processed but not in root, copy to root (for backward compatibility)
        if os.path.exists(processed_path) and not os.path.exists(root_path):
            logger.info(f"Copying {processed_path} to {root_path}")
            shutil.copy2(processed_path, root_path)
        
        # If both exist, use the newer one to update the other
        if os.path.exists(root_path) and os.path.exists(processed_path):
            if os.path.getmtime(root_path) > os.path.getmtime(processed_path):
                logger.info(f"Updating {processed_path} from {root_path}")
                shutil.copy2(root_path, processed_path)
            elif os.path.getmtime(processed_path) > os.path.getmtime(root_path):
                logger.info(f"Updating {root_path} from {processed_path}")
                shutil.copy2(processed_path, root_path)

def check_dependencies(component):
    """
    Check if required dependencies are installed for specific components
    
    Args:
        component: The component to check dependencies for ('pytorch', 'arima', etc.)
        
    Returns:
        bool: True if dependencies are met, False otherwise
    """
    # Define required packages for each component
    required_packages = {
        'pytorch': ['torch', 'torchvision'],
        'arima': ['statsmodels', 'pmdarima'],
        'weighted_arima': ['statsmodels', 'pmdarima', 'joblib'],
        'dashboard': ['plotly', 'dash'],
    }
    
    # If component doesn't have specific requirements, return True
    if component not in required_packages:
        return True
    
    # Check if all required packages are installed
    missing_packages = []
    for package in required_packages.get(component, []):
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    # If there are missing packages, log and print a message
    if missing_packages:
        message = f"Missing dependencies for {component}: {', '.join(missing_packages)}. Please install with: pip install {' '.join(missing_packages)}"
        logger.warning(message)
        print(message)
        return False
    
    return True

def run_full_pipeline(skip_time_series=False, use_existing_rf=False, use_weighted_arima=False, skip_arima=False):
    """
    Run the full predictive ordering pipeline
    
    Args:
        skip_time_series (bool): Skip the PyTorch time series model training
        use_existing_rf (bool): Use existing Random Forest model instead of retraining
        
    Returns:
        bool: True if successful, False otherwise
    """
    pipeline_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"\n{'*'*80}")
    logger.info(f"STARTING PIZZA PREDICTIVE ORDERING PIPELINE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'*'*80}")
    print(f"\n{'*'*80}")
    print(f"STARTING PIZZA PREDICTIVE ORDERING PIPELINE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'*'*80}")
    
    # Verify data files
    if not verify_data_files():
        return False
    
    # Create directories
    create_directories()
    
    # Step 1: Integrate the datasets
    if not run_data_integration():
        logger.error("ERROR: Data integration failed. Exiting pipeline.")
        print("ERROR: Data integration failed. Exiting pipeline.")
        return False
        
    # Step 1b: Market basket analysis has been removed
    logger.info("Market Basket Analysis component has been removed from the pipeline.")
    print("Market Basket Analysis component has been removed from the pipeline.")
    
    # Step 2: Train the Random Forest model (with option to use existing model)
    if os.path.exists(os.path.join("src", "models", "rf_model.py")):
        try:
            logger.info("Using refactored RF model module")
            print("Using refactored RF model module")
            from src.models.rf_model import run_rf_forecasting
            run_rf_forecasting(use_existing=use_existing_rf)
        except Exception as e:
            logger.error(f"Error using refactored RF model module: {e}")
            print(f"Error using refactored RF model module: {e}")
            # Fall back to the legacy script
            rf_cmd = "python3 rf_model_update.py"
            if use_existing_rf:
                rf_cmd += " --use-existing"
            if not run_command(rf_cmd, "Random Forest Model Processing"):
                logger.warning("WARNING: Random Forest model processing failed. Continuing pipeline.")
                print("WARNING: Random Forest model processing failed. Continuing pipeline.")
    else:
        # Use the legacy script
        rf_cmd = "python3 rf_model_update.py"
        if use_existing_rf:
            rf_cmd += " --use-existing"
        if not run_command(rf_cmd, "Random Forest Model Processing"):
            logger.warning("WARNING: Random Forest model processing failed. Continuing pipeline.")
            print("WARNING: Random Forest model processing failed. Continuing pipeline.")
    
    # Step 3: Train the PyTorch time series model (optional)
    if not skip_time_series:
        # Check dependencies first
        if check_dependencies('pytorch'):
            # Check if refactored module exists
            if os.path.exists(os.path.join("src", "models", "time_series.py")):
                try:
                    logger.info("Using refactored time series module")
                    print("Using refactored time series module")
                    from src.models.time_series import run_time_series_forecasting
                    run_time_series_forecasting()
                except Exception as e:
                    logger.error(f"Error using refactored time series module: {e}")
                    print(f"Error using refactored time series module: {e}")
                    # Fall back to legacy script
                    if not run_command("python3 pytorch_time_series.py", "PyTorch Time Series Model Training"):
                        logger.warning("WARNING: PyTorch time series model training failed. Continuing pipeline.")
                        print("WARNING: PyTorch time series model training failed. Continuing pipeline.")
            else:
                # Use legacy script
                if not run_command("python3 pytorch_time_series.py", "PyTorch Time Series Model Training"):
                    logger.warning("WARNING: PyTorch time series model training failed. Continuing pipeline.")
                    print("WARNING: PyTorch time series model training failed. Continuing pipeline.")
        else:
            logger.warning("WARNING: PyTorch dependencies missing. Skipping time series model training.")
            print("WARNING: PyTorch dependencies missing. Skipping time series model training.")
    else:
        logger.info("\nSkipping PyTorch time series model training.")
        print("\nSkipping PyTorch time series model training.")
        # Create a fallback PyTorch forecasts file from RF forecasts to avoid errors
        if os.path.exists(RF_FORECASTS_FILE) and not os.path.exists(PYTORCH_FORECASTS_FILE):
            logger.info("Creating PyTorch forecasts file from Random Forest forecasts as a fallback")
            print("Creating PyTorch forecasts file from Random Forest forecasts as a fallback")
            shutil.copy(RF_FORECASTS_FILE, PYTORCH_FORECASTS_FILE)
    
    # Step 4: Run ARIMA models (if enabled)
    if not skip_arima:
        # First try the weighted ARIMA model if requested
        if use_weighted_arima:
            # Check dependencies first
            if check_dependencies('weighted_arima'):
                if os.path.exists(os.path.join("src", "models", "weighted_arima_model.py")):
                    try:
                        logger.info("Using Weighted ARIMA model")
                        print("Using Weighted ARIMA model")
                        from src.models.weighted_arima_model import run_weighted_arima_forecasting
                        
                        # Check for data file explicitly to avoid potential errors
                        data_file = COMBINED_DATA_FILE
                        if not os.path.exists(data_file):
                            alt_data_file = os.path.join(ROOT_DIR, "combined_pizza_data.csv")
                            if os.path.exists(alt_data_file):
                                logger.info(f"Using alternative data file for weighted ARIMA: {alt_data_file}")
                                print(f"Using alternative data file for weighted ARIMA: {alt_data_file}")
                                data_file = alt_data_file
                                
                        run_weighted_arima_forecasting(data_file=data_file, use_existing=use_existing_rf)
                    except Exception as e:
                        logger.error(f"Error using Weighted ARIMA model: {e}")
                        print(f"Error using Weighted ARIMA model: {e}")
                        # Fall back to standard ARIMA model if dependencies are met
                        if check_dependencies('arima') and os.path.exists(os.path.join("src", "models", "arima_model.py")):
                            try:
                                logger.info("Falling back to standard ARIMA model")
                                print("Falling back to standard ARIMA model")
                                from src.models.arima_model import run_arima_forecasting
                                run_arima_forecasting(use_existing=use_existing_rf)
                            except Exception as e:
                                logger.error(f"Error using ARIMA model: {e}")
                                print(f"Error using ARIMA model: {e}")
                else:
                    # Use command line wrapper
                    arima_cmd = "python3 -m src.models.weighted_arima_model_wrapper"
                    if use_existing_rf:
                        arima_cmd += " --use-existing"
                    if not run_command(arima_cmd, "Weighted ARIMA Model Processing"):
                        logger.warning("WARNING: Weighted ARIMA model processing failed. Continuing pipeline.")
                        print("WARNING: Weighted ARIMA model processing failed. Continuing pipeline.")
            else:
                logger.warning("WARNING: Weighted ARIMA dependencies missing. Trying standard ARIMA model.")
                print("WARNING: Weighted ARIMA dependencies missing. Trying standard ARIMA model.")
                # Try standard ARIMA if dependencies are met
                if check_dependencies('arima'):
                    if os.path.exists(os.path.join("src", "models", "arima_model.py")):
                        try:
                            logger.info("Using standard ARIMA model")
                            print("Using standard ARIMA model")
                            from src.models.arima_model import run_arima_forecasting
                            run_arima_forecasting(use_existing=use_existing_rf)
                        except Exception as e:
                            logger.error(f"Error using ARIMA model: {e}")
                            print(f"Error using ARIMA model: {e}")
                    else:
                        # Use command line wrapper
                        arima_cmd = "python3 -m src.models.arima_model_wrapper"
                        if use_existing_rf:
                            arima_cmd += " --use-existing"
                        if not run_command(arima_cmd, "ARIMA Model Processing"):
                            logger.warning("WARNING: ARIMA model processing failed. Continuing pipeline.")
                            print("WARNING: ARIMA model processing failed. Continuing pipeline.")
        else:
            # Try standard ARIMA model
            if check_dependencies('arima'):
                if os.path.exists(os.path.join("src", "models", "arima_model.py")):
                    try:
                        logger.info("Using standard ARIMA model")
                        print("Using standard ARIMA model")
                        from src.models.arima_model import run_arima_forecasting
                        run_arima_forecasting(use_existing=use_existing_rf)
                    except Exception as e:
                        logger.error(f"Error using ARIMA model: {e}")
                        print(f"Error using ARIMA model: {e}")
                else:
                    # Use command line wrapper
                    arima_cmd = "python3 -m src.models.arima_model_wrapper"
                    if use_existing_rf:
                        arima_cmd += " --use-existing"
                    if not run_command(arima_cmd, "ARIMA Model Processing"):
                        logger.warning("WARNING: ARIMA model processing failed. Continuing pipeline.")
                        print("WARNING: ARIMA model processing failed. Continuing pipeline.")
            else:
                logger.warning("WARNING: ARIMA dependencies missing. Skipping ARIMA models.")
                print("WARNING: ARIMA dependencies missing. Skipping ARIMA models.")
    else:
        logger.info("\nSkipping ARIMA model training.")
        print("\nSkipping ARIMA model training.")
    
    # Step 4b: Inventory management has been removed
    logger.info("Inventory management component has been removed from the pipeline.")
    print("Inventory management component has been removed from the pipeline.")
    
    # Step 5: Run integrated forecasting with weighted ARIMA (if enabled)
    if use_weighted_arima and not skip_arima:
        # Check dependencies first
        if check_dependencies('weighted_arima'):
            if os.path.exists(os.path.join("src", "models", "integrated_forecasting_with_weighted_arima.py")):
                try:
                    logger.info("Using integrated forecasting with weighted ARIMA")
                    print("Using integrated forecasting with weighted ARIMA")
                    from src.models.integrated_forecasting_with_weighted_arima import IntegratedForecasterWithWeightedARIMA
                    forecaster = IntegratedForecasterWithWeightedARIMA()
                    forecaster.run_integrated_optimization(use_existing=use_existing_rf)
                    # Skip regular profit optimization since integrated forecasting handles it
                    logger.info("Skipping regular profit optimization as it's handled by integrated forecasting")
                    print("Skipping regular profit optimization as it's handled by integrated forecasting")
                    profit_optimization_done = True
                except Exception as e:
                    logger.error(f"Error using integrated forecasting: {e}")
                    print(f"Error using integrated forecasting: {e}")
                    profit_optimization_done = False
            else:
                # Use command line wrapper
                integrated_cmd = "python3 -m src.models.integrated_forecasting_with_weighted_arima"
                if use_existing_rf:
                    integrated_cmd += " --use-existing"
                if run_command(integrated_cmd, "Integrated Forecasting with Weighted ARIMA"):
                    # Skip regular profit optimization since integrated forecasting handles it
                    logger.info("Skipping regular profit optimization as it's handled by integrated forecasting")
                    print("Skipping regular profit optimization as it's handled by integrated forecasting")
                    profit_optimization_done = True
                else:
                    logger.warning("WARNING: Integrated forecasting failed. Falling back to regular profit optimization.")
                    print("WARNING: Integrated forecasting failed. Falling back to regular profit optimization.")
                    profit_optimization_done = False
        else:
            logger.warning("WARNING: Weighted ARIMA dependencies missing. Falling back to regular profit optimization.")
            print("WARNING: Weighted ARIMA dependencies missing. Falling back to regular profit optimization.")
            profit_optimization_done = False
    else:
        profit_optimization_done = False
    
    # Step 6: Run profit optimization (if not already handled by integrated forecasting)
    if not profit_optimization_done:
        # Check if refactored module exists
        if os.path.exists(os.path.join("src", "optimization", "profit_optimizer.py")):
            try:
                logger.info("Using refactored profit optimization module")
                print("Using refactored profit optimization module")
                from src.optimization.profit_optimizer import run_profit_optimization
                
                # Make sure we have all the data files needed
                data_file = COMBINED_DATA_FILE
                rf_forecast_file = RF_FORECASTS_FILE
                pytorch_forecast_file = PYTORCH_FORECASTS_FILE
                
                # Check for data files and use alternatives if needed
                if not os.path.exists(data_file):
                    alt_data_file = os.path.join(ROOT_DIR, "combined_pizza_data.csv")
                    if os.path.exists(alt_data_file):
                        logger.info(f"Using alternative data file for profit optimization: {alt_data_file}")
                        print(f"Using alternative data file for profit optimization: {alt_data_file}")
                        data_file = alt_data_file
                
                if not os.path.exists(rf_forecast_file):
                    alt_rf_forecast_file = os.path.join(ROOT_DIR, "rf_forecasts.csv")
                    if os.path.exists(alt_rf_forecast_file):
                        logger.info(f"Using alternative RF forecast file: {alt_rf_forecast_file}")
                        print(f"Using alternative RF forecast file: {alt_rf_forecast_file}")
                        rf_forecast_file = alt_rf_forecast_file
                
                if not os.path.exists(pytorch_forecast_file):
                    alt_pytorch_forecast_file = os.path.join(ROOT_DIR, "pytorch_forecasts.csv")
                    if os.path.exists(alt_pytorch_forecast_file):
                        logger.info(f"Using alternative PyTorch forecast file: {alt_pytorch_forecast_file}")
                        print(f"Using alternative PyTorch forecast file: {alt_pytorch_forecast_file}")
                        pytorch_forecast_file = alt_pytorch_forecast_file
                
                run_profit_optimization(data_file=data_file, rf_forecast_file=rf_forecast_file, pytorch_forecast_file=pytorch_forecast_file)
            except Exception as e:
                logger.error(f"Error using refactored profit optimization module: {e}")
                print(f"Error using refactored profit optimization module: {e}")
                # Fall back to legacy script
                if not run_command("python3 profit_optimization.py", "Profit Optimization"):
                    logger.warning("WARNING: Profit optimization failed. Continuing pipeline.")
                    print("WARNING: Profit optimization failed. Continuing pipeline.")
        else:
            # Use legacy script
            if not run_command("python3 profit_optimization.py", "Profit Optimization"):
                logger.warning("WARNING: Profit optimization failed. Continuing pipeline.")
                print("WARNING: Profit optimization failed. Continuing pipeline.")
    
    # Step 7: Launch the dashboard
    # Check if dashboard dependencies are installed
    if check_dependencies('dashboard'):
        # Check if refactored dashboard exists
        if os.path.exists(os.path.join("ui", "dashboard.py")):
            dashboard_cmd = "python3 -m ui.dashboard"
            logger.info("Using refactored dashboard module")
            print("Using refactored dashboard module")
        else:
            # Fall back to legacy dashboard
            dashboard_cmd = "python3 plotly_dashboard.py"
            logger.info("Using legacy dashboard module")
            print("Using legacy dashboard module")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"LAUNCHING DASHBOARD: {dashboard_cmd}")
        logger.info(f"{'='*80}")
        print(f"\n{'='*80}")
        print(f"LAUNCHING DASHBOARD: {dashboard_cmd}")
        print(f"{'='*80}")
        print("Dashboard will be available at: http://localhost:8050")
    else:
        logger.warning("WARNING: Dashboard dependencies missing. Skipping dashboard launch.")
        print("WARNING: Dashboard dependencies missing. Skipping dashboard launch.")
        print("Install the required dependencies with: pip install plotly dash")
        # Return success without launching dashboard
        return True
    
    # Calculate total runtime
    total_runtime = time.time() - pipeline_start
    logger.info(f"\n{'*'*80}")
    logger.info(f"PIPELINE COMPLETED in {total_runtime:.2f} seconds")
    logger.info(f"{'*'*80}")
    print(f"\n{'*'*80}")
    print(f"PIPELINE COMPLETED in {total_runtime:.2f} seconds")
    print(f"{'*'*80}")
    
    # Launch the dashboard (this will block the script until the dashboard is closed)
    return subprocess.run(dashboard_cmd, shell=True)

def main():
    """
    Parse arguments and run the pipeline
    """
    parser = argparse.ArgumentParser(description='Pizza Predictive Ordering Pipeline')
    parser.add_argument('--skip-time-series', action='store_true', 
                        help='Skip the PyTorch time series model training (faster execution)')
    parser.add_argument('--use-existing-rf', action='store_true',
                        help='Use existing Random Forest model instead of retraining')
    parser.add_argument('--use-weighted-arima', action='store_true',
                        help='Use Weighted ARIMA model instead of standard ARIMA')
    parser.add_argument('--skip-arima', action='store_true',
                        help='Skip the ARIMA model training')
    
    args = parser.parse_args()
    
    run_full_pipeline(
        skip_time_series=args.skip_time_series, 
        use_existing_rf=args.use_existing_rf,
        use_weighted_arima=args.use_weighted_arima,
        skip_arima=args.skip_arima
    )

if __name__ == "__main__":
    main()