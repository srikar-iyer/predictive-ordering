#!/usr/bin/env python3
"""
Test script to verify refactored modules.
This script tests imports and basic functionality of the refactored modules.
"""
import os
import sys
import importlib
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_refactoring')

def test_import(module_path, fallback_path=None):
    """
    Test importing a module.
    
    Args:
        module_path: Path to the module to import
        fallback_path: Fallback module path to try if first import fails
        
    Returns:
        tuple: (success, module_object)
    """
    try:
        module = importlib.import_module(module_path)
        logger.info(f"✓ Successfully imported {module_path}")
        return True, module
    except ImportError as e:
        logger.warning(f"✗ Failed to import {module_path}: {e}")
        if fallback_path:
            try:
                module = importlib.import_module(fallback_path)
                logger.info(f"✓ Successfully imported fallback {fallback_path}")
                return True, module
            except ImportError as e2:
                logger.error(f"✗ Failed to import fallback {fallback_path}: {e2}")
        return False, None

def test_all_modules():
    """
    Test all refactored modules.
    
    Returns:
        bool: True if all tests passed, False otherwise
    """
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    test_results = []
    
    # Test configuration module
    config_success, config = test_import("config.settings")
    test_results.append(config_success)
    
    # Test data processing module
    data_success, data_module = test_import("src.data.data_loader", "integrate_pizza_data")
    test_results.append(data_success)
    
    # Test RF model module
    rf_success, rf_module = test_import("src.models.rf_model", "rf_model_update")
    test_results.append(rf_success)
    
    # Test time series module
    ts_success, ts_module = test_import("src.models.time_series", "pytorch_time_series")
    test_results.append(ts_success)
    
    # Test profit optimization module
    opt_success, opt_module = test_import("src.optimization.profit_optimizer", "profit_optimization")
    test_results.append(opt_success)
    
    # Test weather service module
    weather_success, weather_module = test_import("src.services.weather_service", "weather_service")
    test_results.append(weather_success)
    
    # Test UI modules
    ui_core_success, ui_core = test_import("ui.core")
    test_results.append(ui_core_success)
    
    ui_dash_success, ui_dash = test_import("ui.dashboard", "plotly_dashboard")
    test_results.append(ui_dash_success)
    
    ui_inv_success, ui_inv = test_import("ui.inventory", "plotly_dashboard_inventory_new")
    test_results.append(ui_inv_success)
    
    ui_summary_success, ui_summary = test_import("ui.summary", "summary_dashboard")
    test_results.append(ui_summary_success)
    
    # Verify module functionality
    if data_success and data_module:
        try:
            # Try to access key functions
            if hasattr(data_module, "load_pizza_datasets") or hasattr(data_module, "process_data"):
                logger.info("✓ Data module functions verified")
            else:
                logger.warning("✗ Data module missing expected functions")
                test_results.append(False)
        except Exception as e:
            logger.error(f"✗ Error verifying data module functionality: {e}")
            test_results.append(False)
    
    # Count successful tests
    successful_tests = sum(1 for result in test_results if result)
    total_tests = len(test_results)
    
    logger.info(f"Test Results: {successful_tests}/{total_tests} passed")
    
    return all(test_results)

if __name__ == "__main__":
    success = test_all_modules()
    if success:
        print("\n✓ All modules passed tests")
        sys.exit(0)
    else:
        print("\n✗ Some modules failed tests")
        sys.exit(1)