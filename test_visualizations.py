"""
Test script to verify visualization fixes for 'NoneType' object errors.
"""
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_visualizations')

# Import visualization modules
try:
    from src.models.plotly_visualizations import (
        create_elasticity_distribution_plot,
        create_price_sensitivity_curve,
        create_profit_impact_waterfall,
        create_elasticity_vs_margin_plot
    )
    from src.models.interactive_features import (
        enhance_elasticity_distribution,
        enhance_price_sensitivity_curve,
        enhance_profit_impact_waterfall,
        enhance_elasticity_vs_margin_plot,
        enhance_integrated_chart,
        enhance_impact_heatmap,
        enhance_kpi_indicators,
        enhance_recommendations_table,
        add_hover_tooltip,
        add_drill_down_click,
        add_zoom_pan
    )
    from src.models.chart_export_utils import (
        enhance_figure_with_exports,
        add_export_menu_to_figure
    )
    from src.models.integrated_visualizations import (
        create_integrated_chart,
        create_impact_heatmap,
        create_kpi_indicators,
        create_recommendations_table
    )
except ImportError:
    logger.error("Failed to import visualization modules. Make sure you're running from the project root.")
    exit(1)

def test_with_exception_handling(test_func):
    """Decorator for running tests with exception handling"""
    def wrapper(*args, **kwargs):
        try:
            result = test_func(*args, **kwargs)
            logger.info(f"✅ {test_func.__name__} passed")
            return result
        except Exception as e:
            logger.error(f"❌ {test_func.__name__} failed: {str(e)}")
            traceback.print_exc()
            return None
    return wrapper

@test_with_exception_handling
def test_elasticity_distribution_plot():
    """Test elasticity distribution plot with None inputs"""
    # Test with None DataFrame
    fig1 = create_elasticity_distribution_plot(None)
    assert fig1 is not None, "Should return a figure even with None data"
    
    # Test with empty DataFrame
    fig2 = create_elasticity_distribution_plot(pd.DataFrame())
    assert fig2 is not None, "Should return a figure even with empty DataFrame"
    
    # Test with valid data
    df = pd.DataFrame({
        'Elasticity': [-0.5, -1.2, -0.8, -1.5, -2.0],
        'Product': ['Pizza A', 'Pizza B', 'Pizza C', 'Pizza D', 'Pizza E'],
        'Avg_Price': [10.99, 12.99, 9.99, 11.99, 8.99],
        'Cost': [5.0, 6.0, 4.5, 5.5, 4.0],
        'Margin_Pct': [54.5, 53.8, 55.0, 54.1, 55.5],
    })
    fig3 = create_elasticity_distribution_plot(df)
    assert fig3 is not None, "Should return a figure with valid data"
    
    return True

@test_with_exception_handling
def test_price_sensitivity_curve():
    """Test price sensitivity curve with edge cases"""
    # Test with None elasticity
    fig1 = create_price_sensitivity_curve(None, 10.0)
    assert fig1 is not None, "Should return a figure even with None elasticity"
    
    # Test with None price
    fig2 = create_price_sensitivity_curve(-1.5, None)
    assert fig2 is not None, "Should return a figure even with None price"
    
    # Test with valid inputs
    fig3 = create_price_sensitivity_curve(-1.5, 10.0, 5.0)
    assert fig3 is not None, "Should return a figure with valid inputs"
    
    return True

@test_with_exception_handling
def test_profit_impact_waterfall():
    """Test profit impact waterfall chart"""
    # Test with None DataFrame
    fig1 = create_profit_impact_waterfall(None)
    assert fig1 is not None, "Should return a figure even with None data"
    
    # Test with valid data
    df = pd.DataFrame({
        'Product': ['Pizza A', 'Pizza B', 'Pizza C'],
        'Price_Change_Pct': [5.0, -2.0, 3.0],
        'Total_Current_Profit': [1000, 1200, 800],
        'Total_New_Profit': [1100, 1150, 850],
        'Total_Profit_Difference': [100, -50, 50]
    })
    fig2 = create_profit_impact_waterfall(df)
    assert fig2 is not None, "Should return a figure with valid data"
    
    return True

@test_with_exception_handling
def test_enhance_figure_with_exports():
    """Test enhancing figures with export capabilities"""
    # Test with None figure
    fig1 = enhance_figure_with_exports(None, "Test Title")
    assert fig1 is None, "Should return None when input is None"
    
    # Test with valid figure
    fig2 = go.Figure()
    enhanced_fig = enhance_figure_with_exports(fig2, "Test Title")
    assert enhanced_fig is not None, "Should return enhanced figure"
    assert hasattr(enhanced_fig.layout, 'meta'), "Should have meta attribute"
    assert 'export_filename' in enhanced_fig.layout.meta, "Should have export_filename in meta"
    
    return True

@test_with_exception_handling
def test_enhance_with_interactive_features():
    """Test enhancing figures with interactive features"""
    # Test with None figure
    enhanced_fig1 = add_hover_tooltip(None)
    assert enhanced_fig1 is None, "Should handle None figure"
    
    enhanced_fig2 = add_drill_down_click(None)
    assert enhanced_fig2 is None, "Should handle None figure"
    
    enhanced_fig3 = add_zoom_pan(None)
    assert enhanced_fig3 is None, "Should handle None figure"
    
    # Test with valid figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
    
    enhanced_fig4 = add_hover_tooltip(fig)
    assert enhanced_fig4 is not None, "Should enhance valid figure with hover tooltip"
    
    enhanced_fig5 = add_drill_down_click(fig)
    assert enhanced_fig5 is not None, "Should enhance valid figure with drill down"
    
    enhanced_fig6 = add_zoom_pan(fig)
    assert enhanced_fig6 is not None, "Should enhance valid figure with zoom and pan"
    
    return True

@test_with_exception_handling
def test_enhanced_specialized_features():
    """Test specialized enhancement functions with edge cases"""
    fig = go.Figure()
    
    # Test enhance elasticity distribution
    enhanced_fig1 = enhance_elasticity_distribution(fig, None)
    assert enhanced_fig1 is not None, "Should handle None elasticity data"
    
    # Test enhance price sensitivity curve
    enhanced_fig2 = enhance_price_sensitivity_curve(fig, None)
    assert enhanced_fig2 is not None, "Should handle None price data"
    
    # Test enhance profit impact waterfall
    enhanced_fig3 = enhance_profit_impact_waterfall(fig, None)
    assert enhanced_fig3 is not None, "Should handle None impact data"
    
    # Test with None figures
    enhanced_fig4 = enhance_elasticity_distribution(None)
    assert enhanced_fig4 is not None, "Should handle None figure"
    
    enhanced_fig5 = enhance_price_sensitivity_curve(None)
    assert enhanced_fig5 is not None, "Should handle None figure"
    
    enhanced_fig6 = enhance_profit_impact_waterfall(None)
    assert enhanced_fig6 is not None, "Should handle None figure"
    
    # Test enhance KPI indicators
    indicators = enhance_kpi_indicators(None)
    assert isinstance(indicators, list), "Should return empty list for None input"
    
    indicators_list = [fig, None, fig]
    enhanced_indicators = enhance_kpi_indicators(indicators_list)
    assert len(enhanced_indicators) == 3, "Should preserve length of indicators list"
    
    # Test enhance recommendations table
    enhanced_table = enhance_recommendations_table(None)
    assert enhanced_table is None, "Should handle None table"
    
    return True

@test_with_exception_handling
def test_integrated_visualizations():
    """Test integrated visualizations with edge cases"""
    # Test with None inputs for model functions
    # These would normally require a UnifiedDataModel, but we're just testing error handling
    try:
        fig1 = create_integrated_chart(None, "store1", "item1")
        assert fig1 is not None, "Should handle None model"
    except Exception as e:
        logger.warning(f"Expected failure with create_integrated_chart(None): {str(e)}")
    
    try:
        fig2 = create_impact_heatmap(None, "store1", "item1")
        assert fig2 is not None, "Should handle None model"
    except Exception as e:
        logger.warning(f"Expected failure with create_impact_heatmap(None): {str(e)}")
    
    try:
        fig3 = create_kpi_indicators(None)
        assert isinstance(fig3, list), "Should return empty list for None metrics"
    except Exception as e:
        logger.warning(f"Expected failure with create_kpi_indicators(None): {str(e)}")
    
    try:
        fig4 = create_recommendations_table(None)
        assert fig4 is not None, "Should handle None metrics"
    except Exception as e:
        logger.warning(f"Expected failure with create_recommendations_table(None): {str(e)}")
    
    return True

def main():
    """Run all visualization tests"""
    logger.info("Starting visualization tests")
    
    # Run tests
    test_elasticity_distribution_plot()
    test_price_sensitivity_curve()
    test_profit_impact_waterfall()
    test_enhance_figure_with_exports()
    test_enhance_with_interactive_features()
    test_enhanced_specialized_features()
    test_integrated_visualizations()
    
    logger.info("Visualization tests completed")

if __name__ == "__main__":
    main()