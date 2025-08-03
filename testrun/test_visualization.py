"""
Test script to verify Plotly visualizations with the fixed hovertemplate.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path so we can import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import visualization modules
from src.models.plotly_visualizations import create_elasticity_distribution_plot
from src.models.interactive_features import enhance_elasticity_distribution

def test_elasticity_distribution():
    """Test elasticity distribution plot with hover fixes"""
    # Create sample elasticity data
    data = {
        'Store_Id': [104.0] * 20,
        'Item': range(1, 21),
        'Product': [f"Product {i}" for i in range(1, 21)],
        'Elasticity': np.random.normal(-1.5, 0.5, 20),
        'R_Squared': np.random.uniform(0.1, 0.9, 20),
        'Is_Significant': [True, False] * 10,
        'Current_Price': np.random.uniform(2.99, 9.99, 20),
        'Cost': np.random.uniform(1.5, 5.0, 20),
        'Margin_Pct': np.random.uniform(20, 60, 20)
    }
    
    elasticity_df = pd.DataFrame(data)
    
    # Create the elasticity distribution plot
    fig = create_elasticity_distribution_plot(elasticity_df)
    
    # Check if the plot has hoverinfo set
    has_hover_info = False
    if hasattr(fig, 'data') and len(fig.data) > 0:
        for trace in fig.data:
            if hasattr(trace, 'hoverinfo') and trace.hoverinfo == 'x+y+text':
                has_hover_info = True
                break
    
    print(f"Elasticity distribution plot has hoverinfo set: {has_hover_info}")
    
    # Test with enhancement
    enhanced_fig = enhance_elasticity_distribution(fig, elasticity_df)
    
    # Check if the enhanced plot has hoverinfo set
    enhanced_has_hover_info = False
    if hasattr(enhanced_fig, 'data') and len(enhanced_fig.data) > 0:
        for trace in enhanced_fig.data:
            if hasattr(trace, 'hoverinfo') and trace.hoverinfo == 'x+y+text':
                enhanced_has_hover_info = True
                break
    
    print(f"Enhanced elasticity distribution plot has hoverinfo set: {enhanced_has_hover_info}")
    
    return fig, enhanced_fig

if __name__ == "__main__":
    # Run tests
    fig, enhanced_fig = test_elasticity_distribution()
    
    # Output results
    print("\nTest completed!")
    print("To view the figures, you would normally use fig.show() in a Jupyter notebook environment.")