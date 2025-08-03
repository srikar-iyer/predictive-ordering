# Inventory Management Module

This module enhances the pizza predictive ordering system with comprehensive inventory management capabilities. It integrates real on-hand inventory data and provides data-driven recommendations to optimize inventory levels.

## Key Features

### 1. Data Integration
- Incorporates `optional_on_hand_data.csv` as the source of truth for current inventory levels
- Uses actual on-hand counts, historical movement data, and inventory metrics 
- No hardcoded values - all calculations are data-driven

### 2. Inventory Management Tab
- **Inventory Visualization**: Visual chart showing inventory levels, safety stock, and target stock
- **Manual Inventory Count Updates**: Ability to manually enter current inventory counts
- **Smart Recommendations**: Data-driven order recommendations with clear business reasoning

### 3. Inventory Performance Analytics
- **Inventory Turnover Rate**: Measures how quickly inventory sells using actual sales data
- **Dynamic Target Setting**: Sets target turnover rates based on historical performance
- **Cost Impact Analysis**: Calculates carrying costs for excess inventory and opportunity costs for stockouts

### 4. User-Friendly Interface
- **Business-Focused Terminology**: Uses non-technical terms that business users can understand
- **Visual Indicators**: Badges identify items that need attention (needs reordering, fast/slow movers)
- **Action-Oriented Recommendations**: Clear guidance on what actions to take

## Technical Implementation

### Data Processing
The system dynamically calculates:
- Inventory turnover rates based on historical sales
- Safety stock levels based on actual demand patterns
- Target stock levels for optimal inventory management
- Stockout risk based on current inventory levels and demand
- Carrying costs for excess inventory
- Lost sales opportunity costs for low inventory

### Key Files
- `integrate_on_hand_data.py`: Integrates the on-hand inventory data into the combined dataset
- `plotly_dashboard_inventory.py`: Core inventory management functionality
- `plotly_dashboard.py`: Main dashboard with inventory management tabs

### Using the System
1. View current inventory levels in the "Inventory Management" tab
2. Update inventory counts manually when physical counts are performed
3. Follow ordering recommendations to maintain optimal inventory levels
4. Monitor inventory performance in the "Inventory Performance" tab
5. Track cost impacts of inventory decisions

All calculations adapt to your actual sales patterns and inventory data - no assumptions or hardcoded values are used.