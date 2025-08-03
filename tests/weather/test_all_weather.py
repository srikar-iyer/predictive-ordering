#!/usr/bin/env python3
"""
Comprehensive test of the weather API functionality
"""

from weather_service import WeatherService

def test_all_weather_functions():
    """Run tests for all major weather service functions"""
    ws = WeatherService()
    
    # Test 1: Current weather
    print("\n===== Testing get_current_weather =====")
    
    # Test with place name
    print("\nTesting with place name (New York):")
    try:
        current = ws.get_current_weather("New York")
        print(f"Temperature: {current['temperature']:.1f}°C / {current['temperature_f']:.1f}°F")
        print(f"Weather category: {current['weather_category']} (code: {current['weather_code']})")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test with ZIP code
    print("\nTesting with ZIP code (10001):")
    try:
        current = ws.get_current_weather("10001")
        print(f"Temperature: {current['temperature']:.1f}°C / {current['temperature_f']:.1f}°F")
        print(f"Weather category: {current['weather_category']} (code: {current['weather_code']})")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test with lat/lon
    print("\nTesting with latitude/longitude (40.7128, -74.0060):")
    try:
        current = ws.get_current_weather(40.7128, -74.0060)
        print(f"Temperature: {current['temperature']:.1f}°C / {current['temperature_f']:.1f}°F")
        print(f"Weather category: {current['weather_category']} (code: {current['weather_code']})")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test 2: Weather forecast
    print("\n===== Testing get_weather_forecast =====")
    
    print("\nTesting 3-day forecast:")
    try:
        forecast = ws.get_weather_forecast("New York", days=3)
        for day in forecast:
            print(f"{day['date']}: {day['temperature']:.1f}°C / {day['temperature_f']:.1f}°F - {day['weather_category']}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test 3: Weather impact
    print("\n===== Testing analyze_weather_impact =====")
    
    for category in ["Normal", "Heavy Rain", "Snow", "Storm"]:
        impact = ws.analyze_weather_impact(category)
        print(f"{category}: {impact:.2f} impact factor")
    
    # Test with product type and temperature
    print("\nTesting with product type and temperature:")
    impact = ws.analyze_weather_impact("Normal", "frozen pizza", 85)
    print(f"Normal weather, frozen pizza, 85°F: {impact:.2f} impact factor")
    
    # Test 4: Weather adjusted demand
    print("\n===== Testing get_weather_adjusted_demand =====")
    
    try:
        import numpy as np
        base_demand = [100, 120, 110, 95, 105]
        forecast = ws.get_weather_forecast("New York", days=5)
        adjusted, factors = ws.get_weather_adjusted_demand(base_demand, forecast, "frozen pizza")
        
        print("Base demand vs. Adjusted demand:")
        for i in range(len(base_demand)):
            print(f"Day {i+1}: {base_demand[i]} → {adjusted[i]:.1f} (factor: {factors[i]:.2f})")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test 5: Detailed weather impact
    print("\n===== Testing get_detailed_weather_impact =====")
    
    try:
        impact = ws.get_detailed_weather_impact("New York", product_type="frozen pizza")
        if impact:
            print(f"Overall impact factor: {impact['overall_impact_factor']:.2f}")
            print(f"Overall impact: {impact['overall_impact_direction']} by {impact['overall_impact_percent']}")
            
            print("\nDaily impacts:")
            for day in impact['daily_impacts'][:3]:  # Show first 3 days only
                print(f"{day['date']}: {day['weather_category']} at {day['temperature']:.1f}°F → {day['impact_factor']:.2f} ({day['impact_direction']} by {day['impact_percent']})")
        else:
            print("No impact data returned")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Starting comprehensive weather API test...")
    test_all_weather_functions()
    print("\nAll tests completed.")