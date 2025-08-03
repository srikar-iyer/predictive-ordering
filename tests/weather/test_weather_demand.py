from weather_service import WeatherService

ws = WeatherService()
print("Testing get_weather_adjusted_demand:")
try:
    import numpy as np
    base_demand = [100, 120, 110, 105, 95]
    forecast = ws.get_weather_forecast("New York", days=5)
    adjusted, factors = ws.get_weather_adjusted_demand(base_demand, forecast, "frozen pizza")
    print(f"Base demand: {base_demand}")
    print(f"Adjustment factors: {factors}")
    print(f"Adjusted demand: {adjusted}")
except Exception as e:
    print(f"Error: {str(e)}")
