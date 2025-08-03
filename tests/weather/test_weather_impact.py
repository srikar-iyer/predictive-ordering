from weather_service import WeatherService

ws = WeatherService()
print("Testing analyze_weather_impact:")
try:
    print(ws.analyze_weather_impact("Heavy Rain", "frozen pizza", 75))
except Exception as e:
    print(f"Error: {str(e)}")
