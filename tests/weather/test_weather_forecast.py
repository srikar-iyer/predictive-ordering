from weather_service import WeatherService

ws = WeatherService()
print("Testing get_weather_forecast:")
try:
    print(ws.get_weather_forecast("New York"))
except Exception as e:
    print(f"Error: {str(e)}")
