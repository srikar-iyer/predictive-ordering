from weather_service import WeatherService

ws = WeatherService()
print("Testing get_current_weather with ZIP code:")
try:
    print(ws.get_current_weather("10001"))
except Exception as e:
    print(f"Error: {str(e)}")
