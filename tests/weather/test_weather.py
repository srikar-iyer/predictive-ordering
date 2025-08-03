from weather_service import WeatherService

ws = WeatherService()
print("Testing get_current_weather:")
try:
    print(ws.get_current_weather("New York"))
except Exception as e:
    print(f"Error: {str(e)}")
