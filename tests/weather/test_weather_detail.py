from weather_service import WeatherService

ws = WeatherService()
print("Testing get_detailed_weather_impact:")
try:
    impact = ws.get_detailed_weather_impact("New York", product_type="frozen pizza")
    if impact:
        for key in impact.keys():
            if key != "current_weather" and key != "daily_impacts":
                print(f"{key}: {impact[key]}")
            else:
                print(f"{key}: ...")
    else:
        print("No impact data returned")
except Exception as e:
    print(f"Error: {str(e)}")