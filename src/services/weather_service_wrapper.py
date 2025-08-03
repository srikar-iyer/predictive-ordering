#!/usr/bin/env python3
"""
Wrapper module for weather service to maintain compatibility with the original codebase.
"""
import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.services.weather_service import get_weather_service, WeatherService

def main():
    """
    Main function for testing the weather service.
    """
    parser = argparse.ArgumentParser(description='Weather Service')
    parser.add_argument('--location', type=str, default="10001",
                        help='Location (ZIP code, coordinates, or place name)')
    parser.add_argument('--forecast', action='store_true',
                        help='Get weather forecast instead of current weather')
    parser.add_argument('--days', type=int, default=7,
                        help='Number of days for forecast')
    
    args = parser.parse_args()
    
    # Create weather service
    weather_service = WeatherService()
    
    try:
        if args.forecast:
            # Get forecast
            forecast = weather_service.get_weather_forecast(args.location, days=args.days)
            print(forecast)
        else:
            # Get current weather
            weather = weather_service.get_current_weather(args.location)
            print(weather)
            
            # Show impact analysis
            impact = weather_service.analyze_weather_impact(weather)
            print("\nWeather Impact Analysis:")
            print(impact)
            
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()