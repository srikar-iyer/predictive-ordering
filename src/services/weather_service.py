#!/usr/bin/env python3
"""
Weather service module for retrieving and processing weather data.
This module provides weather data for the predictive ordering system.
"""
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import json

# Setup conditional imports to make the module more robust
OPENMETEO_AVAILABLE = True
try:
    import requests_cache
    from retry_requests import retry
    import openmeteo_requests
except ImportError:
    OPENMETEO_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('weather_service')


class WeatherService:
    """
    Service for retrieving weather data for predictive ordering system.
    Uses the OpenMeteo API to get current weather and forecast data.
    
    Example:
    ```
    weather_service = WeatherService()
    
    # Get weather using latitude and longitude
    current_weather = weather_service.get_current_weather(40.9148, -74.3838)
    forecast = weather_service.get_weather_forecast(40.9148, -74.3838)
    
    # Get weather using a ZIP code
    current_weather = weather_service.get_current_weather("10001")
    
    # Get weather using a place name
    current_weather = weather_service.get_current_weather("New York, NY")
    ```
    """
    
    def __init__(self, cache_dir='.cache'):
        """
        Initialize the weather service
        
        Args:
            cache_dir: Directory to store the cache
        """
        # Set up the weather service
        self.cache_dir = cache_dir
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        
        # Initialize API client if dependencies are available
        if OPENMETEO_AVAILABLE:
            try:
                # Setup the Open-Meteo API client with cache and retry on error
                self.cache_session = requests_cache.CachedSession(cache_dir, expire_after=3600)  # Cache for 1 hour
                self.retry_session = retry(self.cache_session, retries=5, backoff_factor=0.2)
                self.openmeteo = openmeteo_requests.Client(session=self.retry_session)
                logger.info("Weather service initialized with OpenMeteo client")
            except Exception as e:
                logger.error(f"Error initializing OpenMeteo client: {str(e)}")
                self.openmeteo = None
        else:
            self.openmeteo = None
            logger.warning("OpenMeteo dependencies not available. Using fallback methods.")
        
        # Weather condition mappings to our system's categories
        # Weather codes from https://open-meteo.com/en/docs
        self.weather_mapping = {
            # Clear
            0: "Normal",  # Clear sky
            1: "Normal",  # Mainly clear
            2: "Normal",  # Partly cloudy
            3: "Normal",  # Overcast
            
            # Fog
            45: "Normal",  # Fog
            48: "Normal",  # Depositing rime fog
            
            # Drizzle
            51: "Heavy Rain",  # Light drizzle
            53: "Heavy Rain",  # Moderate drizzle
            55: "Heavy Rain",  # Dense drizzle
            56: "Heavy Rain",  # Light freezing drizzle
            57: "Heavy Rain",  # Dense freezing drizzle
            
            # Rain
            61: "Heavy Rain",  # Slight rain
            63: "Heavy Rain",  # Moderate rain
            65: "Heavy Rain",  # Heavy rain
            66: "Heavy Rain",  # Light freezing rain
            67: "Heavy Rain",  # Heavy freezing rain
            
            # Snow
            71: "Snow",  # Slight snow fall
            73: "Snow",  # Moderate snow fall
            75: "Snow",  # Heavy snow fall
            77: "Snow",  # Snow grains
            
            # Rain showers
            80: "Heavy Rain",  # Slight rain showers
            81: "Heavy Rain",  # Moderate rain showers
            82: "Heavy Rain",  # Violent rain showers
            
            # Snow showers
            85: "Snow",  # Slight snow showers
            86: "Snow",  # Heavy snow showers
            
            # Thunderstorm
            95: "Storm",  # Thunderstorm: Slight or moderate
            96: "Storm",  # Thunderstorm with slight hail
            99: "Storm"   # Thunderstorm with heavy hail
        }

    def _map_weather_condition(self, weather_code):
        """
        Map OpenMeteo weather code to our system's weather categories
        
        Args:
            weather_code: OpenMeteo weather code
            
        Returns:
            str: Mapped weather category
        """
        return self.weather_mapping.get(weather_code, "Normal")
    
    def _convert_temp_to_fahrenheit(self, temp_celsius):
        """
        Convert temperature from Celsius to Fahrenheit
        
        Args:
            temp_celsius: Temperature in Celsius
            
        Returns:
            float: Temperature in Fahrenheit
        """
        return (temp_celsius * 9/5) + 32
    
    def _get_coordinates_from_zipcode(self, zipcode):
        """
        Convert a ZIP code to latitude and longitude coordinates.
        
        Args:
            zipcode: US ZIP code
            
        Returns:
            tuple: (latitude, longitude) or None if not found
        """
        try:
            # Use a free geocoding service to convert ZIP code to coordinates
            url = f"https://api.zippopotam.us/us/{zipcode}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                latitude = float(data['places'][0]['latitude'])
                longitude = float(data['places'][0]['longitude'])
                return latitude, longitude
            else:
                logger.warning(f"Failed to get coordinates for ZIP code {zipcode}: {response.status_code}")
                # Return default coordinates for New York City as fallback
                return 40.7128, -74.0060
                
        except Exception as e:
            logger.error(f"Error getting coordinates from ZIP code: {str(e)}")
            # Return default coordinates for New York City as fallback
            return 40.7128, -74.0060
    
    def _get_coordinates_from_place(self, place_name):
        """
        Convert a place name to latitude and longitude coordinates.
        
        Args:
            place_name: Name of the place (city, address, etc.)
            
        Returns:
            tuple: (latitude, longitude) or None if not found
        """
        try:
            # Use OpenStreetMap Nominatim API (please be considerate with usage)
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                "q": place_name,
                "format": "json",
                "limit": 1
            }
            headers = {
                "User-Agent": "PizzaPredictiveOrdering/1.0"  # Be polite and identify your application
            }
            
            response = requests.get(url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    latitude = float(data[0]['lat'])
                    longitude = float(data[0]['lon'])
                    return latitude, longitude
                else:
                    logger.warning(f"Place not found: {place_name}")
            else:
                logger.warning(f"Failed to get coordinates for {place_name}: {response.status_code}")
            
            # Return default coordinates for New York City as fallback
            return 40.7128, -74.0060
                
        except Exception as e:
            logger.error(f"Error getting coordinates from place name: {str(e)}")
            # Return default coordinates for New York City as fallback
            return 40.7128, -74.0060
    
    def get_current_weather(self, location, fallback=True):
        """
        Get current weather for a location.
        
        Args:
            location: Can be coordinates (lat, lon) tuple, ZIP code string, or place name string
            fallback: Whether to use fallback data if API fails
            
        Returns:
            dict: Weather data including temperature, condition, etc.
        """
        # Convert location to coordinates if needed
        if isinstance(location, tuple) and len(location) == 2:
            latitude, longitude = location
        elif isinstance(location, str):
            # Check if it's a ZIP code (5 digits)
            if location.isdigit() and len(location) == 5:
                latitude, longitude = self._get_coordinates_from_zipcode(location)
            else:
                # Assume it's a place name
                latitude, longitude = self._get_coordinates_from_place(location)
        else:
            raise ValueError("Location must be coordinates tuple, ZIP code string, or place name string")
        
        # Use OpenMeteo client if available
        if self.openmeteo is not None:
            try:
                # Make API request to Open-Meteo
                params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": ["temperature_2m", "weather_code"],
                    "temperature_unit": "celsius"
                }
                
                response = self.openmeteo.weather_api(self.base_url, params=params)
                
                # Process the response
                current = response[0].current()
                
                # Map the weather code to our system's categories
                weather_code = current.variables(0).value()
                weather_condition = self._map_weather_condition(weather_code)
                
                # Get temperature in both Celsius and Fahrenheit
                temp_celsius = current.variables(1).value()
                temp_fahrenheit = self._convert_temp_to_fahrenheit(temp_celsius)
                
                # Return the weather data
                return {
                    "condition": weather_condition,
                    "temperature_c": temp_celsius,
                    "temperature_f": temp_fahrenheit,
                    "weather_code": weather_code,
                    "source": "OpenMeteo API",
                    "coordinates": (latitude, longitude),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
            except Exception as e:
                logger.error(f"Error getting weather data: {str(e)}")
                if not fallback:
                    raise
        
        if fallback:
            # Generate mock weather data as fallback
            logger.warning("Using fallback weather data")
            return self._generate_mock_weather_data(latitude, longitude)
        
        raise RuntimeError("Failed to get weather data and fallback is disabled")
    
    def get_weather_forecast(self, location, days=7, fallback=True):
        """
        Get weather forecast for a location.
        
        Args:
            location: Can be coordinates (lat, lon) tuple, ZIP code string, or place name string
            days: Number of days to forecast (max 7)
            fallback: Whether to use fallback data if API fails
            
        Returns:
            dict: Forecast data for each day
        """
        # Limit days to 7 (OpenMeteo free tier limit)
        days = min(days, 7)
        
        # Convert location to coordinates if needed
        if isinstance(location, tuple) and len(location) == 2:
            latitude, longitude = location
        elif isinstance(location, str):
            # Check if it's a ZIP code (5 digits)
            if location.isdigit() and len(location) == 5:
                latitude, longitude = self._get_coordinates_from_zipcode(location)
            else:
                # Assume it's a place name
                latitude, longitude = self._get_coordinates_from_place(location)
        else:
            raise ValueError("Location must be coordinates tuple, ZIP code string, or place name string")
        
        # Use OpenMeteo client if available
        if self.openmeteo is not None:
            try:
                # Make API request to Open-Meteo
                params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
                    "temperature_unit": "celsius",
                    "forecast_days": days
                }
                
                response = self.openmeteo.weather_api(self.base_url, params=params)
                
                # Process the response
                daily = response[0].daily()
                
                # Extract the dates
                forecast_dates = daily.variables(0).days()
                
                # Create forecast data
                forecast = []
                
                for i in range(len(forecast_dates)):
                    date = forecast_dates[i].strftime("%Y-%m-%d")
                    weather_code = daily.variables(1).value(i)
                    temp_max_c = daily.variables(2).value(i)
                    temp_min_c = daily.variables(3).value(i)
                    precipitation = daily.variables(4).value(i)
                    
                    # Map weather code to our categories
                    weather_condition = self._map_weather_condition(weather_code)
                    
                    # Add to forecast
                    forecast.append({
                        "date": date,
                        "condition": weather_condition,
                        "weather_code": weather_code,
                        "temp_max_c": temp_max_c,
                        "temp_min_c": temp_min_c,
                        "temp_max_f": self._convert_temp_to_fahrenheit(temp_max_c),
                        "temp_min_f": self._convert_temp_to_fahrenheit(temp_min_c),
                        "precipitation_mm": precipitation
                    })
                
                return {
                    "forecast": forecast,
                    "source": "OpenMeteo API",
                    "coordinates": (latitude, longitude),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
            except Exception as e:
                logger.error(f"Error getting forecast data: {str(e)}")
                if not fallback:
                    raise
        
        if fallback:
            # Generate mock forecast data as fallback
            logger.warning("Using fallback forecast data")
            return self._generate_mock_forecast_data(latitude, longitude, days)
        
        raise RuntimeError("Failed to get forecast data and fallback is disabled")
    
    def _generate_mock_weather_data(self, latitude, longitude):
        """
        Generate mock weather data for testing and fallback.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            dict: Mock weather data
        """
        # Set random seed based on coordinates for consistent behavior
        seed = int((latitude + 180) * 100 + (longitude + 360))
        np.random.seed(seed)
        
        # Generate random weather condition
        weather_types = ["Normal", "Heavy Rain", "Snow", "Storm"]
        weights = [0.7, 0.15, 0.1, 0.05]  # Normal weather is more common
        weather_condition = np.random.choice(weather_types, p=weights)
        
        # Generate random temperature based on weather condition
        if weather_condition == "Normal":
            temp_celsius = np.random.uniform(15, 30)  # Pleasant temperature
        elif weather_condition == "Heavy Rain":
            temp_celsius = np.random.uniform(10, 20)  # Cooler, rainy
        elif weather_condition == "Snow":
            temp_celsius = np.random.uniform(-5, 5)   # Cold, snowy
        else:  # Storm
            temp_celsius = np.random.uniform(10, 25)  # Variable for storms
        
        # Reset random seed to avoid affecting other components
        np.random.seed(None)
        
        return {
            "condition": weather_condition,
            "temperature_c": temp_celsius,
            "temperature_f": self._convert_temp_to_fahrenheit(temp_celsius),
            "weather_code": None,  # No specific code for mock data
            "source": "Mock Data (API not available)",
            "coordinates": (latitude, longitude),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _generate_mock_forecast_data(self, latitude, longitude, days):
        """
        Generate mock forecast data for testing and fallback.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            days: Number of days to forecast
            
        Returns:
            dict: Mock forecast data
        """
        # Set random seed based on coordinates for consistent behavior
        seed = int((latitude + 180) * 100 + (longitude + 360))
        np.random.seed(seed)
        
        # Generate forecast for each day
        forecast = []
        
        # Start with today's date
        current_date = datetime.now()
        
        # Weather patterns with some continuity
        weather_types = ["Normal", "Heavy Rain", "Snow", "Storm"]
        weights = [0.7, 0.15, 0.1, 0.05]  # Normal weather is more common
        
        # Start with a random weather
        current_weather = np.random.choice(weather_types, p=weights)
        
        for i in range(days):
            # Date for this forecast
            date = current_date + timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            
            # Weather has 70% chance to stay the same, 30% chance to change
            if np.random.random() > 0.7 or i == 0:
                current_weather = np.random.choice(weather_types, p=weights)
            
            # Generate temperatures based on weather
            if current_weather == "Normal":
                temp_max_c = np.random.uniform(20, 30)
                temp_min_c = np.random.uniform(15, temp_max_c - 2)
            elif current_weather == "Heavy Rain":
                temp_max_c = np.random.uniform(15, 25)
                temp_min_c = np.random.uniform(10, temp_max_c - 2)
            elif current_weather == "Snow":
                temp_max_c = np.random.uniform(0, 5)
                temp_min_c = np.random.uniform(-10, temp_max_c - 2)
            else:  # Storm
                temp_max_c = np.random.uniform(15, 28)
                temp_min_c = np.random.uniform(12, temp_max_c - 3)
            
            # Generate precipitation
            if current_weather == "Normal":
                precipitation = np.random.uniform(0, 1)
            elif current_weather == "Heavy Rain":
                precipitation = np.random.uniform(5, 20)
            elif current_weather == "Snow":
                precipitation = np.random.uniform(2, 15)
            else:  # Storm
                precipitation = np.random.uniform(10, 30)
            
            # Add to forecast
            forecast.append({
                "date": date_str,
                "condition": current_weather,
                "weather_code": None,  # No specific code for mock data
                "temp_max_c": temp_max_c,
                "temp_min_c": temp_min_c,
                "temp_max_f": self._convert_temp_to_fahrenheit(temp_max_c),
                "temp_min_f": self._convert_temp_to_fahrenheit(temp_min_c),
                "precipitation_mm": precipitation
            })
        
        # Reset random seed to avoid affecting other components
        np.random.seed(None)
        
        return {
            "forecast": forecast,
            "source": "Mock Data (API not available)",
            "coordinates": (latitude, longitude),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def analyze_weather_impact(self, weather_data, product_category=None):
        """
        Analyze the potential impact of weather on sales.
        
        Args:
            weather_data: Weather data from get_current_weather or get_weather_forecast
            product_category: Optional product category to get specific impact
            
        Returns:
            dict: Analysis of weather impact on sales
        """
        # Define default weather impacts by weather condition
        weather_impacts = {
            "Normal": {
                "frozen_pizza": 1.0,  # Neutral impact
                "pizza": 1.0,
                "ice_cream": 1.2,     # Slight boost
                "soup": 0.9,          # Slight decrease
                "coffee": 1.0,
                "overall": 1.0
            },
            "Heavy Rain": {
                "frozen_pizza": 1.3,  # Significant boost
                "pizza": 1.2,         # Delivery boost
                "ice_cream": 0.8,     # Decrease
                "soup": 1.2,          # Boost
                "coffee": 1.1,        # Slight boost
                "overall": 1.1
            },
            "Snow": {
                "frozen_pizza": 1.5,  # Major boost
                "pizza": 0.8,         # Decrease (hard to deliver)
                "ice_cream": 0.5,     # Major decrease
                "soup": 1.4,          # Major boost
                "coffee": 1.2,        # Boost
                "overall": 1.2
            },
            "Storm": {
                "frozen_pizza": 1.8,  # Extreme boost
                "pizza": 0.6,         # Major decrease (delivery issues)
                "ice_cream": 0.6,     # Major decrease
                "soup": 1.5,          # Major boost
                "coffee": 1.3,        # Significant boost
                "overall": 1.3
            }
        }
        
        # Extract weather condition
        if "forecast" in weather_data:
            # For forecast data, use the first day
            condition = weather_data["forecast"][0]["condition"]
        else:
            # For current weather data
            condition = weather_data["condition"]
        
        # Get impact data for this weather condition
        impact_data = weather_impacts.get(condition, weather_impacts["Normal"])
        
        # Return specific category impact or general impact
        if product_category and product_category in impact_data:
            impact = impact_data[product_category]
            impact_description = self._get_impact_description(impact)
            
            return {
                "weather_condition": condition,
                "product_category": product_category,
                "sales_impact_factor": impact,
                "impact_description": impact_description,
                "source": weather_data.get("source", "Unknown")
            }
        else:
            # Return overall impact
            overall_impact = {}
            for category, impact in impact_data.items():
                overall_impact[category] = {
                    "sales_impact_factor": impact,
                    "impact_description": self._get_impact_description(impact)
                }
                
            return {
                "weather_condition": condition,
                "sales_impacts": overall_impact,
                "source": weather_data.get("source", "Unknown")
            }
    
    def _get_impact_description(self, impact_factor):
        """
        Get a textual description of the impact factor.
        
        Args:
            impact_factor: Numeric impact factor
            
        Returns:
            str: Description of the impact
        """
        if impact_factor >= 1.5:
            return "Major increase in demand"
        elif impact_factor >= 1.2:
            return "Significant increase in demand"
        elif impact_factor >= 1.1:
            return "Slight increase in demand"
        elif impact_factor > 0.9:
            return "Neutral impact on demand"
        elif impact_factor > 0.7:
            return "Slight decrease in demand"
        elif impact_factor > 0.5:
            return "Significant decrease in demand"
        else:
            return "Major decrease in demand"
    
    def adjust_demand_forecast(self, forecast_df, location, product_type="frozen_pizza"):
        """
        Adjust a demand forecast based on weather predictions.
        
        Args:
            forecast_df: DataFrame with demand forecasts
            location: Location to get weather forecast for
            product_type: Type of product for weather impact
            
        Returns:
            DataFrame: Adjusted forecast DataFrame
        """
        if not isinstance(forecast_df, pd.DataFrame):
            raise ValueError("forecast_df must be a pandas DataFrame")
            
        if 'Date' not in forecast_df.columns or 'Forecast' not in forecast_df.columns:
            raise ValueError("forecast_df must contain 'Date' and 'Forecast' columns")
        
        # Make a copy of the forecast DataFrame
        adjusted_df = forecast_df.copy()
        
        try:
            # Ensure Date is datetime
            if not pd.api.types.is_datetime64_any_dtype(adjusted_df['Date']):
                adjusted_df['Date'] = pd.to_datetime(adjusted_df['Date'])
                
            # Get weather forecast
            weather_data = self.get_weather_forecast(location, days=7)
            
            if "forecast" not in weather_data:
                logger.warning("No weather forecast available for adjustment")
                return adjusted_df
                
            # Create a mapping of date to impact factor
            weather_impacts = {}
            
            for day in weather_data["forecast"]:
                date = datetime.strptime(day["date"], "%Y-%m-%d").date()
                
                # Analyze weather impact for this day
                impact_analysis = self.analyze_weather_impact(
                    {"condition": day["condition"]}, 
                    product_category=product_type
                )
                
                # Store the impact factor
                weather_impacts[date] = impact_analysis["sales_impact_factor"]
            
            # Apply the adjustment factors to the forecast
            for i, row in adjusted_df.iterrows():
                forecast_date = row['Date'].date()
                
                if forecast_date in weather_impacts:
                    impact_factor = weather_impacts[forecast_date]
                    
                    # Apply the impact factor to the forecast
                    adjusted_df.at[i, 'Original_Forecast'] = row['Forecast']
                    adjusted_df.at[i, 'Forecast'] = row['Forecast'] * impact_factor
                    adjusted_df.at[i, 'Weather_Impact'] = impact_factor
                    adjusted_df.at[i, 'Weather_Condition'] = next(
                        (day["condition"] for day in weather_data["forecast"] 
                         if datetime.strptime(day["date"], "%Y-%m-%d").date() == forecast_date), 
                        "Unknown"
                    )
            
            logger.info("Applied weather adjustments to forecast")
            return adjusted_df
            
        except Exception as e:
            logger.error(f"Error adjusting forecast for weather: {str(e)}")
            return forecast_df  # Return original forecast on error


def get_weather_service():
    """
    Factory function to get a weather service instance.
    
    Returns:
        WeatherService: A configured weather service
    """
    return WeatherService()


def main():
    """
    Main function for testing the weather service.
    """
    import argparse
    
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
            print(json.dumps(forecast, indent=2))
        else:
            # Get current weather
            weather = weather_service.get_current_weather(args.location)
            print(json.dumps(weather, indent=2))
            
            # Show impact analysis
            impact = weather_service.analyze_weather_impact(weather)
            print("\nWeather Impact Analysis:")
            print(json.dumps(impact, indent=2))
            
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()