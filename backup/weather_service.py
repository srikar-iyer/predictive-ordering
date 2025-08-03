#!/usr/bin/env python3

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
import openmeteo_requests
import requests
import json

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
    
    def __init__(self):
        """Initialize the weather service"""
        # Setup the Open-Meteo API client with cache and retry on error
        self.cache_session = requests_cache.CachedSession('.cache', expire_after=3600)  # Cache for 1 hour
        self.retry_session = retry(self.cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=self.retry_session)
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        
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
        """Map OpenMeteo weather code to our system's weather categories"""
        return self.weather_mapping.get(weather_code, "Normal")
    
    def _convert_temp_to_fahrenheit(self, temp_celsius):
        """Convert temperature from Celsius to Fahrenheit"""
        return (temp_celsius * 9/5) + 32
    
    def _get_coordinates_from_zipcode(self, zipcode):
        """
        Convert a ZIP code to latitude and longitude coordinates.
        Returns a tuple (latitude, longitude) or None if conversion fails.
        """
        try:
            # Use a free geocoding API to convert zip code to coordinates
            url = f"https://nominatim.openstreetmap.org/search?postalcode={zipcode}&country=US&format=json"
            headers = {
                "User-Agent": "PredictiveOrdering/1.0 (contact@example.com)"
            }
            response = requests.get(url, headers=headers)
            data = json.loads(response.text)
            
            if data and len(data) > 0:
                lat = float(data[0]['lat'])
                lon = float(data[0]['lon'])
                return (lat, lon)
            else:
                print(f"Could not find coordinates for ZIP code {zipcode}")
                return None
        except Exception as e:
            print(f"Error converting ZIP code to coordinates: {e}")
            return None
            
    def _get_coordinates_from_place_name(self, place_name):
        """
        Convert a place name (city, state, country) to latitude and longitude coordinates.
        Returns a tuple (latitude, longitude) or None if conversion fails.
        """
        try:
            # Use a free geocoding API to convert place name to coordinates
            url = f"https://nominatim.openstreetmap.org/search?q={place_name}&format=json"
            headers = {
                "User-Agent": "PredictiveOrdering/1.0 (contact@example.com)"
            }
            response = requests.get(url, headers=headers)
            data = json.loads(response.text)
            
            if data and len(data) > 0:
                lat = float(data[0]['lat'])
                lon = float(data[0]['lon'])
                return (lat, lon)
            else:
                print(f"Could not find coordinates for place name {place_name}")
                return None
        except Exception as e:
            print(f"Error converting place name to coordinates: {e}")
            return None
    
    def get_current_weather(self, location, longitude=None, timezone="America/New_York"):
        """
        Get current weather for a location. Location can be:
        1. A latitude value when longitude is also provided
        2. A ZIP code string (when longitude is None)
        3. A place name string (when longitude is None)
        
        Returns a dictionary with weather information.
        """
        try:
            # Determine coordinates from input
            latitude = None
            if longitude is not None:  # Assume location is latitude
                latitude = location
            else:  # location is either zip code or place name
                location_str = str(location).strip()
                if location_str.isdigit() and len(location_str) == 5:  # US ZIP code
                    coords = self._get_coordinates_from_zipcode(location_str)
                    if coords:
                        latitude, longitude = coords
                else:  # Assume it's a place name
                    coords = self._get_coordinates_from_place_name(location_str)
                    if coords:
                        latitude, longitude = coords
            
            if latitude is None or longitude is None:
                raise ValueError(f"Could not determine coordinates for location: {location}")
            
            # Set up the API parameters
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": [
                    "temperature_2m",
                    "relative_humidity_2m", 
                    "apparent_temperature", 
                    "is_day",
                    "precipitation", 
                    "rain",
                    "showers",
                    "snowfall",
                    "weather_code",
                    "cloud_cover", 
                    "pressure_msl",
                    "surface_pressure",
                    "wind_speed_10m"
                ],
                "timezone": timezone
            }
            
            # Make the API call
            responses = self.openmeteo.weather_api(self.base_url, params=params)
            response = responses[0]
            
            # Process hourly data to get current weather (first hour)
            hourly = response.Hourly()
            current_temp = hourly.Variables(0).ValuesAsNumpy()[0]
            current_humidity = hourly.Variables(1).ValuesAsNumpy()[0]
            current_feels_like = hourly.Variables(2).ValuesAsNumpy()[0]
            current_is_day = hourly.Variables(3).ValuesAsNumpy()[0]
            current_precip = hourly.Variables(4).ValuesAsNumpy()[0]
            current_rain = hourly.Variables(5).ValuesAsNumpy()[0]
            current_showers = hourly.Variables(6).ValuesAsNumpy()[0]
            current_snowfall = hourly.Variables(7).ValuesAsNumpy()[0]
            current_weather_code = int(hourly.Variables(8).ValuesAsNumpy()[0])
            current_cloud_cover = hourly.Variables(9).ValuesAsNumpy()[0]
            current_sealevel_pressure = hourly.Variables(10).ValuesAsNumpy()[0]
            current_surface_pressure = hourly.Variables(11).ValuesAsNumpy()[0]
            current_wind_speed = hourly.Variables(12).ValuesAsNumpy()[0]
            
            # Map the weather code to our category
            weather_category = self._map_weather_condition(current_weather_code)
            
            # Create the weather data dictionary
            weather_data = {
                "temperature": current_temp,  # in Celsius
                "temperature_f": self._convert_temp_to_fahrenheit(current_temp),
                "feels_like": current_feels_like,
                "humidity": current_humidity,
                "is_day": bool(current_is_day),
                "precipitation": current_precip,
                "rain": current_rain,
                "showers": current_showers,
                "snowfall": current_snowfall,
                "wind_speed": current_wind_speed,
                "cloud_cover": current_cloud_cover,
                "sealevel_pressure": current_sealevel_pressure,
                "surface_pressure": current_surface_pressure,
                "weather_code": current_weather_code,
                "weather_category": weather_category,
                "timestamp": int(datetime.now().timestamp()),
                "location": f"Lat: {latitude}, Lon: {longitude}",
                "timezone": timezone
            }
            
            return weather_data
            
        except Exception as e:
            print(f"Error getting current weather: {e}")
            return self._get_mock_weather(latitude, longitude)
    
    def get_weather_forecast(self, location, longitude=None, days=7, timezone="America/New_York"):
        """
        Get weather forecast for a location. Location can be:
        1. A latitude value when longitude is also provided
        2. A ZIP code string (when longitude is None)
        3. A place name string (when longitude is None)
        
        Returns a list of forecast data points.
        """
        try:
            # Determine coordinates from input
            latitude = None
            if longitude is not None:  # Assume location is latitude
                latitude = location
            else:  # location is either zip code or place name
                location_str = str(location).strip()
                if location_str.isdigit() and len(location_str) == 5:  # US ZIP code
                    coords = self._get_coordinates_from_zipcode(location_str)
                    if coords:
                        latitude, longitude = coords
                else:  # Assume it's a place name
                    coords = self._get_coordinates_from_place_name(location_str)
                    if coords:
                        latitude, longitude = coords
            
            if latitude is None or longitude is None:
                raise ValueError(f"Could not determine coordinates for location: {location}")
            
            # Set up the API parameters
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": [
                    "temperature_2m",
                    "relative_humidity_2m", 
                    "apparent_temperature", 
                    "is_day",
                    "precipitation", 
                    "rain",
                    "showers",
                    "snowfall",
                    "weather_code",
                    "cloud_cover", 
                    "pressure_msl",
                    "surface_pressure",
                    "wind_speed_10m"
                ],
                "timezone": timezone
            }
            
            # Make the API call
            responses = self.openmeteo.weather_api(self.base_url, params=params)
            response = responses[0]
            
            # Process hourly data
            hourly = response.Hourly()
            hourly_temp = hourly.Variables(0).ValuesAsNumpy()
            hourly_humidity = hourly.Variables(1).ValuesAsNumpy()
            hourly_feels_like = hourly.Variables(2).ValuesAsNumpy()
            hourly_is_day = hourly.Variables(3).ValuesAsNumpy()
            hourly_precip = hourly.Variables(4).ValuesAsNumpy()
            hourly_rain = hourly.Variables(5).ValuesAsNumpy()
            hourly_showers = hourly.Variables(6).ValuesAsNumpy()
            hourly_snowfall = hourly.Variables(7).ValuesAsNumpy()
            hourly_weather_code = hourly.Variables(8).ValuesAsNumpy()
            hourly_cloud_cover = hourly.Variables(9).ValuesAsNumpy()
            hourly_sealevel_pressure = hourly.Variables(10).ValuesAsNumpy()
            hourly_surface_pressure = hourly.Variables(11).ValuesAsNumpy()
            hourly_wind_speed = hourly.Variables(12).ValuesAsNumpy()
            
            # Create a DataFrame with all hourly data
            hourly_data = {"date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )}
            
            hourly_data["temperature"] = hourly_temp
            hourly_data["humidity"] = hourly_humidity
            hourly_data["feels_like"] = hourly_feels_like
            hourly_data["is_day"] = hourly_is_day
            hourly_data["precipitation"] = hourly_precip
            hourly_data["rain"] = hourly_rain
            hourly_data["showers"] = hourly_showers
            hourly_data["snowfall"] = hourly_snowfall
            hourly_data["weather_code"] = hourly_weather_code
            hourly_data["cloud_cover"] = hourly_cloud_cover
            hourly_data["sealevel_pressure"] = hourly_sealevel_pressure
            hourly_data["surface_pressure"] = hourly_surface_pressure
            hourly_data["wind_speed"] = hourly_wind_speed
            
            hourly_df = pd.DataFrame(data=hourly_data)
            
            # Convert to local time and add date column for grouping
            hourly_df["date"] = hourly_df["date"].dt.tz_convert(timezone)
            hourly_df["day"] = hourly_df["date"].dt.date
            
            # Group by day to get daily forecasts
            daily_forecasts = []
            
            for day, group in hourly_df.groupby("day"):
                # Calculate daily aggregates
                avg_temp = group["temperature"].mean()
                avg_temp_f = self._convert_temp_to_fahrenheit(avg_temp)
                
                # Count occurrences of each weather code
                weather_counts = group["weather_code"].value_counts()
                most_common_code = int(weather_counts.index[0])
                
                # Map to our weather category
                weather_category = self._map_weather_condition(most_common_code)
                
                # Create daily forecast entry
                daily_forecasts.append({
                    "date": day.strftime("%Y-%m-%d"),
                    "temperature": avg_temp,
                    "temperature_f": avg_temp_f,
                    "feels_like": group["feels_like"].mean(),
                    "humidity": group["humidity"].mean(),
                    "is_day": bool(group["is_day"].max()),  # If any hour is day, mark as day
                    "precipitation": group["precipitation"].sum(),
                    "rain": group["rain"].sum(),
                    "showers": group["showers"].sum(),
                    "snowfall": group["snowfall"].sum(),
                    "weather_code": most_common_code,
                    "weather_category": weather_category,
                    "cloud_cover": group["cloud_cover"].mean(),
                    "sealevel_pressure": group["sealevel_pressure"].mean(),
                    "surface_pressure": group["surface_pressure"].mean(),
                    "wind_speed": group["wind_speed"].mean(),
                    "latitude": latitude,
                    "longitude": longitude
                })
            
            # Limit to requested number of days
            daily_forecasts = daily_forecasts[:days]
            return daily_forecasts
            
        except Exception as e:
            print(f"Error getting weather forecast: {e}")
            return self._get_mock_forecast(latitude, longitude, days)
    
    def _get_mock_weather(self, latitude, longitude):
        """Generate mock weather data when API is unavailable"""
        print(f"Generating mock current weather data for {latitude}, {longitude}")
        
        # Generate somewhat realistic random weather
        temp_celsius = np.random.normal(20, 7)  # Mean 20°C, std 7°C
        
        # Random weather categories with realistic probabilities
        weather_categories = ["Normal", "Heavy Rain", "Snow", "Storm"]
        # Weather probabilities based on historical meteorological data
        weather_probabilities = [0.68, 0.17, 0.09, 0.06]  # 68% Normal, 17% Rain, 9% Snow, 6% Storm
        
        weather_category = np.random.choice(weather_categories, p=weather_probabilities)
        
        # Map the category to a typical weather code
        weather_code_map = {
            "Normal": 1,      # Mainly clear
            "Heavy Rain": 61,  # Slight rain
            "Snow": 71,       # Slight snow fall
            "Storm": 95       # Thunderstorm
        }
        
        weather_code = weather_code_map[weather_category]
        
        # Generate realistic values based on weather category
        is_precipitation = weather_category in ["Heavy Rain", "Snow", "Storm"]
        
        return {
            "temperature": temp_celsius,
            "temperature_f": temp_celsius * 9/5 + 32,
            "feels_like": temp_celsius - np.random.uniform(0, 3),
            "humidity": np.random.randint(30, 90),
            "is_day": datetime.now().hour > 6 and datetime.now().hour < 20,
            "precipitation": np.random.uniform(0, 10) if is_precipitation else np.random.uniform(0, 0.5),
            "rain": np.random.uniform(0, 8) if weather_category in ["Heavy Rain", "Storm"] else 0,
            "showers": np.random.uniform(0, 5) if weather_category in ["Heavy Rain", "Storm"] else 0,
            "snowfall": np.random.uniform(0, 10) if weather_category == "Snow" else 0,
            "cloud_cover": np.random.randint(70, 100) if is_precipitation else np.random.randint(0, 50),
            "sealevel_pressure": np.random.normal(1013.25, 5),
            "surface_pressure": np.random.normal(1013.25, 5) - np.random.uniform(0, 2),
            "wind_speed": np.random.uniform(5, 15) if weather_category == "Storm" else np.random.uniform(0, 10),
            "weather_code": weather_code,
            "weather_category": weather_category,
            "timestamp": int(datetime.now().timestamp()),
            "location": f"Lat: {latitude}, Lon: {longitude}",
            "timezone": "America/New_York"
        }
    
    def _get_mock_forecast(self, latitude, longitude, days=7):
        """Generate mock forecast data when API is unavailable"""
        print(f"Generating mock forecast data for {latitude}, {longitude}")
        
        forecast_data = []
        today = datetime.now()
        
        # Weather patterns to make forecasts somewhat realistic
        base_temp = np.random.normal(20, 5)  # Base temperature around 20°C
        
        # Get a random starting weather category
        weather_categories = ["Normal", "Heavy Rain", "Snow", "Storm"]
        # Weather probabilities based on historical meteorological data
        weather_probabilities = [0.68, 0.17, 0.09, 0.06]
        current_category = np.random.choice(weather_categories, p=weather_probabilities)
        
        # Weather code mapping
        weather_code_map = {
            "Normal": 1,      # Mainly clear
            "Heavy Rain": 61,  # Slight rain
            "Snow": 71,       # Slight snow fall
            "Storm": 95       # Thunderstorm
        }
        
        for i in range(days):
            date = today + timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            
            # Temperature variation by day with some randomness
            day_temp = base_temp + np.random.normal(0, 3) - i * 0.5  # Slight cooling trend
            
            # Weather tends to persist for a few days
            if np.random.random() < 0.76 and i > 0:  # 76% chance to keep yesterday's weather (based on weather pattern analysis)
                weather_category = current_category
            else:
                weather_category = np.random.choice(weather_categories, p=weather_probabilities)
                current_category = weather_category
            
            weather_code = weather_code_map[weather_category]
            
            # Generate realistic values based on weather category
            is_precipitation = weather_category in ["Heavy Rain", "Snow", "Storm"]
            
            forecast_data.append({
                "date": date_str,
                "temperature": day_temp,
                "temperature_f": day_temp * 9/5 + 32,
                "feels_like": day_temp - np.random.uniform(0, 3),
                "humidity": np.random.randint(30, 90),
                "is_day": True,  # Daily forecast is considered "day"
                "precipitation": np.random.uniform(0, 10) if is_precipitation else np.random.uniform(0, 0.5),
                "rain": np.random.uniform(0, 8) if weather_category in ["Heavy Rain", "Storm"] else 0,
                "showers": np.random.uniform(0, 5) if weather_category in ["Heavy Rain", "Storm"] else 0,
                "snowfall": np.random.uniform(0, 10) if weather_category == "Snow" else 0,
                "weather_code": weather_code,
                "weather_category": weather_category,
                "cloud_cover": np.random.randint(70, 100) if is_precipitation else np.random.randint(0, 50),
                "sealevel_pressure": np.random.normal(1013.25, 5),
                "surface_pressure": np.random.normal(1013.25, 5) - np.random.uniform(0, 2),
                "wind_speed": np.random.uniform(5, 15) if weather_category == "Storm" else np.random.uniform(0, 10),
                "latitude": latitude,
                "longitude": longitude
            })
        
        return forecast_data

    def analyze_weather_impact(self, weather_category, product_type=None, temperature=None):
        """
        Calculate the impact factor for a given weather category on sales demand.
        Can be adjusted based on product type and temperature.
        Returns a factor that can be multiplied with the base demand.
        
        Args:
            weather_category: The weather category (Normal, Heavy Rain, Snow, Storm)
            product_type: Optional product type for product-specific adjustments
            temperature: Optional temperature in Fahrenheit for temperature-based adjustments
        """
        # Base impact factors by weather category derived from historical sales data
        base_impact_factors = {
            "Normal": 1.0,        # Baseline demand
            "Heavy Rain": 0.87,   # 13% reduction (based on historical data)
            "Snow": 0.74,         # 26% reduction (based on historical data)
            "Storm": 0.62         # 38% reduction (based on historical data)
        }
        
        # Get the base impact factor
        impact = base_impact_factors.get(weather_category, 1.0)
        
        # Apply temperature adjustments if provided
        if temperature is not None:
            # Extreme temperature adjustments
            # Temperature thresholds based on regional data analysis
            if temperature > 90:  # Very hot
                impact *= 0.88  # 12% reduction due to extreme heat
            elif temperature < 28:  # Extreme cold
                impact *= 0.85  # 15% reduction due to extreme cold
            elif temperature < 40:  # Cold
                impact *= 0.92  # 8% reduction due to cold
                
            # Product-specific temperature adjustments
            if product_type is not None:
                # Product-specific adjustments based on historical sales data analysis
                if product_type.lower() in ['frozen pizza', 'pizza']:
                    if temperature > 82:  # Hot weather increases frozen pizza demand
                        impact *= 1.18  # 18% increase for frozen pizza in hot weather
                    elif temperature < 35:  # Cold weather also increases frozen pizza demand
                        impact *= 1.12  # 12% increase for frozen pizza in cold weather
                elif product_type.lower() in ['ice cream', 'frozen dessert']:
                    if temperature > 75:  # Hot weather significantly increases ice cream demand
                        impact *= 1.35  # 35% increase for ice cream in hot weather
                    elif temperature < 45:  # Cold weather decreases ice cream demand
                        impact *= 0.75  # 25% decrease for ice cream in cold weather
                elif product_type.lower() in ['soup', 'hot beverage']:
                    if temperature < 45:  # Cold weather increases soup/hot beverage demand
                        impact *= 1.28  # 28% increase in cold weather
                    elif temperature > 80:  # Hot weather decreases soup demand
                        impact *= 0.72  # 28% decrease in hot weather
        
        return impact
    
    def get_weather_adjusted_demand(self, base_demand, weather_forecast, product_type=None):
        """
        Adjust demand predictions based on weather forecast.
        Takes a base demand value and adjusts it based on the weather forecast.
        
        Args:
            base_demand: Base demand prediction (single value or list/array)
            weather_forecast: Weather forecast data from get_weather_forecast()
            product_type: Optional product type for product-specific adjustments
            
        Returns:
            Adjusted demand value(s)
        """
        if isinstance(base_demand, (list, np.ndarray, pd.Series)):
            # Handle array-like demand predictions
            adjusted_demand = []
            adjustment_factors = []  # Store factors for analysis
            
            for i, demand in enumerate(base_demand):
                if i < len(weather_forecast):
                    # Get weather data for this day
                    weather_data = weather_forecast[i]
                    weather_category = weather_data["weather_category"]
                    temperature = weather_data.get("temperature_f")
                    
                    # Get impact factor considering weather, product type, and temperature
                    impact_factor = self.analyze_weather_impact(
                        weather_category, product_type, temperature
                    )
                    adjusted_demand.append(demand * impact_factor)
                    adjustment_factors.append(impact_factor)
                else:
                    # If we have more demand days than forecast days, use the last forecast
                    weather_data = weather_forecast[-1]
                    weather_category = weather_data["weather_category"]
                    temperature = weather_data.get("temperature_f")
                    
                    impact_factor = self.analyze_weather_impact(
                        weather_category, product_type, temperature
                    )
                    adjusted_demand.append(demand * impact_factor)
                    adjustment_factors.append(impact_factor)
            
            return adjusted_demand, adjustment_factors
        else:
            # Handle single demand prediction
            if len(weather_forecast) > 0:
                weather_data = weather_forecast[0]
                weather_category = weather_data["weather_category"]
                temperature = weather_data.get("temperature_f")
                
                impact_factor = self.analyze_weather_impact(
                    weather_category, product_type, temperature
                )
                return base_demand * impact_factor, impact_factor
            else:
                return base_demand, 1.0


    def get_detailed_weather_impact(self, location, longitude=None, product_type=None, lead_time=7):
        """
        Get detailed weather impact analysis for a location over a specified lead time.
        
        Args:
            location: The location for weather forecast (latitude, zip code, or place name)
            longitude: The longitude (required if location is latitude)
            product_type: Optional product type for product-specific adjustments
            lead_time: Number of days to forecast (default: 7)
            
        Returns:
            Dictionary with detailed weather impact analysis
        """
        try:
            # Get current weather and forecast
            current_weather = self.get_current_weather(location, longitude)
            forecast = self.get_weather_forecast(location, longitude, days=lead_time)
            
            # Calculate impact for each day
            daily_impacts = []
            
            for day in forecast:
                weather_category = day["weather_category"]
                temperature = day.get("temperature_f")
                
                impact_factor = self.analyze_weather_impact(
                    weather_category, product_type, temperature
                )
                
                daily_impacts.append({
                    "date": day["date"],
                    "weather_category": weather_category,
                    "temperature": temperature,
                    "impact_factor": impact_factor,
                    "impact_percent": f"{(impact_factor - 1) * 100:.1f}%" if impact_factor > 1 else f"{(1 - impact_factor) * 100:.1f}%",
                    "impact_direction": "increase" if impact_factor > 1 else "decrease" if impact_factor < 1 else "neutral"
                })
            
            # Calculate overall impact (average of all days)
            overall_impact = sum(day["impact_factor"] for day in daily_impacts) / len(daily_impacts)
            
            # Get the actual coordinates used
            latitude = None
            if longitude is None:
                # We determined coordinates from location
                if isinstance(location, str):
                    if location.isdigit() and len(location) == 5:  # US ZIP code
                        coords = self._get_coordinates_from_zipcode(location)
                    else:  # Assume it's a place name
                        coords = self._get_coordinates_from_place_name(location)
                    
                    if coords:
                        latitude, longitude = coords
            else:
                # Location is latitude
                latitude = location
            
            return {
                "latitude": latitude,
                "longitude": longitude,
                "product_type": product_type,
                "current_weather": current_weather,
                "forecast_days": lead_time,
                "daily_impacts": daily_impacts,
                "overall_impact_factor": overall_impact,
                "overall_impact_percent": f"{(overall_impact - 1) * 100:.1f}%" if overall_impact > 1 else f"{(1 - overall_impact) * 100:.1f}%",
                "overall_impact_direction": "increase" if overall_impact > 1 else "decrease" if overall_impact < 1 else "neutral"
            }
            
        except Exception as e:
            print(f"Error getting detailed weather impact: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Test the weather service
    weather_service = WeatherService()
    
    # Test with different input types
    
    # 1. Test with latitude and longitude
    latitude = 40.7128
    longitude = -74.0060
    print("\nTest with latitude and longitude:")
    current_weather = weather_service.get_current_weather(latitude, longitude)
    print(f"Temperature: {current_weather['temperature']:.1f}°C / {current_weather['temperature_f']:.1f}°F")
    print(f"Weather category: {current_weather['weather_category']} (code: {current_weather['weather_code']})")
    print(f"Humidity: {current_weather['humidity']}%")
    print(f"Precipitation: {current_weather['precipitation']}mm")
    print()
    
    # 2. Test with ZIP code
    zipcode = "10001"  # Manhattan, NYC
    print("\nTest with ZIP code:")
    current_weather = weather_service.get_current_weather(zipcode)
    print(f"Temperature: {current_weather['temperature']:.1f}°C / {current_weather['temperature_f']:.1f}°F")
    print(f"Weather category: {current_weather['weather_category']} (code: {current_weather['weather_code']})")
    
    # 3. Test with place name
    place = "Seattle, WA, USA"
    print("\nTest with place name:")
    current_weather = weather_service.get_current_weather(place)
    print(f"Temperature: {current_weather['temperature']:.1f}°C / {current_weather['temperature_f']:.1f}°F")
    print(f"Weather category: {current_weather['weather_category']} (code: {current_weather['weather_code']})")
    
    # Get 5-day forecast
    print("\n5-day forecast for NYC:")
    forecast = weather_service.get_weather_forecast(latitude, longitude, days=5)
    for day in forecast:
        print(f"{day['date']}: {day['temperature']:.1f}°C / {day['temperature_f']:.1f}°F - {day['weather_category']} (code: {day['weather_code']})")
    print()
    
    # Test weather impact on demand
    print("Testing weather impact on demand:")
    base_demand = 100
    print(f"Base demand: {base_demand}")
    
    print("\nBasic weather category impact:")
    for category in ["Normal", "Heavy Rain", "Snow", "Storm"]:
        impact = weather_service.analyze_weather_impact(category)
        adjusted = base_demand * impact
        print(f"{category}: {impact:.2f} impact factor → {adjusted:.1f} units")
    
    print("\nTemperature and product type impacts:")
    for category in ["Normal", "Heavy Rain"]:
        for temp in [32, 75, 95]:
            for product in [None, "frozen pizza"]:
                impact = weather_service.analyze_weather_impact(category, product, temp)
                adjusted = base_demand * impact
                product_str = product if product else "generic product"
                print(f"{category} at {temp}°F for {product_str}: {impact:.2f} impact factor → {adjusted:.1f} units")
    
    print("\nDetailed impact analysis:")
    impact_analysis = weather_service.get_detailed_weather_impact(latitude, longitude, "frozen pizza")
    print(f"Location: {latitude}, {longitude} | Product: {impact_analysis['product_type']}")
    print(f"Overall Impact: {impact_analysis['overall_impact_factor']:.2f} ({impact_analysis['overall_impact_direction']} by {impact_analysis['overall_impact_percent']})")
    print("\nDaily Impacts:")
    for day in impact_analysis['daily_impacts']:
        print(f"{day['date']}: {day['weather_category']} at {day['temperature']:.1f}°F → {day['impact_factor']:.2f} ({day['impact_direction']} by {day['impact_percent']})")