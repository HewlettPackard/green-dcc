import os
import requests
import json

# Define the locations with their coordinates
locations = {
    "US-NY-NYIS": (40.71, -74.01),  # New York City, New York, USA
    "US-MIDA-PJM": (39.95, -75.16),  # Philadelphia, Pennsylvania, USA
    "US-TEX-ERCO": (32.78, -96.79),  # Dallas Texas, USA
    "US-CAL-CISO": (37.40, -121.93),  # San Jose, California, USA
    "DE": (50.12, 8.68),  # Frankrut, Germany
    "CA-ON": (43.70, -79.42),  # Toronto, Ontario, Canada
    "SG": (1.30, 103.83),  # Singapore
    "AU-VIC": (-37.81, 144.96),  # Melbourne, Victoria, Australia
    "AU-NSW": (-33.87, 151.21),  # Sydney, New South Wales, Australia
    "CL-SEN": (-33.45, -70.65),  # Santiago, Chile
    "BR-CS": (-23.55, -46.63),  # São Paulo, Brazil
    "ZA": (-26.20, 28.04),  # Johannesburg, South Africa
    "KR": (37.56, 126.97),  # Seoul, South Korea
    "IN-WE": (19.08, 72.88),  # Mumbai, India
    "JP-TK": (35.69, 139.77),  # Tokyo, Japan
    "GB": (51.51, -0.13)  # London, United Kingdom
}

# Define the years of interest
years = [2020, 2021, 2022, 2023, 2024]

# Base directory to store weather data
data_dir = "data/weather"
os.makedirs(data_dir, exist_ok=True)

# Fetch weather data for each location and year
for location, (lat, lon) in locations.items():
    location_dir = os.path.join(data_dir, location)
    os.makedirs(location_dir, exist_ok=True)
    
    for year in years:
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        url = (f"https://archive-api.open-meteo.com/v1/archive?"
               f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
               "&hourly=temperature_2m,relative_humidity_2m,cloudcover,windspeed_10m"
               "&timezone=auto")
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            file_path = os.path.join(location_dir, f"{year}.json")
            with open(file_path, "w") as file:
                json.dump(data, file, indent=4)
            print(f"Saved: {file_path}")
        else:
            print(f"Failed to fetch data for {location} in {year}")