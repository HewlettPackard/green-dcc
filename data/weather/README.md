# README: Hourly Weather Data Organization

## Folder Structure
The weather dataset is organized by **location and year**, making it easy to navigate and retrieve data for specific locations and time periods.

### **Structure:**
```
data/weather/
│── [LOCATION]/
│   ├── [YEAR].json
```

### **Example:**
```
data/weather/
│── US-NY-NYIS/
│   ├── 2020.json
│   ├── 2021.json
│   ├── 2022.json
│   ├── 2023.json
│   ├── 2024.json
│
│── DE/
│   ├── 2020.json
│   ├── 2021.json
│   ├── 2022.json
│   ├── 2023.json
│   ├── 2024.json
```

## **Selected Locations & Coordinates**
The following locations have been selected for weather data extraction, corresponding to key data center regions:

| Acronym | Full Region Name | Latitude | Longitude |
|---------|-----------------|----------|-----------|
| US-NY-NYIS | New York City, New York, USA | 40.71 | -74.01 |
| US-MIDA-PJM | Philadelphia, Pennsylvania, USA | 39.95 | -75.16 |
| US-TEX-ERCO | Dallas, Texas, USA | 32.78 | -96.79 |
| US-CAL-CISO | San Jose, California, USA | 37.40 | -121.93 |
| DE | Frankfurt, Germany | 50.12 | 8.68 |
| CA-ON | Toronto, Ontario, Canada | 43.70 | -79.42 |
| SG | Singapore | 1.30 | 103.83 |
| AU-VIC | Melbourne, Victoria, Australia | -37.81 | 144.96 |
| AU-NSW | Sydney, New South Wales, Australia | -33.87 | 151.21 |
| CL-SEN | Santiago, Chile | -33.45 | -70.65 |
| BR-CS | São Paulo, Brazil | -23.55 | -46.63 |
| ZA | Johannesburg, South Africa | -26.20 | 28.04 |
| KR | Seoul, South Korea | 37.56 | 126.97 |
| IN-WE | Mumbai, India | 19.08 | 72.88 |
| JP-TK | Tokyo, Japan | 35.69 | 139.77 |
| GB | London, United Kingdom | 51.51 | -0.13 |

## **Weather Data Format (JSON Files)**
Each JSON file contains **hourly weather data** for the selected location and year. Below is a sample structure of a weather data file:

### **JSON File Structure:**
```json
{
    "latitude": 40.738136,
    "longitude": -74.04254,
    "generationtime_ms": 216.50171279907227,
    "utc_offset_seconds": -14400,
    "timezone": "America/New_York",
    "timezone_abbreviation": "GMT-4",
    "elevation": 27.0,
    "hourly_units": {
        "time": "iso8601",
        "temperature_2m": "°C",
        "relative_humidity_2m": "%",
        "cloudcover": "%",
        "windspeed_10m": "km/h"
    },
    "hourly": {
        "time": ["2024-01-01T00:00", "2024-01-01T01:00", "2024-01-01T02:00", ...],
        "temperature_2m": [...],
        "relative_humidity_2m": [...],
        "cloudcover": [...],
        "windspeed_10m": [...]
    }
}
```

### **Explanation of Key Fields:**
| Field | Description |
|------------|------------------------------------------------|
| **latitude, longitude** | Coordinates of the weather station/data source. |
| **generationtime_ms** | Time taken to generate the response from Open-Meteo. |
| **utc_offset_seconds** | UTC offset of the location. |
| **timezone, timezone_abbreviation** | Local timezone of the location. |
| **elevation** | Elevation of the location (meters above sea level). |
| **hourly_units** | Units of measurement for each weather variable. |
| **hourly.time** | List of timestamps in ISO 8601 format. |
| **hourly.temperature_2m** | Hourly temperature at 2 meters above ground (°C). |
| **hourly.relative_humidity_2m** | Hourly relative humidity at 2 meters (%). |
| **hourly.cloudcover** | Hourly cloud cover percentage (%). |
| **hourly.windspeed_10m** | Hourly wind speed at 10 meters (km/h). |

## **Usage**
This dataset is designed for **data center workload balancing simulations**, providing insights into:
- Weather-related energy efficiency impacts.
- Regional cooling needs based on temperature and humidity.
- Renewable energy availability influenced by wind speed and cloud cover.

## **Additional Notes**
- Some files may have missing data in certain time periods.
- Data is sourced from Open-Meteo’s historical API.
- Ensure to adjust for local time zones when analyzing the data.

## **Data Source**
The weather data is obtained from **Open-Meteo's Historical Weather API** ([https://open-meteo.com](https://open-meteo.com)), which provides free access to hourly meteorological data worldwide.

## **Contact**
For further details, refer to Open-Meteo’s API documentation or reach out to dataset maintainers.

