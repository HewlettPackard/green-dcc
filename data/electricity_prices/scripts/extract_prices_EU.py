#%%

import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timedelta

# Base URL for RTE France electricity prices
BASE_URL = "http://eco2mix.rte-france.com/curves/getDonneesMarche"
ZONE_KEY = "DE"  # France

def fetch_price_data(start_date, end_date):
    url = f"{BASE_URL}?dateDeb={start_date.strftime('%d/%m/%Y')}&dateFin={end_date.strftime('%d/%m/%Y')}&mode=NORM"
    response = requests.get(url)
    if not response.ok:
        raise Exception(f"Error fetching data: {response.status_code}")
    return response.content

def parse_prices(xml_content):
    root = ET.fromstring(xml_content)
    data = []
    for daily_market_data in root.iterfind("donneesMarche"):
        date_str = daily_market_data.get("date")
        day = datetime.strptime(date_str, "%Y-%m-%d")
        
        for daily_zone_data in daily_market_data:
            if daily_zone_data.get("perimetre") != ZONE_KEY:
                continue
            
            for value in daily_zone_data:
                price = None if value.text == "ND" else float(value.text)
                if price is None:
                    continue
                
                hour = int(value.attrib["periode"])
                timestamp = day + timedelta(hours=hour)
                data.append([timestamp, price])
    return data

def get_yearly_prices(year):
    all_data = []
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    
    # Fetch data month by month to avoid large requests
    current_date = start_date
    while current_date <= end_date:
        month_end = current_date.replace(day=28) + timedelta(days=4)
        month_end = month_end - timedelta(days=month_end.day)
        month_end = min(month_end, end_date)
        
        xml_data = fetch_price_data(current_date, month_end)
        monthly_data = parse_prices(xml_data)
        all_data.extend(monthly_data)
        
        current_date = month_end + timedelta(days=1)
    
    df = pd.DataFrame(all_data, columns=["Datetime", "Price (EUR/MWh)"])
    return df

# Fetch 2023 electricity prices for France
df_prices = get_yearly_prices(2023)

# Save to CSV
df_prices.to_csv(f"{ZONE_KEY}_electricity_prices_2023.csv", index=False)
print(f"Data saved to {ZONE_KEY}_electricity_prices_2023.csv")

#%%  Extract only one year
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timedelta

# Base URL for RTE-France electricity prices
BASE_URL = "http://eco2mix.rte-france.com/curves/getDonneesMarche"

# Map each dashboard location to its API codes
zones = {
    "FR": ["FR"],
    "GB": ["GB"],
    "BE": ["BE"],
    "DE-LU": ["DE", "DL"],  # Germany+Luxembourg: combine codes "DE" and "DL"
    "CH": ["CH"],
    "ES": ["ES"],
    "PT": ["PT"],
    "NL": ["NL"],
    "AT": ["AT"]
}

def fetch_price_data(start_date, end_date):
    url = (
        f"{BASE_URL}?dateDeb={start_date.strftime('%d/%m/%Y')}"
        f"&dateFin={end_date.strftime('%d/%m/%Y')}&mode=NORM"
    )
    response = requests.get(url)
    if not response.ok:
        raise Exception(f"Error fetching data: {response.status_code}")
    return response.content

def parse_prices(xml_content, valid_zone_codes):
    root = ET.fromstring(xml_content)
    data = []
    for daily_market_data in root.iterfind("donneesMarche"):
        date_str = daily_market_data.get("date")
        if not date_str:
            continue
        day = datetime.strptime(date_str, "%Y-%m-%d")
        for daily_zone_data in daily_market_data:
            zone_code = daily_zone_data.get("perimetre")
            # Only include data if the zone code is in the list for the selected location
            if zone_code not in valid_zone_codes:
                continue
            for value in daily_zone_data:
                if value.text == "ND":
                    continue
                try:
                    price = float(value.text)
                except (ValueError, TypeError):
                    continue
                hour = int(value.attrib["periode"])
                timestamp = day + timedelta(hours=hour)
                data.append([timestamp, price])
    return data

def get_yearly_prices(year, valid_zone_codes):
    all_data = []
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    current_date = start_date

    # Process month by month to avoid overly large requests
    while current_date <= end_date:
        # Calculate last day of the current month
        month_end = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        if month_end > end_date:
            month_end = end_date
        print(f"Fetching data from {current_date.strftime('%d/%m/%Y')} to {month_end.strftime('%d/%m/%Y')}")
        xml_data = fetch_price_data(current_date, month_end)
        monthly_data = parse_prices(xml_data, valid_zone_codes)
        all_data.extend(monthly_data)
        current_date = month_end + timedelta(days=1)
    
    df = pd.DataFrame(all_data, columns=["Datetime", "Price (EUR/MWh)"])
    return df

# Loop over each zone and fetch the 2023 data
for zone_label, zone_codes in zones.items():
    print(f"\nProcessing zone: {zone_label} (API codes: {zone_codes})")
    df_prices = get_yearly_prices(2023, zone_codes)
    filename = f"{zone_label}_electricity_prices_2023.csv"
    df_prices.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

#%% Extract multiple years
import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timedelta

# Base URL for RTE-France electricity prices
BASE_URL = "http://eco2mix.rte-france.com/curves/getDonneesMarche"

# Mapping of locations to API zone codes.
# Note: For Germany+Luxembourg we use "GE" as the folder name and combine codes "DE" and "DL".
zones = {
    "FR": ["FR"], # France
    "GB": ["GB"], # Great Britain
    "BE": ["BE"], # Belgium
    "DE-LU": ["DE", "DL"],  # Germany+Luxembourg: combine codes "DE" and "DL"
    "CH": ["CH"], # Switzerland
    "ES": ["ES"], # Spain
    "PT": ["PT"], # Portugal
    "NL": ["NL"], # Netherlands
    "AT": ["AT"]  # Austria
}

# # Only GB
# zones = {
#     "GB": ["GB"], # Great Britain
# }
# Countries extracted: France, Great Britain, Belgium, Germany+Luxembourg, Switzerland, Spain, Portugal, Netherlands, Austria

def fetch_price_data(start_date, end_date):
    """Fetches raw XML data from the RTE-France API for a given date range."""
    url = (
        f"{BASE_URL}?dateDeb={start_date.strftime('%d/%m/%Y')}"
        f"&dateFin={end_date.strftime('%d/%m/%Y')}&mode=NORM"
    )
    response = requests.get(url)
    if not response.ok:
        raise Exception(f"Error fetching data: {response.status_code}")
    return response.content

def parse_prices(xml_content, valid_zone_codes):
    """Parses XML content and returns a list of [timestamp, price] entries for the given zone codes."""
    root = ET.fromstring(xml_content)
    data = []
    for daily_market_data in root.iterfind("donneesMarche"):
        date_str = daily_market_data.get("date")
        if not date_str:
            continue
        day = datetime.strptime(date_str, "%Y-%m-%d")
        for daily_zone_data in daily_market_data:
            zone_code = daily_zone_data.get("perimetre")
            if zone_code not in valid_zone_codes:
                continue
            for value in daily_zone_data:
                if value.text == "ND":
                    continue
                try:
                    price = float(value.text)
                except (ValueError, TypeError):
                    continue
                hour = int(value.attrib["periode"])
                timestamp = day + timedelta(hours=hour)
                data.append([timestamp, price])
    return data

def get_yearly_prices(year, valid_zone_codes):
    """Fetches the electricity price data for an entire year using month-by-month requests."""
    all_data = []
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    current_date = start_date

    # Loop through the year in month-long chunks
    while current_date <= end_date:
        # Calculate the last day of the current month
        month_end = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        if month_end > end_date:
            month_end = end_date
        print(f"Fetching data from {current_date.strftime('%d/%m/%Y')} to {month_end.strftime('%d/%m/%Y')}")
        xml_data = fetch_price_data(current_date, month_end)
        monthly_data = parse_prices(xml_data, valid_zone_codes)
        all_data.extend(monthly_data)
        current_date = month_end + timedelta(days=1)
    
    df = pd.DataFrame(all_data, columns=["Datetime", "Price (EUR/MWh)"])
    return df

# Years to process
years = [2020, 2021, 2022, 2023, 2024]

# Loop over each location and year, saving the data in a folder structure like: FR/2020/FR_electricity_prices_2020.csv
for zone_label, zone_codes in zones.items():
    for year in years:
        print(f"\nProcessing zone: {zone_label} for year: {year}")
        df_prices = get_yearly_prices(year, zone_codes)
        folder_path = os.path.join(zone_label, str(year))
        os.makedirs(folder_path, exist_ok=True)
        file_name = f"{zone_label}_electricity_prices_{year}.csv"
        file_path = os.path.join(folder_path, file_name)
        df_prices.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")

# %%
