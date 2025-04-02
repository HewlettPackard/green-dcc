# The electricity price is extracted from the price set by the market operator (https://www.eskom.co.za/).
# The price is in the local currency and the unit is in MWh. The data is saved in a CSV file with two columns: Datetime and Price (local currency/MWh).
# The CSV file is saved in a folder named after the country code and year. The folder is saved in the processed folder of the electricity_prices directory.
# The script is run for each country to extract the electricity prices for each year.
# Zone: South Africa (ZA)
# Currency: ZAR
# Timezone: Africa/Johannesburg

from datetime import datetime, timedelta
import pandas as pd
import os

# TOU definitions
def get_season_and_period(dt):
    month = dt.month
    hour = dt.hour
    weekday = dt.weekday()

    # High-demand season means the TOU Period from 1 June to 31 August of each year.
    high_demand = month in [6, 7, 8]

    if high_demand:
        if weekday < 5: # Weekday
            if 6 <= hour < 9 or 17 <= hour < 19:
                return "high", "peak"
            elif 9 <= hour < 17 or 19 <= hour < 22:
                return "high", "standard"
            else:
                return "high", "off-peak"
        else: # Weekend
            # If saturday:
            if weekday == 5:
                if 7 <= hour < 12 or 18 <= hour < 20:
                    return "high", "standard"
                else:
                    return "high", "off-peak"
            # If sunday:
            else:
                return "high", "off-peak"
    else:
        if weekday < 5:
            if 7 <= hour < 10 or 18 <= hour < 20:
                return "low", "peak"
            elif 6 <= hour < 7 or 10 <= hour < 18 or 20 <= hour < 22:
                return "low", "standard"
            else:
                return "low", "off-peak"
        else: # Weekend
            if weekday == 5:
                if 7 <= hour < 12 or 18 <= hour < 20:
                    return "low", "standard"
                else:
                    return "low", "off-peak"
            else:
                return "low", "off-peak"

# Price table for Megaflex tariff, Non-local Authority, Transmission zone < 300km, >=132kV (VAT incl), in ZAR/MWh
# Source: https://www.eskom.co.za/Pages/Tariffs.aspx

prices = {
    2020: {
        "low": {
            "peak": 1222.00,
            "standard": 841.00,
            "off-peak": 534.00
        },
        "high": {
            "peak": 3747.00,
            "standard": 1135.00,
            "off-peak": 616.00
        }
    },
    2021: {
        "low": {
            "peak": 1407.00,
            "standard": 968.00,
            "off-peak": 614.00
        },
        "high": {
            "peak": 4311.00,
            "standard": 1306.00,
            "off-peak": 709.00
        }
    },
    2022: {
        "low": {
            "peak": 1542.00,
            "standard": 1061.00,
            "off-peak": 673.00
        },
        "high": {
            "peak": 4726.00,
            "standard": 1431.00,
            "off-peak": 777.00
        }
    },
    2023: {
        "low": {
            "peak": 1829.00,
            "standard": 1259.00,
            "off-peak": 799.00
        },
        "high": {
            "peak": 5607.00,
            "standard": 1698.00,
            "off-peak": 922.00
        }
    },
    2024: {
        "low": {
            "peak": 2063.00,
            "standard": 1420.00,
            "off-peak": 901.00
        },
        "high": {
            "peak": 6321.00,
            "standard": 1914.00,
            "off-peak": 1041.00
        }
    }
}

# Now we generate the prices for each hour and for each year
# The files should be saved in the processed folder of the electricity_prices directory
# data/electricity_prices/processed/ZA/2020/ZA_electricity_prices_2020.csv
# data/electricity_prices/processed/ZA/2021/ZA_electricity_prices_2021.csv
# etc.

for year in range(2020, 2025):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31, 23)  # Up to last hour of the year
    hours = pd.date_range(start=start_date, end=end_date, freq="H")
    
    all_rows = []

    for hour in hours:
        season, period = get_season_and_period(hour)
        price = prices[year][season][period]

        all_rows.append([hour, price])

    df_out = pd.DataFrame(all_rows, columns=["Datetime", "Price (ZAR/MWh)"])
    # Create the folder if it does not exist
    os.makedirs(f"data/electricity_prices/processed/ZA/{year}",
                exist_ok=True)
    df_out.to_csv(f"data/electricity_prices/processed/ZA/{year}/ZA_electricity_prices_{year}.csv", index=False)
    print(f"Saved: data/electricity_prices/processed/ZA/{year}/ZA_electricity_prices_{year}.csv")
