#%%
import os
import pandas as pd
from datetime import datetime, timedelta
import zoneinfo

# Tokyo timezone
ZONE_INFO = zoneinfo.ZoneInfo("Asia/Tokyo")

def fetch_fiscal_year_data(fiscal_year):
    """
    Downloads the JEPX CSV file for the given fiscal year.
    Each fiscal file covers from April 1 of the fiscal year to March 31 of the next year.
    """
    url = f"http://www.jepx.jp/market/excel/spot_{fiscal_year}.csv"
    df = pd.read_csv(url, encoding="shift-jis")
    # Select only the relevant columns: 0: Date, 1: Period, 6 to 14: Prices for various zones.
    df = df.iloc[:, [0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
    df.columns = [
        "Date",
        "Period",
        "JP-HKD",
        "JP-TH",
        "JP-TK",
        "JP-CB",
        "JP-HR",
        "JP-KN",
        "JP-CG",
        "JP-SK",
        "JP-KY",
    ]
    # Convert the Date column to a date object
    df["Date"] = pd.to_datetime(df["Date"], format="%Y/%m/%d").dt.date
    # Create a datetime column from Date and Period.
    # Each period represents a 30-minute slot.
    df["datetime"] = df.apply(
        lambda row: datetime.combine(row["Date"], datetime.min.time())
        .replace(tzinfo=ZONE_INFO)
        + timedelta(minutes=30 * (row["Period"] - 1)),
        axis=1,
    )
    return df

def get_tokyo_prices_for_year(year):
    """
    Tokyo prices are provided in fiscal CSV files.
    For a given calendar year, we need to extract two parts:
      - January 1 to March 31 from the fiscal file of (year-1)
      - April 1 to December 31 from the fiscal file of (year)
    The two parts are then concatenated and sorted.
    """
    parts = []
    # Part 1: January 1 – March 31 from previous fiscal file
    start_part1 = datetime(year, 1, 1).date()
    end_part1   = datetime(year, 3, 31).date()
    fiscal_year1 = year - 1
    try:
        df1 = fetch_fiscal_year_data(fiscal_year1)
        df1_filtered = df1[(df1["Date"] >= start_part1) & (df1["Date"] <= end_part1)]
        parts.append(df1_filtered)
    except Exception as e:
        print(f"Error fetching data for fiscal year {fiscal_year1}: {e}")

    # Part 2: April 1 – December 31 from current fiscal file
    start_part2 = datetime(year, 4, 1).date()
    end_part2   = datetime(year, 12, 31).date()
    fiscal_year2 = year
    try:
        df2 = fetch_fiscal_year_data(fiscal_year2)
        df2_filtered = df2[(df2["Date"] >= start_part2) & (df2["Date"] <= end_part2)]
        parts.append(df2_filtered)
    except Exception as e:
        print(f"Error fetching data for fiscal year {fiscal_year2}: {e}")

    if parts:
        df_year = pd.concat(parts)
        df_year.sort_values("datetime", inplace=True)
        return df_year
    else:
        return pd.DataFrame()

# Process years 2020 to 2024 and save the results in the folder structure.
years = [2020, 2021, 2022, 2023, 2024]
zone_label = "JP-TK"

for year in years:
    print(f"Processing Tokyo electricity prices for {year}...")
    df_year = get_tokyo_prices_for_year(year)
    if df_year.empty:
        print(f"No data available for {year}")
        continue

    # Filter so that only hourly data remains (i.e. minute equals 0)
    df_hourly = df_year[df_year["datetime"].dt.minute == 0].copy()

    # Prepare the final DataFrame:
    # 1. Keep only the datetime and JP-TK columns.
    # 2. Convert JP-TK values from JPY/kWh to JPY/MWh by multiplying by 1000 and rounding to nearest 10.
    df_out = df_hourly[["datetime", "JP-TK"]].copy()
    df_out.rename(columns={"datetime": "Datetime", "JP-TK": "Price (JPY/kWh)"}, inplace=True)
    df_out["Price (JPY/MWh)"] = (df_out["Price (JPY/kWh)"] * 1000).round(-1)
    # Keep only the desired two columns.
    df_out = df_out[["Datetime", "Price (JPY/MWh)"]]
    # Remove timezone information for consistency.
    df_out["Datetime"] = df_out["Datetime"].dt.tz_localize(None)

    # Create folder structure: JP-TK/2020/, JP-TK/2021/, etc.
    folder_path = os.path.join(zone_label, str(year))
    os.makedirs(folder_path, exist_ok=True)
    file_name = f"{zone_label}_electricity_prices_{year}.csv"
    file_path = os.path.join(folder_path, file_name)
    df_out.to_csv(file_path, index=False)
    print(f"Saved data to {file_path}")



#%%

#%%