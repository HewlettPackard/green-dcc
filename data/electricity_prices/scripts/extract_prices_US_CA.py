#%%

import os
import io
import time
import zipfile
import pandas as pd
from datetime import datetime, timedelta
import requests
from tqdm import tqdm

# Config
START_YEAR = 2020
END_YEAR = 2024
OUTPUT_ROOT = "data/electricity_prices/US-CAL-CISO"
NODE = "TH_NP15_GEN-APND"  # You can change this to any hub/node
'''
You can replace the NODE with any of these valid trading hubs:

Name	Meaning
TH_NP15_GEN-APND	Northern California (generator side)
TH_SP15_GEN-APND	Southern California (generator side)
TH_ZP26_GEN-APND	Central California (Zone Path 26)
'''
BASE_URL = "http://oasis.caiso.com/oasisapi/SingleZip"

# Query params for Day-Ahead hourly prices
def build_url(start, end):
    return (
        f"{BASE_URL}?"
        f"resultformat=6&"
        f"queryname=PRC_LMP&"
        f"version=1&"
        f"startdatetime={start}&"
        f"enddatetime={end}&"
        f"market_run_id=DAM&"
        f"node={NODE}&"
        f"grp_type=ALL"
    )

# Extract ZIP response into DataFrame
def fetch_price_data(start, end):
    url = build_url(start, end)
    for _ in range(3):  # Retry if server is slow
        try:
            r = requests.get(url)
            if r.status_code != 200:
                time.sleep(2)
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f)
            return df
        except:
            time.sleep(2)
    return pd.DataFrame()

# Loop through date range in chunks
def fetch_yearly_data(year):
    all_data = []
    start_date = datetime(year, 1, 1)
    end_date = datetime(year+1, 1, 1)

    date_list = pd.date_range(start=start_date, end=end_date, freq="D")

    for current in tqdm(date_list, desc=f"Fetching {year}"):
        start_str = current.strftime("%Y%m%dT00:00-0000")
        end_str = current.strftime("%Y%m%dT23:59-0000")
        df = fetch_price_data(start_str, end_str)
        
        if not df.empty:
            df.columns = [col.strip().upper() for col in df.columns]
            df = df[(df["LMP_TYPE"] == "LMP") & (df["NODE"] == NODE)].copy()
            df["datetime"] = pd.to_datetime(df["INTERVALSTARTTIME_GMT"])
            df["price"] = pd.to_numeric(df["MW"], errors="coerce")
            data = df[["datetime", "price"]].dropna()
            all_data.append(data)
        time.sleep(0.25)

    if all_data:
        df_all = pd.concat(all_data).sort_values("datetime")
        df_all.rename(columns={"datetime": "Datetime", "price": "Price (USD/MWh)"}, inplace=True)
        df_all["Datetime"] = df_all["Datetime"].dt.tz_convert(None)
        return df_all
    else:
        return pd.DataFrame()

# Save yearly CSVs
def run():
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n--- Processing {year} ---")
        df = fetch_yearly_data(year)
        if df.empty:
            print(f"No data for {year}")
            continue
        year_dir = os.path.join(OUTPUT_ROOT, str(year))
        os.makedirs(year_dir, exist_ok=True)
        path = os.path.join(year_dir, f"US-CAL-CISO_electricity_prices_{year}.csv")
        df.to_csv(path, index=False)
        print(f"Saved: {path}")

if __name__ == "__main__":
    run()

#%%


#%%