# The zip files with the prices are extracted from: https://www.nems.emcsg.com/nems-prices

# Now I will process the files to obtain the expected folder structure and .csv format
# The files will be saved in the following format:
# electricity_price/
# ├── SG
# │   ├── 2020
# │   │   ├── SG_electricity_price_2020.csv
# │   ├── 2021
# │   │   ├── SG_electricity_price_2021.csv
# ...
# And the file format will be:
# Datetime,Price ($/MWh)
# 2020-01-01 00:00:00+08:00,0.020

#%%
import os
import zipfile
import pandas as pd
from datetime import datetime, timedelta

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "SG/zips")
  # Folder where your 5 ZIPs are located
OUTPUT_BASE = os.path.join(BASE_DIR, "SG")    # Output folder name
YEARS = [2020, 2021, 2022, 2023, 2024]        # Years to process


def period_to_hour(period):
    """Convert Singapore market 48-period to timestamp offset (hour + minute)"""
    return timedelta(minutes=(int(period) - 1) * 30)


def process_zip(zip_path, year):
    all_data = []

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if not file.endswith(".csv"):
                continue
            with zip_ref.open(file) as f:
                try:
                    df = pd.read_csv(f)
                except:
                    df = pd.read_csv(f, encoding="latin1")

                # Clean columns
                df.columns = [col.strip() for col in df.columns]

                required_cols = ["DATE", "PERIOD", "USEP ($/MWh)"]
                if not all(col in df.columns for col in required_cols):
                    continue  # skip malformed files

                for _, row in df.iterrows():
                    try:
                        date_str = row["DATE"].strip()
                        period = int(row["PERIOD"])
                        price = float(row["USEP ($/MWh)"])
                        try:
                            date_obj = datetime.strptime(date_str, "%d %b %Y")
                        except:
                            date_obj = datetime.strptime(date_str, "%d-%b-%Y")
                        dt = date_obj + period_to_hour(period)

                        # Keep only full hours (periods 1, 3, 5, ..., 47)
                        if dt.minute == 0:
                            all_data.append([dt, price])
                    except:
                        raise ValueError(f"Error processing row: {row}")

    df_out = pd.DataFrame(all_data, columns=["Datetime", "Price (SGD/MWh)"])
    df_out.sort_values("Datetime", inplace=True)
    return df_out


for year in YEARS:
    print(f"Processing {year}...")
    zip_filename = f"USEP_from_01-Jan-{year}_to_31-Dec-{year}.zip"
    zip_path = os.path.join(INPUT_DIR, zip_filename)
    if not os.path.exists(zip_path):
        print(f"ZIP file not found for {year}: {zip_path}")
        continue

    df = process_zip(zip_path, year)
    if df.empty:
        print(f"No data extracted for {year}")
        continue

    out_folder = os.path.join(OUTPUT_BASE, str(year))
    os.makedirs(out_folder, exist_ok=True)
    out_file = os.path.join(out_folder, f"SG_electricity_prices_{year}.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")




#%%