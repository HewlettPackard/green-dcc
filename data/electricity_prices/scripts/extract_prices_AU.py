#%%
import os
import requests

base_url = "https://aemo.com.au/aemo/data/nem/priceanddemand"
zones = {
    "AU-NSW": "NSW1",
    "AU-VIC": "VIC1",
}

years = range(2020, 2025)
months = [f"{m:02d}" for m in range(1, 13)]

for zone_code, region in zones.items():
    output_dir = f"data/electricity_prices/raw/{zone_code}"
    os.makedirs(output_dir, exist_ok=True)

    for year in years:
        for month in months:
            filename = f"PRICE_AND_DEMAND_{year}{month}_{region}.csv"
            url = f"{base_url}/{filename}"
            filepath = os.path.join(output_dir, filename)

            if os.path.exists(filepath):
                print(f"[✓] Already downloaded: {filepath}")
                continue

            try:
                print(f"Downloading {url}...")
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }

                response = requests.get(url, headers=headers, timeout=20)
                # response = requests.get(url, timeout=20)
                response.raise_for_status()

                with open(filepath, "wb") as f:
                    f.write(response.content)

                print(f"[✓] Saved to: {filepath}")
            except requests.exceptions.RequestException as e:
                print(f"[!] Failed to download {url}: {e}")


#%%
import pandas as pd
from pathlib import Path

# Set raw and processed data directories
raw_dir = Path("data/electricity_prices/raw")
processed_dir = Path("data/electricity_prices/processed")

# Define function to process files for a region
def process_region(region_code):
    files = sorted((raw_dir / f"AU-{region_code}").glob("*.csv"))
    yearly_data = {}

    for file in files:
        year = file.stem.split("_")[3][:4]
        df = pd.read_csv(file)

        # Parse datetime and floor to the hour
        df["Datetime"] = pd.to_datetime(df["SETTLEMENTDATE"])
        df["Hour"] = df["Datetime"].dt.floor("H")
        df["Price"] = df["RRP"]

        # Compute hourly average price
        hourly_avg = (
            df.groupby("Hour")["Price"]
            .mean()
            .reset_index()
            .rename(columns={"Hour": "Datetime", "Price": "Price (USD/MWh)"})
        )

        if year not in yearly_data:
            yearly_data[year] = []
        yearly_data[year].append(hourly_avg)

    # Concatenate and save
    for year, dfs in yearly_data.items():
        df_year = pd.concat(dfs).sort_values("Datetime").reset_index(drop=True)
        out_path = processed_dir / f"AU-{region_code}" / year
        out_path.mkdir(parents=True, exist_ok=True)
        df_year.to_csv(out_path / f"AU-{region_code}_electricity_prices_{year}.csv", index=False)

# Process NSW and VIC
process_region("NSW")
process_region("VIC")


#%%


#%%