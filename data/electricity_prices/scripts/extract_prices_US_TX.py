import os
import pandas as pd
from datetime import date, timedelta
from gridstatus import Ercot, Markets

BASE_FOLDER = "US-TEX-ERCO"
os.makedirs(BASE_FOLDER, exist_ok=True)

# Initialize ISO
iso = Ercot()

# Define hub or zone you want (e.g. HB_HUBAVG, HB_BUSAVG, HB_NORTH)
TARGET_LOCATION = "HB_HUBAVG"

# Loop through years
for year in range(2020, 2025):
    print(f"Processing {year}...")
    try:
        df = iso.get_dam_spp(year=year)

        # Filter only for the target location
        df = df[df["Location"] == TARGET_LOCATION]

        # Select and rename columns
        df = df[["Interval Start", "SPP"]].rename(
            columns={"Interval Start": "Datetime", "SPP": "Price (USD/MWh)"}
        )
        df = df.sort_values("Datetime")

        # Save
        year_folder = os.path.join(BASE_FOLDER, str(year))
        os.makedirs(year_folder, exist_ok=True)
        out_path = os.path.join(year_folder, f"{BASE_FOLDER}_electricity_prices_{year}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

    except Exception as e:
        print(f"Failed {year}: {e}")