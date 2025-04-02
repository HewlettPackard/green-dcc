#%%
import gridstatus
import pandas as pd
import requests
SETTINGS_URL = "https://dataminer2.pjm.com/config/settings.json"

def get_subscription_key():
    r = requests.get(SETTINGS_URL)
    return r.json()["subscriptionKey"]

SUBSCRIPTION_KEY = get_subscription_key()

iso = gridstatus.PJM(SUBSCRIPTION_KEY)

print(iso.markets)
#%%
import os
from datetime import date
from gridstatus import PJM

SUBSCRIPTION_KEY = get_subscription_key()
pjm = PJM(api_key=SUBSCRIPTION_KEY)

BASE_DIR = "US-MIDA-PJM"
YEARS = range(2020, 2025)

for year in YEARS:
    print(f"Processing year {year}...")
    
    start = date(year, 1, 1)
    end = date(year, 12, 31)

    # Download LMP data
    df = pjm.get_lmp(
        market="REAL_TIME_HOURLY",
        start=start,
        end=end,
        location_type="ZONE"
    )

    # Filter for DOM (Dominion)
    df_dom = df[df["Location Name"] == "DOM"].copy()
    if df_dom.empty:
        print(f"No data for year {year}")
        continue

    # Process columns
    df_dom["Datetime"] = df_dom["Interval Start"].dt.tz_convert("UTC").dt.tz_localize(None)
    df_dom["Price (USD/MWh)"] = df_dom["LMP"]
    final_df = df_dom[["Datetime", "Price (USD/MWh)"]].sort_values("Datetime")

    # Save to correct folder
    out_folder = os.path.join(BASE_DIR, str(year))
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f"US-MIDA-PJM_electricity_prices_{year}.csv")
    final_df.to_csv(out_path, index=False)
    
    print(f"Saved: {out_path}")

#%%
#%%
