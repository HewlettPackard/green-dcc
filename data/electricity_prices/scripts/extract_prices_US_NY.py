from gridstatus import NYISO
from datetime import date, timedelta
import pandas as pd
import os

# Setup
iso = NYISO()
start_year = 2020
end_year = 2024
output_base = "./data/electricity_prices/US-NY-NYIS"

# Collect data year by year
for year in range(start_year, end_year + 1):
    print(f"Processing {year}...")
    start = date(year, 1, 1)
    end = date(year, 12, 31)

    # Fetch hourly day-ahead LMPs at the "CAPITL" zone (Albany area, common reference hub)
    try:
        df = iso.get_lmp(
            start=start,
            end=end,
            market="DAY_AHEAD_HOURLY",
            locations=["CAPITL"],  # Capital zone
        )
    except Exception as e:
        print(f"Error fetching {year}: {e}")
        continue

    if df.empty:
        print(f"No data for {year}")
        continue

    # Format to standard output
    df = df.rename(columns={"Interval Start": "Datetime", "LMP": "Price (USD/MWh)"})
    df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.tz_convert(None)
    df = df[["Datetime", "Price (USD/MWh)"]].sort_values("Datetime")

    # Save
    out_dir = os.path.join(output_base, str(year))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"US-NY-NYIS_electricity_prices_{year}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
