#%%
#%%
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---
# Directory where the standardized files are stored.
# The standardized files already have "Datetime (UTC)" and "Price (USD/MWh)" columns.
standardized_dir = Path("../standardized")

# Dynamically detect all region folders from the standardized directory
location_codes = sorted([f.name for f in standardized_dir.iterdir() if f.is_dir()])
print("Detected locations:", location_codes)

# Dictionary to store DataFrames per location
location_data = {}

# --- Data Loading ---
# Loop through each region folder and read all CSV files (across years)
for loc in tqdm(location_codes, desc="Loading standardized data"):
    loc_dir = standardized_dir / loc
    # Find all CSV files under the region folder (recursively)
    csv_files = list(loc_dir.rglob("*.csv"))
    if not csv_files:
        print(f"No CSV files found for {loc}, skipping...")
        continue
    
    df_list = []
    for file in csv_files:
        # Read the file; "Datetime (UTC)" is parsed as datetime
        df = pd.read_csv(file, parse_dates=["Datetime (UTC)"])
        df_list.append(df)
    
    # Concatenate data, remove duplicate timestamps, and sort by time
    if df_list:
        df_loc = pd.concat(df_list).drop_duplicates("Datetime (UTC)").sort_values("Datetime (UTC)").reset_index(drop=True)
        location_data[loc] = df_loc

# --- Plotting ---
plt.figure(figsize=(14, 8))
for loc, df_loc in location_data.items():
    # Apply smoothing with a rolling mean (e.g., 24-hour window)
    df_loc["Smoothed Price (USD/MWh)"] = df_loc["Price (USD/MWh)"].rolling(window=2, center=True).mean()

    plt.plot(df_loc["Datetime (UTC)"], df_loc["Smoothed Price (USD/MWh)"], label=loc)

plt.xlabel("Datetime (UTC)")
plt.ylabel("Price (USD/MWh)")
plt.title("Hourly Electricity Prices (Standardized to UTC, USD)")
plt.legend()
plt.tight_layout()

# Save the plot to a file
plt.savefig("standardized_electricity_prices.png")
plt.show()

#%%
from pathlib import Path
import pandas as pd

# Define paths
standardized_dir = Path("../standardized")

# Expected years and hourly count (including leap year)
expected_years = list(range(2020, 2025))
hours_per_year = {
    2020: 8784,  # Leap year
    2021: 8760,
    2022: 8760,
    2023: 8760,
    2024: 8784,  # Leap year
}

report = []
missing_days = []

# Traverse regions
for region in sorted(f.name for f in standardized_dir.iterdir() if f.is_dir()):
    region_path = standardized_dir / region
    for year in expected_years:
        file = region_path / str(year) / f"{region}_electricity_prices_{year}.csv"
        if not file.exists():
            report.append((region, year, "Missing CSV"))
            continue

        df = pd.read_csv(file, parse_dates=["Datetime (UTC)"])
        row_count = len(df)
        expected_rows = hours_per_year[year]

        if row_count != expected_rows:
            report.append((region, year, f"Unexpected row count: {row_count} vs {expected_rows}"))
            # Find which days are missing
            full_range = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31 23:00:00", freq="H", tz="UTC")
            missing = full_range.difference(df["Datetime (UTC)"])
            if not missing.empty:
                for m in missing:
                    missing_days.append((region, year, m.strftime("%Y-%m-%d %H:%M:%S %Z")))

        # Check for NaNs or unexpected price ranges
        if df["Price (USD/MWh)"].isna().any():
            report.append((region, year, "NaN in price column"))

        if df["Price (USD/MWh)"].lt(-100).any():
            report.append((region, year, "Price below -100 USD/MWh"))

        if df["Price (USD/MWh)"].gt(10000).any():
            report.append((region, year, "Price above 10,000 USD/MWh"))

# Prepare results
df_report = pd.DataFrame(report, columns=["Region", "Year", "Issue"])
df_missing_days = pd.DataFrame(missing_days, columns=["Region", "Year", "Missing Datetime"])

print(df_report)


#%%