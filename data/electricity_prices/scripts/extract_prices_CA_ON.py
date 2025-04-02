import pandas as pd
import os
from datetime import datetime

# Paths
input_dir  = "data/electricity_prices/raw/CA-ON"
output_base_dir = "data/electricity_prices/processed/CA-ON"
YEARS = [2020, 2021, 2022, 2023, 2024]

# Process files from 2020 to 2024
processed_files = []
for year in range(2020, 2025):
    input_file = os.path.join(input_dir, f"PUB_PriceHOEPPredispOR_{year}.csv")
    output_dir = os.path.join(output_base_dir, str(year))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"CA-ON_electricity_prices_{year}.csv")

    # Read the CSV, skip the first 3 metadata rows
    df = pd.read_csv(input_file, skiprows=3)

    # Drop rows with missing HOEP or Date
    df = df[["Date", "Hour", "HOEP"]].dropna()

    # Fix Hour format: Hour 1 = 00:00, Hour 2 = 01:00, ..., Hour 24 = 23:00
    df["Hour"] = df["Hour"].astype(int) - 1
    df["Datetime"] = pd.to_datetime(df["Date"]) + pd.to_timedelta(df["Hour"], unit="h")

    # Keep only necessary columns
    df = df[["Datetime", "HOEP"]]
    df.rename(columns={"HOEP": "Price (CAD/MWh)"}, inplace=True)

    # Sort by time
    df = df.sort_values("Datetime")

    # Save to CSV
    df.to_csv(output_file, index=False)
    processed_files.append(output_file)
