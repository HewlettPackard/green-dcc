#%%
import pandas as pd

# Load the marginal cost data
file_path = "../raw/CL/CL_2020-01-01_2024-01-01_costos-marginales-reales.csv"
df = pd.read_csv(file_path)

# Display structure and sample
df.columns, df.head(3)


#%%
import os
from datetime import timedelta

# Split datetime into date and hour
df["Date"] = df["item"].str.extract(r"(\d{4}-\d{2}-\d{2})")
df["Hour"] = df["item"].str.extract(r"(\d{2})$").astype(int)

# Shift date and set hour=0 where Hour == 24
mask = df["Hour"] == 24
df.loc[mask, "Date"] = pd.to_datetime(df.loc[mask, "Date"]) + timedelta(days=1)
df.loc[mask, "Hour"] = 0


# Combine into proper datetime
df["Datetime"] = pd.to_datetime(df["Date"]) + pd.to_timedelta(df["Hour"], unit="h") - timedelta(hours=1)

# Drop temp columns and reorder
df = df[["Datetime", "Quillota"]]

# Optional: sort if needed
df = df.sort_values("Datetime")

# Constants (in CLP/MWh)
TRANSMISSION_COST = 15_000  # CLP/MWh
ANCILLARY_COST = 3_000      # CLP/MWh
VAT = 0.19

# Final price calculation
df['MarginalPrice'] = df['Quillota'] * 1000  # from CLP/kWh to CLP/MWh
df['FinalPrice'] = (df['MarginalPrice'] + TRANSMISSION_COST + ANCILLARY_COST) * (1 + VAT)

# Extract year-wise and replicate 2023 for 2024
final_data = {}
for year in [2020, 2021, 2022, 2023]:
    year_df = df[df['Datetime'].dt.year == year][['Datetime', 'FinalPrice']].copy()
    # Change the columns to Datetime and Price (CLP/MWh)
    year_df.columns = ['Datetime', 'Price (CLP/MWh)']
    final_data[year] = year_df

# Replicate 2023 for 2024
if 2023 in final_data:
    df_2024 = final_data[2023].copy()
    df_2024['Datetime'] = df_2024['Datetime'] + pd.DateOffset(years=1)
    df_2024.columns = ['Datetime', 'Price (CLP/MWh)']
    final_data[2024] = df_2024

# Save the data into different folders/files
output_base = "../processed/CL"

for year, year_df in final_data.items():
    year_folder = f"{output_base}/{year}"
    os.makedirs(year_folder, exist_ok=True)

    out_file = f"{year_folder}/CL_electricity_prices_{year}.csv"
    year_df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

#%%


#%%


#%%