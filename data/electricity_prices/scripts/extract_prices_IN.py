from pathlib import Path
import pandas as pd

# Output directory
output_root = Path("data/electricity_prices/processed/IN")
raw_files_path = Path("data/electricity_prices/raw/IN")

extracted_files = list(raw_files_path.glob("*.xlsx"))

# Clean and reprocess all files, skipping invalid 'Hour' rows
all_data = []

for file in extracted_files:
    df = pd.read_excel(file, skiprows=4)
    df = df.rename(columns={
        df.columns[0]: "Date",
        df.columns[1]: "Hour",
        df.columns[6]: "Price (INR/MWh)",
    })

    df = df[["Date", "Hour", "Price (INR/MWh)"]]

    # Drop invalid rows (non-date strings or hour not numeric)
    df = df[df["Date"].apply(lambda x: isinstance(x, str) and "-" in x)]
    df = df[df["Hour"].apply(lambda x: isinstance(x, (int, float)) or str(x).isdigit())]
    # Print df shape
    
    df["Datetime"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce") \
                      + pd.to_timedelta(df["Hour"].astype(int), unit="h")
    df = df.dropna(subset=["Datetime", "Price (INR/MWh)"])
    df = df[["Datetime", "Price (INR/MWh)"]]
    all_data.append(df)

# Combine and deduplicate
df_all = pd.concat(all_data).drop_duplicates("Datetime").sort_values("Datetime").reset_index(drop=True)

# Save one file per year
for year, df_year in df_all.groupby(df_all["Datetime"].dt.year):
    print(df_year.shape)
    out_dir = output_root / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    # df_year.to_csv(
    #     out_dir / f"IN_electricity_prices_{year}.csv",
    #     index=False
    # )

# ----------------------------
# SIMULATE YEAR 2020 USING 2021 DATA
# ----------------------------
df_2021 = df_all[df_all["Datetime"].dt.year == 2021].copy()
df_2021["Datetime"] = df_2021["Datetime"] - pd.DateOffset(years=1)

# Save simulated 2020 data
out_dir_2020 = output_root / "2020"
out_dir_2020.mkdir(parents=True, exist_ok=True)
# df_2021.to_csv(
#     out_dir_2020 / "IN_electricity_prices_2020.csv",
#     index=False
# )

# ----------------------------
# FILL GAP IN MAY 2022 USING MAY 2021 DATA
# ----------------------------

# Load back processed 2021 and 2022
df_2021 = pd.read_csv(output_root / "2021" / "IN_electricity_prices_2021.csv", parse_dates=["Datetime"])
df_2022 = pd.read_csv(output_root / "2022" / "IN_electricity_prices_2022.csv", parse_dates=["Datetime"])

# Define gap period to fill
gap_start = pd.Timestamp("2022-05-17 07:00:00")
gap_end = pd.Timestamp("2022-05-24 04:00:00")

# Take May 2021 and shift it to 2022
may_2021 = df_2021[df_2021["Datetime"].dt.month == 5].copy()
may_2021["Datetime"] = may_2021["Datetime"].apply(lambda dt: dt.replace(year=2022))

# Extract only rows that fall within the 2022 gap
filler = may_2021[(may_2021["Datetime"] >= gap_start) & (may_2021["Datetime"] <= gap_end)].copy()

# Remove existing 2022 data inside the gap (if any)
df_2022 = df_2022[~df_2022["Datetime"].between(gap_start, gap_end)]

# Append the filler data and sort
df_2022_filled = pd.concat([df_2022, filler], ignore_index=True)
df_2022_filled = df_2022_filled.sort_values("Datetime").reset_index(drop=True)

# Save updated 2022 file
df_2022_filled.to_csv(
    output_root / "2022" / "IN_electricity_prices_2022.csv",
    index=False
)