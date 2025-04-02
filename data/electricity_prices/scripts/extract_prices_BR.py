import pandas as pd
import os
from pathlib import Path

# ---- CONFIG ----
INPUT_FILE = Path("data/electricity_prices/raw/BR/BR_Historico_do_Preco_Horario_-_17_de_abril_de_2018_a_7_de_marco_de_2025.xlsx")
OUTPUT_DIR = Path("data/electricity_prices/processed/BR/")
TARGET_YEARS = [2020, 2021, 2022, 2023, 2024]
TARGET_REGION = "SUDESTE"  # Most data centers are located in São Paulo

# ---- LOAD FILE ----
df = pd.read_excel(INPUT_FILE)

# ---- CLEANING ----
# Melt wide format to long
df_long = df.melt(id_vars=["Hora", "Submercado"], var_name="Date", value_name="Price")
df_long = df_long[df_long["Submercado"] == TARGET_REGION].copy()

# Build datetime
df_long["Datetime"] = pd.to_datetime(df_long["Date"]) + pd.to_timedelta(df_long["Hora"], unit="h")

# Clean prices: remove non-numeric, handle missing
df_long["Price"] = pd.to_numeric(df_long["Price"], errors="coerce")
df_long.dropna(subset=["Price"], inplace=True)

# Keep only needed columns
df_long = df_long[["Datetime", "Price"]]

# Change the name of the columns to  Datetime, Price (R$/MWh)
df_long.columns = ["Datetime", "Price (R$/MWh)"]

# ---- EXPORT PER YEAR ----
for year in TARGET_YEARS:
    df_year = df_long[df_long["Datetime"].dt.year == year].copy()
    if df_year.empty:
        continue

    year_folder = OUTPUT_DIR / str(year)
    year_folder.mkdir(parents=True, exist_ok=True)

    out_file = year_folder / f"BR_electricity_prices_{year}.csv"
    df_year.to_csv(out_file, index=False)

    print(f"Saved: {out_file}")
