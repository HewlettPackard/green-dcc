import os
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

INPUT_FOLDER = "data/electricity_prices/raw/KR/files"
OUTPUT_BASE = "data/electricity_prices/processed/KR"
TIMEZONE = ZoneInfo("Asia/Seoul")

def process_excel_file(filepath):
    print(f"Processing {filepath}")
    df = pd.read_excel(filepath, skiprows=1)  # skip header rows

    all_rows = []

    for _, row in df.iterrows():
        date_str = str(int(row["구분"]))
        if pd.isna(date_str) or not str(date_str).startswith("20"):
            continue

        for hour in range(1, 25):
            price = row[f"{hour}h"]
            if pd.isna(price):
                continue

            dt = datetime.strptime(date_str, "%Y%m%d") + timedelta(hours=hour - 1)

            all_rows.append([dt, price * 1000])  # Convert from KRW/kWh → KRW/MWh

    df_out = pd.DataFrame(all_rows, columns=["Datetime", "Price (KRW/MWh)"])
    df_out.sort_values("Datetime", inplace=True)

    return df_out

def main():
    for file in os.listdir(INPUT_FOLDER):
        if not file.endswith(".xlsx"):
            continue

        filepath = os.path.join(INPUT_FOLDER, file)
        year = file[-9:-5]  # Extract '202X' from filename

        df_processed = process_excel_file(filepath)

        output_folder = os.path.join(OUTPUT_BASE, year)
        os.makedirs(output_folder, exist_ok=True)

        output_file = os.path.join(output_folder, f"KR_electricity_prices_{year}.csv")
        df_processed.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

if __name__ == "__main__":
    main()
