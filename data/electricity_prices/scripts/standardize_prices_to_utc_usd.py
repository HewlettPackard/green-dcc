#%%
from pathlib import Path
import pandas as pd
import pytz
from tqdm import tqdm


# === Config ===
processed_dir = Path("../processed")
output_dir = Path("../standardized")
output_dir.mkdir(parents=True, exist_ok=True)

# --- Currencies to USD ---
# Extracted from oanda.com on 2025-03-27
conversion_rates = {
    "USD": 1.0,
    "EUR": 1.07764,
    "AUD": 0.63056,
    "BRL": 0.17488,
    "R$":  0.17488,       # Alias for BRL
    "CAD": 0.70067,
    "CHF": 1.13136,
    "INR": 0.01166,
    "JPY": 0.00665,
    "KRW": 0.00068,
    "SGD": 0.74694,
    "CLP": 0.00108,
    "ZAR": 0.05474,
}

# --- Timezones per region ---
timezone_mapping = {
    "AT": "Europe/Vienna",
    "AU-NSW": "Australia/Sydney",
    "AU-VIC": "Australia/Melbourne",
    "BE": "Europe/Brussels",
    "BR": "America/Sao_Paulo",
    "CA-ON": "America/Toronto",
    "CH": "Europe/Zurich",
    "CL-SIC": "America/Santiago",
    "DE-LU": "Europe/Berlin",
    "ES": "Europe/Madrid",
    "FR": "Europe/Paris",
    "GB": "Europe/London",
    "IN": "Asia/Kolkata",
    "JP-TK": "Asia/Tokyo",
    "KR": "Asia/Seoul",
    "NL": "Europe/Amsterdam",
    "PT": "Europe/Lisbon",
    "SG": "Asia/Singapore",
    "US-CAL-CISO": "America/Los_Angeles",
    "US-MIDA-PJM": "America/New_York",
    "US-NY-NYIS": "America/New_York",
    "US-TEX-ERCO": "America/Chicago",
    "ZA": "Africa/Johannesburg",
}

# === Detect region folders ===
regions = sorted([f.name for f in processed_dir.iterdir() if f.is_dir()])

# Remove the regions with "incompleted" in the name
regions = [region for region in regions if "incompleted" not in region]

for region in tqdm(regions, desc="Standardizing regions"):
    tz = timezone_mapping.get(region)
    if not tz:
        print(f"[!] Missing timezone for region: {region}, skipping.")
        continue

    files = list((processed_dir / region).rglob("*.csv"))
    for file in files:
        df = pd.read_csv(file)
        if "Datetime" not in df.columns:
            print(f"[!] Missing 'Datetime' column in {file}, skipping.")
            continue

        # Detect currency
        price_col = [col for col in df.columns if col.startswith("Price (") and col.endswith("/MWh)")]
        if not price_col:
            print(f"[!] No price column found in {file}, skipping.")
            continue
        price_col = price_col[0]
        currency = price_col.split(" ")[1].split("/")[0].strip("()").upper()

        if currency not in conversion_rates:
            print(f"[!] Currency {currency} not found in conversion table for {file}")
            continue

        # --- Datetime parsing ---
        has_tz_offset = df["Datetime"].astype(str).str.contains(r"[+-]\d{2}:\d{2}").any()
        if has_tz_offset:
            df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", utc=True)
        else:
            df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
            try:
                df["Datetime"] = df["Datetime"].dt.tz_localize(
                    pytz.timezone(tz), nonexistent='shift_forward', ambiguous='NaT'
                )
            except Exception as e:
                print(f"[!] Error localizing timezone in {file}: {e}")
                continue

        df["Datetime"] = df["Datetime"].dt.tz_convert("UTC")
        df["Price (USD/MWh)"] = df[price_col] * conversion_rates[currency]

        # --- Smoothing ---
        df = df.sort_values("Datetime").dropna(subset=["Datetime", "Price (USD/MWh)"])
        df["Price (USD/MWh)"] = df["Price (USD/MWh)"].rolling(window=2, center=True, min_periods=1).mean()

        # --- Extrapolation ---
        df = df.set_index("Datetime")
        df = df[~df.index.duplicated(keep="first")]
        year = df.index[len(df) // 2].year

        # Determine base minute and frequency
        base_minute = df.index[0].minute
        step = df.index[1] - df.index[0]

        start = pd.Timestamp(f"{year}-01-01 00:{base_minute:02d}", tz="UTC")
        end = pd.Timestamp(f"{year}-12-31 23:{base_minute:02d}", tz="UTC")

        full_index = pd.date_range(start=start, end=end, freq=step, tz="UTC")
        df = df.reindex(full_index)
        df["Price (USD/MWh)"] = df["Price (USD/MWh)"].interpolate(method="time", limit_direction="both")

        # --- Add local time column ---
        local_tz = pytz.timezone(timezone_mapping[region])
        df["Datetime (Local)"] = df.index.tz_convert(local_tz)

        # --- Final format ---
        df = df.reset_index().rename(columns={"index": "Datetime (UTC)"})
        df = df[["Datetime (UTC)", "Datetime (Local)", "Price (USD/MWh)"]].dropna()

        # --- Save ---
        out_region_dir = output_dir / region / str(year)
        out_region_dir.mkdir(parents=True, exist_ok=True)
        output_file = out_region_dir / f"{region}_electricity_prices_{year}.csv"
        df.to_csv(output_file, index=False)

#%%
