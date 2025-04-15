#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Define ===
data_dir = "standardized"  # Update this to your actual root path
years = ["2023", "2024"]  # You can include multiple years if desired

# Define display names for regions
location_labels = {
    "US-NY-NYIS": "New York/USA",
    "US-MIDA-PJM": "Philadelphia/USA",
    "US-TEX-ERCO": "Dallas/USA",
    "US-CAL-CISO": "San Francisco/USA",
    "DE-LU": "Frankfurt/GE",
    "CA-ON": "Toronto/CA",
    "SG": "Singapore/SG",
    "AU-VIC": "Melbourne/AU",
    "AU-NSW": "Sydney/AU",
    "CL-SEN": "Santiago/CL",
    "BR-SP": "São Paulo/BR",
    "ZA": "Johannesburg/ZA",
    "KR": "Seoul/KR",
    "IN": "Mumbai/IN",
    "JP-TK": "Tokyo/JP",
    "GB": "London/UK"
}

# Collect all relevant files
all_data = []
for region_code in os.listdir(data_dir):
    region_path = os.path.join(data_dir, region_code)
    if not os.path.isdir(region_path):
        continue

    for year in years:
        file_path = os.path.join(region_path, year, f"{region_code}_electricity_prices_{year}.csv")
        if not os.path.isfile(file_path):
            continue
        df = pd.read_csv(file_path, parse_dates=["Datetime (UTC)", "Datetime (Local)"])
        # If region code is not in location_labels, do not use that region
        if region_code not in location_labels:
            print(f"Region code {region_code} not found in location_labels. Skipping...")
            continue
        df["location"] = location_labels.get(region_code, region_code)
        df["time_of_day"] = df["Datetime (UTC)"].dt.floor("H").dt.strftime("%H:%M")
        all_data.append(df)

# Combine all locations
full_dataset = pd.concat(all_data)

# Group by time_of_day and location to compute average price
mean_prices = full_dataset.groupby(["time_of_day", "location"])["Price (USD/MWh)"].mean().reset_index()

# === Plot ===
linestyles = ['-', '--', '-.']
plt.figure(figsize=(12, 6))

for i, location in enumerate(mean_prices["location"].unique()):
    subset = mean_prices[mean_prices["location"] == location]
    color = f"C{i % 10}"
    linestyle = linestyles[i % len(linestyles)]
    plt.plot(subset["time_of_day"], subset["Price (USD/MWh)"],
             label=location, color=color, linestyle=linestyle, alpha=0.9)

# Improve x-axis readability
xticks_indices = np.linspace(0, len(mean_prices["time_of_day"].unique()) - 1, num=24, dtype=int)
xticks_labels = mean_prices["time_of_day"].unique()[xticks_indices]
plt.xticks(xticks_labels, rotation=45)
plt.xlim(0, len(mean_prices["time_of_day"].unique()) - 1)

# Labels and aesthetics
plt.xlabel("Time of Day (HH:MM) (UTC)")
plt.ylabel("Average Electricity Price (USD/MWh)")
plt.title("Typical Daily Electricity Price Variation Across Locations")
plt.grid(linestyle="--", alpha=0.7)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()

# Optional: save
plt.savefig("../../assets/figures/electricity_price_patterns.svg", format="svg")
plt.show()
#%%
