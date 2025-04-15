#%%
import os
import random
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib

# Set global style
sns.set_theme(style="whitegrid")
matplotlib.rcParams.update({"font.size": 16})

# Define the path where weather data is stored
data_path = "."

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
    "IN-WE": "Mumbai/IN",
    "JP-TK": "Tokyo/JP",
    "GB": "London/UK"
}

# Get list of locations
locations = [loc for loc in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, loc))]

# Drop the deprecated location
locations.remove("deprecated")

# Create a list to store all data
all_data = []

# Loop through each location
for location in locations:
    location_path = os.path.join(data_path, location)
    location_data = []

    # Loop through each year's data file
    for file in sorted(os.listdir(location_path)):
        if file.endswith(".json"):
            file_path = os.path.join(location_path, file)
            with open(file_path, "r") as f:
                data = json.load(f)

            # Extract hourly temperature data
            timestamps = data["hourly"]["time"]
            temperatures = data["hourly"]["temperature_2m"]

            # Convert to DataFrame
            df = pd.DataFrame({"datetime": timestamps, "temperature": temperatures})
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["date"] = df["datetime"].dt.strftime("%m-%d")  # Extract only month and day
            df["location"] = location_labels.get(location, location)  # Use human-readable labels
            location_data.append(df)

    # Concatenate all years for the location
    if location_data:
        combined_df = pd.concat(location_data)
        all_data.append(combined_df)

# Combine all locations into a single dataset
full_dataset = pd.concat(all_data)

# Group by date and location to compute mean daily temperature
mean_temperature = full_dataset.groupby(["date", "location"])["temperature"].mean().reset_index()
#%%
# Define different line styles to avoid duplicates
linestyles = ['-', '--', '-.']
plt.figure(figsize=(12, 6))
color_map = {}

for i, location in enumerate(mean_temperature["location"].unique()):
    subset = mean_temperature[mean_temperature["location"] == location]
    color = f"C{i % 10}"  # Use matplotlib's default color cycle
    linestyle = linestyles[i % len(linestyles)]  # Cycle through linestyles
    plt.plot(subset["date"], subset["temperature"], label=location, color=color, linestyle=linestyle, alpha=0.9)

# Reduce the number of x-axis ticks
xticks_indices = np.linspace(0, len(mean_temperature["date"].unique()) - 1, num=12, dtype=int)
xticks_labels = mean_temperature["date"].unique()[xticks_indices]
plt.xticks(xticks_labels, rotation=45)

# Finalize plot
plt.xlabel("Day of Year (MM-DD)")
plt.ylabel("Average Daily Temperature (°C)")
plt.title("Typical Daily Temperature Trends Across Locations")
plt.grid(linestyle="--", alpha=0.7) 
plt.xlim(0, len(mean_temperature["date"].unique()))

# Place the legend outside the plot
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()

# Save the figure as SVG
plt.savefig("../../assets/figures/temperature_trends.svg", format="svg", bbox_inches='tight')
plt.show()


    #%%
