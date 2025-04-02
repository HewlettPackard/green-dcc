#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the path where carbon intensity data is stored
data_path = "."

location_labels = {
    "US-NY-NYIS": "NY/USA",
    "US-MIDA-PJM": "Philadelphia/USA",
    "US-TEX-ERCO": "Dallas/USA",
    "US-CAL-CISO": "San Jose/USA",
    "DE": "Frankfurt/GE",
    "CA-ON": "Toronto/CA",
    "SG": "Singapore/SG",
    "AU-VIC": "Melbourne/AU",
    "AU-NSW": "Sydney/AU",
    "CL-SEN": "Santiago/CL",
    "BR-CS": "São Paulo/BR",
    "ZA": "Johannesburg/ZA",
    "KR": "Seoul/KR",
    "IN-WE": "Mumbai/IN",
    "JP-TK": "Tokyo/JP",
    "GB": "London/UK"
}

# Get list of locations
locations = [loc for loc in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, loc))]

# Drop the deprecated location if exists
if "deprecated" in locations:
    locations.remove("deprecated")

# Create a list to store all data
all_data = []

# Loop through each location
for location in locations:
    location_path = os.path.join(data_path, location)
    location_data = []

    # Loop through each year's data file
    for year in sorted(os.listdir(location_path)):
        year_path = os.path.join(location_path, year)
        if os.path.isdir(year_path):
            for file in sorted(os.listdir(year_path)):
                if file.endswith(".csv"):
                    file_path = os.path.join(year_path, file)
                    df = pd.read_csv(file_path, parse_dates=["Datetime (UTC)"])
                    df["date"] = df["Datetime (UTC)"].dt.strftime("%m-%d")  # Extract only month and day
                    df["location"] = location_labels.get(location, location)  # Use human-readable labels
                    location_data.append(df)

    # Concatenate all years for the location
    if location_data:
        combined_df = pd.concat(location_data)
        all_data.append(combined_df)

# Combine all locations into a single dataset
full_dataset = pd.concat(all_data)

# Group by date and location to compute mean daily carbon intensity
mean_carbon_intensity = full_dataset.groupby(["date", "location"])["Carbon Intensity gCO₂eq/kWh (direct)"].mean().reset_index()
#%%
# Define different line styles to avoid duplicates
linestyles = ['-', '--', '-.']
plt.figure(figsize=(12, 6))
color_map = {}

for i, location in enumerate(mean_carbon_intensity["location"].unique()):
    subset = mean_carbon_intensity[mean_carbon_intensity["location"] == location]
    color = f"C{i % 10}"  # Use matplotlib's default color cycle
    linestyle = linestyles[i % len(linestyles)]  # Cycle through linestyles
    plt.plot(subset["date"], subset["Carbon Intensity gCO₂eq/kWh (direct)"], label=location, color=color, linestyle=linestyle, alpha=0.7)

# Reduce the number of x-axis ticks
xticks_indices = np.linspace(0, len(mean_carbon_intensity["date"].unique()) - 1, num=12, dtype=int)
xticks_labels = mean_carbon_intensity["date"].unique()[xticks_indices]
plt.xticks(xticks_labels, rotation=45, fontsize=8)

# Finalize plot
plt.xlabel("Day of Year (MM-DD)")
plt.ylabel("Average Daily Carbon Intensity (gCO₂eq/kWh)")
plt.title("Typical Daily Carbon Intensity Trends Across Locations")
plt.grid(linestyle="--", alpha=0.7)
plt.xlim(0, len(mean_carbon_intensity["date"].unique()))

# Place the legend outside the plot
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.show()
#%%
all_data = []
# Loop through each location
for location in locations:
    location_path = os.path.join(data_path, location)
    location_data = []

    # Loop through each year's data file
    for year in sorted(os.listdir(location_path)):
        year_path = os.path.join(location_path, year)
        if os.path.isdir(year_path):
            for file in sorted(os.listdir(year_path)):
                if file.endswith(".csv"):
                    file_path = os.path.join(year_path, file)
                    df = pd.read_csv(file_path, parse_dates=["Datetime (UTC)"])
                    df["time_of_day"] = df["Datetime (UTC)"].dt.strftime("%H:%M")  # Extract only hour and minute
                    df["location"] = location_labels.get(location, location)  # Use human-readable labels
                    location_data.append(df)

    # Concatenate all years for the location
    if location_data:
        combined_df = pd.concat(location_data)
        all_data.append(combined_df)

# Combine all locations into a single dataset
full_dataset = pd.concat(all_data)

# Group by time_of_day and location to compute mean carbon intensity
mean_carbon_intensity = full_dataset.groupby(["time_of_day", "location"])["Carbon Intensity gCO₂eq/kWh (direct)"].mean().reset_index()

# Define different line styles to avoid duplicates
linestyles = ['-', '--', '-.']
plt.figure(figsize=(12, 6))
color_map = {}

for i, location in enumerate(mean_carbon_intensity["location"].unique()):
    subset = mean_carbon_intensity[mean_carbon_intensity["location"] == location]
    color = f"C{i % 10}"  # Use matplotlib's default color cycle
    linestyle = linestyles[i % len(linestyles)]  # Cycle through linestyles
    plt.plot(subset["time_of_day"], subset["Carbon Intensity gCO₂eq/kWh (direct)"], label=location, color=color, linestyle=linestyle, alpha=0.7)

# Reduce the number of x-axis ticks
xticks_indices = np.linspace(0, len(mean_carbon_intensity["time_of_day"].unique()) - 1, num=24, dtype=int)
xticks_labels = mean_carbon_intensity["time_of_day"].unique()[xticks_indices]
plt.xticks(xticks_labels, rotation=45, fontsize=8)
plt.xlim(0, len(mean_carbon_intensity["time_of_day"].unique())-1)

# Finalize plot
plt.xlabel("Time of Day (HH:MM) (UTC)")
plt.ylabel("Average Carbon Intensity (gCO₂eq/kWh)")
plt.title("Typical Daily Carbon Intensity Variation Across Locations")
plt.grid(linestyle="--", alpha=0.7)

# Place the legend outside the plot
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.show()
#%%