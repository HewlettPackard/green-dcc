#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
#%%
# Folder path
PATH = 'data/Weather'

# Initialize dictionary for temperature data
temperature_dict = {}

# Loop through each file in the folder
for location in os.listdir(PATH):
    if location.endswith('.epw'):  # Only process .epw files
        # Extract location name (removing file extension and other characters)
        location_name = location.replace('.epw', '').replace('USA_', '').replace('_', ' ').replace('-', ' ')
        
        # Read the data
        weather_data = pd.read_csv(f'{PATH}/{location}', skiprows=8, header=None).values
        temperature_data = weather_data[:, 6].astype(float)  # Column 6 contains temperature
        
        # Store in the dictionary
        temperature_dict[location_name] = temperature_data
#%%
# Plotting
plt.figure(figsize=(12, 8))

for location, temperatures in temperature_dict.items():
    plt.plot(temperatures, label=location)

# Add labels and title
plt.xlabel('Hour of the Year')
plt.ylabel('Temperature (°C)')
plt.title('Comparative Temperature Data Across Locations')
plt.legend(loc='best')
plt.grid()

# Show plot
plt.show()
#%%
# Folder path
PATH = 'data/Weather'

# Initialize dictionary for temperature data
temperature_dict = {}

# Loop through each file in the folder
for location in os.listdir(PATH):
    if location.endswith('.epw'):  # Only process .epw files
        # Extract location name (removing file extension and other characters)
        location_name = location.replace('.epw', '').replace('USA_', '').replace('_', ' ').replace('-', ' ')
        
        # Read the data
        weather_data = pd.read_csv(f'{PATH}/{location}', skiprows=8, header=None).values
        
        # Filter rows where month (column 1) is 7 (July)
        july_data = weather_data[(weather_data[:, 1] == 7) & (weather_data[:, 2] < 7)]  # Column 1 (index 0) has month info
        
        # Extract temperature data (column 6)
        temperature_data = july_data[:, 6].astype(float)
        
        # Store in the dictionary
        temperature_dict[location_name] = temperature_data

# Plotting
plt.figure(figsize=(12, 6))
import numpy as np
# List of markers for differentiation
markers = ['o', 's', 'D', '^', 'v', '>', '<', 'h', 'p', '*', 'x']

# Plot each location with markers
for idx, (location, temperatures) in enumerate(temperature_dict.items()):
    marker = markers[idx % len(markers)]  # Cycle through markers
    plt.plot(temperatures, label=location, marker=marker, markevery=10)  # Use `markevery` to space out markers
    print(f'Location: {location}, Avg: {np.mean(temperatures):.3f}')


# Add labels and title
plt.xlabel('Hour of the Month (July)')
plt.ylabel('Temperature (°C)')
plt.title('Comparative Temperature Data for July Across Locations')
plt.legend(loc='best')
plt.grid()

# Show plot
plt.show()
#%%