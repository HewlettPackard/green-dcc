import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('temperature_log.csv')

# Check if the data has duplicate headers within the file
# If so, we need to clean it up
if df.iloc[0].str.contains('Timestamp').any():
    # Data has embedded headers, needs cleaning
    with open('temperature_log.csv', 'r') as file:
        lines = file.readlines()
    
    # Keep the first header line and filter out other header lines
    clean_lines = [lines[0]]
    for line in lines[1:]:
        if not line.startswith('Timestamp,Inlet Temperature,Outlet Temperature'):
            clean_lines.append(line)
    
    # Parse the cleaned data
    clean_data = ''.join(clean_lines)
    df = pd.read_csv(pd.io.StringIO(clean_data))

# Convert temperature columns to numeric values
df['Inlet Temperature'] = pd.to_numeric(df['Inlet Temperature'], errors='coerce')
df['Outlet Temperature'] = pd.to_numeric(df['Outlet Temperature'], errors='coerce')

# Drop any rows with NaN values
df = df.dropna(subset=['Inlet Temperature', 'Outlet Temperature'])

# Limit to first 500 data points
df = df.head(10000)
print(f"Plotting first {len(df)} data points")

# Create a counter column for x-axis
df['Measurement'] = range(1, len(df) + 1)

# Create the plot with two lines
plt.figure(figsize=(12, 6))

# Plot Inlet Temperature
plt.plot(df['Measurement'], df['Inlet Temperature'], 
         color='blue', marker=None, linestyle='-', linewidth=2, label='Inlet Temperature')

# Plot Outlet Temperature
plt.plot(df['Measurement'], df['Outlet Temperature'], 
         color='red', marker=None, linestyle='-', linewidth=2, label='Outlet Temperature')

# Add titles and labels
plt.title('Inlet vs Outlet Temperature Comparison', fontsize=16)
plt.xlabel('Measurement Number', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Set the y-axis limit between 0 and 70
plt.ylim(bottom=0, top=70)

# Calculate average temperature difference
avg_diff = (df['Outlet Temperature'] - df['Inlet Temperature']).mean()
plt.annotate(f'Avg Temp Difference: {avg_diff:.2f}°C', 
             xy=(0.02, 0.95), xycoords='axes fraction',
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.savefig('temperature_comparison.png', dpi=300)
plt.show()

# Simple summary statistics
print(f"\nAverage Inlet Temperature: {df['Inlet Temperature'].mean():.2f}°C")
print(f"Average Outlet Temperature: {df['Outlet Temperature'].mean():.2f}°C")
print(f"Average Temperature Difference: {avg_diff:.2f}°C")