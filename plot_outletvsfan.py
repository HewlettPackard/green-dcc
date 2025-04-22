import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import matplotlib.dates as mdates

def plot_thermal_control_dynamics():
    """
    Creates visualizations of thermal control dynamics based on logged data.
    Shows the relationship between fan speed, temperature, and system oscillation.
    Limited to first 100000 points for clarity.
    """
    try:
        # Load the data files with error handling
        print("Loading thermal control data...")
        thermal_data = pd.read_csv('thermal_control_data.csv')
        
        # Check for array values in fan_ratio column
        if isinstance(thermal_data['fan_ratio'].iloc[0], str) and '[' in thermal_data['fan_ratio'].iloc[0]:
            print("Converting array fan ratios to scalar values...")
            # Convert string representations of arrays to float averages
            thermal_data['fan_ratio'] = thermal_data['fan_ratio'].apply(
                lambda x: float(np.mean(eval(x))) if isinstance(x, str) and '[' in x else float(x)
            )
        
        print("Loading temperature data...")
        try:
            temp_data = pd.read_csv('temperature_log.csv')
        except pd.errors.ParserError as e:
            print(f"Error parsing temperature_log.csv: {e}")
            print("Trying with different parsing options...")
            # Try with different parsing options
            temp_data = pd.read_csv('temperature_log.csv', on_bad_lines='skip')
        
        # Convert timestamps to datetime
        thermal_data['timestamp'] = pd.to_datetime(thermal_data['timestamp'])
        temp_data['timestamp'] = pd.to_datetime(temp_data['timestamp'])
        
        # Ensure data is sorted by timestamp
        thermal_data = thermal_data.sort_values('timestamp')
        temp_data = temp_data.sort_values('timestamp')
        
        # Add a sequential sample number
        thermal_data['sample'] = range(len(thermal_data))
        temp_data['sample'] = range(len(temp_data))
        
        # Limit to first 100000 points
        thermal_data = thermal_data.head(100000)
        temp_data = temp_data.head(100000)
        
        print(f"Limiting analysis to first {len(thermal_data)} thermal data points and {len(temp_data)} temperature data points")
        
        # Create figures folder if it doesn't exist
        if not os.path.exists('figures'):
            os.makedirs('figures')

        print("Creating fan speed vs temperature plot...")
        # PLOT 1: Fan Speed vs Outlet Temperature Over Time
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # First axis for temperature
        color = 'tab:red'
        ax1.set_xlabel('Sample Number')
        ax1.set_ylabel('Outlet Temperature (°C)', color=color)
        ax1.plot(temp_data['sample'], temp_data['outlet_temperature'], color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Second axis for fan speed ratio
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Fan Speed Ratio', color=color)
        ax2.plot(thermal_data['sample'], thermal_data['fan_ratio'], color=color, linewidth=2, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add title and grid
        plt.title('Outlet Temperature and Fan Speed Ratio Over Time (First 100000 Points)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Add annotation highlighting oscillation if enough data points
        if len(temp_data) > 100:
            try:
                max_idx = temp_data['outlet_temperature'].idxmax()
                max_temp = temp_data.loc[max_idx, 'outlet_temperature']
                max_sample = temp_data.loc[max_idx, 'sample']
                
                ax1.annotate('Temperature Peak', xy=(max_sample, max_temp),
                            xytext=(max_sample+50, max_temp+5),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                            fontsize=12)
                
                # Find corresponding fan response
                after_peak = thermal_data[thermal_data['sample'] >= max_sample]
                if len(after_peak) > 10:
                    sample_range = after_peak.iloc[1:10]
                    if not sample_range.empty:
                        corresponding_fan = sample_range['fan_ratio'].max()
                        fan_max_idx = sample_range['fan_ratio'].idxmax()
                        fan_max_sample = thermal_data.loc[fan_max_idx, 'sample']
                        
                        ax2.annotate('Fan Response', xy=(fan_max_sample, corresponding_fan),
                                    xytext=(fan_max_sample+50, corresponding_fan-0.1),
                                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                                    fontsize=12)
            except Exception as e:
                print(f"Warning: Could not add annotations to time series plot: {e}")
        
        # Save the figure
        plt.tight_layout()
        plt.savefig('figures/temperature_fan_time_series.png', dpi=300)
        print("Fan speed vs temperature plot saved.")
        
        print("Creating phase portrait plot...")
        # PLOT 2: Phase Portrait (Fan Speed vs Temperature)
        plt.figure(figsize=(10, 8))
        
        # We need to merge the datasets to get fan speed and temperature at the same points
        # Create aligned data by matching closest timestamps
        aligned_data = []
        
        for idx, thermal_row in thermal_data.iterrows():
            thermal_time = thermal_row['timestamp']
            # Find closest temperature measurement by time
            temp_idx = (temp_data['timestamp'] - thermal_time).abs().idxmin()
            temp_row = temp_data.loc[temp_idx]
            
            aligned_data.append({
                'timestamp': thermal_time,
                'fan_ratio': thermal_row['fan_ratio'],
                'outlet_temperature': temp_row['outlet_temperature'],
                'has_gpus': thermal_row['has_gpus']
            })
        
        aligned_df = pd.DataFrame(aligned_data)
        
        # Plot data points colored by GPU presence
        gpu_data = aligned_df[aligned_df['has_gpus'] == 1]
        cpu_data = aligned_df[aligned_df['has_gpus'] == 0]
        
        if not gpu_data.empty:
            plt.scatter(gpu_data['outlet_temperature'], gpu_data['fan_ratio'], 
                       color='red', alpha=0.7, label='With GPUs', s=30)
            
            # Draw trajectory lines
            plt.plot(gpu_data['outlet_temperature'], gpu_data['fan_ratio'], 
                    'r-', alpha=0.3)
        
        if not cpu_data.empty:
            plt.scatter(cpu_data['outlet_temperature'], cpu_data['fan_ratio'], 
                       color='blue', alpha=0.7, label='CPU Only', s=30)
            
            # Draw trajectory lines
            plt.plot(cpu_data['outlet_temperature'], cpu_data['fan_ratio'], 
                    'b-', alpha=0.3)
        
        plt.xlabel('Outlet Temperature (°C)', fontsize=12)
        plt.ylabel('Fan Speed Ratio', fontsize=12)
        plt.title('Phase Portrait: Fan Speed vs Temperature (First 100000 Points)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('figures/phase_portrait.png', dpi=300)
        print("Phase portrait plot saved.")
        
        print("Creating power vs temperature plot...")
        # PLOT 3: Power vs Temperature with Fan Speed represented by color
        plt.figure(figsize=(12, 8))
        
        # Calculate total power
        thermal_data['total_power'] = thermal_data['cpu_power'] + thermal_data['gpu_power']
        
        # Create aligned data with power, temperature and fan speed
        power_temp_data = []
        
        for idx, thermal_row in thermal_data.iterrows():
            thermal_time = thermal_row['timestamp']
            # Find closest temperature measurement by time
            temp_idx = (temp_data['timestamp'] - thermal_time).abs().idxmin()
            temp_row = temp_data.loc[temp_idx]
            
            power_temp_data.append({
                'total_power': thermal_row['total_power'],
                'outlet_temperature': temp_row['outlet_temperature'],
                'fan_ratio': thermal_row['fan_ratio']
            })
        
        power_temp_df = pd.DataFrame(power_temp_data)
        
        # Create scatter plot with fan speed as color
        scatter = plt.scatter(power_temp_df['total_power'], power_temp_df['outlet_temperature'], 
                             c=power_temp_df['fan_ratio'], cmap='viridis', alpha=0.7, s=50)
        
        # Add color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Fan Speed Ratio', fontsize=12)
        
        plt.xlabel('Total Power (W)', fontsize=12)
        plt.ylabel('Outlet Temperature (°C)', fontsize=12)
        plt.title('Power vs Temperature Colored by Fan Speed (First 100000 Points)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add annotations for significant regions
        if len(power_temp_df) > 0:
            try:
                max_power_idx = power_temp_df['total_power'].idxmax()
                max_temp_idx = power_temp_df['outlet_temperature'].idxmax()
                
                max_power = power_temp_df.loc[max_power_idx, 'total_power']
                max_power_temp = power_temp_df.loc[max_power_idx, 'outlet_temperature']
                
                max_temp = power_temp_df.loc[max_temp_idx, 'outlet_temperature']
                max_temp_power = power_temp_df.loc[max_temp_idx, 'total_power']
                
                plt.annotate('Highest Power Point', 
                            xy=(max_power, max_power_temp),
                            xytext=(max_power-400, max_power_temp+5),
                            arrowprops=dict(facecolor='black', shrink=0.05),
                            fontsize=10)
                
                plt.annotate('Highest Temperature Point', 
                            xy=(max_temp_power, max_temp),
                            xytext=(max_temp_power-400, max_temp-5),
                            arrowprops=dict(facecolor='black', shrink=0.05),
                            fontsize=10)
            except Exception as e:
                print(f"Warning: Could not add annotations to power-temperature plot: {e}")
        
        plt.tight_layout()
        plt.savefig('figures/power_vs_temperature.png', dpi=300)
        print("Power vs temperature plot saved.")
        
        print("Creating temperature delta plot...")
        # PLOT 4: Delta Temperature vs Time (showing oscillation clearly)
        plt.figure(figsize=(12, 6))
        
        plt.plot(temp_data['sample'], temp_data['delta_temperature'], 'g-', linewidth=2)
        plt.xlabel('Sample Number', fontsize=12)
        plt.ylabel('Temperature Delta (°C)', fontsize=12)
        plt.title('Temperature Delta (Outlet - Inlet) Over Time (First 100000 Points)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add horizontal line at typical CPU-only delta and GPU+CPU delta
        if 'has_gpus' in temp_data.columns:
            cpu_only_data = temp_data[temp_data['has_gpus'] == 0]
            gpu_data = temp_data[temp_data['has_gpus'] == 1]
            
            if not cpu_only_data.empty:
                cpu_avg_delta = cpu_only_data['delta_temperature'].mean()
                plt.axhline(y=cpu_avg_delta, color='b', linestyle='--', 
                           label=f'Avg CPU-only Delta: {cpu_avg_delta:.2f}°C')
            
            if not gpu_data.empty:
                gpu_avg_delta = gpu_data['delta_temperature'].mean()
                plt.axhline(y=gpu_avg_delta, color='r', linestyle='--', 
                           label=f'Avg GPU+CPU Delta: {gpu_avg_delta:.2f}°C')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig('figures/delta_temperature.png', dpi=300)
        print("Temperature delta plot saved.")
        
        print(f"All plots saved to the 'figures' directory.")
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    plot_thermal_control_dynamics()