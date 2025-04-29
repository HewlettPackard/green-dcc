import matplotlib.pyplot as plt
import pandas as pd

def plot_system_resources(csv_path="system_resources.csv"):
    """
    Simple function to plot CPU, memory, and GPU load from a CSV file.
    No timestamp is used - data points are plotted sequentially.
    
    Args:
        csv_path (str): Path to the CSV file
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    df = df.head(1000)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot CPU, Memory, and GPU loads - using index as x-axis
    plt.plot(df['cpu_load'], label='CPU Load (%)', color='blue')
    plt.plot(df['mem_load'], label='Memory Load (%)', color='green')
    plt.plot(df['gpu_load'], label='GPU Load (%)', color='red')
    
    # Add labels and title
    plt.xlabel('Sample Number')
    plt.ylabel('Load (%)')
    plt.title('System Resource Usage')
    plt.legend()
    plt.grid(True)
    
    # Set y-axis range from 0 to 100 for percentage
    plt.ylim(0, 100)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage:
plot_system_resources("system_resources_gpu.csv")