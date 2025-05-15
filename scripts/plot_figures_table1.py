
# ## 1. Setup and Imports

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import os
import argparse
import logging

# Configure plotting style (optional)
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100 # Adjust resolution if needed

# Configure logging (optional, less critical in notebooks but good practice)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# ## 2. Configuration

# **IMPORTANT:** Set the `results_dir` variable below to the directory containing your evaluation CSV files (e.g., `logs/table1_eval_20250509_103000`).

# In[2]:


# --- User Configuration ---
RESULTS_DIR = "logs/table1_eval_20250512_175950" # Path to the directory containing results
RAW_CSV_SUFFIX = "_raw.csv"
SUMMARY_CSV_SUFFIX = ".csv" # Suffix for the formatted summary CSV

# Output directory for plots (defaults to RESULTS_DIR)
PLOT_OUTPUT_DIR = RESULTS_DIR

# Ensure plot output directory exists
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# Evaluation duration (used in titles, might need adjustment if different runs used different durations)
EVALUATION_DURATION_DAYS = 7 # Default used in generate script


# In[3]:

# --- Load Data ---

# Extract timestamp from the directory name (assuming format logs/TYPE_TIMESTAMP)
try:
    timestamp_str = os.path.basename(RESULTS_DIR.rstrip('/\\')) # Get last part of path
    # Attempt to find timestamp format YYYYMMDD_HHMMSS
    import re
    match = re.search(r'(\d{8}_\d{6})', timestamp_str)
    if match:
        timestamp_suffix = "_" + match.group(1)
        logging.info(f"Extracted timestamp suffix: {timestamp_suffix}")
    else:
        # Fallback if timestamp not found in directory name
        logging.warning(f"Could not automatically extract timestamp from RESULTS_DIR: {RESULTS_DIR}. Assuming no timestamp in filenames.")
        timestamp_suffix = "" # Assume no timestamp in filename if pattern not found
except Exception as e:
    logging.warning(f"Error extracting timestamp from RESULTS_DIR: {e}. Assuming no timestamp in filenames.")
    timestamp_suffix = ""


# Construct filenames using the extracted timestamp (if any)
raw_results_filename = f"results_table1{timestamp_suffix}{RAW_CSV_SUFFIX}"
summary_results_filename = f"results_table1{timestamp_suffix}{SUMMARY_CSV_SUFFIX}"

raw_results_path = os.path.join(RESULTS_DIR, raw_results_filename)
summary_results_path = os.path.join(RESULTS_DIR, summary_results_filename)

# --- Load Raw Data ---
try:
    df_raw_all = pd.read_csv(raw_results_path)
    logging.info(f"Loaded raw results ({len(df_raw_all)} rows) from: {raw_results_path}")
    # Display first few rows and info
    print("\nRaw Data Head:")
    print(df_raw_all.head())
    print("\nRaw Data Info:")
    df_raw_all.info()
except FileNotFoundError:
    logging.error(f"Raw results file not found: {raw_results_path}")
    df_raw_all = None
except Exception as e:
     logging.error(f"Error loading raw results CSV: {e}")
     df_raw_all = None

# --- Load Formatted Summary Data ---
try:
    summary_df_formatted = pd.read_csv(summary_results_path)
    logging.info(f"Loaded formatted summary results from: {summary_results_path}")
    print("\nFormatted Summary Data:")
    print(summary_df_formatted.to_string(index=False))
except FileNotFoundError:
    logging.error(f"Formatted summary results file not found: {summary_results_path}")
    summary_df_formatted = None
except Exception as e:
     logging.error(f"Error loading formatted summary results CSV: {e}")
     summary_df_formatted = None

# --- Recalculate Numerical Summary ---
summary_df_recalc = None
if df_raw_all is not None:
    try:
        cols_to_agg = [col for col in df_raw_all.columns if col not in ['Controller', 'Seed', 'timestep', 'datacenter']] # Adjust if needed
        if not cols_to_agg: # Check if list is empty
             logging.warning("No columns found for aggregation in raw data.")
        else:
             agg_dict_plot = {col: ['mean', 'std'] for col in cols_to_agg}
             summary_df_recalc = df_raw_all.groupby("Controller").agg(agg_dict_plot)
             summary_df_recalc.columns = ["_".join(col).strip() for col in summary_df_recalc.columns.values]
             summary_df_recalc = summary_df_recalc.reset_index()
             logging.info("Recalculated numerical summary (mean/std) from raw data.")
             print("\nRecalculated Summary (Mean/Std):")
             print(summary_df_recalc.head())
    except Exception as e:
        logging.error(f"Could not recalculate summary statistics from raw data: {e}")
        summary_df_recalc = None

summary_for_plots = summary_df_recalc if summary_df_recalc is not None else summary_df_formatted

# ## 4. Plotting Functions

# In[4]:


# --- Paste the plotting functions here ---

def plot_bar_chart_comparison(summary_df, metrics_to_plot, title, filename):
    """Generates a grouped bar chart comparing controllers across selected metrics."""
    if summary_df is None: logging.error("Summary DataFrame not available for bar chart."); return
    try:
        n_controllers = len(summary_df['Controller'])
        plot_metrics_found = []
        bars_data = []

        # Check for _mean and _std columns
        has_std_dev = any('_std' in col for col in summary_df.columns)

        for metric in metrics_to_plot:
            metric_mean_col = f"{metric}_mean"
            metric_std_col = f"{metric}_std"

            if metric_mean_col not in summary_df.columns:
                logging.warning(f"Metric '{metric}' (column '{metric_mean_col}') not found for bar chart.")
                continue

            try:
                means = summary_df[metric_mean_col].astype(float)
                if has_std_dev and metric_std_col in summary_df.columns:
                    std_devs = summary_df[metric_std_col].astype(float).fillna(0)
                else:
                    std_devs = None # No error bars if std not available
                plot_metrics_found.append(metric)
                bars_data.append({'means': means, 'stds': std_devs})
            except Exception as e:
                 logging.warning(f"Could not process metric '{metric}' for bar chart: {e}")

        if not plot_metrics_found:
            logging.error("No valid metrics found to plot in bar chart.")
            return

        n_metrics = len(plot_metrics_found)
        bar_width = 0.8 / n_metrics
        index = np.arange(n_controllers)
        fig, ax = plt.subplots(figsize=(max(10, n_controllers * 1.2), 7)) # Adjust width dynamically

        for i, metric_name in enumerate(plot_metrics_found):
            data = bars_data[i]
            label = metric_name.replace('(', '\n(') # Wrap long labels
            ax.bar(index + i * bar_width, data['means'], bar_width, yerr=data['stds'], label=label, capsize=4)

        ax.set_xlabel('Controller', fontweight='bold')
        ax.set_ylabel('Metric Value', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xticks(index + bar_width * (n_metrics - 1) / 2)
        ax.set_xticklabels(summary_df['Controller'], rotation=45, ha="right")
        ax.legend(title="Metrics", bbox_to_anchor=(1.04, 1), loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(filename, bbox_inches='tight')
        # plt.close(fig)
        logging.info(f"Bar chart saved to {filename}")
    except Exception as e: logging.error(f"Failed to generate bar chart {filename}: {e}")

#%%

def plot_spider_chart(summary_df, metrics_to_normalize, title, filename):
    """Generates a spider (radar) chart comparing controllers."""
    if summary_df is None: logging.error("Summary DataFrame not available for spider chart."); return
    try:
        controllers = summary_df['Controller'].tolist()
        plot_metrics = [] # Metrics actually found and plotted
        normalized_data = pd.DataFrame(index=controllers)

        # Define normalization directions
        metrics_lower_is_better = ["Total Energy Cost (USD)", "Total Energy (kWh)", "Total CO2 (kg)",
                                   "Total SLA Violations", "SLA Violation Rate (%)", "Total Transmission Cost (USD)",
                                   "Total Water Usage (L)", "Average PUE", "Total Tasks Deferred"]
        metrics_higher_is_better = ["Avg CPU Util (%)", "Avg GPU Util (%)", "Avg MEM Util (%)"]

        for metric_base_name in metrics_to_normalize:
            mean_col = f"{metric_base_name}_mean"
            # Handle case where input summary_df might be the formatted one
            if mean_col not in summary_df.columns:
                 if metric_base_name in summary_df.columns: # Check if base name exists (formatted df)
                      try: # Attempt to parse mean from formatted string 'mean ± (std)'
                           values = summary_df[metric_base_name].str.split(' ').str[0].astype(float)
                           logging.warning(f"Parsing mean from formatted column '{metric_base_name}' for spider chart.")
                      except Exception:
                           logging.warning(f"Could not parse mean from formatted column '{metric_base_name}'. Skipping metric.")
                           continue
                 else:
                      logging.warning(f"Metric '{metric_base_name}' (column '{mean_col}') not found for spider chart. Skipping.")
                      continue
            else:
                 values = summary_df[mean_col].astype(float) # Use _mean column

            plot_metrics.append(metric_base_name) # Add metric name if valid values obtained
            min_val, max_val = values.min(), values.max(); val_range = max(1e-9, max_val - min_val)
            if metric_base_name in metrics_lower_is_better: norm_values = (max_val - values) / val_range # 1 is best (min value)
            elif metric_base_name in metrics_higher_is_better: norm_values = (values - min_val) / val_range # 1 is best (max value)
            else: logging.warning(f"Norm direction unknown for '{metric_base_name}'. Using 0.5."); norm_values = np.ones(len(values)) * 0.5
            normalized_data[metric_base_name] = norm_values.values

        n_vars = len(plot_metrics)
        if n_vars < 3: logging.warning("Spider chart needs at least 3 valid metrics."); return

        angles = [n / float(n_vars) * 2 * pi for n in range(n_vars)]; angles += angles[:1]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        # Use wrapped labels for clarity
        labels = [name.replace("(", "\n(").replace(")", ")\n").replace("%", "\%") for name in plot_metrics]
        plt.xticks(angles[:-1], labels)
        ax.set_yticks(np.arange(0, 1.1, 0.2)); ax.set_ylim(0, 1)

        for controller in controllers:
            if controller in normalized_data.index:
                data = normalized_data.loc[controller].values.flatten().tolist(); data += data[:1]
                ax.plot(angles, data, linewidth=1.5, linestyle='solid', label=controller, marker='o', markersize=4)
            else:
                logging.warning(f"Controller '{controller}' not found in normalized data for spider chart.")

        plt.legend(loc='lower right', bbox_to_anchor=(1.3, -0.1)) # Adjust legend position
        plt.title(title, size=16, y=1.1); ax.grid(True)
        plt.savefig(filename, bbox_inches='tight')
        # plt.close(fig)
        logging.info(f"Spider chart saved to {filename}")
    except Exception as e: logging.error(f"Failed to generate spider chart {filename}: {e}", exc_info=True)


def plot_timeseries(df_raw, y_metric, title, ylabel, filename, hue_col="Controller", aggregate_dc=False):
    """Generates a time-series line plot. Aggregates across DCs if needed."""
    if df_raw is None: logging.error("Raw DataFrame not available for time-series plot."); return
    if y_metric not in df_raw.columns: logging.warning(f"Column '{y_metric}' not found for time-series plot {filename}. Skipping."); return

    try:
        plt.figure(figsize=(14, 7))
        plot_df = df_raw

        # If raw data is per-DC, aggregate sum or mean across DCs for a global view
        if aggregate_dc and 'datacenter' in df_raw.columns:
            # Sum metrics like cost, energy, carbon, sla counts
            if y_metric in ["energy_cost", "energy_kwh", "carbon_kg", "sla_met", "sla_violated", "total_water_usage_L", "Total Tasks Deferred"]: # Add others if needed
                 plot_df = df_raw.groupby(['Controller', 'Seed', 'timestep'])[y_metric].sum().reset_index()
                 ylabel = f"Total {ylabel} (All DCs)"
            # Average metrics like util, price, ci, temp, setpoint
            elif y_metric in ["price_per_kwh", "ci", "weather", "cpu_util", "gpu_util", "mem_util", "hvac_setpoint", "running_tasks", "pending_tasks"]:
                 plot_df = df_raw.groupby(['Controller', 'Seed', 'timestep'])[y_metric].mean().reset_index()
                 ylabel = f"Avg {ylabel} (All DCs)"
            else:
                 logging.warning(f"Unknown aggregation method for metric '{y_metric}' when aggregate_dc=True.")
                 # Default to mean aggregation or skip? Let's default to mean
                 plot_df = df_raw.groupby(['Controller', 'Seed', 'timestep'])[y_metric].mean().reset_index()
                 ylabel = f"Avg {ylabel} (All DCs)"


        # Plotting average across seeds, std dev as shaded region
        sns.lineplot(data=plot_df, x="timestep", y=y_metric, hue=hue_col, errorbar='sd') # Use errorbar='sd'

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Timestep (15 min intervals)")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title=hue_col, bbox_to_anchor=(1.04, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        logging.info(f"Time-series plot saved to {filename}")

    except Exception as e: logging.error(f"Failed to generate time-series plot {filename}: {e}", exc_info=True)


def plot_distributions(df_raw, y_metric, title, ylabel, filename, hue_col="Controller", plot_type='box'):
    """Generates box or violin plots for metric distributions across timesteps."""
    if df_raw is None: logging.error("Raw DataFrame not available for distribution plot."); return
    if y_metric not in df_raw.columns: logging.warning(f"Column '{y_metric}' not found for distribution plot {filename}. Skipping."); return

    try:
        plt.figure(figsize=(max(8, len(df_raw[hue_col].unique()) * 1.0), 6)) # Dynamic width
        if plot_type == 'box':
            sns.boxplot(data=df_raw, y=y_metric, x=hue_col, showfliers=False) # Hide outliers for clarity
        elif plot_type == 'violin':
            sns.violinplot(data=df_raw, y=y_metric, x=hue_col, cut=0) # Cut prevents tails going beyond data range
        else: logging.error(f"Unknown plot_type '{plot_type}' for distributions."); return

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Controller")
        plt.ylabel(ylabel)
        plt.xticks(rotation=30, ha='right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        logging.info(f"Distribution plot saved to {filename}")
    except Exception as e: logging.error(f"Failed to generate distribution plot {filename}: {e}", exc_info=True)


def plot_bar_chart_comparison_subplots(summary_df, metrics_to_plot, fig_title, filename):
    """
    Generates a figure with subplots, each showing a bar chart for one metric
    comparing all controllers. Includes error bars for standard deviation.
    """
    if summary_df is None:
        logging.error("Summary DataFrame not available for bar chart subplots.")
        return
    if not metrics_to_plot:
        logging.warning("No metrics specified for bar chart subplots.")
        return

    # --- Determine Subplot Layout ---
    n_metrics = len(metrics_to_plot)
    # Arrange in a single column for clarity, especially with many controllers
    nrows = n_metrics
    ncols = 1
    # Adjust figure size based on number of metrics/controllers
    fig_height = max(5, 3 * nrows) # Min height 5, scale with metrics
    fig_width = max(8, len(summary_df['Controller']) * 0.6 + 2) # Scale width with controllers
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharex=True)

    # Ensure axes is always iterable (even if n_metrics is 1)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() # Flatten for easy iteration

    controllers = summary_df['Controller'].tolist()
    x_pos = np.arange(len(controllers))

    has_std_dev = any('_std' in col for col in summary_df.columns)

    # --- Iterate through metrics and create subplots ---
    valid_metrics_plotted = 0
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        metric_mean_col = f"{metric}_mean"
        metric_std_col = f"{metric}_std"

        if metric_mean_col not in summary_df.columns:
            logging.warning(f"Metric '{metric}' (column '{metric_mean_col}') not found for subplot. Skipping.")
            ax.set_title(f"{metric}\n(Data Missing)", fontsize=10, style='italic')
            ax.set_yticks([]) # Hide y-axis for missing data
            continue

        try:
            means = summary_df[metric_mean_col].astype(float)
            if has_std_dev and metric_std_col in summary_df.columns:
                std_devs = summary_df[metric_std_col].astype(float).fillna(0)
            else:
                std_devs = None # No error bars

            # Plot bars with error bars
            bars = ax.bar(x_pos, means, yerr=std_devs, capsize=4, alpha=0.8, edgecolor='black')

            ax.set_ylabel(metric.replace('(', '\n(').replace('%','\%'), fontsize=9) # Metric name as y-label
            ax.set_title(metric, fontsize=11, fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            # Add value labels on top of bars (optional)
            # ax.bar_label(bars, fmt='{:,.2f}', fontsize=8, padding=3)
            valid_metrics_plotted += 1

        except Exception as e:
             logging.warning(f"Could not plot subplot for metric '{metric}': {e}")
             ax.set_title(f"{metric}\n(Plotting Error)", fontsize=10, style='italic')
             ax.set_yticks([])


    if valid_metrics_plotted == 0:
         logging.error("No valid metrics could be plotted.")
        #  plt.close(fig)
         return

    # --- Final Figure Adjustments ---
    # Set shared x-axis labels and rotation only once
    plt.xticks(x_pos, controllers, rotation=45, ha="right")
    fig.suptitle(fig_title, fontsize=16, fontweight='bold', y=0.99) # Adjust y if needed
    fig.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust margins (top for suptitle, bottom for labels)

    try:
        plt.savefig(filename, bbox_inches='tight')
        # plt.close(fig)
        logging.info(f"Bar chart subplots saved to {filename}")
    except Exception as e:
        logging.error(f"Failed to save bar chart subplots {filename}: {e}")
# ## 5. Generate Plots

# In[5]:

NUM_SEEDS = 10 # Number of random seeds to run for each controller

# --- Generate Standard Plots ---
if summary_for_plots is not None:
    timestamp_str = os.path.basename(RESULTS_DIR).split('_')[-1]
    logging.info("Generating standard comparison plots...")

    # 1. Bar Chart (Aggregated KPIs)
    bar_metrics_plot = [
        "Total Energy Cost (USD)", "Total CO2 (kg)", "SLA Violation Rate (%)",
        "Total Transmission Cost (USD)", "Total Water Usage (m3)", "Average PUE",
        "Total Tasks Deferred"
    ]
    plot_bar_chart_comparison(
        summary_df_recalc, # Use recalculated numerical summary
        bar_metrics_plot,
        f"Controller Comparison (Mean ± Std Dev, {EVALUATION_DURATION_DAYS} days, {NUM_SEEDS} seeds)",
        os.path.join(PLOT_OUTPUT_DIR, f"plot_bar_comparison_{timestamp_str}.png")
    )
    
    # --- Generate Bar Chart Subplots ---
    bar_metrics_subplot = [
            "Total Energy Cost (USD)",
            "Total CO2 (kg)",
            "SLA Violation Rate (%)",
            "Total Transmission Cost (USD)",
            "Total Water Usage (m3)",
            "Average PUE",
            "Total Tasks Deferred",
            "Avg CPU Util (%)",
            "Avg GPU Util (%)",
        ]
    plot_bar_chart_comparison_subplots( # <<<--- CALL NEW FUNCTION
        summary_df_recalc, # Pass the df with numerical _mean and _std
        bar_metrics_subplot,
        f"Controller Metric Comparison (Mean ± Std Dev, {EVALUATION_DURATION_DAYS} days, {NUM_SEEDS} seeds)",
        os.path.join(PLOT_OUTPUT_DIR, f"plot_bar_subplots_{timestamp_str}.png")
    )

    # 2. Spider Chart (Normalized Trade-offs)
    spider_metrics_plot = [
        "Total Energy Cost (USD)", "Total CO2 (kg)", "SLA Violation Rate (%)",
        "Avg CPU Util (%)", "Total Tasks Deferred", "Total Transmission Cost (USD)",
        "Total Water Usage (m3)", "Average PUE"
    ]
    plot_spider_chart(
        summary_df_recalc, # Use recalculated numerical summary with _mean columns
        spider_metrics_plot,
        f"Controller Trade-offs (Normalized, Best=1, {EVALUATION_DURATION_DAYS} days)",
        os.path.join(PLOT_OUTPUT_DIR, f"plot_spider_tradeoffs_{timestamp_str}.png")
    )
else:
    logging.warning("Summary DataFrame for plotting is not available.")

if df_raw_all is not None:
    # 3. Time Series Plots (Using Raw Data - Aggregating across DCs)
    logging.info("Generating time-series plots (aggregated across DCs)...")
    plot_timeseries(
        df_raw_all, "energy_kwh", "Total Energy Consumption over Time (All DCs)", "Total Energy (kWh) / 15min",
        os.path.join(PLOT_OUTPUT_DIR, f"plot_ts_total_energy_{timestamp_str}.png"),
        hue_col="Controller", aggregate_dc=True
    )
    plot_timeseries(
        df_raw_all, "carbon_kg", "Total Carbon Emissions over Time (All DCs)", "Total Carbon (kg CO2) / 15min",
        os.path.join(PLOT_OUTPUT_DIR, f"plot_ts_total_carbon_{timestamp_str}.png"),
        hue_col="Controller", aggregate_dc=True
    )
    # Add time-series plot for total pending tasks across all DCs
    plot_timeseries(
        df_raw_all, "pending_tasks", "Total Pending Tasks over Time (All DCs)", "Total Tasks in Queue",
        os.path.join(PLOT_OUTPUT_DIR, f"plot_ts_total_pending_{timestamp_str}.png"),
        hue_col="Controller", aggregate_dc=True
    )
    # Add time-series plot for average CPU utilization across all DCs
    plot_timeseries(
        df_raw_all, "cpu_util", "Average CPU Utilization over Time (All DCs)", "Avg CPU Util (%)",
        os.path.join(PLOT_OUTPUT_DIR, f"plot_ts_avg_cpu_util_{timestamp_str}.png"),
        hue_col="Controller", aggregate_dc=True
    )


    # 4. Distribution Plots (Using Raw Data - showing variability over time)
    logging.info("Generating distribution plots...")
    plot_distributions(
        df_raw_all, "energy_cost", "Distribution of Timestep Energy Cost per Controller",
        "Energy Cost (USD) / 15min",
        os.path.join(PLOT_OUTPUT_DIR, f"plot_dist_energy_cost_{timestamp_str}.png"), plot_type='violin'
    )
    plot_distributions(
        df_raw_all, "hvac_setpoint", "Distribution of HVAC Setpoints per Controller (if available)",
        "Setpoint (°C)",
        os.path.join(PLOT_OUTPUT_DIR, f"plot_dist_setpoint_{timestamp_str}.png"), plot_type='box'
    )
    plot_distributions(
        df_raw_all, "ci", "Distribution of Carbon Intensity Experienced",
        "Carbon Intensity (gCO2/kWh)",
        os.path.join(PLOT_OUTPUT_DIR, f"plot_dist_carbon_intensity_{timestamp_str}.png"), plot_type='box'
    )

else:
     logging.warning("Raw DataFrame not available, skipping time-series and distribution plots.")

logging.info("--- Plotting Complete ---")
# %%
import seaborn as sns

# Configure plotting style (optional)
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100 # Adjust resolution if needed

# 2. Spider Chart (Normalized Trade-offs)
metrics_to_normalize = [
    "Total Energy (kWh)", "Total CO2 (kg)", "Total Water Usage (m3)",  
    "SLA Violation Rate (%)", "Avg CPU Util (%)", "Avg GPU Util (%)", "Avg MEM Util (%)",
    "Total Tasks Deferred",  "Total Transmission Cost (USD)", "Total Energy Cost (USD)"
]

title = f"Controller Trade-offs (Normalized, Best=1, {EVALUATION_DURATION_DAYS} days)",
filename = os.path.join(PLOT_OUTPUT_DIR, f"plot_spider_tradeoffs_{timestamp_str}.png")

"""Generates a spider (radar) chart comparing controllers."""

controllers = summary_df_recalc['Controller'].tolist()
plot_metrics = [] # Metrics actually found and plotted
normalized_data = pd.DataFrame(index=controllers)

# Define normalization directions
metrics_lower_is_better = ["Total Energy Cost (USD)", "Total Energy (kWh)", "Total CO2 (kg)",
                            "Total SLA Violations", "SLA Violation Rate (%)", "Total Transmission Cost (USD)",
                            "Total Water Usage (m3)", "Average PUE", "Total Tasks Deferred", ]
metrics_higher_is_better = ["Avg CPU Util (%)", "Avg GPU Util (%)", "Avg MEM Util (%)"]

for metric_base_name in metrics_to_normalize:
    mean_col = f"{metric_base_name}_mean"
    # Handle case where input summary_df_recalc might be the formatted one
    if mean_col not in summary_df_recalc.columns:
            if metric_base_name in summary_df_recalc.columns: # Check if base name exists (formatted df)
                try: # Attempt to parse mean from formatted string 'mean ± (std)'
                    values = summary_df_recalc[metric_base_name].str.split(' ').str[0].astype(float)
                    logging.warning(f"Parsing mean from formatted column '{metric_base_name}' for spider chart.")
                except Exception:
                    logging.warning(f"Could not parse mean from formatted column '{metric_base_name}'. Skipping metric.")
                    continue
            else:
                logging.warning(f"Metric '{metric_base_name}' (column '{mean_col}') not found for spider chart. Skipping.")
                continue
    else:
            values = summary_df_recalc[mean_col].astype(float) # Use _mean column

    plot_metrics.append(metric_base_name) # Add metric name if valid values obtained
    min_val, max_val = values.min(), values.max()
    val_range = max(1e-9, max_val - min_val)
    
    if metric_base_name in metrics_lower_is_better:
        norm_values = (max_val - values) / val_range # 1 is best (min value)
        
    elif metric_base_name in metrics_higher_is_better:
        norm_values = (values - min_val) / val_range # 1 is best (max value)
        
    else:
        logging.warning(f"Norm direction unknown for '{metric_base_name}'. Using 0.5.")
        norm_values = np.ones(len(values)) * 0.5
        
    normalized_data[metric_base_name] = norm_values.values

n_vars = len(plot_metrics)

angles = [n / float(n_vars) * 2 * pi for n in range(n_vars)]; angles += angles[:1]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
# Use wrapped labels for clarity
labels = [name.replace("(", "\n(").replace(")", ")\n").replace("%", "\%") for name in plot_metrics]
plt.xticks(angles[:-1], labels)
ax.set_yticks(np.arange(0, 1.1, 0.2)); ax.set_ylim(0, 1)

for controller in controllers:
    if controller in normalized_data.index:
        data = normalized_data.loc[controller].values.flatten().tolist(); data += data[:1]
        ax.plot(angles, data, linewidth=1.5, linestyle='solid', label=controller, marker='o', markersize=4)
    else:
        logging.warning(f"Controller '{controller}' not found in normalized data for spider chart.")

plt.legend(loc='lower right', bbox_to_anchor=(1.3, -0.1)) # Adjust legend position
# plt.title(title, size=16, y=1.1)
ax.grid(True)
plt.savefig(filename, bbox_inches='tight')
# plt.close(fig)
logging.info(f"Spider chart saved to {filename}")

#%%
import matplotlib
import textwrap

matplotlib.rcParams.update({
    'font.size': 9,             # Base font size - adjust as needed
    'axes.labelsize': 9,        # Axis labels
    'axes.titlesize': 10,       # Subplot titles
    'xtick.labelsize': 8,       # X-axis tick labels
    'ytick.labelsize': 7,       # Y-axis tick labels
    'legend.fontsize': 8,       # Legend font size
    'figure.titlesize': 12,     # Main figure title (suptitle)
    'font.family': 'sans-serif',     # Or 'sans-serif' - try matching LaTeX font
    # Use Times New Roman if available and desired (often standard for papers)
    # 'font.serif': 'Times New Roman',
    # 'mathtext.fontset': 'stix', # Use STIX fonts for math if using Times
    'figure.dpi': 300,          # Good resolution for PNG fallback
    'savefig.dpi': 300,
    'savefig.format': 'pdf',    # <<<--- PREFERRED FORMAT FOR LATEX
    'savefig.bbox': 'tight',    # Crop whitespace
    'axes.grid': True,          # Keep grid
    'grid.linestyle': '--',
    'grid.alpha': 0.6
})
timestamp_str = os.path.basename(RESULTS_DIR).split('_')[-1] # Assuming timestamp is last part
logging.info("Generating standard comparison plots...")

# --- Generate Subplot Spider Charts ---
metrics_to_normalize = [
    "Total Energy (kWh)", "Total CO2 (kg)", "Total Water Usage (m3)",  
    "SLA Violation Rate (%)", "Avg CPU Util (%)", "Avg GPU Util (%)", "Avg MEM Util (%)",
    "Total Tasks Deferred",  "Total Transmission Cost (USD)", "Total Energy Cost (USD)"
]
# Define controller groups
controllers_rl_ablation = ['RBC (Local Only)', 'RL (Geo Only)', 'RL (Time Only)', 'RL (Geo+Time)']
controllers_rbc_compare = ['RBC (Local Only)', 'RBC (Lowest Carbon)', 'RBC (Lowest Price)', 'RBC (Most Available)', 'RBC (Round Robin)']

title = f"Controller Trade-offs (Normalized, Best=Outer Edge, {EVALUATION_DURATION_DAYS} days)", # Main Title
filename = os.path.join(PLOT_OUTPUT_DIR, f"plot_spider_subplots_{timestamp_str}.png") # Filename


# --- Data Preparation ---
def normalize_metrics(df_subset, metrics, lower_is_better, higher_is_better):
    normalized_subset = pd.DataFrame(index=df_subset['Controller'])
    valid_metrics = []
    for metric_base_name in metrics:
        mean_col = f"{metric_base_name}_mean"
        if mean_col not in df_subset.columns:
            logging.warning(f"Metric '{metric_base_name}' (col '{mean_col}') not found. Skipping.")
            continue

        values = df_subset[mean_col].astype(float)
        min_val, max_val = values.min(), values.max()
        val_range = max(1e-9, max_val - min_val)

        if metric_base_name in lower_is_better: norm_values = (max_val - values) / val_range
        elif metric_base_name in higher_is_better: norm_values = (values - min_val) / val_range
        else: norm_values = np.ones(len(values)) * 0.5; logging.warning(f"Norm direction unknown for '{metric_base_name}'.")

        normalized_subset[metric_base_name] = norm_values.values
        valid_metrics.append(metric_base_name)
    return normalized_subset, valid_metrics

# Define normalization directions (copied from previous script)
metrics_lower_is_better = ["Total Energy Cost (USD)", "Total Energy (kWh)", "Total CO2 (kg)",
                            "Total SLA Violations", "SLA Violation Rate (%)", "Total Transmission Cost (USD)",
                            "Total Water Usage (m3)", "Average PUE", "Total Tasks Deferred"]
metrics_higher_is_better = ["Avg CPU Util (%)", "Avg GPU Util (%)", "Avg MEM Util (%)"]

# Filter and normalize data for each subplot
df_left = summary_df_recalc[summary_df_recalc['Controller'].isin(controllers_rl_ablation)].copy()
df_right = summary_df_recalc[summary_df_recalc['Controller'].isin(controllers_rbc_compare)].copy()

norm_data_left, metrics_left = normalize_metrics(df_left, metrics_to_normalize, metrics_lower_is_better, metrics_higher_is_better)
norm_data_right, metrics_right = normalize_metrics(df_right, metrics_to_normalize, metrics_lower_is_better, metrics_higher_is_better)

# Use the union of valid metrics for consistent axes, ensure order
common_metrics = sorted(list(set(metrics_left) & set(metrics_right)), key=metrics_to_normalize.index)
n_vars = len(common_metrics)

# --- Plotting Setup ---
angles = [n / float(n_vars) * 2 * pi for n in range(n_vars)]
angles += angles[:1] # Close the loop

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), subplot_kw=dict(polar=True)) # Wide but not too tall
# fig.suptitle(title, y=0.99) # Adjust suptitle y position slightly if needed

# Use seaborn color palettes for better distinction
palette_left = sns.color_palette("colorblind", len(controllers_rl_ablation)) # Good starting point
palette_right = sns.color_palette("muted", len(controllers_rbc_compare))    # Another decent option

labels = [ "\n".join(textwrap.wrap(m, width=15)) 
                   for m in common_metrics  ]

# --- Plot Left Subplot (RL Ablation) ---
ax_left = axes[0]
ax_left.set_title("RL Agent Ablation", y=1.15,  weight='bold') # Adjust title position
ax_left.set_xticks(angles[:-1])
ax_left.set_xticklabels(labels, weight='bold') # Bold labels
ax_left.set_yticks(np.arange(0, 1.1, 0.25)) # Fewer ticks maybe? 0, 0.5, 1?
ax_left.set_yticklabels(["Worst", "0.25", "0.5", "0.75", "Best"]) # Label ticks
ax_left.set_ylim(0, 1)

for i, controller in enumerate(controllers_rl_ablation):
    if controller in norm_data_left.index:
        data = norm_data_left.loc[controller, common_metrics].values.flatten().tolist()
        data += data[:1] # Close the loop
        color = palette_left[i]
        ax_left.plot(angles, data, linewidth=1.0, linestyle='solid', label=controller, color=color, marker='o', markersize=3) # Slightly thinner lines/markers
        # --- Fill with transparency ---
        ax_left.fill(angles, data, color=color, alpha=0.15)
        # --- End Fill ---
    else: logging.warning(f"Controller '{controller}' not found for left spider plot.")

# --- Plot Right Subplot (RBC Comparison) ---
ax_right = axes[1]
ax_right.set_title("RBC Comparison", y=1.15,  weight='bold') # Adjust title position
ax_right.set_xticks(angles[:-1])
ax_right.set_xticklabels(labels, weight='bold') # Bold labels
ax_right.set_yticks(np.arange(0, 1.1, 0.25))
ax_right.set_yticklabels(["Worst", "0.25", "0.5", "0.75", "Best"]) # Label ticks
ax_right.set_ylim(0, 1)

for i, controller in enumerate(controllers_rbc_compare):
    if controller in norm_data_right.index:
        data = norm_data_right.loc[controller, common_metrics].values.flatten().tolist()
        data += data[:1]
        color = palette_right[i]
        ax_right.plot(angles, data, linewidth=1.0, linestyle='solid', label=controller, color=color, marker='o', markersize=3)
        ax_right.fill(angles, data, color=color, alpha=0.15)
    else: logging.warning(f"Controller '{controller}' not found for right spider plot.")

# --- Legend ---
# Create a single legend below the plots for two-column figures
handles_left, labels_left = ax_left.get_legend_handles_labels()
handles_right, labels_right = ax_right.get_legend_handles_labels()
fig.legend(handles_left + handles_right, labels_left + labels_right,
            loc='lower center', bbox_to_anchor=(0.5, -0.08), # Adjust position below plots
            ncol=max(len(controllers_rl_ablation), len(controllers_rbc_compare)), # Arrange in columns
            title="Controllers", title_fontsize=8, fontsize=7,)

plt.tight_layout(rect=[0, 0.08, 1, 0.95]) # Adjust rect for suptitle and legend space

try:
    # *** Save as PDF ***
    pdf_filename = filename.replace(".png", ".pdf")
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    logging.info(f"Spider subplot chart saved to {filename}")
except Exception as e:
    logging.error(f"Failed to save spider subplot chart {filename}: {e}", exc_info=True)
# %%
matplotlib.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,       # Default legend item font size
    'legend.title_fontsize': 9, # Default legend title font size
    'figure.titlesize': 12,
    'font.family': 'sans-serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.6
})

# --- Mock Data and Setup (for standalone execution) ---
# Ensure these are defined or mocked if running standalone
RESULTS_DIR = "results_20231027100000" # Example
PLOT_OUTPUT_DIR = "plots"
EVALUATION_DURATION_DAYS = 7

if not os.path.exists(PLOT_OUTPUT_DIR):
    os.makedirs(PLOT_OUTPUT_DIR)

# Mock logging
logging.basicConfig(level=logging.INFO)

# Mock summary_df_recalc
controllers_rl_ablation = ['RBC (Local Only)', 'RL (Geo Only)', 'RL (Time Only)', 'RL (Geo+Time)']
controllers_rbc_compare = ['RBC (Local Only)', 'RBC (Lowest Carbon)', 'RBC (Lowest Price)', 'RBC (Most Available)', 'RBC (Round Robin)']
all_controllers = list(set(controllers_rl_ablation + controllers_rbc_compare))

metrics_to_normalize = [
    "Total Energy (kWh)", "Total CO2 (kg)", "Total Water Usage (m3)",
    "SLA Violation Rate (%)", "Avg CPU Util (%)", "Avg GPU Util (%)", "Avg MEM Util (%)",
    "Total Tasks Deferred", "Total Transmission Cost (USD)", "Total Energy Cost (USD)"
]
data_mock = {
    'Controller': all_controllers
}

summary_df_recalc = pd.DataFrame(data_mock)
# --- End Mock Data ---

timestamp_str = os.path.basename(RESULTS_DIR).split('_')[-1]
logging.info("Generating standard comparison plots...")

# --- Generate Subplot Spider Charts ---
# metrics_to_normalize defined in mock data

# Define controller groups (already defined in mock data)

# Main Title (figure suptitle) - ensure it's a string
figure_title_text = f"Controller Trade-offs (Normalized, Best=Outer Edge, {EVALUATION_DURATION_DAYS} days)"
filename = os.path.join(PLOT_OUTPUT_DIR, f"plot_spider_subplots_{timestamp_str}.png")

metrics_lower_is_better = ["Total Energy Cost (USD)", "Total Energy (kWh)", "Total CO2 (kg)",
                            "Total SLA Violations", "SLA Violation Rate (%)", "Total Transmission Cost (USD)",
                            "Total Water Usage (m3)", "Average PUE", "Total Tasks Deferred"]
metrics_higher_is_better = ["Avg CPU Util (%)", "Avg GPU Util (%)", "Avg MEM Util (%)"]

df_left = summary_df_recalc[summary_df_recalc['Controller'].isin(controllers_rl_ablation)].copy()
df_right = summary_df_recalc[summary_df_recalc['Controller'].isin(controllers_rbc_compare)].copy()

# norm_data_left, metrics_left = normalize_metrics(df_left, metrics_to_normalize, metrics_lower_is_better, metrics_higher_is_better)
# norm_data_right, metrics_right = normalize_metrics(df_right, metrics_to_normalize, metrics_lower_is_better, metrics_higher_is_better)

common_metrics = sorted(list(set(metrics_left) & set(metrics_right)), key=metrics_to_normalize.index)
n_vars = len(common_metrics)

# --- Plotting Setup ---
angles = [n / float(n_vars) * 2 * pi for n in range(n_vars)]
angles += angles[:1]


fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.9), subplot_kw=dict(polar=True)) # Increased height for legends
# fig.suptitle(figure_title_text, y=0.98, weight='bold') # Add suptitle

palette_left = sns.color_palette("colorblind", len(controllers_rl_ablation))
palette_right = sns.color_palette("muted", len(controllers_rbc_compare))

labels = [ "\n".join(textwrap.wrap(m, width=15)) for m in common_metrics]

bold_font_props = {'weight': 'bold'} # For legend item text

# --- Plot Left Subplot (RL Ablation) ---
ax_left = axes[0]
ax_left.set_axisbelow(False)
ax_left.set_title("RL Agent Ablation", y=1.15, weight='bold') # Adjusted y slightly
ax_left.set_xticks(angles[:-1])
ax_left.set_xticklabels(labels, weight='bold')
ax_left.set_yticks(np.arange(0, 1.1, 0.25))
ax_left.set_yticklabels(["Worst", "0.25", "0.5", "0.75", "Best"])
ax_left.set_ylim(-0.1, 1.02)

for i, controller in enumerate(controllers_rl_ablation):
    if controller in norm_data_left.index:
        data = norm_data_left.loc[controller, common_metrics].values.flatten().tolist()
        data += data[:1]
        color = palette_left[i]
        ax_left.plot(angles, data, linewidth=1.0, linestyle='solid', label=controller, color=color, marker='o', markersize=3)
        ax_left.fill(angles, data, color=color, alpha=0.15)
    else: logging.warning(f"Controller '{controller}' not found for left spider plot.")

# --- Plot Right Subplot (RBC Comparison) ---
ax_right = axes[1]
ax_right.set_axisbelow(False)
ax_right.set_title("RBC Comparison", y=1.15, weight='bold') # Adjusted y slightly
ax_right.set_xticks(angles[:-1])
ax_right.set_xticklabels(labels, weight='bold')
ax_right.set_yticks(np.arange(0, 1.1, 0.25))
ax_right.set_yticklabels(["Worst", "0.25", "0.5", "0.75", "Best"])
ax_right.set_ylim(-0.1, 1.02)

for i, controller in enumerate(controllers_rbc_compare):
    if controller in norm_data_right.index:
        data = norm_data_right.loc[controller, common_metrics].values.flatten().tolist()
        data += data[:1]
        color = palette_right[i]
        ax_right.plot(angles, data, linewidth=1.0, linestyle='solid', label=controller, color=color, marker='o', markersize=3)
        ax_right.fill(angles, data, color=color, alpha=0.15)
    else: logging.warning(f"Controller '{controller}' not found for right spider plot.")

# --- Legends ---
# Legend for Left Subplot
# Adjust bbox_to_anchor: (x, y) where x is horizontal position (0.5 for center),
# y is vertical position (<0 to place below axes).
# The y-value might need tuning based on font size and number of items.
leg_left = ax_left.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, -0.43), # Fine-tune this y-value as needed
    ncol=2, # Or len(controllers_rl_ablation) if it fits well
    title="RL Controllers",
    prop=bold_font_props, # For item text
    fontsize=7, # Explicitly set item font size if needed
    title_fontsize=8 # Explicitly set title font size if needed
)
if leg_left: # Check if legend was created (it should be)
    leg_left.get_title().set_fontweight('bold') # Make legend title bold

# Legend for Right Subplot
leg_right = ax_right.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, -0.50), # Fine-tune this y-value
    ncol=2, # Or len(controllers_rbc_compare)
    title="RBC Controllers",
    prop=bold_font_props, # For item text
    fontsize=7,
    title_fontsize=8
)
if leg_right:
    leg_right.get_title().set_fontweight('bold')

# --- Layout and Save ---
# Adjust layout to make space for suptitle and legends
# rect=[left, bottom, right, top]
# We need more space at the bottom for the legends, and some at the top for suptitle.
# Option 1: plt.tight_layout with rect
# plt.tight_layout(rect=[0, 0.15, 1, 0.92]) # bottom=0.15 gives 15% space at bottom

# Option 2: fig.subplots_adjust (often gives more predictable control)
# This might be better as tight_layout can sometimes be tricky with external legends.
fig.subplots_adjust(
    left=0.05,   # Adjust as needed for left padding
    right=0.95,  # Adjust as needed for right padding
    top=0.85,    # Make space for suptitle and subplot titles (y=1.12)
    bottom=0.25  # Make space for legends (bbox_to_anchor y=-0.35)
)


try:
    pdf_filename = filename.replace(".png", ".pdf")
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    # Also save PNG if you want both
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
    logging.info(f"Spider subplot chart saved to {pdf_filename} and {filename}")
except Exception as e:
    logging.error(f"Failed to save spider subplot chart {filename}: {e}", exc_info=True)

plt.show() # Display the plot
#%%