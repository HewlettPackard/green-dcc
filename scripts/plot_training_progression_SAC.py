#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Seaborn Style for Publication ---
# You can choose a style. 'whitegrid' is good, 'ticks' with despine is also common.
# For NeurIPS, often a cleaner, less busy style is preferred.
# sns.set_theme(style="whitegrid")
sns.set_theme(style="ticks") # Cleaner, often preferred for papers
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 10


# --- Configuration (Keep as before) ---
csv_dir = "plots/training_progression"
csv_files = [
    "run-train_20250509_232500-tag-Reward_Episode.csv",
    "run-train_20250512_022856-tag-Reward_Episode.csv",
    "run-train_20250512_023025-tag-Reward_Episode.csv",
    "run-train_20250512_023211-tag-Reward_Episode.csv",
    "run-train_20250512_023315-tag-Reward_Episode.csv"
]
output_figure_path = "plots/training_progression/sac_training_progression.pdf"
max_steps_to_plot = 1_000_000
smoothing_window = 25 # Adjust for desired smoothness
# No figure title in code, will be handled by LaTeX caption
y_axis_label = f'Episode Reward' # More concise
x_axis_label = f'Training Steps'

extrapolate_to_max_steps = True

# --- Data Loading and Processing (Keep as before) ---
all_runs_data_list = []
for i, file_name in enumerate(csv_files):
    file_path = os.path.join(csv_dir, file_name)
    if not os.path.exists(file_path):
        print(f"Warning: File not found - {file_path}")
        continue
    try:
        run_df = pd.read_csv(file_path)
        if 'Step' not in run_df.columns or 'Value' not in run_df.columns:
            if 'step' in run_df.columns and 'value' in run_df.columns:
                run_df.rename(columns={'step': 'Step', 'value': 'Value'}, inplace=True)
            else: continue
        run_df = run_df.sort_values(by='Step').reset_index(drop=True)
        run_df['Value_Smoothed'] = run_df['Value'].rolling(window=smoothing_window, min_periods=1).mean()
        run_df_processed = run_df[run_df['Step'] <= max_steps_to_plot].copy()
        run_df_processed['run_id'] = f"Run_{i+1}"
        all_runs_data_list.append(run_df_processed[['Step', 'Value_Smoothed', 'run_id']])
    except Exception as e: print(f"Error processing {file_path}: {e}")

if not all_runs_data_list:
    print("Error: No data loaded."); exit()
combined_df = pd.concat(all_runs_data_list)

plot_df_final = combined_df # Default to combined if no extrapolation
if extrapolate_to_max_steps and not combined_df.empty:
    num_interp_points = 500
    master_steps = np.linspace(0, max_steps_to_plot, num_interp_points, dtype=int)
    interpolated_runs = []
    for run_id, group in combined_df.groupby('run_id'):
        group = group.sort_values(by='Step')
        if not group.empty:
            interp_values = np.interp(master_steps, group['Step'], group['Value_Smoothed'],
                                      left=group['Value_Smoothed'].iloc[0],
                                      right=group['Value_Smoothed'].iloc[-1])
            temp_df = pd.DataFrame({'Step': master_steps, 'Value_Smoothed': interp_values, 'run_id': run_id})
            interpolated_runs.append(temp_df)
    if interpolated_runs: plot_df_final = pd.concat(interpolated_runs)


# --- Plotting for NeurIPS ---
# Adjust figsize for a typical single-column width or desired size in a two-column layout
# For a single column figure (approx 3.3 inches wide), height might be ~2.5-3 inches
fig_width_inches = 3.3 # Typical NeurIPS column width
fig_height_inches = 2.5
plt.figure(figsize=(fig_width_inches, fig_height_inches))

# Plot the mean line
lineplot = sns.lineplot(
    x='Step',
    y='Value_Smoothed',
    data=plot_df_final,
    estimator='mean',
    errorbar='sd',  # Show standard deviation as shaded area
    n_boot=1, # For smoother CI if bootstrapping (not strictly needed for 'sd')
    color='C0', # Use a standard color (e.g., Seaborn's default blue)
    linewidth=1.5, # Slightly thicker mean line
    err_style="band", # Explicitly 'band' for shaded CI
    # To make CI lighter:
    # err_kws={'alpha': 0.2} # Does not work directly for ci='sd' band, control with lineplot's alpha or facecolor of collection
)

for run_id_str in plot_df_final['run_id'].unique(): # Changed variable name
    run_data = plot_df_final[plot_df_final['run_id'] == run_id_str]
    plt.plot(run_data['Step'], run_data['Value_Smoothed'],
             color='gray', 
             linewidth=0.75,
             alpha=0.3, 
             zorder=1)
    
# No title in the plot itself, use LaTeX caption
# plt.title(plot_title, fontsize=plt.rcParams['figure.titlesize'])

plt.xlabel(x_axis_label, fontsize=plt.rcParams['axes.labelsize'])
plt.ylabel(y_axis_label, fontsize=plt.rcParams['axes.labelsize'])
plt.xlim(-10000, max_steps_to_plot+10000) # Set x-axis limit

plt.xticks(fontsize=plt.rcParams['xtick.labelsize'])
plt.yticks(fontsize=plt.rcParams['ytick.labelsize'])

plt.grid(True, linestyle='--', alpha=0.5) # Lighter grid
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True) # Use math text for scientific notation
# plt.legend(
#     title='SAC',
# )
# Remove top and right spines for a cleaner look
# sns.despine()

plt.tight_layout(pad=0.5) # Adjust padding
os.makedirs(os.path.dirname(output_figure_path), exist_ok=True)
plt.savefig(output_figure_path, bbox_inches='tight', dpi=300) # Save with good DPI
print(f"Plot saved to {output_figure_path}")
plt.show()
# %%
