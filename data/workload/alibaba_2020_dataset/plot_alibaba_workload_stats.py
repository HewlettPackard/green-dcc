#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Set global style
sns.set_theme(style="whitegrid")
matplotlib.rcParams.update({"font.size": 16})

df = pd.read_pickle("result_df_full_year_2020.pkl")

'''
"job_name",           # [0]
"start_time",         # [1]
"end_time",           # [2]
"start_dt",           # [3]
"duration_min",       # [4]
"cpu_usage",          # [5] Need to divide by 100 to obtain CPU usage
"gpu_wrk_util",       # [6] Need to divide by 100 to obtain GPU usage
"avg_mem",            # [7]
"avg_gpu_wrk_mem",    # [8]
"bandwidth_gb",       # [9]
"weekday_name",       # [10]
"weekday_num"         # [11]
'''

durations = []
for tasks in df["tasks_matrix"]:
    for task in tasks:
        durations.append(float(task[4]))  # assuming index 4 is duration_min

plt.figure(figsize=(8, 4))
sns.histplot(durations, bins=50, color="tab:blue", edgecolor="black", alpha=0.7)
plt.xlabel("Task Duration (minutes)")
plt.ylabel("Count")
plt.title("Distribution of Task Durations")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
# plt.savefig("../../../assets/figures/alibaba_task_durations.svg", format="svg")
plt.show()


#%% 
# Extract resource usage
cpu_usage = [float(task[5]) / 100.0 for tasks in df["tasks_matrix"] for task in tasks]
gpu_usage = [float(task[6]) / 100.0 for tasks in df["tasks_matrix"] for task in tasks]
mem_usage = [float(task[7]) for tasks in df["tasks_matrix"] for task in tasks]
bw_usage = [float(task[9]) for tasks in df["tasks_matrix"] for task in tasks]

fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
sns.histplot(cpu_usage, bins=30, ax=axes[0], color='tab:orange', edgecolor="black", alpha=0.7)
sns.histplot(gpu_usage, bins=30, ax=axes[1], color='tab:red', edgecolor="black", alpha=0.7)
sns.histplot(mem_usage, bins=30, ax=axes[2], color='tab:green', edgecolor="black", alpha=0.7)
sns.histplot(bw_usage, bins=30, ax=axes[3], color='tab:purple', edgecolor="black", alpha=0.7)

axes[0].set_title("CPU Usage (cores)")
axes[1].set_title("GPU Usage (# GPUs)")
axes[2].set_title("Memory Usage (GB)")
axes[3].set_title("Bandwidth (GB)")

# Set y-axis log scale
for ax in axes:
    ax.set_yscale('log')
    ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
# plt.savefig("../../../assets/figures/alibaba_resource_usage.svg", format="svg")
plt.show()


#%%
df["weekday"] = df["interval_15m"].dt.dayofweek
df["hour"] = df["interval_15m"].dt.hour
df["task_count"] = df["tasks_matrix"].apply(len)

heatmap_data = df.groupby(["weekday", "hour"])["task_count"].mean().unstack()
sns.heatmap(heatmap_data, cmap="jet", annot=False)
plt.title("Avg Task Count by Hour and Weekday")
plt.xlabel("Hour of Day")
plt.ylabel("Weekday (0=Mon)")
plt.tight_layout()

# plt.savefig("../../../assets/figures/alibaba_task_count_heatmap.svg", format="svg")
#%%
import matplotlib

# Unpack tasks_matrix into a flat DataFrame
tasks = []
for interval, task_list in zip(df["interval_15m"], df["tasks_matrix"]):
    for task in task_list:
        tasks.append({
            "interval_15m": interval,
            "hour": interval.hour,
            "cpu": float(task[5]) / 100.0,
            "gpu": float(task[6]) / 100.0,
            "mem": float(task[7]),
        })

task_df = pd.DataFrame(tasks)

# Crop only to months July and August
task_df = task_df[(task_df["interval_15m"].dt.month == 7) | (task_df["interval_15m"].dt.month == 8)]

# Group by hour
cpu_hourly = task_df.groupby("hour")["cpu"].apply(list)
gpu_hourly = task_df.groupby("hour")["gpu"].apply(list)
mem_hourly = task_df.groupby("hour")["mem"].apply(list)

# Convert to long format for seaborn boxplot
def melt_hourly_series(series):
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in series.items()]))
    return pd.melt(df, var_name='hour', value_name='value')

cpu_melted = melt_hourly_series(cpu_hourly)
gpu_melted = melt_hourly_series(gpu_hourly)
mem_melted = melt_hourly_series(mem_hourly)

# %%
figsize = (16, 4)
# CPU Plot
plt.figure(figsize=figsize)
sns.boxplot(data=cpu_melted, x="hour", y="value", whis=1, fliersize=0, color="tab:orange", boxprops=dict(facecolor="tab:orange", edgecolor="black", linewidth=1.2, alpha=0.7))
plt.title("Hourly Distribution of CPU Requests (>15 min duration)")
plt.xlabel("Hour of Day")
plt.ylabel("CPU Cores Requested")
plt.xticks(rotation=45)
plt.grid(alpha=0.3, linestyle="--")
plt.tight_layout()
# plt.savefig("../../../assets/figures/alibaba_hourly_cpu_requests.svg", format="svg")
plt.show()

# GPU Plot
plt.figure(figsize=figsize, dpi=120)
sns.boxplot(data=gpu_melted, x="hour", y="value", whis=1, fliersize=0, color="tab:red", boxprops=dict(facecolor="tab:red", edgecolor="black", linewidth=1.2, alpha=0.7))
plt.title("Hourly Distribution of GPU Requests (>15 min duration)")
plt.xlabel("Hour of Day")
plt.ylabel("GPU Units Requested")
plt.ylim(0, 1)  # Optional: adjust as needed
plt.xticks(rotation=45)
plt.grid(alpha=0.3, linestyle="--")
plt.tight_layout()
# plt.savefig("../../../assets/figures/alibaba_hourly_gpu_requests.svg", format="svg")
# plt.show()

# Memory Plot
plt.figure(figsize=figsize, dpi=120)
sns.boxplot(data=mem_melted, x="hour", y="value", whis=1, fliersize=0, color="tab:green", boxprops=dict(facecolor="tab:green", edgecolor="black", linewidth=1.2, alpha=0.7))
plt.title("Hourly Distribution of Memory Requests (>15 min duration)")
plt.xlabel("Hour of Day")
plt.ylabel("Memory Requested (GB)")
plt.xticks(rotation=45)
plt.grid(alpha=0.3, linestyle="--")
plt.tight_layout()
# plt.savefig("../../../assets/figures/alibaba_hourly_memory_requests.svg", format="svg")
plt.show()

#%%
# Choose a time window (e.g., 10 hours of simulation)
window_df = df[df["interval_15m"] < df["interval_15m"].iloc[0] + pd.Timedelta(hours=10)]

# Flatten tasks
all_tasks = []
for row in window_df.itertuples():
    for task in row.tasks_matrix:
        all_tasks.append({
            "start_time": pd.to_datetime(task[1], unit="s", origin="unix").tz_localize("Asia/Shanghai").tz_convert("UTC"),
            "end_time": pd.to_datetime(task[2], unit="s", origin="unix").tz_localize("Asia/Shanghai").tz_convert("UTC"),
            "cpu": float(task[5])/100,
            "gpu": float(task[6])/100,
            "mem": float(task[7])
        })

task_df = pd.DataFrame(all_tasks)

# Normalize time to hours since simulation start
start_ref = task_df["start_time"].min()
task_df["start_hr"] = (task_df["start_time"] - start_ref).dt.total_seconds() / 3600
task_df["duration_hr"] = (task_df["end_time"] - task_df["start_time"]).dt.total_seconds() / 3600

# Limit to ~50 tasks
init = 100
endd = init + 100
task_df = task_df.iloc[init:endd:2].copy()
task_df["task_idx"] = range(len(task_df))

# Create color mappers
def get_cmap(values, cmap_name):
    norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
    cmap = cm.get_cmap(cmap_name)
    return [cmap(norm(val)) for val in values], cm.ScalarMappable(norm=norm, cmap=cmap)

# Prepare data for plotting
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

for ax, resource, cmap, label in zip(
    axes,
    ["cpu", "gpu", "mem"],
    ["jet", "jet", "jet"],
    ["CPU Requirement (cores)", "GPU Requirement (# GPUs)", "Memory Requirement (GB)"]
):
    colors, sm = get_cmap(task_df[resource], cmap)
    for j, (_, row) in enumerate(task_df.iterrows()):
        ax.broken_barh([(row["start_hr"], row["duration_hr"])], (j - 0.5, 1.00), facecolors=colors[j], 
                       edgecolor="black", linewidth=0.1)


    ax.set_xlabel("Hours since simulation start")
    ax.set_title(f"Gantt Chart with {label}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(-0.5, len(task_df) - 0.5)

# Shared Y axis label
axes[0].set_ylabel("Task Index")

# Add colorbars
for ax, resource, label in zip(axes, ["cpu", "gpu", "mem"], ["CPU", "GPU", "Memory"]):
    _, sm = get_cmap(task_df[resource], "jet")
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(f"{label} Requirement")

plt.tight_layout()
# plt.savefig("../../../assets/figures/alibaba_task_gantt_resources.svg", format="svg")
plt.show()

#%%