
#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Read the CSV files (Adjust paths as needed)
# Explicitly pass your column names:
columns = [
    "job_name",
    "task_name",
    "inst_num",
    "status",
    "start_time",
    "end_time",
    "plan_cpu",
    "plan_mem",
    "plan_gpu",
    "gpu_type"
]


df_tasks = pd.read_csv("alibaba_2020_dataset/pai_task_table.csv",
                        header=None,       # No header row in the file
                       names=columns)     # Use the list above as the column names

df_tasks.head()

print("Initial dataframe info:")
df_tasks.info()  # or df_tasks.head()

#%%
# 4. Drop rows with any missing values
df_tasks.dropna(inplace=True)

# 5. Convert numeric columns (if not already numeric)
#    For example, plan_cpu or inst_num might still be strings:
numeric_cols = ["inst_num", "start_time", "end_time", "plan_cpu", "plan_mem", "plan_gpu"]
for col in numeric_cols:
    df_tasks[col] = pd.to_numeric(df_tasks[col], errors="coerce")

# 6. After conversion, there could be new NaNs if a column had non-numeric garbage
#    So drop those rows again if needed
df_tasks.dropna(subset=numeric_cols, inplace=True)

# 7. Remove duplicates if any (optional)
df_tasks.drop_duplicates(inplace=True)

# 8. (Optional) Filter out obviously invalid data
#    For example, negative timestamps, or plan_cpu < 0, etc.
df_tasks = df_tasks[
    (df_tasks["start_time"] >= 0) &
    (df_tasks["end_time"] >= df_tasks["start_time"]) &  # end_time should be >= start_time
    (df_tasks["plan_cpu"] >= 0) &
    (df_tasks["plan_mem"] >= 0) &
    (df_tasks["plan_gpu"] >= 0)
]

# Drop the tasks with total duration less than 15 minutes because my simulation timestep is 15 minutes
df_tasks = df_tasks[df_tasks["end_time"] - df_tasks["start_time"] >= 15 * 60]

# 9. Final check
print("Cleaned dataframe info:")
df_tasks.info()


#%%
# 2. Convert start/end times to datetime
df_tasks["arrival_time"] = pd.to_datetime(df_tasks["start_time"], unit="s", origin="unix")
df_tasks["finish_time"]  = pd.to_datetime(df_tasks["end_time"],   unit="s", origin="unix")

# 3. Get earliest and latest times
min_time = df_tasks["arrival_time"].min() + pd.Timedelta(days=10)  # skip the first day
max_time = df_tasks["arrival_time"].min() + pd.Timedelta(days=17)  # skip the first day
print("Earliest start:", min_time)
print("Latest end:   ", max_time)

# Crop the data from min_time to max_time
df_tasks_short = df_tasks[
    (df_tasks["arrival_time"] >= min_time) &
    (df_tasks["finish_time"] <= max_time)
]

# 4. Resample by 15-minute bins based on arrival_time
df_tasks_short.set_index("arrival_time", inplace=True)
count_series = df_tasks_short.resample("15T").size()  # counts how many tasks start in each bin

# 5. Reindex from min_time to max_time in 15-min increments, filling missing bins with zero
full_index = pd.date_range(start=min_time, end=max_time, freq="15T")
# count_series = count_series.reindex(full_index, fill_value=0)

# 6. Plot (line chart of how many tasks start in each 15-min bin)
plt.figure(figsize=(12,6))
sns.lineplot(x=count_series.index, y=count_series.values, marker="o")
plt.title("Count of Tasks Started Every 15 Minutes")
plt.xlabel("Time")
plt.ylabel("Count of Task Starts")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

#%%
# Create a time index from min_time to max_time with 15-minute frequency.
time_bins = pd.date_range(start=min_time, end=max_time, freq="15T")

# Prepare a list to store our per-interval simulation data.
simulation_data = []

df_tasks_short = df_tasks[
    (df_tasks["arrival_time"] >= min_time) &
    (df_tasks["finish_time"] <= max_time)
]

# Loop over each 15-minute interval.
for t in time_bins:
    # Define the current interval [t, t + 15 minutes)
    next_t = t + pd.Timedelta(minutes=15)
    
    # New arrivals: tasks that start in the current interval.
    arrivals = df_tasks_short[(df_tasks_short["arrival_time"] >= t) & (df_tasks_short["arrival_time"] < next_t)]
    
    # Running tasks: tasks that have started before t and have not finished by t.
    running = df_tasks_short[(df_tasks_short["arrival_time"] < t) & (df_tasks_short["finish_time"] > t)]
    
    simulation_data.append({
        "time": t,
        "new_arrivals_count": arrivals.shape[0],
        "running_tasks_count": running.shape[0],
        "new_arrivals_cpu": arrivals["plan_cpu"].sum(),
        "running_tasks_cpu": running["plan_cpu"].sum()
        # You can add similar aggregates for memory or GPU requests if desired.
    })

# Create a DataFrame with the simulation data.
simulation_df = pd.DataFrame(simulation_data)
simulation_df.set_index("time", inplace=True)

print("\nSimulation dataset head:")
print(simulation_df.head())

# ---------------------------
# 4. Plot the results (for example)
# ---------------------------
# Plot the number of new arrivals and running tasks over time.
plt.figure(figsize=(12, 6))
sns.lineplot(data=simulation_df, x=simulation_df.index, y="new_arrivals_count", label="New Arrivals")
sns.lineplot(data=simulation_df, x=simulation_df.index, y="running_tasks_count", label="Running Tasks")
plt.title("Task Arrivals and Running Tasks per 15-Minute Interval")
plt.xlabel("Time")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Optionally, you can also plot the aggregated CPU requests similarly.
plt.figure(figsize=(12, 6))
sns.lineplot(data=simulation_df, x=simulation_df.index, y="new_arrivals_cpu", label="New Arrivals CPU")
sns.lineplot(data=simulation_df, x=simulation_df.index, y="running_tasks_cpu", label="Running Tasks CPU")
plt.title("Aggregated CPU Requests per 15-Minute Interval")
plt.xlabel("Time")
plt.ylabel("Total plan_cpu")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#%%
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Example: Create an "active tasks" timeline using the sweep-line method.
# =============================================================================

# Assume you already have a cleaned df_tasks with these columns:
#   - 'arrival_time': pd.to_datetime(df_tasks["start_time"], unit="s", origin="unix")
#   - 'finish_time':  pd.to_datetime(df_tasks["end_time"],   unit="s", origin="unix")
#
# If not, here's a reminder:
# df_tasks["arrival_time"] = pd.to_datetime(df_tasks["start_time"], unit="s", origin="unix")
# df_tasks["finish_time"]  = pd.to_datetime(df_tasks["end_time"], unit="s", origin="unix")

# 1. Determine the 15-minute bin for each task’s start and finish.
#    For the finish, we want the task to be active in its finishing bin, so we
#    subtract events (–1) at the bin *after* the finish.
df_tasks["start_bin"] = df_tasks["arrival_time"].dt.floor("15T")
df_tasks["finish_bin"] = df_tasks["finish_time"].dt.floor("15T")

# 2. Create event counts: 
#    +1 at the start_bin and -1 at the bin immediately after finish_bin.
df_tasks["finish_bin_plus"] = df_tasks["finish_bin"] + pd.Timedelta(minutes=15)

# Count how many tasks start in each 15-minute bin.
start_events = df_tasks.groupby("start_bin").size()

# Count how many tasks finish (i.e. should stop counting) in each bin.
finish_events = df_tasks.groupby("finish_bin_plus").size()

# 3. Create a combined event series.
# For each bin, net event = (number of starts) - (number of tasks ending in the previous bin)
all_events = start_events.subtract(finish_events, fill_value=0)  # subtract finish events

# 4. Create a full 15-minute time index covering from the earliest start_bin to the latest finish event.
min_bin = df_tasks["start_bin"].min()
max_bin = df_tasks["finish_bin_plus"].max()
full_index = pd.date_range(start=min_bin, end=max_bin, freq="15T")

# Reindex the event series so every 15-minute interval is represented.
all_events = all_events.reindex(full_index, fill_value=0)

# 5. Compute the cumulative sum to get the number of active tasks at each time interval.
active_tasks = all_events.cumsum()

# 6. Plot the resulting active tasks time series as a step plot.
plt.figure(figsize=(12, 6))
plt.plot(active_tasks.index, active_tasks.values, drawstyle="steps-post", marker="o")
plt.xlabel("Time")
plt.ylabel("Number of Active Tasks")
plt.title("Accumulated Active Tasks Over Time (15-Minute Bins)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




#%%
#%%
# 2. Convert task start_time (seconds) to a pandas datetime.
#    The trace "start_time" is an offset in seconds, so we treat it like a Unix timestamp.
df_tasks["arrival_time"] = pd.to_datetime(df_tasks["start_time"], unit="s", origin="unix")

# ----- 3. Extract only the first day -----
# For example, from 1970-01-01 00:00:00 up to (but not including) 1970-01-02 00:00:00
df_tasks_one_day = df_tasks[
    (df_tasks["arrival_time"] >= "1970-01-09") &
    (df_tasks["arrival_time"] <  "1970-01-12")
].copy()

# ----- 4. Floor arrival_time to 15-minute bins -----
df_tasks_one_day["interval_15m"] = df_tasks_one_day["arrival_time"].dt.floor("15T")

# ----- 5. Plot violin for the subset -----
plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df_tasks_one_day,
    x="interval_15m",     # the grouping on the x-axis
    y="plan_cpu",         # numeric values on the y-axis
    showfliers=True,      # to display outliers
    whis=1.5              # determines the whisker length
)

plt.xticks(rotation=45, ha="right")
plt.xlabel("15-Minute Interval")
plt.ylabel("plan_cpu (percentage = cores * 100)")

plt.tight_layout()
plt.show()

#%%