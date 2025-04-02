#%%
import pandas as pd


#%%
# -------------------------------------------------------------------
# 1) Read your CSV and parse 'start_date' as datetime
#    (This dataset example has columns: job_name, start_time, end_time,
#     start_date, duration_min, cpu_usage, gpu_wrk_util, avg_mem, avg_gpu_wrk_mem)
# -------------------------------------------------------------------
df = pd.read_csv("data/workload/alibaba_2020_dataset/extracted_dfas_with_bandwidth.csv")

# If your 'start_date' column is already a datetime with a timezone, 
# you can parse as normal. If not, you might do:
df["start_dt"] = pd.to_datetime(df["start_date"])  # already has +08:00 in your example

# If you wanted to interpret numeric columns (like start_time in seconds)
# as Asia/Shanghai, you'd do something like:
df["start_dt"] = (pd.to_datetime(df["start_time"], unit="s", origin="unix")
                  .dt.tz_localize("Asia/Shanghai"))

# Now convert to UTC+0:
df["start_dt"] = df["start_dt"].dt.tz_convert("UTC")

# -------------------------------------------------------------------
# 2) Extract the day of the week from 'start_dt'
# -------------------------------------------------------------------
df["weekday_name"] = df["start_dt"].dt.day_name()   # e.g. 'Monday', 'Tuesday'
df["weekday_num"]  = df["start_dt"].dt.weekday      # Monday=0, Sunday=6

# -------------------------------------------------------------------
# 3) Create 15-minute intervals by "flooring" the start_dt
# -------------------------------------------------------------------
df["interval_15m"] = df["start_dt"].dt.floor("15T")


#%%
# EDA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cols = ["duration_min", "cpu_usage", "avg_mem"]

# Plot boxplots for each column
plt.figure(figsize=(15, 5))

for i, col in enumerate(cols):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)

plt.tight_layout()
plt.show()

# Plot histograms for each column
plt.figure(figsize=(15, 5))

for i, col in enumerate(cols):
    plt.subplot(1, 3, i + 1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


#%%
import pandas as pd

# Function to remove outliers using the IQR rule
def remove_outliers_iqr(df, columns):
    for col in columns:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define the lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Remove rows with values outside the bounds
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Columns with potential outliers
columns_with_outliers = ["duration_min", "cpu_usage", "avg_mem"]

# Apply the IQR rule to remove outliers
df_cleaned = remove_outliers_iqr(df, columns_with_outliers)

# Check the result
print("Original dataset shape:", df.shape)
print("Cleaned dataset shape:", df_cleaned.shape)



#%%

cols = ["duration_min", "cpu_usage", "avg_mem"]

# Plot boxplots for each column
plt.figure(figsize=(15, 5))

for i, col in enumerate(cols):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(y=df_cleaned[col])
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)

plt.tight_layout()
plt.show()

# Plot histograms for each column
plt.figure(figsize=(15, 5))

for i, col in enumerate(cols):
    plt.subplot(1, 3, i + 1)
    sns.histplot(df_cleaned[col], kde=True, bins=30)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


#%%
# -------------------------------------------------------------------
# 4) Group tasks by this 15-minute interval
# -------------------------------------------------------------------
grouped = df_cleaned.groupby("interval_15m")

# Decide which columns you want in your matrix. 
# We'll include everything plus the new weekday columns.
columns_of_interest = [
    "job_name", 
    "start_time", 
    "end_time",
    "start_dt", 
    "duration_min", 
    "cpu_usage", 
    "gpu_wrk_util", 
    "avg_mem", 
    "avg_gpu_wrk_mem",
    "weekday_name",
    "weekday_num"
]

# -------------------------------------------------------------------
# 5) Build a new DataFrame: one row per 15-min interval,
#    and a "tasks_matrix" column containing a NumPy array of tasks.
# -------------------------------------------------------------------
data = []
for interval_value, group_df in grouped:
    # Convert the subset to a numpy array
    tasks_matrix = group_df[columns_of_interest].to_numpy()
    
    data.append({
        "interval_15m": interval_value,       # the 15-min bin (datetime)
        "tasks_matrix": tasks_matrix          # all tasks that started in this bin
    })

result_df = pd.DataFrame(data)

# (Optional) Sort by the interval if not already sorted:
result_df.sort_values("interval_15m", inplace=True)

# -------------------------------------------------------------------
# 6) Show or inspect the final dataset
# -------------------------------------------------------------------
print(result_df.head())

# For a given row (time bin), you can see the array of tasks:
print("\n--- Example row ---")
first_row = result_df.iloc[0]
print("Interval start:", first_row["interval_15m"])
print("Tasks matrix shape:", first_row["tasks_matrix"].shape)
print("Tasks matrix example:\n", first_row["tasks_matrix"])
# %%

# Assuming result_df has a "interval_15m" column as datetime
# Define the start and end time for the 7-week period
start_time = pd.Timestamp("1970-01-26 00:00:00+00:00")  # Midnight of the desired start
end_time = start_time + pd.Timedelta(weeks=7)           # Add 7 weeks (49 days)

# Filter result_df to include only rows within this range
result_df_cropped = result_df[
    (result_df["interval_15m"] >= start_time) & (result_df["interval_15m"] < end_time)
]

# Print the cropped dataset info
print("Original result_df shape:", result_df.shape)
print("Cropped result_df shape:", result_df_cropped.shape)
print(result_df_cropped.head())

# Check the first and last timestamps in the cropped data
print("Start of cropped data:", result_df_cropped["interval_15m"].min())
print("End of cropped data:", result_df_cropped["interval_15m"].max())

#%%

# Save to a pickle file
result_df_cropped.to_pickle("alibaba_2020_dataset/result_df_cropped.pkl")

print("Saved result_df_cropped to 'result_df_cropped.pkl'")

#%%
#%%
import pandas as pd


#%%
# -------------------------------------------------------------------
# 1) Read your CSV and parse 'start_date' as datetime
#    (Now includes bandwidth_gb)
# -------------------------------------------------------------------
df = pd.read_csv("data/workload/alibaba_2020_dataset/extracted_dfas_with_bandwidth.csv")

# If your 'start_date' column is already a datetime with a timezone, 
# you can parse as normal. If not, you might do:
df["start_dt"] = pd.to_datetime(df["start_date"])  # already has +08:00 in your example

# If you wanted to interpret numeric columns (like start_time in seconds)
# as Asia/Shanghai, you'd do something like:
df["start_dt"] = (pd.to_datetime(df["start_time"], unit="s", origin="unix")
                  .dt.tz_localize("Asia/Shanghai"))

# Now convert to UTC+0:
df["start_dt"] = df["start_dt"].dt.tz_convert("UTC")

# -------------------------------------------------------------------
# 2) Extract the day of the week from 'start_dt'
# -------------------------------------------------------------------
df["weekday_name"] = df["start_dt"].dt.day_name()   # e.g. 'Monday', 'Tuesday'
df["weekday_num"]  = df["start_dt"].dt.weekday      # Monday=0, Sunday=6

# -------------------------------------------------------------------
# 3) Create 15-minute intervals by "flooring" the start_dt
# -------------------------------------------------------------------
df["interval_15m"] = df["start_dt"].dt.floor("15T")


#%%
# EDA (Updated for Bandwidth)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cols = ["duration_min", "cpu_usage", "avg_mem", "bandwidth_gb"]

# Plot boxplots for each column
plt.figure(figsize=(20, 5))

for i, col in enumerate(cols):
    plt.subplot(1, 4, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)

plt.tight_layout()
plt.show()

# Plot histograms for each column
plt.figure(figsize=(20, 5))

for i, col in enumerate(cols):
    plt.subplot(1, 4, i + 1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


#%%
import pandas as pd

# Function to remove outliers using the IQR rule (Updated for Bandwidth)
def remove_outliers_iqr(df, columns):
    for col in columns:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define the lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Remove rows with values outside the bounds
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Columns with potential outliers
columns_with_outliers = ["duration_min", "cpu_usage", "avg_mem", "bandwidth_gb"]

# Apply the IQR rule to remove outliers
df_cleaned = remove_outliers_iqr(df, columns_with_outliers)

# Check the result
print("Original dataset shape:", df.shape)
print("Cleaned dataset shape:", df_cleaned.shape)


#%%

cols = ["duration_min", "cpu_usage", "avg_mem", "bandwidth_gb"]

# Plot boxplots for each column
plt.figure(figsize=(20, 5))

for i, col in enumerate(cols):
    plt.subplot(1, 4, i + 1)
    sns.boxplot(y=df_cleaned[col])
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)

plt.tight_layout()
plt.show()

# Plot histograms for each column
plt.figure(figsize=(20, 5))

for i, col in enumerate(cols):
    plt.subplot(1, 4, i + 1)
    sns.histplot(df_cleaned[col], kde=True, bins=30)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


#%%
# -------------------------------------------------------------------
# 4) Group tasks by this 15-minute interval
# -------------------------------------------------------------------
grouped = df_cleaned.groupby("interval_15m")

# Decide which columns you want in your matrix. 
# We'll include everything plus the new weekday columns and bandwidth.
columns_of_interest = [
    "job_name", 
    "start_time", 
    "end_time",
    "start_dt", 
    "duration_min", 
    "cpu_usage", 
    "gpu_wrk_util", 
    "avg_mem", 
    "avg_gpu_wrk_mem",
    "bandwidth_gb",  # Newly added column
    "weekday_name",
    "weekday_num"
]

# -------------------------------------------------------------------
# 5) Build a new DataFrame: one row per 15-min interval,
#    and a "tasks_matrix" column containing a NumPy array of tasks.
# -------------------------------------------------------------------
data = []
for interval_value, group_df in grouped:
    # Convert the subset to a numpy array
    tasks_matrix = group_df[columns_of_interest].to_numpy()
    
    data.append({
        "interval_15m": interval_value,       # the 15-min bin (datetime)
        "tasks_matrix": tasks_matrix          # all tasks that started in this bin
    })

result_df = pd.DataFrame(data)

# (Optional) Sort by the interval if not already sorted:
result_df.sort_values("interval_15m", inplace=True)

# -------------------------------------------------------------------
# 6) Show or inspect the final dataset
# -------------------------------------------------------------------
print(result_df.head())

# For a given row (time bin), you can see the array of tasks:
print("\n--- Example row ---")
first_row = result_df.iloc[0]
print("Interval start:", first_row["interval_15m"])
print("Tasks matrix shape:", first_row["tasks_matrix"].shape)
print("Tasks matrix example:\n", first_row["tasks_matrix"])
# %%

# Assuming result_df has a "interval_15m" column as datetime
# Define the start and end time for the 7-week period
start_time = pd.Timestamp("1970-01-26 00:00:00+00:00")  # Midnight of the desired start
end_time = start_time + pd.Timedelta(weeks=7)           # Add 7 weeks (49 days)

# Filter result_df to include only rows within this range
result_df_cropped = result_df[
    (result_df["interval_15m"] >= start_time) & (result_df["interval_15m"] < end_time)
]

# Print the cropped dataset info
print("Original result_df shape:", result_df.shape)
print("Cropped result_df shape:", result_df_cropped.shape)
print(result_df_cropped.head())

# Check the first and last timestamps in the cropped data
print("Start of cropped data:", result_df_cropped["interval_15m"].min())
print("End of cropped data:", result_df_cropped["interval_15m"].max())

#%%
# Save to a pickle file
result_df_cropped.to_pickle("data/alibaba_2020_dataset/result_df_cropped_with_bandwidth.pkl")

print("Saved result_df_cropped_with_bandwidth to 'result_df_cropped_with_bandwidth.pkl'")

#%%
# %%
# Generate 1 year (2020) worth of repeated task data from 7-week trace

import copy

# How many weeks do we need to fill all of 2020?
weeks_in_2020 = 52  # Ignore leap day for simplicity
weeks_from_trace = result_df_cropped.copy()

# Sanity check
assert len(weeks_from_trace) > 0, "No data found in the cropped 7-week result_df"

# Store new rows here
new_rows = []

# Starting date for 2020 (UTC)
target_start = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")

# Duration of one full 7-week block
trace_duration = weeks_from_trace["interval_15m"].iloc[-1] - weeks_from_trace["interval_15m"].iloc[0] + pd.Timedelta(minutes=15)

# Repeat trace to cover the year
current_time = target_start
weeks_repeated = 0

while current_time < pd.Timestamp("2021-01-01", tz="UTC"):
    time_shift = current_time - weeks_from_trace["interval_15m"].iloc[0]
    for _, row in weeks_from_trace.iterrows():
        shifted_row = {
            "interval_15m": row["interval_15m"] + time_shift,
            "tasks_matrix": copy.deepcopy(row["tasks_matrix"])  # Copy to prevent mutation
        }
        new_rows.append(shifted_row)

    current_time += trace_duration
    weeks_repeated += 7

print(f"Total weeks generated: ~{weeks_repeated}")
print(f"Total intervals: {len(new_rows)}")

# Build final DataFrame
full_year_df = pd.DataFrame(new_rows)
full_year_df.sort_values("interval_15m", inplace=True)

# Sanity check range
print("First timestamp:", full_year_df["interval_15m"].min())
print("Last timestamp:", full_year_df["interval_15m"].max())

# Save to disk
full_year_df.to_pickle("data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl")
print("✅ Saved extended 1-year dataset to: result_df_full_year_2020.pkl")

# %%
