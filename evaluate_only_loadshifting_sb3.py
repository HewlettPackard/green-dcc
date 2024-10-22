#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from envs.dcrl_env_harl_partialobs_sb3 import DCRL
from tqdm import tqdm
#%%
# Load the trained model
model = PPO.load("/lustre/guillant/green-dcc/models/ppo_agent_ls_20241021-173634_best_model/best_model.zip")

# Define your environment configuration
config = {
    'location': 'ca',
    'cintensity_file': 'CA_NG_&_avgCI.csv',
    'weather_file': 'USA_NY_New.York-LaGuardia.epw',
    'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
    'dc_config_file': 'dc_config_dc1.json',
    'datacenter_capacity_mw': 1.0,
    'flexible_load': 0.4,
    'timezone_shift': 8,
    'month': 7,
    'days_per_episode': 30,
    'partial_obs': True,
    'nonoverlapping_shared_obs_space': True,
    'debug': False,
    'initialize_queue_at_reset': True,
    'agents': ['agent_ls'],
    'workload_baseline': 0.0,
}

# Create the evaluation environment
env = DCRL(config)
env = Monitor(env)

# Number of episodes to evaluate
num_episodes = 1  # Adjust as needed

# Initialize lists to store episode rewards and lengths
episode_rewards = []
episode_lengths = []

# Initialize a list to collect metrics across episodes
metrics_list = []
#%%

for episode in tqdm(range(num_episodes)):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    time_step = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        # Collect metrics from the info dictionary
        common_info = info.get('__common__', {})

        # Convert any arrays to lists for serialization
        for key, value in common_info.items():
            if isinstance(value, np.ndarray):
                common_info[key] = value.tolist()
            elif isinstance(value, dict):
                # Flatten dictionaries if necessary
                for sub_key, sub_value in value.items():
                    new_key = f"{key}_{sub_key}"
                    common_info[new_key] = sub_value
                del common_info[key]

        # Append metrics to the list
        metrics = {'episode': episode + 1, 'time_step': time_step}
        metrics.update(common_info)
        metrics_list.append(metrics)

        time_step += 1

    # Store episode reward and length
    episode_rewards.append(episode_reward)
    episode_lengths.append(time_step)

# Convert metrics_list to a DataFrame and save to CSV
df_metrics = pd.DataFrame(metrics_list)
df_metrics.to_csv('evaluation_metrics.csv', index=False)

# Save episode rewards and lengths
df_episodes = pd.DataFrame({
    'episode': list(range(1, num_episodes + 1)),
    'episode_reward': episode_rewards,
    'episode_length': episode_lengths
})
df_episodes.to_csv('evaluation_episodes.csv', index=False)
#%%
# Save episode rewards and lengths
df_episodes = pd.DataFrame({
    'episode': list(range(1, num_episodes + 1)),
    'episode_reward': episode_rewards,
    'episode_length': episode_lengths
})
df_episodes.to_csv('evaluation_episodes.csv', index=False)
#%%
# Plot Original vs. Shifted Workload Over Time. Plot the index 1000:2000 to zoom in.

# Step 2: Read the metrics data
df_metrics = pd.read_csv('evaluation_metrics.csv')

# Step 3: Filter the DataFrame for timesteps between 1000 and 2000
one_day = 96
init_day = 7
init_timestep = one_day * init_day

plotted_days = 7
plotted_timesteps = one_day * plotted_days
final_timestep = init_timestep + plotted_timesteps

df_filtered = df_metrics[(df_metrics['time_step'] >= init_timestep) & (df_metrics['time_step'] <= final_timestep)].copy()

# Ensure the time_step index is reset for plotting
df_filtered.reset_index(drop=True, inplace=True)

# Step 4: Create a plot with twin y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Step 5: Plot ls_original_workload and ls_shifted_workload on the left y-axis
color1 = 'tab:blue'
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('Workload', color=color1)
line1, = ax1.plot(df_filtered['time_step'], df_filtered['ls_original_workload']*100, label='Original Workload', color='tab:blue', linewidth=2)
line2, = ax1.plot(df_filtered['time_step'], df_filtered['ls_shifted_workload']*100, label='Shifted Workload', color='tab:orange', linestyle='--', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)

# Step 5: Plot bat_CO2_footprint on the right y-axis
ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
color2 = 'tab:red'
ax2.set_ylabel('Battery CO2 Footprint', color=color2)
line3, = ax2.plot(df_filtered['time_step'], df_filtered['bat_avg_CI'], label='Carbon Intensity', color=color2, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color2)

# Step 6: Customize the plot
plt.title('Workload and Battery CO2 Footprint Over Time (Timesteps 1000-2000)')
lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left')
ax1.grid(True)

# Set y-axis limits between 0 and 1 for the workload
ax1.set_ylim([0, 100])

# Step 7: Display the plot
plt.show()
#%% Now plot the value ls_tasks_dropped

plt.figure(figsize=(12, 6))
plt.plot(df_filtered['time_step'], df_filtered['ls_tasks_dropped'], label='Tasks Dropped', color='tab:green', linewidth=2)

# %% Now plot the value of ls_overdue_penalty
plt.figure(figsize=(12, 6))
plt.plot(df_filtered['time_step'], df_filtered['ls_overdue_penalty'], label='Overdue Penalty', color='tab:red', linewidth=2)

# %%
