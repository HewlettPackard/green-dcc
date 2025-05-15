#%%
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
sns.set_theme(style="whitegrid")

from rl_components.agent_net import ActorNet
from envs.task_scheduling_env import TaskSchedulingEnv
from simulation.cluster_manager import DatacenterClusterManager

import datetime
import logging

#%%
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

log_path = f"logs/evaluation_{timestamp}.log"

logger = logging.getLogger("eval_logger")
logger.setLevel(logging.INFO)  # File handler will capture INFO+

# === File handler (full log)
file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
logger.addHandler(file_handler)

# === Console handler (only warnings and errors)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Only show warnings+errors in terminal
console_handler.setFormatter(logging.Formatter(
    "[%(levelname)s] %(message)s"
))
logger.addHandler(console_handler)

def make_eval_env(eval_mode=True):
    from utils.config_loader import load_yaml
    from rewards.predefined.composite_reward import CompositeReward
    from torch.utils.tensorboard import SummaryWriter

    sim_cfg = load_yaml("configs/env/sim_config.yaml")["simulation"]
    dc_cfg = load_yaml("configs/env/datacenters.yaml")["datacenters"]
    reward_cfg = load_yaml("configs/env/reward_config.yaml")["reward"]

    start = pd.Timestamp(datetime.datetime(sim_cfg["year"], sim_cfg["month"], sim_cfg["init_day"],
                                           sim_cfg["init_hour"], 0, tzinfo=datetime.timezone.utc))
    end = start + datetime.timedelta(days=sim_cfg["duration_days"])

    cluster = DatacenterClusterManager(
        config_list=dc_cfg,
        simulation_year=sim_cfg["year"],
        init_day=int(sim_cfg["month"] * 30.5),
        init_hour=sim_cfg["init_hour"],
        strategy=sim_cfg["strategy"],
        tasks_file_path=sim_cfg["workload_path"],
        shuffle_datacenter_order=not eval_mode,
        cloud_provider=sim_cfg["cloud_provider"],
        logger=logger
    )

    reward_fn = CompositeReward(
        components=reward_cfg["components"],
        normalize=reward_cfg.get("normalize", False)
    )

    env = TaskSchedulingEnv(
        cluster_manager=cluster,
        start_time=start,
        end_time=end,
        reward_fn=reward_fn,
        writer=None
    )
    return env


# Load trained actor model
checkpoint_path = "checkpoints/train_20250509_232500/best_checkpoint.pth"  # Adjust path
env = make_eval_env()
obs, _ = env.reset(seed=123)
obs_dim = env.observation_space.shape[0]
# Detect the strategy (manual_rl = agent, else = RBC)
use_actor = env.cluster_manager.strategy == "manual_rl"
act_dim = env.num_dcs + 1

if use_actor:
    logger.info("Using trained actor model for evaluation.")
    actor = ActorNet(obs_dim, act_dim, hidden_dim=64)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
else:
    logger.info("Using RBC for evaluation.")
    

infos_list = []
common_info_list = []
rewards = []
steps = 7 * 96  # 7 days of 15-min intervals
delayed_task_counts = [] 

for step in tqdm(range(steps)):
    if len(obs) == 0:
        actions = []
    elif use_actor:
        # Use trained agent
        obs_tensor = torch.FloatTensor(obs)
        with torch.no_grad():
            logits = actor(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample().numpy().tolist()
        delayed_count = sum(1 for a in actions if a == 0)
        delayed_task_counts.append(delayed_count)
    else:
        # If RBC is used, actions must be empty -> internal logic assigns them
        actions = []
        delayed_task_counts.append(0)  # No delayed logic in RBC

    obs, reward, done, truncated, info = env.step(actions)
    infos_list.append(info["datacenter_infos"])
    common_info_list.append(info["transmission_cost_total_usd"])
    rewards.append(reward)

    if done or truncated:
        obs, _ = env.reset(seed=123)

#%%
info["datacenter_infos"]

#%%

flat_records = []

for t, timestep_info in enumerate(infos_list):
    for dc_name, dc_info in timestep_info.items():
        sla_info = dc_info.get("__sla__", {"met": 0, "violated": 0})
        record = {
            "timestep": t,
            "datacenter": dc_name,
            "energy_cost": dc_info["__common__"]["energy_cost_USD"],
            "energy_kwh": dc_info["__common__"]["energy_consumption_kwh"],
            "carbon_kg": dc_info["__common__"]["carbon_emissions_kg"],
            "price_per_kwh": dc_info["__common__"]["price_USD_kwh"],
            "ci": dc_info["__common__"]["ci"],
            "weather": dc_info["__common__"]["weather"],
            "cpu_util": dc_info["__common__"]["cpu_util_percent"],
            "gpu_util": dc_info["__common__"]["gpu_util_percent"],
            "mem_util": dc_info["__common__"]["mem_util_percent"],
            "running_tasks": dc_info["__common__"]["running_tasks"],
            "pending_tasks": dc_info["__common__"]["pending_tasks"],
            "tasks_assigned": dc_info["__common__"].get("tasks_assigned", 0),
            "sla_met": dc_info["__common__"]['__sla__'].get("met", 0),
            "sla_violated": dc_info["__common__"]['__sla__'].get("violated", 0),
            "dc_cpu_workload_fraction": dc_info["agent_dc"].get("dc_cpu_workload_fraction", 0.0),
            # "transmission_cost": dc_info["__common__"].get("transmission_cost_total_usd", 0.0),
            "hvac_setpoint": dc_info["__common__"].get("hvac_setpoint_c", np.nan),

        }
        flat_records.append(record)

df = pd.DataFrame(flat_records)


#%%
summary = df.groupby("datacenter").agg({
    "energy_cost": "sum",
    "energy_kwh": "sum",
    "carbon_kg": "sum",
    "price_per_kwh": "mean",
    "ci": "mean",
    "weather": "mean",
    "cpu_util": "mean",
    "gpu_util": "mean",
    "mem_util": "mean",
    "running_tasks": "sum", # Maybe 'mean' is better for running tasks over time? Summing might be misleading.
    "pending_tasks": "mean",
    "sla_met": "sum",
    "sla_violated": "sum",
    # "dc_cpu_workload_fraction": "mean", # Already captured by cpu_util
    "hvac_setpoint": "mean", #
}).reset_index()

# Recalculate SLA Violation Rate
summary["SLA Violation Rate (%)"] = (
    summary["sla_violated"] / (summary["sla_met"] + summary["sla_violated"]).replace(0, np.nan)
) * 100
summary.fillna(0, inplace=True) # Fill NaN rates with 0 if no tasks completed

summary = summary.round(2)

# Update Column Names
summary.columns = [
    "Datacenter", "Total Energy Cost (USD)", "Total Energy (kWh)", "Total CO₂ (kg)",
    "Avg Price (USD/kWh)", "Avg Carbon Intensity (gCO₂/kWh)", "Avg Weather (°C)",
    "Avg CPU Util (%)", "Avg GPU Util (%)", "Avg MEM Util (%)", "Total Tasks Finished", # Changed label
    "Avg Pending Tasks", "SLA Met", "SLA Violated", "Avg HVAC Setpoint (°C)", "SLA Violation Rate (%)",
]

print("\n--- Evaluation Summary ---")
summary

#%%
total_cost = sum(common_info_list)
print(f"Total Transmission Cost (USD) for 7 Days: {total_cost:.2f}")

#%%
# Energy price per kWh plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="timestep", y="price_per_kwh", hue="datacenter")
plt.title("Energy Price per kWh over Time")
plt.xlabel("Timestep")
plt.ylabel("USD/kWh")
plt.grid(True)

# Save the figure as pdf
plt.savefig("assets/figures/energy_price_over_time.pdf", bbox_inches='tight')

plt.show()


#%% 
# Plot of "Total Running Tasks"
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="timestep", y="running_tasks", hue="datacenter")
plt.title("Total Running Tasks per Datacenter over Time")
plt.xlabel("Timestep")
plt.ylabel("Number of Tasks")
plt.grid(True)

# Save the figure as pdf
plt.savefig("assets/figures/running_tasks_over_time.pdf", bbox_inches='tight')

plt.show()

#%% Plot of Tasks Assigned per Datacenter over Time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="timestep", y="tasks_assigned", hue="datacenter")
plt.title("Tasks Assigned per Datacenter over Time")
plt.xlabel("Timestep")
plt.ylabel("Number of Assigned Tasks")
plt.grid(True)

# Save the figure as pdf
plt.savefig("assets/figures/tasks_assigned_over_time.pdf", bbox_inches='tight')

plt.show()


#%%
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="timestep", y="energy_cost", hue="datacenter")
plt.title("Energy Cost per Datacenter over Time")
plt.xlabel("Timestep")
plt.ylabel("USD")
plt.grid(True)

# Save the figure as pdf
plt.savefig("assets/figures/energy_cost_over_time.pdf", bbox_inches='tight')

plt.show()


#%%
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="timestep", y="carbon_kg", hue="datacenter")
plt.title("Carbon Emissions (kg) per Datacenter over Time")
plt.ylabel("kg CO₂")
plt.grid(True)
# Save the figure as pdf
plt.savefig("assets/figures/carbon_emissions_over_time.pdf", bbox_inches='tight')
plt.show()


#%% Carbon intensity plot on each location (Datacenter)
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="timestep", y="ci", hue="datacenter")
plt.title("Carbon Intensity (gCO₂/kWh) over Time")
plt.xlabel("Timestep")
plt.ylabel("gCO₂/kWh")
plt.grid(True)
# Save the figure as pdf
plt.savefig("assets/figures/carbon_intensity_over_time.pdf", bbox_inches='tight')
plt.show()



#%%
plt.figure(figsize=(12, 6))
plt.plot(common_info_list)
plt.title("Total Transmission Cost (USD) Over Time")
plt.xlabel("Timestep")
plt.ylabel("Transmission Cost (USD)")
plt.grid(True)

# Save the figure as pdf
plt.savefig("assets/figures/transmission_cost_over_time.pdf", bbox_inches='tight')
plt.show()


#%%
plt.figure(figsize=(12, 6))
plt.plot(delayed_task_counts, label="Delayed Tasks", color="orange")
plt.title("Number of Delayed Tasks per Timestep")
plt.xlabel("Timestep")
plt.ylabel("Number of Delayed Tasks")
plt.grid(True)
plt.legend()
# Save the figure as pdf
plt.savefig("assets/figures/delayed_tasks_over_time.pdf", bbox_inches='tight')
plt.show()



#%% Plot the external temperature
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="timestep", y="weather", hue="datacenter")
plt.title("External Temperature (°C) over Time")
plt.xlabel("Timestep")
plt.ylabel("Temperature (°C)")
plt.grid(True)
# Save the figure as pdf
plt.savefig("assets/figures/external_temperature_over_time.pdf", bbox_inches='tight')

plt.show()


#%%

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# CPU Utilization
sns.lineplot(
    data=df,
    x="timestep",
    y="cpu_util",
    hue="datacenter",
    ax=axes[0]
)
axes[0].set_title("CPU Utilization (%) over Time")
axes[0].set_ylabel("CPU %")
axes[0].grid(True)

# GPU Utilization
sns.lineplot(
    data=df,
    x="timestep",
    y="gpu_util",
    hue="datacenter",
    ax=axes[1]
)
axes[1].set_title("GPU Utilization (%) over Time")
axes[1].set_ylabel("GPU %")
axes[1].grid(True)

# MEM Utilization
sns.lineplot(
    data=df,
    x="timestep",
    y="mem_util",
    hue="datacenter",
    ax=axes[2]
)
axes[2].set_title("Memory Utilization (%) over Time")
axes[2].set_ylabel("MEM %")
axes[2].set_xlabel("Timestep")
axes[2].grid(True)

plt.tight_layout()
# Save the figure as pdf
plt.savefig("assets/figures/utilization_over_time.pdf", bbox_inches='tight')

plt.show()


#%% Plot the HVAC Cooling Setpoint
plt.figure(figsize=(12, 6))
# sns.lineplot(data=df, x="timestep", y="hvac_setpoint", hue="datacenter", marker='o', markersize=2, linestyle='') # Points might be better
sns.lineplot(data=df, x="timestep", y="hvac_setpoint", hue="datacenter")

plt.title("HVAC Cooling Setpoint (°C) per Datacenter over Time")
plt.xlabel("Timestep (15 min intervals)")
plt.ylabel("Setpoint (°C)")

plt.grid(True)
plt.legend(title='Datacenter', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
#%%

#%% Figure 2: Energy Price, Carbon Intensity, and Tasks Assigned over First N Days (Simulation Hours)

# number of days to plot
days = 5
steps_per_day = 96
max_steps = steps_per_day * days

# filter to first N days
df2 = df[df["timestep"] < max_steps].copy()
# add a column for simulation hours (each timestep = 0.25 h)
df2["sim_hour"] = df2["timestep"] * 0.25

# compute dynamic x-ticks at each 24 h interval
total_hours = days * 24
xticks = np.arange(0, total_hours + 1, 24)
xticklabels = [f"{int(h)}h" for h in xticks]

fig, axes = plt.subplots(1, 3, figsize=(15, 3.5), sharex=True)

# Energy Price
sns.lineplot(data=df2, x="sim_hour", y="price_per_kwh", hue="datacenter", ax=axes[0])
axes[0].set_title(f"Energy Price over First {days} Days")
axes[0].set_ylabel("USD/kWh")
axes[0].set_xlabel("Simulation Hours")
axes[0].set_xticks(xticks)
axes[0].set_xticklabels(xticklabels)
axes[0].grid(True)
axes[0].set_xlim(0, total_hours)

# Carbon Intensity
sns.lineplot(data=df2, x="sim_hour", y="ci", hue="datacenter", ax=axes[1])
axes[1].set_title(f"Carbon Intensity over First {days} Days")
axes[1].set_ylabel("gCO₂/kWh")
axes[1].set_xlabel("Simulation Hours")
axes[1].set_xticks(xticks)
axes[1].set_xticklabels(xticklabels)
axes[1].grid(True)

# Tasks Assigned
sns.lineplot(data=df2, x="sim_hour", y="running_tasks", hue="datacenter", ax=axes[2])
axes[2].set_title(f"Tasks Assigned over First {days} Days")
axes[2].set_ylabel("Number of Tasks")
axes[2].set_xlabel("Simulation Hours")
axes[2].set_xticks(xticks)
axes[2].set_xticklabels(xticklabels)
axes[2].grid(True)

# remove individual legends
for ax in axes:
    ax.get_legend().remove()

# mapping from DC IDs to human-readable locations
dc_label_map = {
    "DC1": "California, US",
    "DC2": "Germany",
    "DC3": "Santiago, CL",
    "DC4": "Singapore",
    "DC5": "New South Wales, AU"
}

# add a single legend on the right, with remapped labels
handles, labels = axes[0].get_legend_handles_labels()
labels = [dc_label_map.get(lbl, lbl) for lbl in labels]
fig.legend(handles, labels, title="Datacenter Location", ncols=5,
           bbox_to_anchor=(0.15, -0.06), loc="center left")

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

# Save as pdf
# fig.savefig("assets/figures/energy_price_carbon_intensity_tasks_assigned.pdf", bbox_inches='tight')


#%%
#%% Figure: Transmission Cost, Delayed Tasks, CPU & GPU Utilization Over Time
import numpy as np
#%% Figure: Transmission Cost, Delayed Tasks, CPU & GPU Utilization over First N Days
days = 5
steps_per_day = 96
max_steps = days * steps_per_day

# prepare data for first N days
df2 = df[df["timestep"] < max_steps].copy()
df2["sim_hour"] = df2["timestep"] * 0.25  # 15-min = 0.25h
sim_hours = np.arange(max_steps) * 0.25

# dynamic x-ticks at each 24h
total_hours = days * 24
xticks = np.arange(0, total_hours + 1, 24)
xticklabels = [f"{int(h)}h" for h in xticks]

fig, axes = plt.subplots(2, 2, figsize=(12, 4), sharex=True)

# (a) Transmission Cost
axes[0, 0].plot(sim_hours, common_info_list[:max_steps])
axes[0, 0].set_title("Transmission Cost Over Time")
axes[0, 0].set_ylabel("USD")
axes[0, 0].set_xticks(xticks)
axes[0, 0].set_xticklabels(xticklabels)
axes[0, 0].grid(True)
axes[0, 0].set_xlim(0, total_hours)

# (b) Delayed Tasks
axes[1, 0].plot(sim_hours, delayed_task_counts[:max_steps], color="orange")
axes[1, 0].set_title("Delayed Tasks Over Time")
axes[1, 0].set_ylabel("Count")
axes[1, 0].set_xlabel("Simulation Hours")
axes[1, 0].set_xticks(xticks)
axes[1, 0].set_xticklabels(xticklabels)
axes[1, 0].grid(True)

# (c) CPU Utilization per DC
sns.lineplot(data=df2, x="sim_hour", y="cpu_util", hue="datacenter", ax=axes[0, 1])
axes[0, 1].set_title("CPU Utilization Over Time")
axes[0, 1].set_ylabel("CPU Util (%)")
axes[0, 1].set_xticks(xticks)
axes[0, 1].set_xticklabels(xticklabels)
axes[0, 1].grid(True)

# capture legend handles/labels before removal
handles, labels = axes[0, 1].get_legend_handles_labels()

# (d) GPU Utilization per DC
sns.lineplot(data=df2, x="sim_hour", y="gpu_util", hue="datacenter", ax=axes[1, 1])
axes[1, 1].set_title("GPU Utilization Over Time")
axes[1, 1].set_ylabel("GPU Util (%)")
axes[1, 1].set_xlabel("Simulation Hours")
axes[1, 1].set_xticks(xticks)
axes[1, 1].set_xticklabels(xticklabels)
axes[1, 1].grid(True)

# remove individual legends on CPU/GPU axes
axes[0, 1].get_legend().remove()
axes[1, 1].get_legend().remove()

# mapping DC IDs to medium-length labels
dc_label_map = {
    "DC1": "California (US)",
    "DC2": "Germany",
    "DC3": "Santiago (CL)",
    "DC4": "Singapore",
    "DC5": "NSW (AU)"
}
new_labels = [dc_label_map.get(lbl, lbl) for lbl in labels]

# add single legend on right
fig.legend(handles, new_labels, title="Datacenter Location", ncols=5,
           bbox_to_anchor=(0.05, -0.04), loc="center left")

plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.show()

# Save as pdf
# fig.savefig("assets/figures/transmission_cost_delayed_tasks_utilization.pdf", bbox_inches='tight')

#%%