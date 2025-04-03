#%%
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dc_env.agent_net import ActorNet
from dc_env.dc_scheduling_env import TaskSchedulingEnv
from simulation.datacenter_cluster_manager import DatacenterClusterManager

import datetime
import logging

#%%
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

log_path = f"logs/evaluation_{timestamp}.log"

logger = logging.getLogger("eval_logger")
logger.setLevel(logging.INFO)  # File handler will capture INFO+

# === File handler (full log)
file_handler = logging.FileHandler(log_path, mode="w")
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
    simulation_year = 2023
    simulated_month = 8
    init_day = 1
    init_hour = 5
    init_minute = 0

    start_time = datetime.datetime(simulation_year, simulated_month, init_day, init_hour, init_minute, tzinfo=datetime.timezone.utc)
    end_time = start_time + datetime.timedelta(days=7)
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)

    datacenter_configs = [
        {
            'location': 'US-NY-NYIS', 'dc_id': 1, 'agents': [], 'timezone_shift': -5,
            'dc_config_file': 'dc_config.json', 'weather_file': None, 'cintensity_file': None,
            'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv', 'month': simulated_month,
            'datacenter_capacity_mw': 1.5, 'max_bat_cap_Mw': 3.0, 'days_per_episode': 30,
            'network_cost_per_gb': 0.08, 'total_cpus': 5000, 'total_gpus': 700,
            'total_mem': 5000, 'population_weight': 0.25,
        },
        {
            'location': 'DE-LU', 'dc_id': 2, 'agents': [], 'timezone_shift': 1,
            'dc_config_file': 'dc_config.json', 'weather_file': None, 'cintensity_file': None,
            'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv', 'month': simulated_month,
            'datacenter_capacity_mw': 1.2, 'max_bat_cap_Mw': 2.5, 'days_per_episode': 30,
            'network_cost_per_gb': 0.07, 'total_cpus': 5000, 'total_gpus': 700,
            'total_mem': 5000, 'population_weight': 0.22,
        },
        {
            'location': 'ZA', 'dc_id': 3, 'agents': [], 'timezone_shift': 2,
            'dc_config_file': 'dc_config.json', 'weather_file': None, 'cintensity_file': None,
            'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv', 'month': simulated_month,
            'datacenter_capacity_mw': 1.0, 'max_bat_cap_Mw': 2.0, 'days_per_episode': 30,
            'network_cost_per_gb': 0.06, 'total_cpus': 5000, 'total_gpus': 700,
            'total_mem': 5000, 'population_weight': 0.13,
        },
        {
            'location': 'SG', 'dc_id': 4, 'agents': [], 'timezone_shift': 8,
            'dc_config_file': 'dc_config.json', 'weather_file': None, 'cintensity_file': None,
            'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv', 'month': simulated_month,
            'datacenter_capacity_mw': 1.8, 'max_bat_cap_Mw': 3.5, 'days_per_episode': 30,
            'network_cost_per_gb': 0.09, 'total_cpus': 5000, 'total_gpus': 700,
            'total_mem': 5000, 'population_weight': 0.25,
        },
        {
            'location': 'AU-NSW', 'dc_id': 5, 'agents': [], 'timezone_shift': 11,
            'dc_config_file': 'dc_config.json', 'weather_file': None, 'cintensity_file': None,
            'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv', 'month': simulated_month,
            'datacenter_capacity_mw': 1.4, 'max_bat_cap_Mw': 2.8, 'days_per_episode': 30,
            'network_cost_per_gb': 0.10, 'total_cpus': 5000, 'total_gpus': 700,
            'total_mem': 5000, 'population_weight': 0.15,
        }
    ]
    tasks_file_path = "data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl"
    # tasks_file_path = "data/workload/alibaba_2020_dataset/result_df_cropped.pkl"


    cluster_manager = DatacenterClusterManager(
        config_list=datacenter_configs,
        simulation_year=simulation_year,
        init_day=int(simulated_month*30.5),
        init_hour=init_hour,
        strategy="manual_rl",
        # strategy="lowest_price",
        tasks_file_path=tasks_file_path,
        shuffle_datacenter_order=not eval_mode  # shuffle only during training
    )
    
    cluster_manager.logger = logger
    env = TaskSchedulingEnv(
        cluster_manager=cluster_manager,
        start_time=start_time,
        end_time=end_time,
        carbon_price_per_kg=0.1
    )
    return env



# Load trained actor model
checkpoint_path = "checkpoints/train_20250402_225152/checkpoint_step_1210000.pth"  # Adjust path
env = make_eval_env()
obs, _ = env.reset(seed=123)
obs_dim = env.observation_space.shape[0]
# Detect the strategy (manual_rl = agent, else = RBC)
use_actor = env.cluster_manager.strategy == "manual_rl"
if use_actor:
    logger.info("Using trained actor model for evaluation.")
    actor = ActorNet(obs_dim, env.num_dcs, hidden_dim=64)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
else:
    logger.info("Using RBC for evaluation.")
    

infos_list = []
rewards = []
steps = 7 * 96  # 7 days of 15-min intervals

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
    else:
        # If RBC is used, actions must be empty → internal logic assigns them
        actions = []

    obs, reward, done, truncated, info = env.step(actions)
    infos_list.append(info["datacenter_infos"])
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
    "running_tasks": "sum",
    "pending_tasks": "mean",
    "sla_met": "sum",
    "sla_violated": "sum"
}).reset_index()

summary["SLA Violation Rate (%)"] = (
    summary["sla_violated"] / (summary["sla_met"] + summary["sla_violated"]).replace(0, np.nan)
) * 100

summary = summary.round(2)

summary.columns = [
    "Datacenter",
    "Total Energy Cost (USD)",
    "Total Energy (kWh)",
    "Total CO₂ (kg)",
    "Avg Price (USD/kWh)",
    "Avg Carbon Intensity (gCO₂/kWh)",
    "Avg Weather (°C)",
    "Avg CPU Util (%)",
    "Avg GPU Util (%)",
    "Avg MEM Util (%)",
    "Total Running Tasks",
    "Avg Pending Tasks",
    "SLA Met",
    "SLA Violated",
    "SLA Violation Rate (%)"
]

summary

#%%
import seaborn as sns
import matplotlib.pyplot as plt

#%%
# Energy price per kWh plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="timestep", y="price_per_kwh", hue="datacenter")
plt.title("Energy Price per kWh over Time")
plt.xlabel("Timestep")
plt.ylabel("USD/kWh")
plt.grid(True)
plt.show()

#%% 
# Plot of "Total Running Tasks"
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="timestep", y="running_tasks", hue="datacenter")
plt.title("Total Running Tasks per Datacenter over Time")
plt.xlabel("Timestep")
plt.ylabel("Number of Tasks")
plt.grid(True)
plt.show()

#%% Plot of Tasks Assigned per Datacenter over Time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="timestep", y="tasks_assigned", hue="datacenter")
plt.title("Tasks Assigned per Datacenter over Time")
plt.xlabel("Timestep")
plt.ylabel("Number of Assigned Tasks")
plt.grid(True)
plt.show()

#%%
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="timestep", y="energy_cost", hue="datacenter")
plt.title("Energy Cost per Datacenter over Time")
plt.xlabel("Timestep")
plt.ylabel("USD")
plt.grid(True)
plt.show()


#%%
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="timestep", y="carbon_kg", hue="datacenter")
plt.title("Carbon Emissions (kg) per Datacenter over Time")
plt.ylabel("kg CO₂")
plt.grid(True)
plt.show()


#%% Carbon intensity plot on each location (Datacenter)
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="timestep", y="ci", hue="datacenter")
plt.title("Carbon Intensity (gCO₂/kWh) over Time")
plt.xlabel("Timestep")
plt.ylabel("gCO₂/kWh")
plt.grid(True)
plt.show()


#%%
import matplotlib.pyplot as plt
import seaborn as sns

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
plt.show()


#%%