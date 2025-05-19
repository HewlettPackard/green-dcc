#%%
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
sns.set_theme(style="whitegrid")
import yaml # For loading algo_config if needed
from utils.config_loader import load_yaml

# Import all network types
from rl_components.agent_net import ActorNet, AttentionActorNet # Add AttentionActorNet
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

def make_eval_env(sim_config_path="configs/env/sim_config.yaml", # Add paths as args
                  dc_config_path="configs/env/datacenters.yaml",
                  reward_config_path="configs/env/reward_config.yaml",
                  eval_mode=True):
    from rewards.predefined.composite_reward import CompositeReward

    sim_cfg_full = load_yaml(sim_config_path) # Load full sim_config
    sim_cfg = sim_cfg_full["simulation"]
    dc_cfg = load_yaml(dc_config_path)["datacenters"]
    reward_cfg = load_yaml(reward_config_path)["reward"]

    start = pd.Timestamp(datetime.datetime(sim_cfg["year"], sim_cfg["month"], sim_cfg["init_day"],
                                           sim_cfg["init_hour"], 0, tzinfo=datetime.timezone.utc))
    end = start + datetime.timedelta(days=sim_cfg["duration_days"])

    cluster = DatacenterClusterManager(
        config_list=dc_cfg,
        simulation_year=sim_cfg["year"],
        init_day=int(sim_cfg["month"] * 30.5 + sim_cfg["init_day"]), # Adjusted
        init_hour=sim_cfg["init_hour"],
        strategy=sim_cfg["strategy"], # This will determine if use_actor is True
        tasks_file_path=sim_cfg["workload_path"],
        shuffle_datacenter_order=not eval_mode, # No shuffle in eval
        cloud_provider=sim_cfg["cloud_provider"],
        logger=logger
    )
    reward_fn = CompositeReward(
        components=reward_cfg["components"],
        normalize=False, # Typically False for evaluation rewards
        freeze_stats_after_steps=reward_cfg.get("freeze_stats_after_steps", None)
    )
    env = TaskSchedulingEnv(
        cluster_manager=cluster,
        start_time=start,
        end_time=end,
        reward_fn=reward_fn,
        writer=None,
        sim_config=sim_cfg # Pass the simulation config dictionary
    )
    return env

# --- Configuration for Evaluation ---
SIM_CONFIG_PATH = "configs/env/sim_config.yaml"       # Path to sim_config used for this eval run
DC_CONFIG_PATH = "configs/env/datacenters.yaml"        # Path to datacenters config
REWARD_CONFIG_PATH = "configs/env/reward_config.yaml"  # Path to reward config
CHECKPOINT_PATH = "checkpoints/train_20250516_165831/checkpoint_step_835000.pth"  # <<<< ADJUST THIS PATH
# If evaluating RBC, ensure SIM_CONFIG_PATH's strategy is set, e.g., "lowest_carbon"
# If evaluating RL, ensure SIM_CONFIG_PATH's strategy is "manual_rl"

# --- Determine if evaluating RL agent or RBC based on sim_config ---
# Load sim_config once to decide the mode
temp_sim_cfg = load_yaml(SIM_CONFIG_PATH)["simulation"]
EVAL_STRATEGY = temp_sim_cfg.get("strategy", "manual_rl")
USE_RL_AGENT = (EVAL_STRATEGY == "manual_rl")

# --- Create Environment ---
# The env will now be created with single_action_mode from its sim_config
env = make_eval_env(sim_config_path=SIM_CONFIG_PATH,
                    dc_config_path=DC_CONFIG_PATH,
                    reward_config_path=REWARD_CONFIG_PATH,
                    eval_mode=True)

obs, _ = env.reset(seed=123) # Use a fixed seed for reproducible evaluation

# Get action/obs dim from the environment, which now respects single_action_mode
# and disable_defer_action for its action space.
# However, the *network's* output dimension depends on how it was trained.
single_action_mode_env = env.single_action_mode # From env instance
disable_defer_action_env = env.disable_defer_action # From env instance

actor = None
if USE_RL_AGENT:
    logger.info(f"Strategy: {EVAL_STRATEGY}. Using trained actor model for evaluation from: {CHECKPOINT_PATH}")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        extra_info = checkpoint.get("extra_info", {})

        # Get config from checkpoint to build the correct network
        ckpt_single_action_mode = extra_info.get("single_action_mode", False) # Default to multi if not in ckpt
        ckpt_use_attention = extra_info.get("use_attention", False)
        ckpt_obs_dim = extra_info.get("obs_dim")
        ckpt_act_dim_net = extra_info.get("act_dim") # This is the network's output dim

        if ckpt_obs_dim is None or ckpt_act_dim_net is None:
            raise ValueError("Checkpoint is missing 'obs_dim' or 'act_dim' in 'extra_info'. Please re-save checkpoints with these.")

        logger.info(f"Checkpoint info: single_action_mode={ckpt_single_action_mode}, use_attention={ckpt_use_attention}, obs_dim_net={ckpt_obs_dim}, act_dim_net={ckpt_act_dim_net}")

        # Validate env mode matches checkpoint's training mode for obs/action structure
        if single_action_mode_env != ckpt_single_action_mode:
            logger.warning(f"Environment single_action_mode ({single_action_mode_env}) "
                           f"differs from checkpoint training mode ({ckpt_single_action_mode}). "
                           f"Ensure evaluation sim_config matches training setup for observation structure.")
                           # The env will behave as per its sim_config, but the loaded actor was trained differently.

        if ckpt_use_attention and not ckpt_single_action_mode: # Attention only for multi-task
            # Assuming AttentionActorNet takes same constructor args as in train.py
            # Need to load algo_cfg that was used for training this specific checkpoint
            # For simplicity, assuming default attention params if not in checkpoint (BAD for general use)
            # TODO: Ideally, save relevant attn_cfg in checkpoint's extra_info
            logger.info("Loading AttentionActorNet.")
            actor = AttentionActorNet(obs_dim_per_task=ckpt_obs_dim, act_dim=ckpt_act_dim_net,
                                      embed_dim=extra_info.get("attn_embed_dim", 128),
                                      num_heads=extra_info.get("attn_num_heads", 4),
                                      num_attention_layers=extra_info.get("attn_num_layers",2),
                                      dropout=extra_info.get("attn_dropout",0.1)
                                      )
        else: # MLP Actor (either single_action_mode or multi-task without attention)
            logger.info("Loading ActorNet (MLP).")
            actor = ActorNet(obs_dim=ckpt_obs_dim, act_dim=ckpt_act_dim_net, hidden_dim=extra_info.get("hidden_dim", 64))

        actor.load_state_dict(checkpoint["actor_state_dict"])
        actor.eval()
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found at {CHECKPOINT_PATH}. Cannot evaluate RL agent.")
        USE_RL_AGENT = False # Fallback to not using actor
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}. Cannot evaluate RL agent.")
        USE_RL_AGENT = False
else:
    logger.info(f"Strategy: {EVAL_STRATEGY}. Using Rule-Based Controller for evaluation.")

# --- Evaluation Loop ---
infos_list = []
common_info_list = []
rewards_log = [] # Renamed from rewards
steps_to_eval = 7 * 96  # 7 days for evaluation
delayed_task_counts_log = []

for step in tqdm(range(steps_to_eval), desc=f"Evaluating {EVAL_STRATEGY}"):
    actions_to_env = [] # Default for RBC or no tasks

    if USE_RL_AGENT and actor is not None:
        if single_action_mode_env: # Env expects single action
            # Obs from env is already aggregated if single_action_mode_env is True
            # Ensure actor was trained for this aggregated obs_dim
            if len(obs) == 0 and obs.ndim == 0 : # Handle rare case of scalar 0 if env returns that for no tasks + single mode
                obs_for_actor = np.zeros(ckpt_obs_dim) # Use ckpt_obs_dim
            else:
                obs_for_actor = obs
            
            if obs_for_actor.shape[0] != ckpt_obs_dim:
                 logger.error(f"Obs dim mismatch for actor! Env obs_dim={obs_for_actor.shape[0]}, Ckpt train obs_dim={ckpt_obs_dim}")
                 # Potentially skip or use random action
                 actions_to_env = env.action_space.sample() if not disable_defer_action_env else np.random.randint(env.num_dcs)
            else:
                obs_tensor = torch.FloatTensor(obs_for_actor).unsqueeze(0) # Batch dim
                with torch.no_grad():
                    logits = actor(obs_tensor) # Expects [B, D_obs_agg]
                    # Actor outputs according to ckpt_act_dim_net
                    # Env's step() will interpret this based on disable_defer_action_env
                    actions_to_env = torch.distributions.Categorical(logits=logits).sample().item()
            
            # For logging deferral if single_action_mode
            # The actual deferral decision happens inside env.step based on disable_defer_action_env
            current_tasks_in_env = env.current_tasks # Tasks before step
            is_defer_decision = False
            if not disable_defer_action_env and actions_to_env == 0:
                is_defer_decision = True
            delayed_task_counts_log.append(len(current_tasks_in_env) if is_defer_decision and current_tasks_in_env else 0)

        else: # Multi-task mode for env and actor
            if len(obs) > 0: # obs is a list of per-task obs
                obs_tensor = torch.FloatTensor(obs)
                with torch.no_grad():
                    logits = actor(obs_tensor) # Expects [k_t, D_obs_per_task]
                    actions_list = torch.distributions.Categorical(logits=logits).sample().numpy().tolist()
                actions_to_env = actions_list
                # Log deferral for multi-task mode
                # The actions in actions_list are direct agent outputs (0 means defer if not disabled by agent training)
                delayed_count_this_step = 0
                if not disable_defer_action_env: # Only count '0' if defer was an option for the agent
                     delayed_count_this_step = sum(1 for a in actions_list if a == 0)
                delayed_task_counts_log.append(delayed_count_this_step)

            else: # No tasks
                actions_to_env = []
                delayed_task_counts_log.append(0)
    else: # RBC mode
        actions_to_env = [] # RBCs don't take actions from here
        delayed_task_counts_log.append(0) # RBCs typically don't defer in this framework

    obs, reward, done, truncated, info = env.step(actions_to_env)
    infos_list.append(info["datacenter_infos"])
    common_info_list.append(info["transmission_cost_total_usd"])
    rewards_log.append(reward)

    if done or truncated:
        logger.info(f"Evaluation episode finished at step {step+1}. Resetting.")
        obs, _ = env.reset(seed=123) # Keep seed fixed for multiple episodes if duration_days is short

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