# tests/test_marl_env_random_loop.py

# %%
# --- [Cell 1] Setup: Imports and Path Configuration ---
import sys
import os
import time
import numpy as np
import pandas as pd
import logging # Import standard logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set global plotting style
sns.set_theme(style="whitegrid")

# Add project root to path
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = os.getcwd()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Import the new MARL environment and its dependencies
from envs.sustaincluster_ma_env import SustainClusterMAEnv
from simulation.cluster_manager_ma import DatacenterClusterManagerMA
from utils.config_loader import load_yaml
from rewards.predefined.composite_reward import CompositeReward
from utils.config_logger import setup_logger # <<< IMPORT THE LOGGER SETUP

print("Setup complete. Imports successful.")

# %%
# --- [Cell 2] Logger and Environment Instantiation ---

# --- Logger Setup ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/test_marl_env_random_loop_{timestamp}"
logger = setup_logger(log_dir, enable_logger=True)
if logger:
    logger.propagate = False # Prevent double-logging in some setups
    logger.info("MARL Environment Test Logger Initialized.")
else:
    print("Logger is disabled.")

# --- Configuration ---
dc_config_path = "configs/env/datacenters.yaml"
reward_config_path = "configs/env/reward_config.yaml"
datacenters_config_list = load_yaml(dc_config_path)["datacenters"]

N_DCS_TO_TEST = 3
test_config_list = datacenters_config_list[:N_DCS_TO_TEST]

for config in test_config_list:
    config['simulation_year'] = 2023
    config['agents'] = []
    config['evaluation'] = False

MAX_OPTIONS = 5
CLOUD_PROVIDER = "aws"
WORKLOAD_PATH = "data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl"

# Initialize Cluster Manager with the logger
logger.info("Initializing DatacenterClusterManagerMA...")
cluster_manager = DatacenterClusterManagerMA(
    config_list=test_config_list,
    simulation_year=2023,
    tasks_file_path=WORKLOAD_PATH,
    cloud_provider=CLOUD_PROVIDER,
    max_total_options=MAX_OPTIONS,
    logger=logger # <<< PASS LOGGER
)

# Initialize Reward Function
reward_cfg = load_yaml(reward_config_path)["reward"]
reward_fn = CompositeReward(components=reward_cfg["components"], normalize=False)

# Initialize the Gym Environment with the logger
logger.info("Initializing SustainClusterMAEnv...")
start_time = pd.Timestamp("2023-01-01 00:00:00", tz='UTC')
end_time = start_time + pd.Timedelta(days=2) # Run for 2 simulated days

env = SustainClusterMAEnv(
    cluster_manager_ma=cluster_manager,
    start_time=start_time,
    end_time=end_time,
    reward_fn=reward_fn,
    logger=logger # <<< PASS LOGGER
)

logger.info("\n✅ Environment created successfully.")
logger.info(f"Number of possible agents: {len(env.possible_agents)}")


# %%
# --- [Cell 3] The Random Action Test Loop ---

logger.info("\n--- Starting Random Action Test Loop ---")

# Configuration for the test run
num_episodes = 1
max_steps_per_episode = 24 * 4 * 7 # 2 simulated days (96 steps/day)

# List to store data for plotting at each step
plotting_data = []

for episode in range(num_episodes):
    logger.info(f"\n--- Episode {episode + 1}/{num_episodes} ---")
    
    # Reset the environment
    start_time_reset = time.time()
    observations, info = env.reset(seed=42 + episode)
    logger.info(f"Environment reset in {time.time() - start_time_reset:.4f} seconds.")

    assert isinstance(observations, dict), "Observations should be a dictionary"
    logger.info(f"Initial number of active agents: {len(env.agents)}")
    assert len(observations) == len(env.agents), "Observation dict should have one entry per agent"

    # Loop for the duration of the episode
    for step in range(max_steps_per_episode):
        if not env.agents: # Check if the episode has terminated
            logger.info(f"Episode terminated early at step {step}. All agents are done.")
            break

        # --- Generate Random Actions for all active agents ---
        actions = {}
        for agent_id in env.agents:
            if "manager" in agent_id:
                # For managers, we must only sample from the VALID options.
                # The number of valid options is the number of datacenters.
                num_valid_options = env.cluster_manager_ma.num_dcs
                actions[agent_id] = np.random.randint(0, num_valid_options)
            else: # For workers
                # Worker action space is simple (0 or 1), so .sample() is fine.
                actions[agent_id] = env.action_space(agent_id).sample()
        
        # Take a step in the environment
        start_time_step = time.time()
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        step_duration = time.time() - start_time_step
        
        # --- Collect Data for Plotting ---
        step_data = {"timestep": step}
        # Iterate through each DC node to get its state
        for dc_id, node in env.cluster_manager_ma.nodes.items():
            # Resource Utilization
            total_cores = node.physical_dc_model.total_cores
            avail_cores = node.physical_dc_model.available_cores
            step_data[f"dc_{dc_id}_cpu_util"] = (total_cores - avail_cores) / total_cores * 100 if total_cores > 0 else 0
            
            # GPU Utilization
            total_gpus = node.physical_dc_model.total_gpus
            avail_gpus = node.physical_dc_model.available_gpus
            step_data[f"dc_{dc_id}_gpu_util"] = (total_gpus - avail_gpus) / total_gpus * 100 if total_gpus > 0 else 0
            
            # Running and Queued Tasks
            step_data[f"dc_{dc_id}_running_tasks"] = len(node.physical_dc_model.running_tasks)
            # The "deferred" tasks are those in the worker's queue
            step_data[f"dc_{dc_id}_deferred_tasks"] = len(node.worker_commitment_queue)
        
        plotting_data.append(step_data)
        
        # --- Log Status ---
        # The detailed step-by-step logs now happen inside the simulation classes.
        # Here we log a high-level summary of the step outcome.
        
        log_message = (
            f"[Step {step+1}/{max_steps_per_episode}] | Step Duration: {step_duration:.4f}s | "
            f"Global Reward: {next(iter(rewards.values()), 'N/A'):.3f} | "
            f"Terminated: {terminations['__all__']}"
        )
        logger.info(log_message)
        # Optional: Print to console for real-time feedback
        if (step + 1) % 10 == 0 or step == max_steps_per_episode - 1:
            print(log_message)
            
        # Update observations for the next loop
        observations = next_obs
        
        if terminations["__all__"]:
             break
        
    logger.info(f"\n--- Episode {episode + 1} Finished ---")

logger.info("\n✅ Random action loop completed without crashing.")
print(f"\nTest finished. Detailed logs saved in: {log_dir}")

# %%
# --- [Cell 4] Plotting the Results ---

print("\n--- Generating Plots ---")

# Convert collected data to a DataFrame
plot_df = pd.DataFrame(plotting_data)
plot_df.set_index("timestep", inplace=True)

# Create a figure with 3 rows of subplots
fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
fig.suptitle("MARL Environment Dynamics with Random Actions", fontsize=16)

# --- Plot 1: CPU Utilization ---
for dc_id in env.cluster_manager_ma.nodes.keys():
    axes[0].plot(plot_df.index, plot_df[f"dc_{dc_id}_cpu_util"], label=f"DC {dc_id} CPU Util")
axes[0].set_title("Resource Utilization per Datacenter")
axes[0].set_ylabel("CPU Utilization (%)")
axes[0].legend(loc="upper right")
axes[0].grid(True, linestyle='--', alpha=0.6)

# --- Plot 1: GPU Utilization ---
for dc_id in env.cluster_manager_ma.nodes.keys():
    axes[0].plot(plot_df.index, plot_df[f"dc_{dc_id}_gpu_util"], label=f"DC {dc_id} GPU Util", linestyle='--')
axes[0].set_title("Resource Utilization per Datacenter")
axes[0].set_ylabel("GPU Utilization (%)")
axes[0].legend(loc="upper right")
axes[0].grid(True, linestyle='--', alpha=0.6)

# --- Plot 2: Running Tasks ---
for dc_id in env.cluster_manager_ma.nodes.keys():
    axes[1].plot(plot_df.index, plot_df[f"dc_{dc_id}_running_tasks"], label=f"DC {dc_id} Running")
axes[1].set_title("Active (Running) Tasks per Datacenter")
axes[1].set_ylabel("Number of Running Tasks")
axes[1].legend(loc="upper right")
axes[1].grid(True, linestyle='--', alpha=0.6)

# --- Plot 3: Deferred Tasks (in Worker Queue) ---
for dc_id in env.cluster_manager_ma.nodes.keys():
    axes[2].plot(plot_df.index, plot_df[f"dc_{dc_id}_deferred_tasks"], label=f"DC {dc_id} Deferred")
axes[2].set_title("Locally Deferred Tasks (in Worker Queue) per Datacenter")
axes[2].set_ylabel("Number of Deferred Tasks")
axes[2].set_xlabel("Simulation Timestep")
axes[2].legend(loc="upper right")
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
plt.show()
#%%