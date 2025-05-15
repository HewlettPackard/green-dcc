# generate_table1_metrics.py

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns # Keep imports if needed for other analysis
import matplotlib.pyplot as plt # Keep imports if needed

# --- Local Imports ---
# Adjust paths as needed
from rl_components.agent_net import ActorNet
from envs.task_scheduling_env import TaskSchedulingEnv
from simulation.cluster_manager import DatacenterClusterManager
from utils.config_loader import load_yaml
from rewards.predefined.composite_reward import CompositeReward
from utils.config_logger import setup_logger # Assuming setup_logger exists

import datetime
import logging
import random
import os
import copy # For deep copying config dicts

from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Configuration ---
CONFIG_DIR = "configs/env"
SIM_CONFIG_PATH = os.path.join(CONFIG_DIR, "sim_config.yaml")
DC_CONFIG_PATH = os.path.join(CONFIG_DIR, "datacenters.yaml")
REWARD_CONFIG_PATH = os.path.join(CONFIG_DIR, "reward_config.yaml")
# Path to the standard trained RL agent (e.g., SAC trained on default multi-objective)
DEFAULT_RL_CHECKPOINT_PATH = "../checkpoints/train_20250509_232500/best_checkpoint.pth" # ADJUST THIS

EVALUATION_DURATION_DAYS = 7 # Use 1 month for evaluation
NUM_SEEDS = 10 # Number of random seeds to run for each controller
SEEDS = [i * 10 for i in range(NUM_SEEDS)] # Example: [0, 10, 20, 30, 40]

# Define controllers for Table 1
CONTROLLERS_TO_EVALUATE = [
    {
    "name": "RL (Geo+Time)",
    "strategy": "manual_rl",
    "checkpoint": DEFAULT_RL_CHECKPOINT_PATH,
    "disable_defer": False,
    "local_only": False,
    "is_rl": True,
    },
    {
    "name": "RL (Geo Only)",
    "strategy": "manual_rl",
    "checkpoint": DEFAULT_RL_CHECKPOINT_PATH,
    "disable_defer": True, # Force assignment (no action 0)
    "local_only": False,
    "is_rl": True,
    },
    {
    "name": "RL (Time Only)",
    "strategy": "manual_rl",
    "checkpoint": DEFAULT_RL_CHECKPOINT_PATH,
    "disable_defer": False,
    "local_only": True, # Force assignment only to origin DC (or defer)
    "is_rl": True,
    },
    {
        "name": "RBC (Lowest Carbon)",
        "strategy": "lowest_carbon",
        "is_rl": False,
    },
    {
        "name": "RBC (Lowest Price)",
        "strategy": "lowest_price",
        "is_rl": False,
    },
    {
        "name": "RBC (Round Robin)",
        "strategy": "round_robin",
        "is_rl": False,
    },
    {
        "name": "RBC (Most Available)",
        "strategy": "most_available",
        "is_rl": False,
    },
    {
        "name": "RBC (Local Only)",
        "strategy": "local_only",
        "is_rl": False,
    },
    {
        "name": "RBC (Random)",
        "strategy": "random",
        "is_rl": False,
    }
]

# --- Logger Setup ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/table1_eval_{timestamp}"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"evaluation_table1_{timestamp}.log")
results_csv_path = os.path.join(log_dir, f"results_table1_{timestamp}.csv")

logger = logging.getLogger("table1_eval_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# --- Environment Creation Function (Modified) ---
def make_eval_env(strategy, base_sim_config, base_dc_config, base_reward_config, eval_duration_days, seed, eval_mode=True):
    """Creates the evaluation environment with specified strategy and duration."""

    sim_cfg = copy.deepcopy(base_sim_config) # Avoid modifying original dict
    sim_cfg["strategy"] = strategy # Override strategy
    sim_cfg["duration_days"] = eval_duration_days # Override duration

    # Ensure start date remains consistent across runs unless randomized intentionally
    start = pd.Timestamp(datetime.datetime(sim_cfg["year"], sim_cfg["month"], sim_cfg["init_day"],
                                           sim_cfg["init_hour"], 0, tzinfo=datetime.timezone.utc))
    end = start + datetime.timedelta(days=sim_cfg["duration_days"])

    # Use deep copies if configs might be modified internally, otherwise shallow is fine
    dc_cfg = copy.deepcopy(base_dc_config)
    reward_cfg = copy.deepcopy(base_reward_config)

    cluster = DatacenterClusterManager(
        config_list=dc_cfg,
        simulation_year=sim_cfg["year"],
        init_day=int(sim_cfg["month"] * 30.5), # Consider if start day should also use seed?
        init_hour=sim_cfg["init_hour"],
        strategy=sim_cfg["strategy"],
        tasks_file_path=sim_cfg["workload_path"],
        shuffle_datacenter_order=not eval_mode, # No shuffle during evaluation
        cloud_provider=sim_cfg["cloud_provider"],
        logger=logger # Use the shared logger
    )

    # Use the default multi-objective reward for Table 1 comparisons
    reward_fn = CompositeReward(
        components=reward_cfg["components"],
        normalize=reward_cfg.get("normalize", False)
    )

    env = TaskSchedulingEnv(
        cluster_manager=cluster,
        start_time=start,
        end_time=end,
        reward_fn=reward_fn,
        writer=None # No Tensorboard writer needed for eval runs
    )
    env.reset(seed=seed) # Reset with seed immediately
    return env

# --- Single Evaluation Run Function ---
def run_single_evaluation(controller_config, seed, base_sim_cfg, base_dc_cfg, base_reward_cfg, bar_position):
    """Runs one full evaluation for a given controller and seed."""
    logger.info(f"--- Running Eval: {controller_config['name']} | Seed: {seed} ---")

    strategy = controller_config['strategy']
    env = make_eval_env(strategy, base_sim_cfg, base_dc_cfg, base_reward_cfg, EVALUATION_DURATION_DAYS, seed)

    # --- Load Actor if RL ---
    actor = None
    if controller_config['is_rl']:
        checkpoint_path = controller_config['checkpoint']
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found for {controller_config['name']}: {checkpoint_path}")
            return None # Skip this run

        # Determine observation dim dynamically if possible, or hardcode/load from config
        # For now, reset env once to get shape (might be slightly inefficient)
        temp_obs, _ = env.reset(seed=seed)
        if not temp_obs: # Handle empty initial obs
             temp_obs, _, _, _, _ = env.step([])
        if not temp_obs: logger.warning(f"Env yields empty obs even after step for seed {seed}"); return None
        obs_dim = len(temp_obs[0]) # Get dim from first task's obs vector
        act_dim = env.num_dcs + 1

        actor = ActorNet(obs_dim, act_dim, hidden_dim=64)
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            actor.load_state_dict(checkpoint["actor_state_dict"])
            actor.eval()
            logger.info(f"Loaded actor from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load actor checkpoint {checkpoint_path}: {e}")
            return None

    # --- Simulation Loop ---
    obs, _ = env.reset(seed=seed) # Reset again to ensure consistent start
    num_steps = EVALUATION_DURATION_DAYS * 24 * 4 # 15-min intervals
    all_infos = [] # Store info dict from each step
    total_deferred_tasks_count = 0 # <<<--- Initialize deferred count

    for step in range(num_steps):
        actions = [] # Default for RBCs or empty obs
        step_deferred_count = 0 # <<<--- Deferred count for this step

        if len(obs) > 0:
            if controller_config['is_rl'] and actor is not None:
                # --- RL Agent Action Selection ---
                obs_tensor = torch.FloatTensor(obs)
                with torch.no_grad():
                    logits = actor(obs_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    sampled_actions = dist.sample().numpy().tolist()

                # --- Apply Action Modifications ---
                actions = []
                for i, action in enumerate(sampled_actions):
                    modified_action = action
                    # 1. Disable Defer?
                    if controller_config.get('disable_defer', False) and modified_action == 0:
                        # If defer is disabled, assign the task to the DC where it was created
                        modified_action = int(obs[i][4])
                        # print(f"Defer disabled for task {i}, assigning to origin DC: {modified_action}")

                    # 2. Local Only?
                    if controller_config.get('local_only', False) and modified_action > 0:
                        # Need task origin ID. Assuming it's feature index 4 after time features
                        task_origin_id = int(obs[i][4]) # Ensure correct index!
                        if modified_action != task_origin_id: 
                            modified_action = task_origin_id
                        # print(f"Local only for task {i}, assigning to origin DC: {modified_action}")
                        
                    actions.append(modified_action)
                    if modified_action == 0: # Count deferrals *after* modification
                        step_deferred_count += 1
                # --- End Modifications ---
            # Else: actions remains [] for RBCs
            else:
                actions = []
            
        total_deferred_tasks_count += step_deferred_count # <<<--- Accumulate deferred count
        # Step environment
        # try:
        obs, _, done, truncated, info = env.step(actions)
        all_infos.append(info) # Store the entire info dict
        if done or truncated:
            #logger.info(f"Episode finished early at step {step+1} for seed {seed}.")
            # break # Stop if the environment finishes based on Time_Manager duration
                # For fixed duration eval, we might want to continue if reset logic handles it,
                # but safer to break if TimeManager dictates the end. Let's break.
                if done: # Only break if manager_done triggered it
                    logger.info(f"Simulation duration reached at step {step+1} for seed {seed}.")
                    break
                else: # Reset if truncated for other reasons? Typically not needed in fixed eval.
                    # obs, _ = env.reset(seed=seed) # Optional reset, likely not needed
                    pass


        # except Exception as e:
        #     logger.error(f"Error during env.step at step {step+1}, seed {seed}: {e}")
        #     raise e # Raise to stop execution if critical error occurs

    # --- Aggregate Metrics ---
    if not all_infos:
        logger.warning(f"No info collected for {controller_config['name']}, seed {seed}.")
        return None

    total_energy_cost = 0.0
    total_energy_kwh = 0.0
    total_carbon_kg = 0.0
    total_sla_met = 0
    total_sla_violated = 0
    total_trans_cost = 0.0
    total_water_usage_L = 0.0
    total_ite_energy_kwh = 0.0 # For PUE calculation
    cpu_utils = []
    gpu_utils = []
    mem_utils = []

    for step_info in all_infos:
        total_trans_cost += step_info.get("transmission_cost_total_usd", 0.0)
        for dc_name, dc_info in step_info.get("datacenter_infos", {}).items():
            common = dc_info.get("__common__", {})
            sla = common.get("__sla__", {})
            total_energy_cost += common.get("energy_cost_USD", 0.0)
            total_energy_kwh += common.get("energy_consumption_kwh", 0.0)
            total_carbon_kg += common.get("carbon_emissions_kg", 0.0)
            total_sla_met += sla.get("met", 0)
            total_sla_violated += sla.get("violated", 0)
            # Collect utils for averaging later
            cpu_utils.append(common.get("cpu_util_percent", 0.0))
            gpu_utils.append(common.get("gpu_util_percent", 0.0))
            mem_utils.append(common.get("mem_util_percent", 0.0))
            total_water_usage_L += dc_info.get("agent_dc", {}).get("dc_water_usage", 0.0) # Check key name
            
            ite_power_kw = dc_info.get('agent_dc',{}).get('dc_ITE_total_power_kW')
            total_ite_energy_kwh += ite_power_kw * (15/60.0) # kW * 0.25 hours = kWh


    # Calculate averages and rates
    avg_cpu_util = np.mean(cpu_utils) if cpu_utils else 0.0
    avg_gpu_util = np.mean(gpu_utils) if gpu_utils else 0.0
    avg_mem_util = np.mean(mem_utils) if mem_utils else 0.0
    total_tasks_finished = total_sla_met + total_sla_violated
    sla_violation_rate = (total_sla_violated / total_tasks_finished * 100) if total_tasks_finished > 0 else 0.0
    average_pue = total_energy_kwh / total_ite_energy_kwh if total_ite_energy_kwh > 0 else np.nan # Use NaN if no IT energy
    
    # *** Convert total water usage to m³ ***
    total_water_usage_m3 = total_water_usage_L / 1000.0

    results = {
        "Controller": controller_config['name'],
        "Seed": seed,
        "Total Energy Cost (USD)": total_energy_cost,
        "Total Energy (kWh)": total_energy_kwh,
        "Total CO2 (kg)": total_carbon_kg,
        "Total SLA Violations": total_sla_violated,
        "SLA Violation Rate (%)": sla_violation_rate,
        "Total Transmission Cost (USD)": total_trans_cost,
        "Avg CPU Util (%)": avg_cpu_util,
        "Avg GPU Util (%)": avg_gpu_util,
        "Avg MEM Util (%)": avg_mem_util,
        "Total Water Usage (m3)": total_water_usage_m3,
        "Average PUE": average_pue,
        "Total Tasks Deferred": total_deferred_tasks_count, # Use accumulated count
    }
    logger.info(f"--- Finished Eval: {controller_config['name']} | Seed: {seed} ---")
    return results


# --- Main: load configs once, then parallelize seeds per controller ---
if __name__ == "__main__":
    logger.info("Starting Table 1 Evaluations...")
    logger.info(f"Duration: {EVALUATION_DURATION_DAYS} days | Seeds: {SEEDS}")

    base_sim_config    = load_yaml(SIM_CONFIG_PATH)["simulation"]
    base_dc_config     = load_yaml(DC_CONFIG_PATH)["datacenters"]
    base_reward_config = load_yaml(REWARD_CONFIG_PATH)["reward"]

    # build flat list of all tasks
    tasks = [
        (ctrl, seed)
        for ctrl in CONTROLLERS_TO_EVALUATE
        for seed in SEEDS
    ]

    all_results = []
    max_workers = min(len(tasks), os.cpu_count())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, (ctrl, seed) in enumerate(tasks):
            fut = executor.submit(
                run_single_evaluation,
                ctrl,
                seed,
                base_sim_config,
                base_dc_config,
                base_reward_config,
                idx + 1
            )
            futures[fut] = (ctrl["name"], seed)

        for fut in as_completed(futures):
            name, seed = futures[fut]
            try:
                res = fut.result()
                if res:
                    all_results.append(res)
                else:
                    logger.warning(f"no result {name} seed {seed}")
            except Exception as e:
                logger.error(f"error {name} seed {seed}: {e}")

    if not all_results:
        logger.error("nothing collected, exiting")
        exit()

    # --- Process Results ---
    results_df = pd.DataFrame(all_results)

    # Calculate Mean and Std Dev
    summary_df = results_df.groupby("Controller").agg(
        {
            "Total Energy Cost (USD)": ['mean', 'std'],
            "Total Energy (kWh)": ['mean', 'std'],
            "Total CO2 (kg)": ['mean', 'std'],
            "Total SLA Violations": ['mean', 'std'],
            "SLA Violation Rate (%)": ['mean', 'std'],
            "Total Transmission Cost (USD)": ['mean', 'std'],
            "Avg CPU Util (%)": ['mean', 'std'],
            "Avg GPU Util (%)": ['mean', 'std'],
            "Avg MEM Util (%)": ['mean', 'std'],
            "Total Water Usage (m3)": ['mean', 'std'], # <<<--- UPDATED KEY
            "Average PUE": ['mean', 'std'],          # <<<--- ADDED
            "Total Tasks Deferred": ['mean', 'std'],   # <<<--- ADDED
        }
    )
    
    # Flatten multi-index columns
    summary_df.columns = ["_".join(col).strip() for col in summary_df.columns.values]
    summary_df = summary_df.reset_index()

    # Format output nicely
    formatted_summary_df = summary_df.copy()
    cols_to_drop = [] # Keep track of original cols to drop later

    for col in summary_df.columns:
        if col == "Controller":
            continue # Skip the controller name column

        if '_mean' in col:
            metric_base_name = col.replace('_mean', '')
            std_col = f"{metric_base_name}_std"

            # Check if the corresponding std column exists
            if std_col in summary_df.columns:
                # Format mean ± (std)
                mean_series = summary_df[col].map('{:.2f}'.format)
                std_series = summary_df[std_col].map('{:.2f}'.format)
                formatted_summary_df[metric_base_name] = mean_series + ' ± (' + std_series + ')'

                # Mark original mean and std columns for removal
                cols_to_drop.append(col)
                cols_to_drop.append(std_col)
            else:
                # Only mean exists, just format it and potentially rename
                formatted_summary_df[metric_base_name] = summary_df[col].map('{:.2f}'.format)
                if metric_base_name != col: # If renaming occurred
                     cols_to_drop.append(col)

        elif '_std' in col:
            # If we encounter a std column that wasn't paired with a mean,
            # it means it was already processed. We just need to ensure it's marked for dropping.
             if col not in cols_to_drop:
                 cols_to_drop.append(col)
            # Or handle it as an error/warning if this case shouldn't happen
            # logger.warning(f"Found standalone std column: {col}")

    # Drop all original _mean and _std columns at once after the loop
    # Use errors='ignore' in case a column was already implicitly dropped by renaming
    formatted_summary_df = formatted_summary_df.drop(columns=cols_to_drop, errors='ignore')


    logger.info("\n--- Aggregated Results Table 1 ---")
    logger.info("\n" + formatted_summary_df.to_string(index=False))

    # Save raw results and summary
    try:
        results_df.to_csv(results_csv_path.replace(".csv", "_raw.csv"), index=False)
        formatted_summary_df.to_csv(results_csv_path, index=False)
        logger.info(f"Raw results saved to {results_csv_path.replace('.csv', '_raw.csv')}")
        logger.info(f"Formatted summary saved to {results_csv_path}")
    except Exception as e:
        logger.error(f"Error saving results to CSV: {e}")

    print("\n--- Aggregated Results Table 1 ---")
    print(formatted_summary_df.to_string(index=False))
    print(f"\nFull logs saved to: {log_path}")
    print(f"CSV results saved to directory: {log_dir}")