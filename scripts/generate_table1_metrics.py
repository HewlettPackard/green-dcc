# generate_table1_metrics.py
import sys
import os

# Add the project root directory to the Python path
# This allows imports from rl_components, envs, etc.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Keep if you add plotting back later
from tqdm import tqdm
import seaborn as sns # Keep if you add plotting back later
sns.set_theme(style="whitegrid")
import yaml # For loading algo_config if needed

# --- Local Imports ---
from rl_components.agent_net import ActorNet, AttentionActorNet # Ensure both are importable
from envs.task_scheduling_env import TaskSchedulingEnv
from simulation.cluster_manager import DatacenterClusterManager
from utils.config_loader import load_yaml
from rewards.predefined.composite_reward import CompositeReward
from utils.config_logger import setup_logger
from utils.checkpoint_manager import load_checkpoint_data

import datetime
import logging
import random
import copy


# --- Configuration ---
CONFIG_DIR = "configs/env"
# Base configs for the environment structure
BASE_SIM_CONFIG_PATH = os.path.join(CONFIG_DIR, "sim_config.yaml")
BASE_DC_CONFIG_PATH = os.path.join(CONFIG_DIR, "datacenters.yaml")
BASE_REWARD_CONFIG_PATH = os.path.join(CONFIG_DIR, "reward_config.yaml")
# Path to the algorithm config that might contain disable_defer_action, if not in sim_config
# This is needed if the agent's behavior (like not outputting action 0) is tied to its training config
BASE_ALGO_CONFIG_PATH = os.path.join(CONFIG_DIR, "algorithm_config.yaml") # Example path

DEFAULT_RL_CHECKPOINT_PATH = "checkpoints/train_layer_norm_20250519_165727/best_eval_checkpoint.pth" # <<<< ADJUST THIS

EVALUATION_DURATION_DAYS = 7 # Example: 7 days for Table 1
NUM_SEEDS = 5
SEEDS = [i * 10 for i in range(NUM_SEEDS)] # Example: [0, 10, 20, 30, 40]

# Define controllers for Table 1
CONTROLLERS_TO_EVALUATE = [
    {
        "name": "RL (Geo+Time)", "strategy": "manual_rl", "checkpoint": DEFAULT_RL_CHECKPOINT_PATH,
        "is_rl": True, "eval_disable_defer": True, "eval_local_only": False # Flags for eval-time behavior override
    },
    # {
    #     "name": "RL (Geo Only)", "strategy": "manual_rl", "checkpoint": DEFAULT_RL_CHECKPOINT_PATH,
    #     "is_rl": True, "eval_disable_defer": True, "eval_local_only": False
    # },
    # {
    #     "name": "RL (Time Only)", "strategy": "manual_rl", "checkpoint": DEFAULT_RL_CHECKPOINT_PATH,
    #     "is_rl": True, "eval_disable_defer": False, "eval_local_only": True
    # },
    # {"name": "RBC (Lowest Carbon)", "strategy": "lowest_carbon", "is_rl": False},
    # {"name": "RBC (Lowest Price)", "strategy": "lowest_price", "is_rl": False},
    # {"name": "RBC (Round Robin)", "strategy": "round_robin", "is_rl": False},
    # {"name": "RBC (Most Available)", "strategy": "most_available", "is_rl": False},
    # {"name": "RBC (Local Only)", "strategy": "local_only", "is_rl": False},
    # {"name": "RBC (Random)", "strategy": "random", "is_rl": False},
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
def make_eval_env(strategy_to_eval, base_sim_cfg_dict, base_dc_cfg_dict, base_reward_cfg_dict,
                  eval_duration_days, seed, controller_eval_flags, eval_mode=True):
    """
    Creates the evaluation environment.
    `controller_eval_flags` can contain `eval_disable_defer` to override env behavior.
    """
    # Deepcopy base configs to allow modification per run
    sim_cfg_dict_run = copy.deepcopy(base_sim_cfg_dict)
    dc_cfg_dict_run = copy.deepcopy(base_dc_cfg_dict)
    reward_cfg_dict_run = copy.deepcopy(base_reward_cfg_dict)

    # Override strategy and duration for this specific evaluation run
    sim_cfg_dict_run["simulation"]["strategy"] = strategy_to_eval
    sim_cfg_dict_run["simulation"]["duration_days"] = eval_duration_days

    # If controller_eval_flags specify an override for disable_defer_action, apply it
    # This allows testing an agent trained WITH deferral in an environment where deferral is NOT allowed by the env
    if 'eval_disable_defer' in controller_eval_flags:
        sim_cfg_dict_run["simulation"]["disable_defer_action"] = controller_eval_flags['eval_disable_defer']
        logger.info(f"Env for '{strategy_to_eval}' will run with disable_defer_action: {controller_eval_flags['eval_disable_defer']}")


    # Extract the simulation part for TaskSchedulingEnv
    sim_cfg_for_env = sim_cfg_dict_run["simulation"]

    start = pd.Timestamp(datetime.datetime(sim_cfg_for_env["year"], sim_cfg_for_env["month"], sim_cfg_for_env["init_day"],
                                           sim_cfg_for_env["init_hour"], 0, tzinfo=datetime.timezone.utc))
    end = start + datetime.timedelta(days=sim_cfg_for_env["duration_days"])

    cluster = DatacenterClusterManager(
        config_list=dc_cfg_dict_run["datacenters"], # Use copied and potentially modified config
        simulation_year=sim_cfg_for_env["year"],
        init_day=int(sim_cfg_for_env["month"] * 30.5 + sim_cfg_for_env["init_day"]),
        init_hour=sim_cfg_for_env["init_hour"],
        strategy=sim_cfg_for_env["strategy"],
        tasks_file_path=sim_cfg_for_env["workload_path"],
        shuffle_datacenter_order=not eval_mode,
        cloud_provider=sim_cfg_for_env["cloud_provider"],
        logger=logger
    )
    reward_fn = CompositeReward(
        components=reward_cfg_dict_run["reward"]["components"],
        normalize=False # No reward normalization during eval
    )
    env = TaskSchedulingEnv(
        cluster_manager=cluster, start_time=start, end_time=end,
        reward_fn=reward_fn, writer=None,
        sim_config=sim_cfg_for_env # Pass the potentially modified sim_config
    )
    # Env reset is now done in run_single_evaluation before the loop
    return env

# --- Single Evaluation Run Function ---
def run_single_evaluation(controller_config, seed, base_sim_cfg_dict, base_dc_cfg_dict, base_reward_cfg_dict):
    logger.info(f"--- Running Eval: {controller_config['name']} | Seed: {seed} ---")

    strategy_to_eval = controller_config['strategy']
    # Pass eval-specific flags to environment creator
    controller_eval_flags = {
        'eval_disable_defer': controller_config.get('eval_disable_defer', False),
        # eval_local_only will be handled by modifying agent actions, not env directly
    }
    env = make_eval_env(strategy_to_eval, base_sim_cfg_dict, base_dc_cfg_dict, base_reward_cfg_dict,
                        EVALUATION_DURATION_DAYS, seed, controller_eval_flags)

    actor = None
    ckpt_single_action_mode = False # Default for RBCs or if RL fails to load
    ckpt_act_dim_net = env.num_dcs + 1 # Default for RBCs

    if controller_config['is_rl']:
        checkpoint_path = controller_config['checkpoint']
        if not os.path.exists(checkpoint_path):
            logger.error(f"RL Checkpoint not found for {controller_config['name']}: {checkpoint_path}. Skipping RL run.")
            return None

        try:
            checkpoint_data, loaded_step = load_checkpoint_data(path=checkpoint_path, device="cpu")
            if checkpoint_data is None:
                logger.error(f"Failed to load checkpoint from {checkpoint_path}. Exiting.")
                exit()
                
            extra_info = checkpoint_data.get("extra_info", {})
            ckpt_single_action_mode = extra_info.get("single_action_mode", False) # Default to False if not found
            ckpt_use_attention = extra_info.get("use_attention", False)         # Default to False
            # Use the obs_dim and act_dim stored in the checkpoint for network creation
            # Fallback to env dimensions if not found (for older checkpoints, though less ideal)
            ckpt_obs_dim_net = extra_info.get("obs_dim", env.observation_space.shape[0])
            ckpt_act_dim_net = extra_info.get("act_dim", env.num_dcs)
            ckpt_hidden_dim = extra_info.get("hidden_dim", 128) # Fallback for MLP
            ckpt_use_layer_norm = extra_info.get("use_layer_norm", True) # Fallback for MLP

            logger.info(f"Loaded checkpoint info: single_action_mode={ckpt_single_action_mode}, use_attention={ckpt_use_attention}, obs_dim={ckpt_obs_dim_net}, act_dim={ckpt_act_dim_net}")
            
            
            # --- Initialize Networks based on loaded info ---
            if ckpt_use_attention:
                attn_cfg_eval = { 
                    "embed_dim": extra_info.get("attn_embed_dim", 128), # Try to get from ckpt if saved
                    "num_heads": extra_info.get("attn_num_heads", 4),
                    "num_attention_layers": extra_info.get("attn_num_layers", 2),
                    "dropout": extra_info.get("attn_dropout", 0.1)
                }
                actor = AttentionActorNet(
                    obs_dim_per_task=ckpt_obs_dim_net, # Use the dimension the network was trained with
                    act_dim=ckpt_act_dim_net,
                    embed_dim=attn_cfg_eval.get("embed_dim"),
                    num_heads=attn_cfg_eval.get("num_heads"),
                    num_attention_layers=attn_cfg_eval.get("num_attention_layers"),
                    dropout=attn_cfg_eval.get("dropout")
                ).to("cpu")
            else:
                actor = ActorNet(ckpt_obs_dim_net, ckpt_act_dim_net, ckpt_hidden_dim, ckpt_use_layer_norm).to("cpu")
                

            # Load state dict
            if "actor_state_dict" in checkpoint_data:
                actor.load_state_dict(checkpoint_data["actor_state_dict"])
                actor.eval()
                logger.info("Actor model loaded and set to eval mode.")
            else:
                logger.error("actor_state_dict not found in checkpoint!")
                exit()
            logger.info(f"Successfully loaded actor for {controller_config['name']}")

        except Exception as e:
            logger.error(f"Failed to load actor for {controller_config['name']} from {checkpoint_path}: {e}")
            return None
    # --- Simulation Loop ---
    obs, _ = env.reset(seed=seed) # Env setup based on its own flags (e.g. env.single_action_mode)
    num_steps = EVALUATION_DURATION_DAYS * 24 * 4
    all_infos_step = []
    total_deferred_tasks_in_run = 0

    for step in tqdm(range(num_steps), desc=f"Simulating {controller_config['name']} (Seed {seed})", leave=False):
        actions_to_pass_to_env = []
        step_deferred_count = 0

        is_env_single_action = env.single_action_mode # How the env expects actions
        is_env_defer_disabled = env.disable_defer_action # How the env interprets actions

        if controller_config['is_rl'] and actor is not None:
            # --- RL Agent Action Selection ---
            if ckpt_single_action_mode: # Actor was trained in single action mode
                if not is_env_single_action:
                    logger.warning("Actor trained in single_action_mode but env is in multi_action_mode. This is likely an error.")
                # Obs should be a single aggregated vector from the env
                obs_for_actor_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    logits = actor(obs_for_actor_tensor) # [1, ckpt_act_dim_net]
                    action_from_actor = torch.distributions.Categorical(logits=logits).sample().item()

                # Apply eval-time modifications (disable_defer, local_only) to agent's chosen action
                action_modified = action_from_actor
                if controller_config.get('eval_disable_defer', False) and action_modified == 0 and ckpt_act_dim_net > env.num_dcs:
                    # If actor *could* output 0 (defer) but eval says no defer, pick a random DC
                    action_modified = np.random.randint(1, env.num_dcs + 1) # Map to DC 1..N
                # Local_only for single_action_mode is complex: which task's origin to pick?
                # For simplicity, if local_only and single_action, we might enforce defer if any task would be remote.
                # Or, this specific eval_local_only=True might not make sense with ckpt_single_action_mode=True.
                # For now, we assume the env will handle the single action for all tasks.

                actions_to_pass_to_env = action_modified # Env step expects single int
                if not is_env_defer_disabled and action_modified == 0 and len(env.current_tasks) > 0 :
                    step_deferred_count = len(env.current_tasks) # All tasks deferred

            else: # Actor was trained in multi-task mode (ckpt_single_action_mode is False)
                if is_env_single_action:
                    logger.warning("Actor trained in multi_action_mode but env is in single_action_mode. This is likely an error.")
                if len(obs) > 0: # obs is a list of per-task obs
                    obs_tensor = torch.FloatTensor(obs)
                    with torch.no_grad():
                        logits = actor(obs_tensor) # [k_t, ckpt_act_dim_net]
                        sampled_actions_from_actor = torch.distributions.Categorical(logits=logits).sample().numpy().tolist()

                    # Apply eval-time modifications to each action in the list
                    modified_actions_list = []
                    for i, single_action in enumerate(sampled_actions_from_actor):
                        mod_act = single_action
                        # 1. Eval Disable Defer
                        if controller_config.get('eval_disable_defer', False) and mod_act == 0 and ckpt_act_dim_net > env.num_dcs:
                            # If agent chose defer (0) but eval says no defer, force to origin
                            # Requires per-task obs to have origin_dc_id accessible
                            # Assuming per-task obs[i][4] is origin_dc_id (as in your _get_obs)
                            mod_act = int(obs[i][4]) + 1 # Map origin_id (0-indexed) to action (1-indexed DC)
                        # 2. Eval Local Only
                        if controller_config.get('eval_local_only', False) and mod_act > 0 : # If assigning to a DC
                            task_origin_id_for_action = int(obs[i][4]) + 1 # Map origin_id (0-indexed) to action (1-indexed DC)
                            if mod_act != task_origin_id_for_action:
                                mod_act = task_origin_id_for_action # Force to origin
                        modified_actions_list.append(mod_act)
                        if not is_env_defer_disabled and mod_act == 0 :
                            step_deferred_count += 1
                    actions_to_pass_to_env = modified_actions_list
                else: # No tasks
                    actions_to_pass_to_env = []
        else: # RBC mode or RL agent loading failed
            actions_to_pass_to_env = [] # RBCs don't take actions from here

        total_deferred_tasks_in_run += step_deferred_count
        obs, _, done, truncated, info = env.step(actions_to_pass_to_env)
        all_infos_step.append(info)
        if done or truncated:
            if done: logger.info(f"Sim duration reached at step {step+1} for seed {seed}.")
            break

    # --- Aggregate Metrics ---
    # ... (Aggregation logic remains the same, ensure keys in 'info' are correct) ...
    # Make sure "Total Tasks Deferred" uses total_deferred_tasks_in_run
    # ... (rest of your aggregation and result dict creation, identical to before,
    #      just ensure 'total_deferred_tasks_in_run' is used for "Total Tasks Deferred")
    # Example for the result dict:
    # results = { ... "Total Tasks Deferred": total_deferred_tasks_in_run, ... }
    # ... (rest of the aggregation) ...
    total_energy_cost = sum(s_info.get("total_energy_cost_usd_this_step",0) for s_info in all_infos_step) # Assuming you log this
    total_energy_kwh = sum(s_info.get("total_energy_kwh_this_step",0) for s_info in all_infos_step)
    total_carbon_kg = sum(s_info.get("total_emissions_kg_this_step",0) for s_info in all_infos_step)
    total_sla_met = sum(s_info.get("datacenter_infos",{}).get(dc,{}).get("__common__",{}).get("__sla__",{}).get("met",0) for s_info in all_infos_step for dc in s_info.get("datacenter_infos",{}))
    total_sla_violated = sum(s_info.get("datacenter_infos",{}).get(dc,{}).get("__common__",{}).get("__sla__",{}).get("violated",0) for s_info in all_infos_step for dc in s_info.get("datacenter_infos",{}))
    total_trans_cost = sum(s_info.get("transmission_cost_total_usd", 0.0) for s_info in all_infos_step)

    cpu_utils_all_steps = []
    gpu_utils_all_steps = []
    mem_utils_all_steps = []
    water_usage_all_steps = []
    ite_energy_all_steps = []

    for step_info in all_infos_step:
        for dc_name, dc_info_step in step_info.get("datacenter_infos", {}).items():
            common = dc_info_step.get("__common__", {})
            cpu_utils_all_steps.append(common.get("cpu_util_percent", 0.0))
            gpu_utils_all_steps.append(common.get("gpu_util_percent", 0.0))
            mem_utils_all_steps.append(common.get("mem_util_percent", 0.0))
            water_usage_all_steps.append(dc_info_step.get("agent_dc", {}).get("dc_water_usage", 0.0))
            
            ite_power_kw_step = dc_info_step.get('agent_dc',{}).get('dc_ITE_total_power_kW', 0)
            ite_energy_all_steps.append(ite_power_kw_step * (15.0/60.0)) # kWh


    results = {
        "Controller": controller_config['name'], "Seed": seed,
        "Total Energy Cost (USD)": sum(info_step['datacenter_infos'][dc]['__common__']['energy_cost_USD'] for info_step in all_infos_step for dc in info_step['datacenter_infos']),
        "Total Energy (kWh)": sum(info_step['datacenter_infos'][dc]['__common__']['energy_consumption_kwh'] for info_step in all_infos_step for dc in info_step['datacenter_infos']),
        "Total CO2 (kg)": sum(info_step['datacenter_infos'][dc]['__common__']['carbon_emissions_kg'] for info_step in all_infos_step for dc in info_step['datacenter_infos']),
        "Total SLA Violations": sum(info_step['datacenter_infos'][dc]['__common__']['__sla__']['violated'] for info_step in all_infos_step for dc in info_step['datacenter_infos']),
        "SLA Violation Rate (%)": (sum(info_step['datacenter_infos'][dc]['__common__']['__sla__']['violated'] for info_step in all_infos_step for dc in info_step['datacenter_infos']) / \
                                  (sum(info_step['datacenter_infos'][dc]['__common__']['__sla__']['met'] for info_step in all_infos_step for dc in info_step['datacenter_infos']) + \
                                   sum(info_step['datacenter_infos'][dc]['__common__']['__sla__']['violated'] for info_step in all_infos_step for dc in info_step['datacenter_infos']) + 1e-6)) * 100,
        "Total Transmission Cost (USD)": sum(info_step['transmission_cost_total_usd'] for info_step in all_infos_step),
        "Avg CPU Util (%)": np.mean(cpu_utils_all_steps) if cpu_utils_all_steps else 0,
        "Avg GPU Util (%)": np.mean(gpu_utils_all_steps) if gpu_utils_all_steps else 0,
        "Avg MEM Util (%)": np.mean(mem_utils_all_steps) if mem_utils_all_steps else 0,
        "Total Water Usage (L)": sum(water_usage_all_steps),
        "Average PUE": (sum(info_step['datacenter_infos'][dc]['__common__']['energy_consumption_kwh'] for info_step in all_infos_step for dc in info_step['datacenter_infos'])) / (sum(ite_energy_all_steps) + 1e-6) if sum(ite_energy_all_steps) > 0 else np.nan,
        "Total Tasks Deferred": total_deferred_tasks_in_run,
    }
    logger.info(f"--- Finished Eval: {controller_config['name']} | Seed: {seed} ---")
    return results

# --- Main Evaluation ---
if __name__ == "__main__":
    # ... (Main evaluation loop and result processing remain the same) ...
    logger.info("Starting Table 1 Evaluations...")
    logger.info(f"Duration: {EVALUATION_DURATION_DAYS} days | Seeds: {SEEDS}")
    base_sim_config_dict = load_yaml(BASE_SIM_CONFIG_PATH) # Load as dict
    base_dc_config_dict = load_yaml(BASE_DC_CONFIG_PATH)
    base_reward_config_dict = load_yaml(BASE_REWARD_CONFIG_PATH)
    all_results_list = [] # Changed name
    for controller_cfg in CONTROLLERS_TO_EVALUATE: # Changed name
        seed_results_list = [] # Changed name
        for s in SEEDS: # Changed name
            res = run_single_evaluation(controller_cfg, s, base_sim_config_dict, base_dc_config_dict, base_reward_config_dict)
            if res: seed_results_list.append(res)
            else: logger.warning(f"Skipping results for {controller_cfg['name']} seed {s} due to error.")
        if seed_results_list: all_results_list.extend(seed_results_list)
        else: logger.warning(f"No successful runs for controller: {controller_cfg['name']}")
    if not all_results_list: logger.error("No evaluation results collected. Exiting."); exit()
    results_df_final = pd.DataFrame(all_results_list) # Changed name
    summary_df_final = results_df_final.groupby("Controller").agg( # Changed name
        # ... (aggregation logic remains largely the same, ensure all keys in 'results' dict are covered)
        {"Total Energy Cost (USD)": ['mean', 'std'], "Total Energy (kWh)": ['mean', 'std'], "Total CO2 (kg)": ['mean', 'std'],
         "Total SLA Violations": ['mean', 'std'], "SLA Violation Rate (%)": ['mean', 'std'], "Total Transmission Cost (USD)": ['mean', 'std'],
         "Avg CPU Util (%)": ['mean', 'std'], "Avg GPU Util (%)": ['mean', 'std'], "Avg MEM Util (%)": ['mean', 'std'],
         "Total Water Usage (L)": ['mean', 'std'], "Average PUE": ['mean', 'std'], "Total Tasks Deferred": ['mean', 'std']}
    )
    summary_df_final.columns = ["_".join(col).strip() for col in summary_df_final.columns.values]
    summary_df_final = summary_df_final.reset_index()
    formatted_summary_final_df = summary_df_final.copy() # Changed name
    for col in summary_df_final.columns:
        if col != "Controller":
            if 'mean' in col: mean_val_col = col # Store the mean column name
            elif 'std' in col:
                # Combine mean and std into one "Mean ± (Std)" column
                new_col_name_display = col.replace('_std', '') # For display "Total Energy Cost (USD)"
                # Ensure the corresponding mean column exists from the previous iteration
                if mean_val_col.replace('_mean', '') == new_col_name_display:
                    formatted_summary_final_df[new_col_name_display] = summary_df_final[mean_val_col].map('{:.2f}'.format) + \
                                                                ' ± (' + summary_df_final[col].map('{:.2f}'.format) + ')'
                    formatted_summary_final_df = formatted_summary_final_df.drop(columns=[mean_val_col])
                formatted_summary_final_df = formatted_summary_final_df.drop(columns=[col])

    logger.info("\n--- Aggregated Results Table ---")
    logger.info("\n" + formatted_summary_final_df.to_string(index=False))
    try:
        results_df_final.to_csv(results_csv_path.replace(".csv", "_raw.csv"), index=False)
        formatted_summary_final_df.to_csv(results_csv_path, index=False)
        logger.info(f"Raw results saved to {results_csv_path.replace('.csv', '_raw.csv')}")
        logger.info(f"Formatted summary saved to {results_csv_path}")
    except Exception as e: logger.error(f"Error saving results to CSV: {e}")
    print("\n--- Aggregated Results Table ---")
    print(formatted_summary_final_df.to_string(index=False))
    print(f"\nFull logs saved to: {log_path}")
    print(f"CSV results saved to directory: {log_dir}")