#%%
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

# DEFAULT_RL_CHECKPOINT_PATH =  # <<<< ADJUST THIS
DEFAULT_RL_CHECKPOINT_PATH = "checkpoints/train_multiaction_defer_20250527_210534/best_eval_checkpoint.pth" # <<<< Multi Action + Defer
# DEFAULT_RL_CHECKPOINT_PATH = "checkpoints/train_multiaction_nodefer_20250527_212105/best_eval_checkpoint.pth" # <<<< Multi Action + NO_Defer
# DEFAULT_RL_CHECKPOINT_PATH = 'checkpoints/train_single_action_enable_defer_20250527_222926/best_eval_checkpoint.pth' # <<<< Single Action + Enable Defer
# DEFAULT_RL_CHECKPOINT_PATH = 'checkpoints/train_single_action_disable_defer_20250527_223002/best_eval_checkpoint.pth' # <<<< Single Action + Disable Defer


EVALUATION_DURATION_DAYS = 7 # Example: 7 days for Table 1
NUM_SEEDS = 1
SEEDS = [123] # Example: [0, 10, 20, 30, 40]

# Define controllers for Table 1
CONTROLLERS_TO_EVALUATE = [
    {
        "name": "RL (Geo+Time)", "strategy": "manual_rl", "checkpoint": DEFAULT_RL_CHECKPOINT_PATH,
        "is_rl": True, "eval_local_only": False # Flags for eval-time behavior override
    },
    
    # Local Only{"name": "RBC (Local Only)", "strategy": "local_only", "is_rl": False},
    # {
    #     "name": "RBC (Local Only)", "strategy": "local_only", "is_rl": False,
    #     "eval_disable_defer": False, "eval_local_only": True # RBCs should not defer
    # },
]

#TODO extract the flag eval_disable_defer from the algo_config.yaml

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
                  eval_duration_days, seed,
                  # NEW: Pass resolved disable_defer for env creation
                  env_disable_defer_flag: bool,
                  # NEW: Pass resolved single_action_mode for env creation
                  env_single_action_mode_flag: bool,
                  eval_mode=True):
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

    # --- SET ENV BEHAVIOR BASED ON PASSED FLAGS ---
    sim_cfg_dict_run["simulation"]["disable_defer_action"] = env_disable_defer_flag
    sim_cfg_dict_run["simulation"]["single_action_mode"] = env_single_action_mode_flag
    logger.info(f"Env for '{strategy_to_eval}' will run with disable_defer_action: {env_disable_defer_flag}, single_action_mode: {env_single_action_mode_flag}")


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
controller_config = CONTROLLERS_TO_EVALUATE[0] # Example: just the first controller
seed = SEEDS[0] # Example: just the first seed
base_sim_cfg_dict = load_yaml(BASE_SIM_CONFIG_PATH)
base_dc_cfg_dict = load_yaml(BASE_DC_CONFIG_PATH)
base_reward_cfg_dict = load_yaml(BASE_REWARD_CONFIG_PATH)

logger.info(f"--- Running Eval: {controller_config['name']} | Seed: {seed} ---")

strategy_to_eval = controller_config['strategy']
actor = None

# --- Determine Environment Creation Flags based on Agent Training (if RL) ---
# These will be the defaults for creating the env, can be overridden by controller_config['eval_...'] for specific tests
initial_env_single_action_mode = base_sim_cfg_dict["simulation"].get("single_action_mode", False)
initial_env_disable_defer = base_sim_cfg_dict["simulation"].get("disable_defer_action", False)

ckpt_obs_dim_net = None # Will be set if RL agent is loaded
ckpt_act_dim_net = None # Will be set if RL agent is loaded
loaded_ckpt_single_action_mode = None # From checkpoint extra_info

if controller_config['is_rl']:
    checkpoint_path = controller_config['checkpoint']
    if not os.path.exists(checkpoint_path):
        logger.error(f"RL Checkpoint not found: {checkpoint_path}. Skipping.")
    try:
        checkpoint_data, _ = load_checkpoint_data(path=checkpoint_path, device="cpu")
        if checkpoint_data is None: raise ValueError("Failed to load checkpoint data.")
        
        extra_info = checkpoint_data.get("extra_info", {})
        loaded_ckpt_single_action_mode = extra_info.get("single_action_mode", initial_env_single_action_mode) # Important
        ckpt_use_attention = extra_info.get("use_attention", False)
        ckpt_obs_dim_net = extra_info.get("obs_dim") # Must exist
        ckpt_act_dim_net = extra_info.get("act_dim") # Must exist
        ckpt_hidden_dim = extra_info.get("hidden_dim", 64)
        ckpt_use_layer_norm = extra_info.get("use_layer_norm", False)

        if ckpt_obs_dim_net is None or ckpt_act_dim_net is None:
            logger.error(f"obs_dim or act_dim missing from checkpoint extra_info: {checkpoint_path}")

        # --- Determine how the env should be configured based on agent training & eval flags ---
        # 1. single_action_mode for env creation: MUST match how agent was trained
        env_create_single_action_mode = loaded_ckpt_single_action_mode

        # 2. disable_defer for env creation:
        #    If eval_disable_defer is explicitly set in controller_config, use that.
        #    Otherwise, infer from agent's trained action dimension.
        if 'eval_disable_defer' in controller_config: # Eval override takes precedence
            env_create_disable_defer = controller_config['eval_disable_defer']
        else: # Infer from agent's output dimension
            # Need a temporary way to get num_dcs to compare with ckpt_act_dim_net
            # This is a bit circular. A dummy env or direct config parse might be needed
            # For simplicity, let's assume we can get num_dcs easily for this inference
            # A better way is to store 'trained_disable_defer_flag' in extra_info
            # For now, let's assume if not explicitly overridden, env matches ckpt implicitly
            env_create_disable_defer = extra_info.get("disable_defer_action", initial_env_disable_defer) # Try to get from ckpt
            # Or infer:
            # num_dcs_temp = len(base_dc_cfg_dict['datacenters']) # Approx
            # if ckpt_act_dim_net == num_dcs_temp:
            #    env_create_disable_defer = True
            # else: # Assumes ckpt_act_dim_net == num_dcs_temp + 1
            #    env_create_disable_defer = False

        logger.info(f"For controller '{controller_config['name']}':")
        logger.info(f"  Checkpoint single_action_mode: {loaded_ckpt_single_action_mode}")
        logger.info(f"  Environment will be created with single_action_mode: {env_create_single_action_mode}")
        logger.info(f"  Environment will be created with disable_defer_action: {env_create_disable_defer}")

        # Create the environment with these determined flags
        env = make_eval_env(strategy_to_eval, base_sim_cfg_dict, base_dc_cfg_dict, base_reward_cfg_dict,
                            EVALUATION_DURATION_DAYS, seed,
                            env_disable_defer_flag=env_create_disable_defer, # Pass the resolved flag
                            env_single_action_mode_flag=env_create_single_action_mode)


        # Initialize Networks based on loaded ckpt info
        if ckpt_use_attention:
            # ... (AttentionActorNet instantiation using extra_info)
            attn_cfg_eval = { 
                            "embed_dim": extra_info.get("attn_embed_dim", 128), # Try to get from ckpt if saved
                            "num_heads": extra_info.get("attn_num_heads", 4),
                            "num_attention_layers": extra_info.get("attn_num_layers", 2),
                            "dropout": extra_info.get("attn_dropout", 0.1)
                        }
            actor = AttentionActorNet(ckpt_obs_dim_net, ckpt_act_dim_net, **attn_cfg_eval).to("cpu")
        else:
            actor = ActorNet(ckpt_obs_dim_net, ckpt_act_dim_net, ckpt_hidden_dim, ckpt_use_layer_norm).to("cpu")
        
        actor.load_state_dict(checkpoint_data["actor_state_dict"])
        actor.eval()
        logger.info(f"Successfully loaded actor for {controller_config['name']}")

    except Exception as e:
        logger.error(f"Failed to load or setup RL actor for {controller_config['name']}: {e}")

else: # RBC
    # For RBCs, create env with their default behavior (usually no defer by agent choice)
    # The eval_disable_defer in controller_config for RBCs is more about setting the env constraint
    env_create_disable_defer = controller_config.get('eval_disable_defer', True) # RBCs don't "choose" to defer
    env_create_single_action_mode = initial_env_single_action_mode # RBCs don't have a single/multi mode
    env = make_eval_env(strategy_to_eval, base_sim_cfg_dict, base_dc_cfg_dict, base_reward_cfg_dict,
                        EVALUATION_DURATION_DAYS, seed,
                        {'eval_disable_defer': env_create_disable_defer},
                        env_single_action_mode_flag=env_create_single_action_mode)


#%%
# --- Simulation Loop ---
obs, _ = env.reset(seed=seed) # Env setup based on its own flags (e.g. env.single_action_mode)
num_steps = EVALUATION_DURATION_DAYS * 24 * 4
all_infos_step = []
total_deferred_tasks_in_run = 0
infos_list = []
common_info_list = []
delayed_task_counts_this_run = [] # Specific for this run

for step in tqdm(range(num_steps), desc=f"Simulating {controller_config['name']} (Seed {seed})", leave=False):
    actions_to_pass_to_env = []
    step_deferred_count = 0

    is_env_single_action = env.single_action_mode # How the env expects actions
    is_env_defer_disabled = env.disable_defer_action # How the env interprets actions

    if controller_config['is_rl'] and actor is not None:
        # --- RL Agent Action Selection ---
        if loaded_ckpt_single_action_mode: # Actor was trained in single action mode
            if not is_env_single_action:
                logger.warning("Actor trained in single_action_mode but env is in multi_action_mode. This is likely an error.")
            # Obs should be a single aggregated vector from the env
            obs_for_actor_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                logits = actor(obs_for_actor_tensor) # [1, ckpt_act_dim_net]
                # action_from_actor = torch.distributions.Categorical(logits=logits).sample().item()
                action_from_actor = torch.argmax(logits, dim=-1).item()

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
            delayed_task_counts_this_run.append(step_deferred_count)

        else: # Actor was trained in multi-task mode (ckpt_single_action_mode is False)
            if is_env_single_action:
                logger.warning("Actor trained in multi_action_mode but env is in single_action_mode. This is likely an error.")
            if len(obs) > 0: # obs is a list of per-task obs
                obs_tensor = torch.FloatTensor(obs)
                with torch.no_grad():
                    logits = actor(obs_tensor) # [k_t, ckpt_act_dim_net]
                    # sampled_actions_from_actor = torch.distributions.Categorical(logits=logits).sample().numpy().tolist()
                    sampled_actions_from_actor = torch.argmax(logits, dim=-1).cpu().numpy().tolist()

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
                delayed_task_counts_this_run.append(step_deferred_count)
            else: # No tasks
                actions_to_pass_to_env = []
                delayed_task_counts_this_run.append(0)
    else: # RBC mode or RL agent loading failed
        actions_to_pass_to_env = [] # RBCs don't take actions from here
        delayed_task_counts_this_run.append(0) # RBCs in this setup don't defer

    total_deferred_tasks_in_run += step_deferred_count
    obs, _, done, truncated, info = env.step(actions_to_pass_to_env)
    all_infos_step.append(info)
    
    
    infos_list.append(info["datacenter_infos"])
    common_info_list.append(info["transmission_cost_total_usd"])


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

if 'df_rbc_local' not in locals():
    df_rbc_local = df.copy()
elif 'df_rl_agent' not in locals():
    df_rl_agent = df.copy()

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
sns.lineplot(data=df, x="timestep", y="price_per_kwh", hue="datacenter", palette="colorblind")
plt.title("Energy Price per kWh over Time")
plt.xlabel("Timestep")
plt.ylabel("USD/kWh")
plt.grid(True)

# Save the figure as pdf
# plt.savefig(f"assets/figures/energy_price_over_time_{timestamp}.pdf", bbox_inches='tight')

plt.show()


#%% 
# Plot of "Total Running Tasks"
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="timestep", y="running_tasks", hue="datacenter", palette="colorblind")
plt.title("Total Running Tasks per Datacenter over Time")
plt.xlabel("Timestep")
plt.ylabel("Number of Tasks")
plt.grid(True)

# Save the figure as pdf
# plt.savefig(f"assets/figures/running_tasks_over_time_{timestamp}.pdf", bbox_inches='tight')

plt.show()


#%%
# Assuming 'df_eval_run' is your DataFrame with the evaluation results
# and contains columns: "timestep", "datacenter", "running_tasks"

# Get the unique datacenter names to create a subplot for each
# Ensure they are sorted for consistent plotting order if desired
unique_dcs = sorted(df['datacenter'].unique())
num_unique_dcs = len(unique_dcs)

dc_name_to_location_label = {
    "DC1": "US-CA (CISO)",  # Assuming DC1 is US-CAL-CISO
    "DC2": "Germany (DE-LU)", # Assuming DC2 is DE-LU
    "DC3": "Chile (CL-SIC)",  # Assuming DC3 is CL-SIC
    "DC4": "Singapore (SG)", # Assuming DC4 is SG
    "DC5": "Australia (NSW)",# Assuming DC5 is AU-NSW
    # Add all your DC names and their desired labels
}

if num_unique_dcs == 0:
    print("No datacenter data to plot.")
else:
    # Determine the number of rows for subplots
    # If you specifically want 5, and have 5 DCs, it's straightforward.
    # If num_unique_dcs might be different, you might need to adjust.
    # For exactly 5 vertical subplots:
    if num_unique_dcs > 5:
        logger.warning(f"Found {num_unique_dcs} datacenters, but only plotting the first 5 vertically.")
        unique_dcs_to_plot = unique_dcs[:5]
        n_rows = 5
    elif num_unique_dcs == 0:
        print("No datacenter data found in DataFrame for plotting.")
        n_rows = 0
    else:
        unique_dcs_to_plot = unique_dcs
        n_rows = num_unique_dcs

    if n_rows > 0:
        fig, axes = plt.subplots(n_rows, 1, figsize=(6, 2.5 * n_rows), sharex=True) # nrows, 1 col
        
        # If only one DC, axes might not be an array, so make it one
        if n_rows == 1 and not isinstance(axes, np.ndarray):
            axes = [axes]

        palette = sns.color_palette("colorblind", n_colors=max(10, num_unique_dcs)) # tab10 is good for up to 10 distinct colors

        for i, dc_name in enumerate(unique_dcs_to_plot):
            ax = axes[i]
            dc_data = df[df['datacenter'] == dc_name]
            color_for_dc = palette[unique_dcs.index(dc_name) % len(palette)] # Cycle through palette if more DCs than colors

            location_label = dc_name_to_location_label.get(dc_name, dc_name) # Fallback to dc_name if no label

            sns.lineplot(
                            data=dc_data,
                            x="timestep",
                            y="running_tasks",
                            ax=ax,
                            label=location_label, # Use location label for legend
                            color=color_for_dc,   # Assign specific color
                            errorbar=None         # No confidence interval bands for this plot
                        )   
                     
            ax.set_title(f"Running Tasks on {dc_name}")
            ax.set_ylabel("Number of Tasks")
            ax.grid(True)
            ax.set_xlim(0, df["timestep"].max()) # Set x-limits to the full range of timesteps
            
            if i == n_rows - 1: # Only set x-label for the bottom-most plot
                ax.set_xlabel("Timestep (15 min intervals)")
            # ax.legend() # Optional: legend per subplot if needed, or rely on title

        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust rect to make space for suptitle
        
        # Save the figure
        # Ensure the assets/figures directory exists
        os.makedirs("assets/figures", exist_ok=True)
        plot_filename_per_dc = f"assets/figures/running_tasks_per_dc_rl_{timestamp}.pdf"
        # plt.savefig(plot_filename_per_dc, bbox_inches='tight')
        # logger.info(f"Saved per-DC running tasks plot to: {plot_filename_per_dc}")
        
        plt.show()

#%%
fig, axes = plt.subplots(n_rows, 1, figsize=(6, 2.5 * n_rows), sharex=True) # nrows, 1 col
        
# If only one DC, axes might not be an array, so make it one
if n_rows == 1 and not isinstance(axes, np.ndarray):
    axes = [axes]

palette = sns.color_palette("colorblind", n_colors=max(10, num_unique_dcs)) # tab10 is good for up to 10 distinct colors

for i, dc_name in enumerate(unique_dcs_to_plot):
    ax = axes[i]
    dc_data = df[df['datacenter'] == dc_name]
    color_for_dc = palette[unique_dcs.index(dc_name) % len(palette)] # Cycle through palette if more DCs than colors

    location_label = dc_name_to_location_label.get(dc_name, dc_name) # Fallback to dc_name if no label

    sns.lineplot(
                    data=dc_data,
                    x="timestep",
                    y="price_per_kwh",
                    ax=ax,
                    label=location_label, # Use location label for legend
                    color=color_for_dc,   # Assign specific color
                    errorbar=None         # No confidence interval bands for this plot
                )   
                
    ax.set_title(f"Price per kWh on {dc_name}")
    ax.set_ylabel("USD/kWh")
    ax.grid(True)
    ax.set_xlim(0, df["timestep"].max()) # Set x-limits to the full range of timesteps
    
    if i == n_rows - 1: # Only set x-label for the bottom-most plot
        ax.set_xlabel("Timestep (15 min intervals)")
    # ax.legend() # Optional: legend per subplot if needed, or rely on title

plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust rect to make space for suptitle

# Save the figure
# Ensure the assets/figures directory exists
os.makedirs("assets/figures", exist_ok=True)
plot_filename_per_dc = f"assets/figures/price_per_kwh_per_dc_rl_{timestamp}.pdf"
plt.savefig(plot_filename_per_dc, bbox_inches='tight')
# logger.info(f"Saved per-DC running tasks plot to: {plot_filename_per_dc}")

plt.show()

#%% Plot of Tasks Assigned per Datacenter over Time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="timestep", y="tasks_assigned", hue="datacenter", palette="colorblind")
plt.title("Tasks Assigned per Datacenter over Time")
plt.xlabel("Timestep")
plt.ylabel("Number of Assigned Tasks")
plt.grid(True)

# Save the figure as pdf
# plt.savefig(f"assets/figures/tasks_assigned_over_time_{timestamp}.pdf", bbox_inches='tight')

plt.show()


#%%
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="timestep", y="energy_cost", hue="datacenter", palette="colorblind")
plt.title("Energy Cost per Datacenter over Time")
plt.xlabel("Timestep")
plt.ylabel("USD")
plt.grid(True)

# Save the figure as pdf
# plt.savefig(f"assets/figures/energy_cost_over_time_{timestamp}.pdf", bbox_inches='tight')

plt.show()


#%%
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="timestep", y="carbon_kg", hue="datacenter", palette="colorblind")
plt.title("Carbon Emissions (kg) per Datacenter over Time")
plt.ylabel("kg CO₂")
plt.grid(True)
# Save the figure as pdf
# plt.savefig(f"assets/figures/carbon_emissions_over_time_{timestamp}.pdf", bbox_inches='tight')
plt.show()


#%% Carbon intensity plot on each location (Datacenter)
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="timestep", y="ci", hue="datacenter", palette="colorblind")
plt.title("Carbon Intensity (gCO₂/kWh) over Time")
plt.xlabel("Timestep")
plt.ylabel("gCO₂/kWh")
plt.grid(True)
# Save the figure as pdf
# plt.savefig(f"assets/figures/carbon_intensity_over_time_{timestamp}.pdf", bbox_inches='tight')
plt.show()

#%%
fig, axes = plt.subplots(n_rows, 1, figsize=(6, 2.5 * n_rows), sharex=True) # nrows, 1 col
        
# If only one DC, axes might not be an array, so make it one
if n_rows == 1 and not isinstance(axes, np.ndarray):
    axes = [axes]

palette = sns.color_palette("colorblind", n_colors=max(10, num_unique_dcs)) # tab10 is good for up to 10 distinct colors

for i, dc_name in enumerate(unique_dcs_to_plot):
    ax = axes[i]
    dc_data = df[df['datacenter'] == dc_name]
    color_for_dc = palette[unique_dcs.index(dc_name) % len(palette)] # Cycle through palette if more DCs than colors

    location_label = dc_name_to_location_label.get(dc_name, dc_name) # Fallback to dc_name if no label

    sns.lineplot(
                    data=dc_data,
                    x="timestep",
                    y="ci",
                    ax=ax,
                    label=location_label, # Use location label for legend
                    color=color_for_dc,   # Assign specific color
                    errorbar=None         # No confidence interval bands for this plot
                )   
                
    ax.set_title(f"Carbon Intensity on {dc_name}")
    ax.set_ylabel("gCO₂/kWh")
    ax.grid(True)
    ax.set_xlim(0, df["timestep"].max()) # Set x-limits to the full range of timesteps
    
    if i == n_rows - 1: # Only set x-label for the bottom-most plot
        ax.set_xlabel("Timestep (15 min intervals)")
    # ax.legend() # Optional: legend per subplot if needed, or rely on title

plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust rect to make space for suptitle

# Save the figure
# Ensure the assets/figures directory exists
os.makedirs("assets/figures", exist_ok=True)
plot_filename_per_dc = f"assets/figures/ci_per_dc_rl_{timestamp}.pdf"
# plt.savefig(plot_filename_per_dc, bbox_inches='tight')
# logger.info(f"Saved per-DC running tasks plot to: {plot_filename_per_dc}")

plt.show()


#%% A carbon intensity plot that shows all of the locations in one plot
# Use the legend to show the datacenter names as defined in dc_name_to_location_label
dc_name_to_location_label = {
    "DC1": "US-CA (CISO)",  # Assuming DC1 is US-CAL-CISO
    "DC2": "Germany (DE-LU)", # Assuming DC2 is DE-LU
    "DC3": "Chile (CL-SIC)",  # Assuming DC3 is CL-SIC
    "DC4": "Singapore (SG)", # Assuming DC4 is SG
    "DC5": "Australia (NSW)",# Assuming DC5 is AU-NSW
}

plt.figure(figsize=(8, 4))
sns.lineplot(data=df, x="timestep", y="ci", hue="datacenter", palette="colorblind")

plt.title("Carbon Intensity (gCO₂/kWh) over Time")
plt.xlabel("Timestep (15 min intervals)")
plt.ylabel("gCO₂/kWh")
plt.grid(True)
plt.xlim(0, df["timestep"].max()) # Set x-limits to the full range of timesteps

# Remap legend labels to human-readable names
handles, labels = plt.gca().get_legend_handles_labels()
new_labels = [dc_name_to_location_label.get(lbl, lbl) for lbl in labels]  # skip "datacenter" label
plt.legend(handles, new_labels, title="Datacenter", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plot_filename = f"assets/figures/carbon_intensity_small_over_time_{timestamp}.pdf"
# Save the figure
# plt.savefig(plot_filename, bbox_inches='tight')

plt.show()

#%% The same but for Electricity Price
plt.figure(figsize=(8, 4))
sns.lineplot(data=df, x="timestep", y="price_per_kwh", hue="datacenter", palette="colorblind")
plt.title("Electricity Price (USD/kWh) over Time")
plt.xlabel("Timestep (15 min intervals)")
plt.ylabel("USD/kWh")
plt.grid(True)
plt.xlim(0, df["timestep"].max()) # Set x-limits to the full range of timesteps
# Remap legend labels to human-readable names
handles, labels = plt.gca().get_legend_handles_labels()
new_labels = [dc_name_to_location_label.get(lbl, lbl) for lbl in labels]  # skip "datacenter" label
plt.legend(handles, new_labels, title="Datacenter", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plot_filename = f"assets/figures/electricity_price_small_over_time_{timestamp}.pdf"
# Save the figure
# plt.savefig(plot_filename, bbox_inches='tight')
plt.show()

#%% The same but for External Temperature
plt.figure(figsize=(8, 4))
sns.lineplot(data=df, x="timestep", y="weather", hue="datacenter", palette="colorblind")
plt.title("External Temperature (°C) over Time")
plt.xlabel("Timestep (15 min intervals)")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.xlim(0, df["timestep"].max()) # Set x-limits to the full range of timesteps
# Remap legend labels to human-readable names
handles, labels = plt.gca().get_legend_handles_labels()
new_labels = [dc_name_to_location_label.get(lbl, lbl) for lbl in labels]  # skip "datacenter" label
plt.legend(handles, new_labels, title="Datacenter", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plot_filename = f"assets/figures/external_temperature_small_over_time_{timestamp}.pdf"
# Save the figure
# plt.savefig(plot_filename, bbox_inches='tight')
plt.show()


#%% A vertical plot to show the carbon intensity, electricity price, and external temperature
fig, axes = plt.subplots(3, 1, figsize=(7, 11), sharex=True)

# Carbon Intensity
sns.lineplot(data=df, x="timestep", y="ci", hue="datacenter", palette="colorblind", ax=axes[0])
axes[0].set_title("Carbon Intensity (gCO₂/kWh) over Time")
axes[0].set_ylabel("gCO₂/kWh")
axes[0].grid(True)
axes[0].set_xlim(0, df["timestep"].max()) # Set x-limits to the full range of timesteps
# Electricity Price
sns.lineplot(data=df, x="timestep", y="price_per_kwh", hue="datacenter", palette="colorblind", ax=axes[1])
axes[1].set_title("Electricity Price (USD/kWh) over Time")
axes[1].set_ylabel("USD/kWh")
axes[1].grid(True)
axes[1].set_xlim(0, df["timestep"].max()) # Set x-limits to the full range of timesteps
# External Temperature
sns.lineplot(data=df, x="timestep", y="weather", hue="datacenter", palette="colorblind", ax=axes[2])
axes[2].set_title("External Temperature (°C) over Time")
axes[2].set_ylabel("Temperature (°C)")
axes[2].set_xlabel("Timestep (15 min intervals)")
axes[2].grid(True)
axes[2].set_xlim(0, df["timestep"].max()) # Set x-limits to the full range of timesteps
# Remove individual legends from each subplot
for ax in axes:
    ax.get_legend().remove()

# Remap legend labels to human-readable names using the first axis
handles, labels = axes[0].get_legend_handles_labels()
new_labels = [dc_name_to_location_label.get(lbl, lbl) for lbl in labels]

# Add a single legend below all subplots. Text size is set to 8
fig.legend(
    handles, new_labels, title="Datacenter",
    loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=3,
)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at bottom for legend

# Save the figure
# plot_filename = f"assets/figures/carbon_intensity_electricity_price_external_temperature_{timestamp}.pdf"
# plt.savefig(plot_filename, bbox_inches='tight')
plt.show()


#%%
plt.figure(figsize=(12, 6))
plt.plot(common_info_list)
plt.title("Total Transmission Cost (USD) Over Time")
plt.xlabel("Timestep")
plt.ylabel("Transmission Cost (USD)")
plt.grid(True)

# Save the figure as pdf
# plt.savefig(f"assets/figures/transmission_cost_over_time_{timestamp}.pdf", bbox_inches='tight')
plt.show()


#%%
plt.figure(figsize=(12, 6))
plt.plot(delayed_task_counts_this_run, label="Delayed Tasks", color="orange")
plt.title("Number of Delayed Tasks per Timestep")
plt.xlabel("Timestep")
plt.ylabel("Number of Delayed Tasks")
plt.grid(True)
plt.legend()
# Save the figure as pdf
# plt.savefig("assets/figures/delayed_tasks_over_time.pdf", bbox_inches='tight')
plt.show()



#%% Plot the external temperature
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="timestep", y="weather", hue="datacenter")
plt.title("External Temperature (°C) over Time")
plt.xlabel("Timestep")
plt.ylabel("Temperature (°C)")
plt.grid(True)
# Save the figure as pdf
# plt.savefig("assets/figures/external_temperature_over_time.pdf", bbox_inches='tight')

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
# plt.savefig("assets/figures/utilization_over_time.pdf", bbox_inches='tight')

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

unique_dcs_comp = sorted(df_rbc_local['datacenter'].unique()) # Assume same DCs in both
num_unique_dcs_comp = len(unique_dcs_comp)

# Mapping from DC name (in DataFrame) to Location Label (for plot legend/title)
dc_name_to_location_label_comp = {
    "DC1": "US-CA (CISO)",
    "DC2": "Germany (DE-LU)",
    "DC3": "Chile (CL-SIC)",
    "DC4": "Singapore (SG)",
    "DC5": "Australia (NSW)"
    # Ensure this matches your actual DC identifiers and desired labels
}

if num_unique_dcs_comp > 0:
    fig_comp, axes_comp = plt.subplots(num_unique_dcs_comp, 1, 
                                        figsize=(8, 2.5 * num_unique_dcs_comp), # Adjusted figsize
                                        sharex=True)
    
    if num_unique_dcs_comp == 1 and not isinstance(axes_comp, np.ndarray):
        axes_comp = [axes_comp] # Make it iterable

    # Define distinct colors for RBC and RL, and linestyles
    color_rbc = "steelblue" # Example color for RBC
    color_rl = "darkorange"  # Example color for RL
    palette = sns.color_palette("colorblind", n_colors=max(10, num_unique_dcs_comp))

    for i, dc_name_key in enumerate(unique_dcs_comp):
        ax = axes_comp[i]
        location_label_comp = dc_name_to_location_label_comp.get(dc_name_key, dc_name_key)
        dc_color = palette[i % len(palette)] # Use index 'i' from sorted unique_dcs_keys

        # Plot RBC (Local Only) data - Solid line
        dc_data_rbc = df_rbc_local[df_rbc_local['datacenter'] == dc_name_key]
        if not dc_data_rbc.empty:
            sns.lineplot(data=dc_data_rbc, x="timestep", y="running_tasks", ax=ax, alpha=0.5,
                            color=dc_color, linestyle='-', label="RBC (Local Only)", errorbar=None)

        # Plot RL (Geo+Time) data - Dashed line
        dc_data_rl = df_rl_agent[df_rl_agent['datacenter'] == dc_name_key]
        if not dc_data_rl.empty:
            sns.lineplot(data=dc_data_rl, x="timestep", y="running_tasks", ax=ax, alpha=1,
                            color=dc_color, linestyle='--', label="RL (Geo+Time)", errorbar=None)
        
        # ax.set_title(f"Running Tasks on {location_label_comp}")
        ax.set_ylabel("Number of Tasks")
        ax.grid(True)
        ax.set_xlim(0, df_rbc_local["timestep"].max()) # Assuming both DFs have same timestep range

        if i == 0: # Add legend only to the first subplot to avoid repetition
            ax.legend(loc='upper right')
        else:
            ax.legend().remove() # Remove legends from other subplots


        if i == num_unique_dcs_comp - 1:
            ax.set_xlabel("Timestep (15 min intervals)")

    # fig_comp.suptitle(f"Comparison: Running Tasks per DC - RBC (Local) vs. RL (Geo+Time)", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust rect for suptitle
    
    # Save the comparative figure
    comp_plot_filename = f"assets/figures/running_tasks_comparison_RBC_vs_RL_{timestamp}.pdf"
    plt.savefig(comp_plot_filename, bbox_inches='tight')
    logger.info(f"Saved comparative running tasks plot to: {comp_plot_filename}")
    
    plt.show()

#%%




#%%

#%%


#%%


#%%