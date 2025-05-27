import numpy as np
import torch
import datetime

import pandas as pd
from tqdm import tqdm
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.policy.policy import Policy
from ray.rllib.core.columns import Columns
from ray.rllib.utils.numpy import convert_to_numpy, softmax

# from train_rllib_ppo import TaskSchedulingEnvRLLIB
from ray.tune.registry import register_env

# Import your custom environment
from envs.task_scheduling_env import TaskSchedulingEnv # Adjust path if necessary
# Import make_env to be used by the registration lambda
from utils.config_loader import load_yaml # Assuming this is used by your make_env
import pandas as pd # make_env might use it
from simulation.cluster_manager import DatacenterClusterManager # make_env uses it
from rewards.predefined.composite_reward import CompositeReward # make_env uses it



# --- Environment Creator Function for RLlib ---
def env_creator(env_config_rllib):
    """
    env_config_rllib will contain parameters passed from RLlib's .environment(env_config=...)
    These include the paths to your simulation config files.
    """
    print(f"RLlib env_config received by creator: {env_config_rllib}")
    sim_cfg_path = env_config_rllib.get("sim_config_path", "configs/env/sim_config.yaml")
    dc_cfg_path = env_config_rllib.get("dc_config_path", "configs/env/datacenters.yaml")
    reward_cfg_path = env_config_rllib.get("reward_config_path", "configs/env/reward_config.yaml")

    # --- Using a simplified make_env or direct instantiation ---
    # The make_env from your train_sac.py might be too complex if it sets up loggers/writers
    # specific to that script. For RLlib, it's cleaner if the env is self-contained.

    sim_cfg_full = load_yaml(sim_cfg_path)
    sim_cfg = sim_cfg_full["simulation"] # Extract the simulation part
    dc_cfg = load_yaml(dc_cfg_path)["datacenters"]
    reward_cfg = load_yaml(reward_cfg_path)["reward"]

    # Ensure 'single_action_mode' is true for this training script's purpose
    if not sim_cfg.get("single_action_mode", False):
        print("WARNING: 'single_action_mode' is not true in sim_config.yaml. This RLlib script expects it for simplicity.")
        # sim_cfg["single_action_mode"] = True # Optionally force it

    start = pd.Timestamp(datetime.datetime(sim_cfg["year"], sim_cfg["month"], sim_cfg["init_day"],
                                           sim_cfg["init_hour"], 0, tzinfo=datetime.timezone.utc))
    end = start + datetime.timedelta(days=sim_cfg["duration_days"])

    cluster = DatacenterClusterManager(
        config_list=dc_cfg,
        simulation_year=sim_cfg["year"],
        init_day=int(sim_cfg["month"] * 30.5 + sim_cfg["init_day"]),
        init_hour=sim_cfg["init_hour"],
        strategy="manual_rl", # Must be manual_rl for agent control
        tasks_file_path=sim_cfg["workload_path"],
        shuffle_datacenter_order=sim_cfg.get("shuffle_datacenters", True),
        cloud_provider=sim_cfg["cloud_provider"],
        logger=None # RLlib workers usually handle their own logging
    )

    reward_fn_instance = CompositeReward(
        components=reward_cfg["components"],
        normalize=reward_cfg.get("normalize", False),
        freeze_stats_after_steps=reward_cfg.get("freeze_stats_after_steps", None)
    )

    # Pass the sim_cfg dictionary to TaskSchedulingEnv for single_action_mode etc.
    env = TaskSchedulingEnv(
        cluster_manager=cluster,
        start_time=start,
        end_time=end,
        reward_fn=reward_fn_instance,
        writer=None, # RLlib handles its own TensorBoard logging
        sim_config=sim_cfg # Pass the simulation config dict
    )
    print(f"GreenDCC Env Created. Obs Space: {env.observation_space}, Act Space: {env.action_space}")
    return env

env_name = "GreenDCC_RLlib_Env"
register_env(env_name, env_creator)
NUM_SEEDS = 5
# checkpoint_dir = "/lustre/guillant/rllib_checkpoints/checkpoint_000070" # PPO
# checkpoint_dir = "/lustre/guillant/rllib_checkpoints/checkpoint_000090" # IMPALA
# checkpoint_dir = "/lustre/guillant/rllib_checkpoints/checkpoint_000099"   # APPO
# checkpoint_dir = "~/ray_results/GreenDCC_PPO_SingleAction/PPO_GreenDCC_RLlib_Env_ae88e_00000_0_2025-05-21_16-00-14/checkpoint_000099" # PPO
checkpoint_dir = "~/ray_results/GreenDCC_IMPALA_SingleAction/IMPALA_GreenDCC_RLlib_Env_b01ce_00000_0_2025-05-21_18-09-07/checkpoint_000002" # IMPALA

rl_module  = Algorithm.from_checkpoint(checkpoint_dir).get_module()

env = env_creator({})
summary_all_seeds = []

print()
print("Running evaluation for", checkpoint_dir.split('/')[-2].split('_')[0])

for i in tqdm(range(NUM_SEEDS)):
    obs, _ = env.reset(seed=i*10)
    done = False
    step = 0
    infos_list = []
    common_info_list = []
    total_deffered_tasks = 0
    while not done:

        input_dict = {Columns.OBS: torch.from_numpy(obs).unsqueeze(0)}
        rl_module_out = rl_module.forward_inference(input_dict)

        logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])
        actions = []
        # action = np.random.choice(5, p=softmax(logits[0]))
        action = np.argmax(logits)

        obs, reward, done, truncated, info = env.step(action)
        step += 1
        # print(infbo['datacenter_infos']['DC1'])

        infos_list.append(info["datacenter_infos"])
        common_info_list.append(info["transmission_cost_total_usd"])

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
                "water_usage": dc_info.get("agent_dc", {}).get("dc_water_usage", 0.0),
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
                # "dc_cpu_workload_fraction": dc_info["agent_dc"].get("dc_cpu_workload_fraction", 0.0),
                # "transmission_cost": dc_info["__common__"].get("transmission_cost_total_usd", 0.0),
            }
            flat_records.append(record)

    df = pd.DataFrame(flat_records)

    summary = df.agg({
        "energy_cost": "sum",
        "carbon_kg": "sum",
        "energy_kwh": "sum",
        "water_usage": "sum",

        "price_per_kwh": "mean",
        "ci": "mean",
        "weather": "mean",
        "sla_met": "sum",
        "sla_violated": "sum",

        "cpu_util": "mean",
        "gpu_util": "mean",
        "mem_util": "mean",
        "running_tasks": "sum",
        "pending_tasks": "mean",
        
    })

    summary["SLA Violation Rate (%)"] = (
        summary["sla_violated"] / (summary["sla_met"] + summary["sla_violated"])
    ) * 100

    total_cost = sum(common_info_list)
    summary["Total TX Cost"] = total_cost
    summary["Total deffered tasks"] = total_deffered_tasks
    summary_all_seeds.append(summary)

summary_df = pd.DataFrame(summary_all_seeds)

summary_df = summary_df.agg(['mean', 'std'])
print(summary_df)
summary_df.to_csv('summary_ppo.csv', index=False)

# Print for Latex
columns = [
"energy_cost",
"carbon_kg",
"energy_kwh",
"water_usage",
"SLA Violation Rate (%)",
"cpu_util",
"gpu_util",
"pending_tasks",
"Total TX Cost",
"Total deffered tasks"
]


summary_df = summary_df.set_index(pd.Index(['mean', 'std']))

summary_df['carbon_kg'] /= 1000
summary_df['energy_kwh'] /= 1000
summary_df['water_usage'] /= 1000

res = ''
for col in columns:
    mean = summary_df.loc['mean', col]
    std = summary_df.loc['std', col]
    res += f"{mean:.1f}±{std:.1f} & "

print(res)