import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.policy.policy import Policy
from ray.rllib.core.columns import Columns
from ray.rllib.utils.numpy import convert_to_numpy, softmax

from train_rllib import TaskSchedulingEnvRLLIB

NUM_SEEDS = 10
# checkpoint_dir = "/lustre/guillant/rllib_checkpoints/checkpoint_000070" # PPO
# checkpoint_dir = "/lustre/guillant/rllib_checkpoints/checkpoint_000090" # IMPALA
# checkpoint_dir = "/lustre/guillant/rllib_checkpoints/checkpoint_000099"   # APPO
checkpoint_dir = "/lustre/guillant/new_green-dcc/results/test/PPO_TaskSchedulingEnvRLLIB_14669_00000_0_2025-05-16_03-30-14/checkpoint_000002" # 
rl_module  = Algorithm.from_checkpoint(checkpoint_dir).get_module()

env = TaskSchedulingEnvRLLIB({})
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
        logits = logits.reshape((750, 6))
        actions = []
        for i in range(len(env.current_tasks)):
            action = np.random.choice(6, p=softmax(logits[i]))
            # action = np.argmax(logits[i])
            if action == 0:
                total_deffered_tasks += 1
            actions.append(action)

        obs, reward, done, truncated, info = env.step(actions)
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