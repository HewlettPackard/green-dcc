#%%
from tqdm import tqdm
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from envs.heirarchical_env_cont import HeirarchicalDCRL, DEFAULT_CONFIG
from envs.truly_heirarchical_env import TrulyHeirarchicalDCRL
from utils.hierarchical_workload_optimizer import WorkloadOptimizer
import glob
import pickle
from baselines.rbc_baselines import RBCBaselines
#%%
# Load the checkpoint
FOLDER = 'results/TrulyPPO/PPO_TrulyHeirarchicalDCRL_e153d_00000_0_2024-09-04_00-23-16'
CHECKPOINT_PATH = sorted(glob.glob(FOLDER + '/checkpoint_*'))[-1]

print(f'Loading checkpoint: {CHECKPOINT_PATH}')
trainer = Algorithm.from_checkpoint(CHECKPOINT_PATH)

# Access policies
policies = {
    "high_level_policy": trainer.get_policy("high_level_policy"),
    "DC1_ls_policy": trainer.get_policy("DC1_ls_policy"),
    "DC2_ls_policy": trainer.get_policy("DC2_ls_policy"),
    "DC3_ls_policy": trainer.get_policy("DC3_ls_policy"),
}

# Set the models to evaluation mode (for PyTorch)
for policy_name, policy in policies.items():
    if hasattr(policy.model, 'eval'):
        policy.model.eval()
#%%
dc_location_mapping = {
    'DC1': DEFAULT_CONFIG['config1']['location'].upper(),
    'DC2': DEFAULT_CONFIG['config2']['location'].upper(),
    'DC3': DEFAULT_CONFIG['config3']['location'].upper(),
}

# Initialize environment and baselines
env = TrulyHeirarchicalDCRL(DEFAULT_CONFIG)

# Set the number of iterations for evaluation
max_iterations = 4*24*7  # Example: 7 days with 4 intervals per hour

# Update environment config to match evaluation length
DEFAULT_CONFIG['config1']['days_per_episode'] = int(max_iterations/(4*24))
DEFAULT_CONFIG['config2']['days_per_episode'] = int(max_iterations/(4*24))
DEFAULT_CONFIG['config3']['days_per_episode'] = int(max_iterations/(4*24))

# Define different agent strategies
strategies = {
    0: "PPO Agent",
    1: "One-step Greedy",
    2: "Multi-step Greedy",
    3: "Equal Workload Distribution",
    4: "Do Nothing"
}
#%%
results_all = []

for strategy_id, strategy_name in strategies.items():
    # Reset the metrics dictionary for each strategy
    metrics = {
        "original_workload_DC1": [],
        "original_workload_DC2": [],
        "original_workload_DC3": [],
        "shifted_workload_DC1": [],
        "shifted_workload_DC2": [],
        "shifted_workload_DC3": [],
        "energy_consumption_DC1": [],
        "energy_consumption_DC2": [],
        "energy_consumption_DC3": [],
        "carbon_emissions_DC1": [],
        "carbon_emissions_DC2": [],
        "carbon_emissions_DC3": [],
        "external_temperature_DC1": [],
        "external_temperature_DC2": [],
        "external_temperature_DC3": [],
        "water_consumption_DC1": [],
        "water_consumption_DC2": [],
        "water_consumption_DC3": [],
        "carbon_intensity_DC1": [],
        "carbon_intensity_DC2": [],
        "carbon_intensity_DC3": [],
    }

    env = TrulyHeirarchicalDCRL(DEFAULT_CONFIG)
    
    rbc_baseline = RBCBaselines(env)
    greedy_optimizer = WorkloadOptimizer(env.datacenters.keys())

    done = False
    obs, _ = env.reset(seed=43)
    total_reward = 0
    
    with tqdm(total=max_iterations, ncols=150) as pbar:
        while not done:
            if strategy_id == 0:  # PPO Agent
                actions = {
                    "high_level_policy": policies["high_level_policy"].compute_single_action(obs["high_level_policy"], explore=False)[0],
                    "DC1_ls_policy": policies["DC1_ls_policy"].compute_single_action(obs["DC1_ls_policy"], explore=False)[0],
                    "DC2_ls_policy": policies["DC2_ls_policy"].compute_single_action(obs["DC2_ls_policy"], explore=False)[0],
                    "DC3_ls_policy": policies["DC3_ls_policy"].compute_single_action(obs["DC3_ls_policy"], explore=False)[0],
                }
            elif strategy_id == 1:  # One-step Greedy
                hier_obs = env.get_original_observation()
                ci = [hier_obs[dc]['ci'] for dc in env.datacenters]
                high_level_action = np.zeros(3)
                
                sender_idx = np.argmax(ci)
                receiver_idx = np.argmin(ci)

                if sender_idx == 0 and receiver_idx == 1:
                    high_level_action[0] = 1.0  # Transfer from DC1 to DC2
                elif sender_idx == 0 and receiver_idx == 2:
                    high_level_action[1] = 1.0  # Transfer from DC1 to DC3
                elif sender_idx == 1 and receiver_idx == 0:
                    high_level_action[0] = -1.0  # Transfer from DC2 to DC1
                elif sender_idx == 1 and receiver_idx == 2:
                    high_level_action[2] = 1.0  # Transfer from DC2 to DC3
                elif sender_idx == 2 and receiver_idx == 0:
                    high_level_action[1] = -1.0  # Transfer from DC3 to DC1
                elif sender_idx == 2 and receiver_idx == 1:
                    high_level_action[2] = -1.0  # Transfer from DC3 to DC2
                
                actions = {
                    "high_level_policy": high_level_action,
                    "DC1_ls_policy": np.array([0], dtype=np.float32),
                    "DC2_ls_policy": np.array([0], dtype=np.float32),
                    "DC3_ls_policy": np.array([0], dtype=np.float32),
                }
            elif strategy_id == 2:  # Multi-step Greedy
                high_level_action = rbc_baseline.multi_step_greedy()
                actions = {
                    "high_level_policy": high_level_action,
                    "DC1_ls_policy": np.array([0], dtype=np.float32),
                    "DC2_ls_policy": np.array([0], dtype=np.float32),
                    "DC3_ls_policy": np.array([0], dtype=np.float32),
                }
            elif strategy_id == 3:  # Equal Workload Distribution
                high_level_action = rbc_baseline.equal_workload_distribution()
                actions = {
                    "high_level_policy": high_level_action,
                    "DC1_ls_policy": np.array([0], dtype=np.float32),
                    "DC2_ls_policy": np.array([0], dtype=np.float32),
                    "DC3_ls_policy": np.array([0], dtype=np.float32),
                }
            else:  # Do Nothing
                actions = {
                    "high_level_policy": np.zeros(3, dtype=np.float32),  # No transfer
                    "DC1_ls_policy": np.array([0], dtype=np.float32),
                    "DC2_ls_policy": np.array([0], dtype=np.float32),
                    "DC3_ls_policy": np.array([0], dtype=np.float32),
                }

            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated['__all__'] or truncated['__all__']

            metrics["original_workload_DC1"].append(env.low_level_infos['DC1']['agent_ls']['ls_original_workload'])
            metrics["original_workload_DC2"].append(env.low_level_infos['DC2']['agent_ls']['ls_original_workload'])
            metrics["original_workload_DC3"].append(env.low_level_infos['DC3']['agent_ls']['ls_original_workload'])
            
            metrics["shifted_workload_DC1"].append(env.low_level_infos['DC1']['agent_ls']['ls_shifted_workload'])
            metrics["shifted_workload_DC2"].append(env.low_level_infos['DC2']['agent_ls']['ls_shifted_workload'])
            metrics["shifted_workload_DC3"].append(env.low_level_infos['DC3']['agent_ls']['ls_shifted_workload'])

            metrics["energy_consumption_DC1"].append(env.low_level_infos['DC1']['agent_bat']['bat_total_energy_without_battery_KWh'])
            metrics["energy_consumption_DC2"].append(env.low_level_infos['DC2']['agent_bat']['bat_total_energy_without_battery_KWh'])
            metrics["energy_consumption_DC3"].append(env.low_level_infos['DC3']['agent_bat']['bat_total_energy_without_battery_KWh'])

            metrics["carbon_emissions_DC1"].append(env.low_level_infos['DC1']['agent_bat']['bat_CO2_footprint'])
            metrics["carbon_emissions_DC2"].append(env.low_level_infos['DC2']['agent_bat']['bat_CO2_footprint'])
            metrics["carbon_emissions_DC3"].append(env.low_level_infos['DC3']['agent_bat']['bat_CO2_footprint'])

            metrics["external_temperature_DC1"].append(env.low_level_infos['DC1']['agent_dc']['dc_exterior_ambient_temp'])
            metrics["external_temperature_DC2"].append(env.low_level_infos['DC2']['agent_dc']['dc_exterior_ambient_temp'])
            metrics["external_temperature_DC3"].append(env.low_level_infos['DC3']['agent_dc']['dc_exterior_ambient_temp'])

            metrics["water_consumption_DC1"].append(env.low_level_infos['DC1']['agent_dc']['dc_water_usage'])
            metrics["water_consumption_DC2"].append(env.low_level_infos['DC2']['agent_dc']['dc_water_usage'])
            metrics["water_consumption_DC3"].append(env.low_level_infos['DC3']['agent_dc']['dc_water_usage'])

            metrics["carbon_intensity_DC1"].append(env.low_level_infos['DC1']['agent_bat']['bat_avg_CI'])
            metrics["carbon_intensity_DC2"].append(env.low_level_infos['DC2']['agent_bat']['bat_avg_CI'])
            metrics["carbon_intensity_DC3"].append(env.low_level_infos['DC3']['agent_bat']['bat_avg_CI'])

            total_reward += reward['high_level_policy']
            pbar.update(1)

    results_all.append(metrics)
    
    print(f'Strategy: {strategy_name}')
    print(f'Total reward: {total_reward:.3f}')
    print(f'Average energy consumption: {(np.mean(metrics["energy_consumption_DC1"]) + np.mean(metrics["energy_consumption_DC2"]) + np.mean(metrics["energy_consumption_DC3"]))/3:.3f} Kwh')
    print(f'Average carbon emissions: {(np.mean(metrics["carbon_emissions_DC1"]) + np.mean(metrics["carbon_emissions_DC2"]) + np.mean(metrics["carbon_emissions_DC3"]))/3:.3f} MgCO2')
    print(f'Average water consumption: {(np.mean(metrics["water_consumption_DC1"]) + np.mean(metrics["water_consumption_DC2"]) + np.mean(metrics["water_consumption_DC3"]))/3:.3f} m3')

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Assuming you have a list of metrics dictionaries, one for each strategy
win_size = 8

# Define controllers/strategies names
controllers = ['RL', 'One-step Greedy', 'Multi-step Greedy', 'Equal Distributed', 'Do nothing']

# Assuming `results_all` is a list of metrics dictionaries, where each element corresponds to a different strategy
for strategy_idx, metrics in enumerate(results_all):
    # Convert lists to numpy arrays for easier manipulation
    metrics_np = {key: np.array(value) for key, value in metrics.items()}

    # Smoothing the data using a uniform filter
    smoothed_metrics = {key: uniform_filter1d(value, size=win_size) for key, value in metrics_np.items()}

    # Plot the original workload vs shifted workload
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_metrics["original_workload_DC1"][:4*24*7] * 100, label=f'Original {dc_location_mapping["DC1"]}', linestyle='--', linewidth=2, alpha=1)
    plt.plot(smoothed_metrics["shifted_workload_DC1"][:4*24*7] * 100, label=f'Shifted {dc_location_mapping["DC1"]}', linestyle='-', linewidth=2, alpha=0.7)
    plt.plot(smoothed_metrics["original_workload_DC2"][:4*24*7] * 100, label=f'Original {dc_location_mapping["DC2"]}', linestyle='--', linewidth=2, alpha=0.9)
    plt.plot(smoothed_metrics["shifted_workload_DC2"][:4*24*7] * 100, label=f'Shifted {dc_location_mapping["DC2"]}', linestyle='-', linewidth=2, alpha=0.6)
    plt.plot(smoothed_metrics["original_workload_DC3"][:4*24*7] * 100, label=f'Original {dc_location_mapping["DC3"]}', linestyle='--', linewidth=2, alpha=0.8)
    plt.plot(smoothed_metrics["shifted_workload_DC3"][:4*24*7] * 100, label=f'Shifted {dc_location_mapping["DC3"]}', linestyle='-', linewidth=2, alpha=0.5)
    plt.title(f'Original vs Shifted Workload for {controllers[strategy_idx]} Controller')
    plt.xlabel('Time Step')
    plt.ylabel('Workload (%)')
    plt.legend()
    plt.grid('on', linestyle='--', alpha=0.5)
    plt.ylim(-10, 111)
    plt.show()

    # # Plot energy consumption for each strategy
    # plt.figure(figsize=(10, 6))
    # plt.plot(smoothed_metrics["energy_consumption_DC1"][:4*24*7], label=dc_location_mapping['DC1'], linestyle='--', linewidth=2, alpha=1)
    # plt.plot(smoothed_metrics["energy_consumption_DC2"][:4*24*7], label=dc_location_mapping['DC2'], linestyle='-.', linewidth=2, alpha=0.9)
    # plt.plot(smoothed_metrics["energy_consumption_DC3"][:4*24*7], label=dc_location_mapping['DC3'], linestyle='-', linewidth=2, alpha=0.7)
    # plt.title(f'Energy Consumption for {controllers[strategy_idx]} Controller')
    # plt.xlabel('Time Step')
    # plt.ylabel('Energy Consumption (Kwh)')
    # plt.legend()
    # plt.grid('on', linestyle='--', alpha=0.5)
    # plt.show()

    # # Plot carbon emissions for each strategy
    # plt.figure(figsize=(10, 6))
    # plt.plot(smoothed_metrics["carbon_emissions_DC1"][:4*24*7] / 1e6, label=dc_location_mapping['DC1'], linestyle='--', linewidth=2, alpha=1)
    # plt.plot(smoothed_metrics["carbon_emissions_DC2"][:4*24*7] / 1e6, label=dc_location_mapping['DC2'], linestyle='-.', linewidth=2, alpha=0.9)
    # plt.plot(smoothed_metrics["carbon_emissions_DC3"][:4*24*7] / 1e6, label=dc_location_mapping['DC3'], linestyle='-', linewidth=2, alpha=0.7)
    # plt.title(f'Carbon Emissions for {controllers[strategy_idx]} Controller')
    # plt.xlabel('Time Step')
    # plt.ylabel('Carbon Emissions (MgCO2)')
    # plt.legend()
    # plt.grid('on', linestyle='--', alpha=0.5)
    # plt.show()

# # Plot carbon intensity for each datacenter for the first strategy as an example
# plt.figure(figsize=(10, 6))
# plt.plot(smoothed_metrics["carbon_intensity_DC1"][:4*24*7], label=dc_location_mapping['DC1'], linestyle='--', linewidth=2, alpha=1)
# plt.plot(smoothed_metrics["carbon_intensity_DC2"][:4*24*7], label=dc_location_mapping['DC2'], linestyle='-.', linewidth=2, alpha=0.9)
# plt.plot(smoothed_metrics["carbon_intensity_DC3"][:4*24*7], label=dc_location_mapping['DC3'], linestyle='-', linewidth=2, alpha=0.7)
# plt.title('Carbon Intensity for Each Datacenter (First Strategy)')
# plt.xlabel('Time Step')
# plt.ylabel('Carbon Intensity (gCO2/kWh)')
# plt.legend(dc_location_mapping.values())
# plt.grid('on', linestyle='--', alpha=0.5)
# plt.show()


#%%


#%%