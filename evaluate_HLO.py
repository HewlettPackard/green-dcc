'''Code used to evaluate the High Level Only Baseline for the NeuirsIPS 2024 submission'''
#%%
from tqdm import tqdm

import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from envs.heirarchical_env_cont import HeirarchicalDCRL, DEFAULT_CONFIG
from utils.hierarchical_workload_optimizer import WorkloadOptimizer
import glob
import pickle
from baselines.rbc_baselines import RBCBaselines

#%%
# trainer_single = Algorithm.from_checkpoint('./results/SingleStep/PPO_HeirarchicalDCRLWithHysterisis_59fd7_00000_0_2024-05-14_18-39-53/checkpoint_000350')
# trainer_multi = Algorithm.from_checkpoint('./results/MultiStep/PPO_HeirarchicalDCRLWithHysterisisMultistep_659f8_00000_0_2024-05-14_18-40-12/checkpoint_005145')

FOLDER = 'results/PPO/PPO_HeirarchicalDCRL_6eae7_00000_0_2024-09-03_19-33-44'
CHECKPOINT_PATH = sorted(glob.glob(FOLDER + '/checkpoint_*'))[-1]

print(f'Loading checkpoing: {CHECKPOINT_PATH}')
trainer = Algorithm.from_checkpoint(CHECKPOINT_PATH)

# print("Trained weights:")
# print(trainer.get_weights())

# # Load the specific policy state
# with open(f'{CHECKPOINT_PATH}/policies/default_policy/policy_state.pkl', 'rb') as f:
#     policy_state = pickle.load(f)

# # Load the policy state into the trainer
# trainer.set_weights(policy_state)

# # Verify the policy is loaded correctly
# print(trainer.get_weights())

# Access the default policy (single-agent setup)
policy = trainer.get_policy()

# Check if the policy's model has an eval() method (this is specific to PyTorch models)
if hasattr(policy.model, 'eval'):
    policy.model.eval()  # Set the model to evaluation mode
else:
    print("The model does not support eval mode, or it's not necessary for this type of policy.")

#%
# obtain the locations from DEFAULT_CONFIG
dc_location_mapping = {
    'DC1': DEFAULT_CONFIG['config1']['location'].upper(),
    'DC2': DEFAULT_CONFIG['config2']['location'].upper(),
    'DC3': DEFAULT_CONFIG['config3']['location'].upper(),
}
env = HeirarchicalDCRL(DEFAULT_CONFIG)

# Initialize the RBCBaselines with the environment
rbc_baseline = RBCBaselines(env)

greedy_optimizer = WorkloadOptimizer(env.datacenters.keys())
#%%
def compare_transfer_actions(actions1, actions2):
    """Compare transfer actions for equality on specific keys."""
    # Check if both actions have the same set of transfers
    if set(actions1.keys()) != set(actions2.keys()):
        return False

    # Iterate through each transfer action and compare
    for key in actions1:
        action1 = actions1[key]
        action2 = actions2[key]

        # Check the specific keys within each transfer action
        if (action1['receiver'] != action2['receiver'] or
            action1['sender'] != action2['sender'] or
            not np.array_equal(action1['workload_to_move'], action2['workload_to_move'])):
            return False

    return True

max_iterations = 4*24*7

# TODO: change the max iterations in the DEFAULT_CONFIG using the parameter max_iterations
DEFAULT_CONFIG['config1']['days_per_episode'] = int(max_iterations/(4*24))
DEFAULT_CONFIG['config2']['days_per_episode'] = int(max_iterations/(4*24))
DEFAULT_CONFIG['config3']['days_per_episode'] = int(max_iterations/(4*24))

results_all = []

# Initialize lists to store the 'current_workload' metric
workload_DC1 = [[], [], [], [], []]
workload_DC2 = [[], [], [], [], []]
workload_DC3 = [[], [], [], [], []]

# List to store the energy consumption
energy_consumption_DC1 = [[], [], [], [], []]
energy_consumption_DC2 = [[], [], [], [], []]
energy_consumption_DC3 = [[], [], [], [], []]

# Other lists to store the 'carbon_emissions' metric
carbon_emissions_DC1 = [[], [], [], [], []]
carbon_emissions_DC2 = [[], [], [], [], []]
carbon_emissions_DC3 = [[], [], [], [], []]

# Other lists to store the 'external_temperature' metric
external_temperature_DC1 = [[], [], [], [], []]
external_temperature_DC2 = [[], [], [], [], []]
external_temperature_DC3 = [[], [], [], [], []]

# List to store the water consumption metric
water_consumption_DC1 = [[], [], [], [], []]
water_consumption_DC2 = [[], [], [], [], []]
water_consumption_DC3 = [[], [], [], [], []]

# List to store the carbon intensity of each datacenter
carbon_intensity_DC1 = [[], [], [], [], []]
carbon_intensity_DC2 = [[], [], [], [], []]
carbon_intensity_DC3 = [[], [], [], [], []]

# Another list to store the carbon intensity of each datacenter
carbon_intensity = []

# 5 Different agents (One-step RL, Multi-step RL, One-step Greedy, Multi-step Greedy, Do nothing)

for i in [0, 1, 2, 3, 4]:
    env = HeirarchicalDCRL(DEFAULT_CONFIG)
    if i == 2 or i == 3:
        rbc_baseline = RBCBaselines(env)
    done = False
    obs, _ = env.reset(seed=43)

    actions_list = []
    rewards_list = []
    total_reward = 0
    
    with tqdm(total=max_iterations, ncols=150) as pbar:
        while not done:
            if i == 0:
                # print('One-step RL')
                # if obs = {'DC1': {'curr_workload': array([1.]), 'ci': array([-0.38324634])}, 'DC2': {'curr_workload': array([1.]), 'ci': array([0.73191553])}, 'DC3': {'curr_workload': array([0.]), 'ci': array([-0.55756748])}}, I want to explore the actions of the agents under random values of the observation
                # obs = {'DC1': {'curr_workload': np.random.uniform(0, 1, size=(1,)), 'ci': np.random.uniform(-1, 1, size=(1,))},
                #        'DC2': {'curr_workload': np.random.uniform(0, 1, size=(1,)), 'ci': np.random.uniform(-1, 1, size=(1,))},
                #        'DC3': {'curr_workload': np.random.uniform(0, 1, size=(1,)), 'ci': np.random.uniform(-1, 1, size=(1,))}
                #        }
                actions = trainer.compute_single_action(obs, explore=False)
                # print(obs)
                # print(actions)
            elif i == 1:
                # One-step greedy
                # print('One-step Greedy')
                hier_obs = env.get_original_observation()
                ci = [hier_obs[dc]['ci'] for dc in env.datacenters]
                # denorm_ci = [env.low_level_infos[dc_key]['agent_bat']['bat_avg_CI'] for dc_key in env.datacenters.keys()]
                # carbon_intensity.append(denorm_ci)
                # actions = {'receiver': np.argmin(ci), 'sender': np.argmax(ci), 'workload_to_move': np.array([1.])}
                # actions = {'transfer_1': actions}
                
                # Continuous action space
                actions = np.zeros(3)
                
                # Identify the indices of the sender and receiver based on carbon intensity
                sender_idx = np.argmax(ci)  # Data center with the highest carbon intensity
                receiver_idx = np.argmin(ci)  # Data center with the lowest carbon intensity

                # Map sender-receiver pairs to the action array indices
                if sender_idx == 0 and receiver_idx == 1:
                    actions[0] = 0.9  # Transfer from DC1 to DC2
                elif sender_idx == 0 and receiver_idx == 2:
                    actions[1] = 0.9  # Transfer from DC1 to DC3
                elif sender_idx == 1 and receiver_idx == 0:
                    actions[0] = -0.9  # Transfer from DC2 to DC1
                elif sender_idx == 1 and receiver_idx == 2:
                    actions[2] = 0.9  # Transfer from DC2 to DC3
                elif sender_idx == 2 and receiver_idx == 0:
                    actions[1] = -0.9  # Transfer from DC3 to DC1
                elif sender_idx == 2 and receiver_idx == 1:
                    actions[2] = -0.9  # Transfer from DC3 to DC2
            elif i == 2:
                # Multi-step greedy
                actions = rbc_baseline.multi_step_greedy()
            elif i == 3:
                # Equal workload distribution
                # Use the equal workload distribution method
                actions = rbc_baseline.equal_workload_distribution()
            else:
                # Do nothing
                # Continuous action space
                actions = np.zeros(3)  # All transfers are set to 0.0

            
            obs, reward, terminated, done, info = env.step(actions)
            
            # Obtain the 'current_workload' metric for each datacenter using the low_level_infos -> agent_ls -> ls_original_workload
            workload_DC1[i].append(env.low_level_infos['DC1']['agent_ls']['ls_original_workload'])
            workload_DC2[i].append(env.low_level_infos['DC2']['agent_ls']['ls_original_workload'])
            workload_DC3[i].append(env.low_level_infos['DC3']['agent_ls']['ls_original_workload'])
            
            # Obtain the energy consumption
            energy_consumption_DC1[i].append(env.low_level_infos['DC1']['agent_bat']['bat_total_energy_without_battery_KWh'])
            energy_consumption_DC2[i].append(env.low_level_infos['DC2']['agent_bat']['bat_total_energy_without_battery_KWh'])
            energy_consumption_DC3[i].append(env.low_level_infos['DC3']['agent_bat']['bat_total_energy_without_battery_KWh'])

            # Obtain the 'carbon_emissions' metric for each datacenter using the low_level_infos -> agent_bat -> bat_CO2_footprint
            carbon_emissions_DC1[i].append(env.low_level_infos['DC1']['agent_bat']['bat_CO2_footprint'])
            carbon_emissions_DC2[i].append(env.low_level_infos['DC2']['agent_bat']['bat_CO2_footprint'])
            carbon_emissions_DC3[i].append(env.low_level_infos['DC3']['agent_bat']['bat_CO2_footprint'])

            # Obtain the 'external_temperature' metric for each datacenter using the low_level_infos -> agent_dc -> dc_exterior_ambient_temp
            external_temperature_DC1[i].append(env.low_level_infos['DC1']['agent_dc']['dc_exterior_ambient_temp'])
            external_temperature_DC2[i].append(env.low_level_infos['DC2']['agent_dc']['dc_exterior_ambient_temp'])
            external_temperature_DC3[i].append(env.low_level_infos['DC3']['agent_dc']['dc_exterior_ambient_temp'])
            
            # Obtain the 'water_consumption' metric for each datacenter using the low_level_infos -> agent_dc -> dc_water_usage
            water_consumption_DC1[i].append(env.low_level_infos['DC1']['agent_dc']['dc_water_usage'])
            water_consumption_DC2[i].append(env.low_level_infos['DC2']['agent_dc']['dc_water_usage'])
            water_consumption_DC3[i].append(env.low_level_infos['DC3']['agent_dc']['dc_water_usage'])
            
            # Obtain the carbon intensity of each datacenter using the low_level_infos -> agent_bat -> bat_avg_CI
            carbon_intensity_DC1[i].append(env.low_level_infos['DC1']['agent_bat']['bat_avg_CI'])
            carbon_intensity_DC2[i].append(env.low_level_infos['DC2']['agent_bat']['bat_avg_CI'])
            carbon_intensity_DC3[i].append(env.low_level_infos['DC3']['agent_bat']['bat_avg_CI'])
            
            total_reward += reward
    
            # actions_list.append(actions['transfer_1'])
            rewards_list.append(reward)
            
            pbar.update(1)

    results_all.append((actions_list, rewards_list))
    # print(f'Not computed workload: {env.not_computed_workload:.2f}')
    # pbar.close()

    print(f'Total reward: {total_reward:.3f}')
    print(f'Average energy consumption: {(np.mean(energy_consumption_DC1[i]) + np.mean(energy_consumption_DC2[i]) + np.mean(energy_consumption_DC3[i]))/3:.3f} Kwh')
    print(f'Average carbon emissions: {(np.mean(carbon_emissions_DC1[i]) + np.mean(carbon_emissions_DC2[i]) + np.mean(carbon_emissions_DC3[i]))/3:.3f} MgCO2')
    print(f'Average water consumption: {(np.mean(water_consumption_DC1[i]) + np.mean(water_consumption_DC2[i]) + np.mean(water_consumption_DC3[i]))/3:.3f} m3')
#%%
# First of all, let's smooth the metrics before plotting.
# We can smooth the metrics using the moving average method.
# We will use a window of 1 hour (4 timestep) for the moving average.

win_size = 8
workload_DC1 = np.array(workload_DC1)
workload_DC2 = np.array(workload_DC2)
workload_DC3 = np.array(workload_DC3)

energy_consumption_DC1 = np.array(energy_consumption_DC1)
energy_consumption_DC2 = np.array(energy_consumption_DC2)
energy_consumption_DC3 = np.array(energy_consumption_DC3)

carbon_emissions_DC1 = np.array(carbon_emissions_DC1)
carbon_emissions_DC2 = np.array(carbon_emissions_DC2)
carbon_emissions_DC3 = np.array(carbon_emissions_DC3)

external_temperature_DC1 = np.array(external_temperature_DC1)
external_temperature_DC2 = np.array(external_temperature_DC2)
external_temperature_DC3 = np.array(external_temperature_DC3)

carbon_intensity_DC1 = np.array(carbon_intensity_DC1)
carbon_intensity_DC2 = np.array(carbon_intensity_DC2)
carbon_intensity_DC3 = np.array(carbon_intensity_DC3)

# Smooth the 'current_workload' metric, remeber that workload_DC1.shape=(num_controllers, time_steps).
# Use scipy.ndimage.filters with 1D filter to only smooth the time dimension.
from scipy.ndimage import uniform_filter1d
smoothed_workload_DC1 = uniform_filter1d(workload_DC1, size=win_size, axis=1)
smoothed_workload_DC2 = uniform_filter1d(workload_DC2, size=win_size, axis=1)
smoothed_workload_DC3 = uniform_filter1d(workload_DC3, size=win_size, axis=1)

# Smooth the energy consumption metric
energy_consumption_DC1 = uniform_filter1d(energy_consumption_DC1, size=win_size, axis=1)
energy_consumption_DC2 = uniform_filter1d(energy_consumption_DC2, size=win_size, axis=1)
energy_consumption_DC3 = uniform_filter1d(energy_consumption_DC3, size=win_size, axis=1)

# Smooth the 'carbon_emissions' metric
smoothed_carbon_emissions_DC1 = uniform_filter1d(carbon_emissions_DC1, size=win_size, axis=1)
smoothed_carbon_emissions_DC2 = uniform_filter1d(carbon_emissions_DC2, size=win_size, axis=1)
smoothed_carbon_emissions_DC3 = uniform_filter1d(carbon_emissions_DC3, size=win_size, axis=1)

# Smooth the 'external_temperature' metric
smoothed_external_temperature_DC1 = uniform_filter1d(external_temperature_DC1, size=win_size, axis=1)
smoothed_external_temperature_DC2 = uniform_filter1d(external_temperature_DC2, size=win_size, axis=1)
smoothed_external_temperature_DC3 = uniform_filter1d(external_temperature_DC3, size=win_size, axis=1)

# Smooth the 'carbon_intensity' metric
smoothed_carbon_intensity_DC1 = uniform_filter1d(carbon_intensity_DC1, size=win_size, axis=1)
smoothed_carbon_intensity_DC2 = uniform_filter1d(carbon_intensity_DC2, size=win_size, axis=1)
smoothed_carbon_intensity_DC3 = uniform_filter1d(carbon_intensity_DC3, size=win_size, axis=1)
 
#%%
import matplotlib.pyplot as plt
# Plot the 'current_workload' metric
# controllers = ['One-step RL', 'Multi-step RL', 'One-step Greedy', 'Multi-step Greedy', 'Do nothing']
controllers = ['RL', 'One-step Greedy', 'Multi-step Greedy', 'Equal Distributed', 'Do nothing']

for i in range(len(controllers)):
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_workload_DC1[i][:4*24*7]*100, label=dc_location_mapping['DC1'], linestyle='--', linewidth=2, alpha=1)
    plt.plot(smoothed_workload_DC2[i][:4*24*7]*100, label=dc_location_mapping['DC2'], linestyle='-.', linewidth=2, alpha=0.9)
    plt.plot(smoothed_workload_DC3[i][:4*24*7]*100, label=dc_location_mapping['DC3'], linestyle='-', linewidth=2, alpha=0.7)
    plt.title(f'Current Workload for {controllers[i]} Controller')
    plt.xlabel('Time Step')
    plt.ylabel('Current Workload (%)')
    plt.legend()
    plt.grid('on', linestyle='--', alpha=0.5)
    plt.ylim(-10, 111)
    plt.show()


#%% Plot the energy consumption metric
for i in range(len(controllers)):
    plt.figure(figsize=(10, 6))
    plt.plot(energy_consumption_DC1[i][:4*24*7], label=dc_location_mapping['DC1'], linestyle='--', linewidth=2, alpha=1)
    plt.plot(energy_consumption_DC2[i][:4*24*7], label=dc_location_mapping['DC2'], linestyle='-.', linewidth=2, alpha=0.9)
    plt.plot(energy_consumption_DC3[i][:4*24*7], label=dc_location_mapping['DC3'], linestyle='-', linewidth=2, alpha=0.7)
    plt.title(f'Current Workload for {controllers[i]} Controller')
    plt.xlabel('Time Step')
    plt.ylabel('Energy Consumption (Kwh)')
    plt.legend()
    plt.grid('on', linestyle='--', alpha=0.5)
    # plt.ylim(0, 101)
    plt.show()

# Print the sum energy consumption of the different controllers
#%% Plot the 'carbon_emissions' metric

for i in range(len(controllers)):
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_carbon_emissions_DC1[i][:4*24*7]/1e6, label=dc_location_mapping['DC1'], linestyle='--', linewidth=2, alpha=1)
    plt.plot(smoothed_carbon_emissions_DC2[i][:4*24*7]/1e6, label=dc_location_mapping['DC2'], linestyle='-.', linewidth=2, alpha=0.9)
    plt.plot(smoothed_carbon_emissions_DC3[i][:4*24*7]/1e6, label=dc_location_mapping['DC3'], linestyle='-', linewidth=2, alpha=0.7)
    plt.title(f'Carbon Emissions for {controllers[i]} Controller')
    plt.xlabel('Time Step')
    plt.ylabel('Carbon Emissions (MgCO2)')
    plt.legend()
    plt.grid('on', linestyle='--', alpha=0.5)
    # plt.ylim(0.2, 1)
    plt.show()

#%% Let's plot the carbon intensity for each datacenter
# First, adapt the carbon_intensity list to be a numpy array. On each time step, the carbon intensity of each datacenter is stored in a list
# We need to convert this list to a numpy array to plot it np.array(carbon_intensity).shape = (time_steps, num_datacenters)
# Only plot the first week (:4*24*7)

plt.figure(figsize=(10, 6))
plt.plot(smoothed_carbon_intensity_DC1[0][:4*24*7], label=dc_location_mapping['DC1'], linestyle='--', linewidth=2, alpha=1)
plt.plot(smoothed_carbon_intensity_DC2[0][:4*24*7], label=dc_location_mapping['DC2'], linestyle='-.', linewidth=2, alpha=0.9)
plt.plot(smoothed_carbon_intensity_DC3[0][:4*24*7], label=dc_location_mapping['DC3'], linestyle='-', linewidth=2, alpha=0.7)
plt.title('Carbon Intensity for Each Datacenter')
plt.xlabel('Time Step')
plt.ylabel('Carbon Intensity (gCO2/kWh)')
plt.legend(dc_location_mapping.values())
plt.grid('on', linestyle='--', alpha=0.5)
plt.show()

#%% 
'''
Now I want to plot the sum of carbon emissions for each controller on each timestep on the 3 locations (DC1, DC2, and DC3)
So, we can compare the sum of carbon emissions on each timestep for each controller.
The sum of carbon emissions is calculated as the sum of carbon emissions for each datacenter on each timestep.
For example, for the first controller (RL), the sum of carbon emissions on each timestep is calculated as:
sum_carbon_emissions = carbon_emissions_DC1[0] + carbon_emissions_DC2[0] + carbon_emissions_DC3[0]
'''
sum_carbon_emissions = []
for i in range(len(controllers)):
    sum_carbon_emissions.append(np.array(smoothed_carbon_emissions_DC1[i]) + np.array(smoothed_carbon_emissions_DC2[i]) + np.array(smoothed_carbon_emissions_DC3[i]))

# Plot the sum of carbon emissions for each controller on the same figure with different colors
plt.figure(figsize=(10, 6))
linestyles = ['--', '-.', '-', '--', '-.']
for i in range(len(controllers)):
    plt.plot(sum_carbon_emissions[i][:4*24*7]/1e6, label=controllers[i], linestyle=linestyles[i], linewidth=2, alpha=0.9)
    
plt.title('Sum of Carbon Emissions for Each Controller')
plt.xlabel('Time Step')
plt.ylabel('Sum of Carbon Emissions (MgCO2)')
plt.legend()
plt.grid('on', linestyle='--', alpha=0.5)
plt.show()

# %% Plot to represent the external temperature for each datacenter
i = 0
plt.figure(figsize=(10, 6))
plt.plot(external_temperature_DC1[i][:4*24*7], label=dc_location_mapping['DC1'], linestyle='-', linewidth=2, alpha=1)
plt.plot(external_temperature_DC2[i][:4*24*7], label=dc_location_mapping['DC2'], linestyle='-', linewidth=2, alpha=1)
plt.plot(external_temperature_DC3[i][:4*24*7], label=dc_location_mapping['DC3'], linestyle='-', linewidth=2, alpha=1)
plt.title(f'External Temperature on Each Controller')
plt.xlabel('Time Step')
plt.ylabel('External Temperature (°C)')
plt.legend()
plt.grid('on', linestyle='--', alpha=0.5)
plt.show()

# %%
import pandas as pd
# Organize the data by agent type
agent_data = {
    "RL Agent": {
        "Total Reward": [653.303, 691.818, 628.551, 617.627, 603.921],
        "Average Energy Consumption (Kwh)": [268.892, 261.257, 261.902, 272.090, 275.621],
        "Average Carbon Emissions (MgCO2)": [157786.788, 154413.867, 159954.490, 160911.155, 162111.437],
        "Average Water Consumption (m3)": [792.488, 765.893, 768.503, 786.667, 782.350]
    },
    "Single Step Greedy": {
        "Total Reward": [312.827, 337.403, 219.580, 206.932, 219.580],
        "Average Energy Consumption (Kwh)": [347.027, 337.556, 355.307, 359.854, 355.307],
        "Average Carbon Emissions (MgCO2)": [187604.068, 185451.872, 195770.192, 196877.847, 195770.192],
        "Average Water Consumption (m3)": [947.795, 918.471, 948.334, 950.352, 948.334]
    },
    "Multi Step Greedy": {
        "Total Reward": [-126.149, -51.072, -239.828, -196.286, -196.286],
        "Average Energy Consumption (Kwh)": [428.776, 409.079, 438.723, 434.761, 434.761],
        "Average Carbon Emissions (MgCO2)": [226047.543, 219472.642, 236003.023, 232189.772, 232189.772],
        "Average Water Consumption (m3)": [1097.301, 1049.803, 1102.135, 1091.683, 1091.683]
    },
    "Equally Distributed": {
        "Total Reward": [583.282, 607.073, 537.161, 545.987, 537.329],
        "Average Energy Consumption (Kwh)": [270.646, 264.787, 277.266, 277.186, 280.364],
        "Average Carbon Emissions (MgCO2)": [163918.946, 161835.392, 167957.960, 167185.077, 167943.239],
        "Average Water Consumption (m3)": [796.941, 774.887, 798.342, 785.642, 790.977]
    },
    "Do Nothing": {
        "Total Reward": [565.175, 502.696, 545.987, 537.329, 537.329],
        "Average Energy Consumption (Kwh)": [274.301, 280.768, 277.186, 280.364, 280.364],
        "Average Carbon Emissions (MgCO2)": [165504.682, 170976.241, 167957.960, 167943.239, 167943.239],
        "Average Water Consumption (m3)": [803.632, 804.786, 798.342, 790.977, 790.977]
    }
}

# Calculate mean and std for each agent
agent_summary_stats = {
    agent: {metric: {"Mean": np.mean(values), "Std": np.std(values)} for metric, values in metrics.items()}
    for agent, metrics in agent_data.items()
}

# Convert to DataFrame for better visualization
agent_summary_df = pd.concat({agent: pd.DataFrame(metrics).T for agent, metrics in agent_summary_stats.items()}, axis=1)


#%%