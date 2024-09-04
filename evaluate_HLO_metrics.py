import numpy as np
from tqdm import tqdm
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

FOLDER = 'results/PPO/PPO_HeirarchicalDCRL_1a980_00000_0_2024-09-02_17-23-43'
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

# Function to run a single evaluation loop
def run_evaluation_loop(trainer, rbc_baseline, agent_type, max_iterations=4*24*7, seed=123):
    # Initialize the environment
    env = HeirarchicalDCRL(DEFAULT_CONFIG)
    done = False
    obs, _ = env.reset(seed=seed)
    
    total_reward = 0
    
    workload_DC1, workload_DC2, workload_DC3 = [], [], []
    energy_consumption_DC1, energy_consumption_DC2, energy_consumption_DC3 = [], [], []
    carbon_emissions_DC1, carbon_emissions_DC2, carbon_emissions_DC3 = [], [], []
    water_consumption_DC1, water_consumption_DC2, water_consumption_DC3 = [], [], []

    with tqdm(total=max_iterations, ncols=150) as pbar:
        while not done:
            if agent_type == 0:  # One-step RL
                actions = trainer.compute_single_action(obs, explore=False)
            elif agent_type == 1:  # One-step greedy
                hier_obs = env.get_original_observation()
                ci = [hier_obs[dc]['ci'] for dc in env.datacenters]
                sender_idx = np.argmax(ci)  # Data center with the highest carbon intensity
                receiver_idx = np.argmin(ci)  # Data center with the lowest carbon intensity
                actions = np.zeros(3)
                if sender_idx == 0 and receiver_idx == 1:
                    actions[0] = 1.0
                elif sender_idx == 0 and receiver_idx == 2:
                    actions[1] = 1.0
                elif sender_idx == 1 and receiver_idx == 0:
                    actions[0] = -1.0
                elif sender_idx == 1 and receiver_idx == 2:
                    actions[2] = 1.0
                elif sender_idx == 2 and receiver_idx == 0:
                    actions[1] = -1.0
                elif sender_idx == 2 and receiver_idx == 1:
                    actions[2] = -1.0
            elif agent_type == 2:  # Multi-step greedy
                actions = rbc_baseline.multi_step_greedy()
            elif agent_type == 3:  # Equal workload distribution
                actions = rbc_baseline.equal_workload_distribution()
            else:  # Do nothing
                actions = np.zeros(3)
            
            obs, reward, terminated, done, info = env.step(actions)
            total_reward += reward

            workload_DC1.append(env.low_level_infos['DC1']['agent_ls']['ls_original_workload'])
            workload_DC2.append(env.low_level_infos['DC2']['agent_ls']['ls_original_workload'])
            workload_DC3.append(env.low_level_infos['DC3']['agent_ls']['ls_original_workload'])

            energy_consumption_DC1.append(env.low_level_infos['DC1']['agent_bat']['bat_total_energy_without_battery_KWh'])
            energy_consumption_DC2.append(env.low_level_infos['DC2']['agent_bat']['bat_total_energy_without_battery_KWh'])
            energy_consumption_DC3.append(env.low_level_infos['DC3']['agent_bat']['bat_total_energy_without_battery_KWh'])

            carbon_emissions_DC1.append(env.low_level_infos['DC1']['agent_bat']['bat_CO2_footprint'])
            carbon_emissions_DC2.append(env.low_level_infos['DC2']['agent_bat']['bat_CO2_footprint'])
            carbon_emissions_DC3.append(env.low_level_infos['DC3']['agent_bat']['bat_CO2_footprint'])

            water_consumption_DC1.append(env.low_level_infos['DC1']['agent_dc']['dc_water_usage'])
            water_consumption_DC2.append(env.low_level_infos['DC2']['agent_dc']['dc_water_usage'])
            water_consumption_DC3.append(env.low_level_infos['DC3']['agent_dc']['dc_water_usage'])

            pbar.update(1)
    
    metrics = {
        'total_reward': total_reward,
        'workload_DC1': workload_DC1,
        'workload_DC2': workload_DC2,
        'workload_DC3': workload_DC3,
        'energy_consumption_DC1': energy_consumption_DC1,
        'energy_consumption_DC2': energy_consumption_DC2,
        'energy_consumption_DC3': energy_consumption_DC3,
        'carbon_emissions_DC1': carbon_emissions_DC1,
        'carbon_emissions_DC2': carbon_emissions_DC2,
        'carbon_emissions_DC3': carbon_emissions_DC3,
        'water_consumption_DC1': water_consumption_DC1,
        'water_consumption_DC2': water_consumption_DC2,
        'water_consumption_DC3': water_consumption_DC3,
    }

    return metrics

# Function to run multiple simulations and compute averages and standard deviations
def run_multiple_simulations(num_runs, trainer, rbc_baseline):
    all_metrics = []
    agent_types = [0, 1, 2, 3, 4]  # Different agents

    for agent_type in agent_types:
        agent_metrics = []
        for _ in range(num_runs):
            metrics = run_evaluation_loop(trainer, rbc_baseline, agent_type)
            agent_metrics.append(metrics)
        
        # Calculate average and std for each metric across all runs
        avg_metrics = {key: np.mean([m[key] for m in agent_metrics], axis=0) for key in agent_metrics[0]}
        std_metrics = {key: np.std([m[key] for m in agent_metrics], axis=0) for key in agent_metrics[0]}
        
        print(f"Agent Type {agent_type} - Averages:")
        for key in avg_metrics:
            print(f"{key}: {np.mean(avg_metrics[key]):.3f} ± {np.mean(std_metrics[key]):.3f}")
        
        all_metrics.append({'avg': avg_metrics, 'std': std_metrics})
    
    return all_metrics

# Run the simulations
num_runs = 10  # Number of simulations to average over
all_metrics = run_multiple_simulations(num_runs, trainer, rbc_baseline)

# %%
