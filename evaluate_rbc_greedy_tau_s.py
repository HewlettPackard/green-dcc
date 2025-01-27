import numpy as np
from tqdm import tqdm
from envs.heirarchical_env_cont_random_location import HeirarchicalDCRL, DEFAULT_CONFIG
from utils.hierarchical_workload_optimizer import WorkloadOptimizer
from baselines.rbc_baselines import RBCBaselines
import random

# Define the function to evaluate RBCs with varying \tau_s

def evaluate_rbc_greedy():
    results = {}
    
    # Define the range of \tau_s values
    tau_s_values = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    # Number of runs for each \tau_s value
    num_runs = 10
    for run in range(num_runs):
        # Randomize baselines for workload and temperature
        DEFAULT_CONFIG['config1']['workload_baseline'] = random.uniform(-0.4, 0.4)
        DEFAULT_CONFIG['config2']['workload_baseline'] = random.uniform(-0.4, 0.4)
        DEFAULT_CONFIG['config3']['workload_baseline'] = random.uniform(-0.4, 0.4)

        DEFAULT_CONFIG['config1']['temperature_baseline'] = random.uniform(-5.0, 5.0)
        DEFAULT_CONFIG['config2']['temperature_baseline'] = random.uniform(-5.0, 5.0)
        DEFAULT_CONFIG['config3']['temperature_baseline'] = random.uniform(-5.0, 5.0)

        max_iterations = 4*24*30  # 1 month of simulation at 15-min intervals

        DEFAULT_CONFIG['config1']['days_per_episode'] = int(max_iterations/(4*24))
        DEFAULT_CONFIG['config2']['days_per_episode'] = int(max_iterations/(4*24))
        DEFAULT_CONFIG['config3']['days_per_episode'] = int(max_iterations/(4*24))

        for tau_s in tau_s_values:
            print(f"Evaluating for \u03c4_s = {tau_s}")
    
            # Set the \tau_s value in DEFAULT_CONFIG
            DEFAULT_CONFIG['max_util'] = tau_s

            # Initialize metrics for aggregation
            energy_consumption, carbon_emissions, water_usage = [], [], []

            print(f"Run {run + 1} for \u03c4_s = {tau_s}")

            # Initialize the environment
            env = HeirarchicalDCRL(DEFAULT_CONFIG, random_locations=False)
            rbc_baseline = RBCBaselines(env)

            done = False
            truncated = False
            obs, _ = env.reset(seed=run)

            with tqdm(total=max_iterations, ncols=150) as pbar:  # 1 week of simulation at 15-min intervals
                while not done and not truncated:
                    actions = rbc_baseline.multi_step_greedy(variable='ci')  # Replace 'ci' with 'curr_temperature' for Temperature Greedy
                    obs, reward, done, truncated, info = env.step(actions)

                    # Aggregate metrics
                    energy_consumption.append(
                        sum([env.low_level_infos[dc]['agent_bat']['bat_total_energy_without_battery_KWh']
                            for dc in env.low_level_infos])
                    )
                    carbon_emissions.append(
                        sum([env.low_level_infos[dc]['agent_bat']['bat_CO2_footprint']
                            for dc in env.low_level_infos])
                    )
                    water_usage.append(
                        sum([env.low_level_infos[dc]['agent_dc']['dc_water_usage']
                            for dc in env.low_level_infos])
                    )
                    
                    pbar.update(1)

            # Aggregate and store results for this \tau_s value
            results[tau_s] = {
                "avg_energy": np.mean(energy_consumption),
                "std_energy": np.std(energy_consumption),
                "avg_carbon": np.mean(carbon_emissions),
                "std_carbon": np.std(carbon_emissions),
                "avg_water": np.mean(water_usage),
                "std_water": np.std(water_usage)
            }

    # Print results
    print("\nFinal Results:")
    for tau_s, metrics in results.items():
        print(f"\u03c4_s = {tau_s}: Energy = {metrics['avg_energy']:.2f} +- {metrics['std_energy']:.2f} kWh, "
              f"Carbon = {metrics['avg_carbon']:.2f} +- {metrics['std_carbon']:.2f} kg, "
              f"Water = {metrics['avg_water']:.2f} +- {metrics['std_water']:.2f} L")

if __name__ == "__main__":
    evaluate_rbc_greedy()
