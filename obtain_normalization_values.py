import os
import random
import warnings
import copy
import torch
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
from multiprocessing import Pool, cpu_count
from functools import partial

# Suppress gym warnings if desired
warnings.filterwarnings("ignore", category=UserWarning)

# Import your custom environment
# Adjust the import path based on your project structure
from envs.heirarchical_env_cont_random_location import HeirarchicalDCRL, DEFAULT_CONFIG

def run_multiple_episodes(num_episodes, config, random_locations, seed=None):
    """
    Runs a specified number of episodes and collects carbon_footprint and water_usage.
    
    Args:
        num_episodes (int): Number of episodes to run.
        config (dict): Configuration dictionary for the environment.
        random_locations (bool): Whether to randomize datacenter locations.
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        tuple: Two lists containing carbon footprints and water usages.
    """
    carbon_list = []
    water_list = []
    
    # Set seeds for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    
    # Initialize the environment
    env = HeirarchicalDCRL(config, random_locations=random_locations)
    
    for _ in range(num_episodes):
        episode_carbon, episode_water = run_episode(env)
        carbon_list.extend(episode_carbon)
        water_list.extend(episode_water)
    
    env.close()
    return carbon_list, water_list

def run_episode(env):
    """
    Runs a single episode with random actions and collects carbon_footprint and water_usage.
    
    Args:
        env (HeirarchicalDCRL): The environment instance.
        
    Returns:
        tuple: Two lists containing carbon footprints and water usages for the episode.
    """
    carbon_list = []
    water_list = []
    obs, info = env.reset()
    done = False
    while not done:
        # Sample random action
        action = env.action_space.sample()
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Collect carbon_footprint and water_usage from low_level_infos
        for dc_id in env.low_level_infos:
            dc_info = env.low_level_infos[dc_id]
            carbon_footprint = dc_info.get('agent_bat', {}).get('bat_CO2_footprint', 0.0)
            water_usage = dc_info.get('agent_dc', {}).get('dc_water_usage', 0.0)
            carbon_list.append(carbon_footprint)
            water_list.append(water_usage)
    return carbon_list, water_list

def init_worker():
    """
    Initialize worker process.
    Suppress any unwanted warnings or setup required for each worker.
    """
    warnings.filterwarnings("ignore", category=UserWarning)

def main():
    """
    Main function to run multiple episodes in parallel and compute statistics for reward components.
    """
    # Number of total episodes to run
    total_episodes = 500  # Adjust based on your requirements
    
    # Number of parallel workers (use number of CPU cores)
    num_workers = cpu_count()
    print(f"Using {num_workers} parallel workers.")
    
    # Episodes per worker
    episodes_per_worker = total_episodes // num_workers
    remaining_episodes = total_episodes % num_workers
    
    # Configuration for each worker
    worker_configs = [copy.deepcopy(DEFAULT_CONFIG) for _ in range(num_workers)]
    
    # Seeds for reproducibility (optional)
    seeds = [random.randint(0, 1000000) for _ in range(num_workers)]
    
    # Prepare arguments for each worker
    worker_args = []
    for i in range(num_workers):
        # Distribute remaining episodes among the first few workers
        n_eps = episodes_per_worker + (1 if i < remaining_episodes else 0)
        worker_args.append((n_eps, worker_configs[i], True, seeds[i]))
    
    # Partial function to unpack arguments
    run_func = run_multiple_episodes
    
    # Initialize multiprocessing Pool
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        # Map the function to the worker arguments
        results = pool.starmap(run_func, worker_args)
    
    # Aggregate results from all workers
    all_carbon = []
    all_water = []
    for carbon, water in results:
        all_carbon.extend(carbon)
        all_water.extend(water)
    
    # Convert to numpy arrays for statistics
    all_carbon = np.array(all_carbon)
    all_water = np.array(all_water)
    
    # Compute statistics
    carbon_mean = np.mean(all_carbon)
    carbon_std = np.std(all_carbon)
    
    water_mean = np.mean(all_water)
    water_std = np.std(all_water)
    
    # Print the results
    print("\n--- Reward Components Statistics ---")
    print(f"Carbon Footprint: Mean = {carbon_mean:.2f}, Std = {carbon_std:.2f}")
    print(f"Water Usage: Mean = {water_mean:.2f}, Std = {water_std:.2f}")
    
    # Optionally, save the statistics to a file
    output_dir = "reward_stats_mp"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "reward_stats.txt"), "w") as f:
        f.write("--- Reward Components Statistics ---\n")
        f.write(f"Carbon Footprint: Mean = {carbon_mean:.2f}, Std = {carbon_std:.2f}\n")
        f.write(f"Water Usage: Mean = {water_mean:.2f}, Std = {water_std:.2f}\n")
    
    # Optionally, save the raw data for further analysis
    np.save(os.path.join(output_dir, "carbon_footprint.npy"), all_carbon)
    np.save(os.path.join(output_dir, "water_usage.npy"), all_water)
    
    print(f"\nStatistics saved in the '{output_dir}' directory.")
    
if __name__ == "__main__":
    main()
