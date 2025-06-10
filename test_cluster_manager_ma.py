# tests/test_cluster_manager_ma.py

# %%
# --- [Cell 1] Setup: Imports and Path Configuration ---
import sys
import os
import numpy as np
import pandas as pd
from collections import deque

# Add the project root to the Python path
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = os.getcwd()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Import the new MARL cluster manager and other necessary components
from simulation.cluster_manager_ma import DatacenterClusterManagerMA
from utils.config_loader import load_yaml
from rl_components.task import Task

# Helper to pretty print dictionaries
def print_dict(d, title=""):
    print(f"--- {title} ---")
    if not d:
        print(" (Empty)")
        return
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")
    print("-" * (len(title) + 8))

print("Setup complete. Imports successful.")


# %%
# --- [Cell 2] Configuration and Instantiation of DatacenterClusterManagerMA ---

# Load datacenter configurations from YAML
dc_config_path = "configs/env/datacenters.yaml"
datacenters_config_list = load_yaml(dc_config_path)["datacenters"]
# Let's test with a smaller cluster for clarity, e.g., the first 3 DCs
N_DCS_TO_TEST = 3
test_config_list = datacenters_config_list[:N_DCS_TO_TEST]

# Add necessary simulation parameters to each DC's config
for config in test_config_list:
    config['simulation_year'] = 2023
    config['agents'] = [] # Not used by SustainDC directly, but good practice
    config['evaluation'] = False

# MARL-specific configuration
# This must be >= N_DCS_TO_TEST
MAX_OPTIONS = 5 
CLOUD_PROVIDER = "aws"
WORKLOAD_PATH = "data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl"

print(f"Instantiating DatacenterClusterManagerMA with {N_DCS_TO_TEST} datacenters...")
cluster_manager = DatacenterClusterManagerMA(
    config_list=test_config_list,
    simulation_year=2023,
    tasks_file_path=WORKLOAD_PATH,
    cloud_provider=CLOUD_PROVIDER,
    max_total_options=MAX_OPTIONS,
    logger=None # Pass a real logger here if you want detailed logs
)

# Reset the cluster to a specific time
print("\nResetting the cluster manager...")
init_seed = 42
init_year, init_day, init_hour = 2023, 180, 12
cluster_manager.reset(init_year, init_day, init_hour, init_seed)
current_sim_time = pd.Timestamp(f"{init_year}-01-01 00:00:00", tz='UTC') + pd.Timedelta(days=init_day, hours=init_hour)


print("\nCluster Manager instance created and reset successfully.")
print(f"Number of nodes: {cluster_manager.num_dcs}")
print(f"Cloud provider for network: {cluster_manager.cloud_provider}")
print(f"Manager option padding size: {cluster_manager.max_total_options}")


# %%
# --- [Cell 3] Test a Single Timestep ---

print(f"\n==================== TESTING TIMESTEP: {current_sim_time} ====================")

# 1. Manually call the task origination part of the step logic
print("\n--- [B] Task Origination ---")
newly_arrived_tasks_by_origin = cluster_manager._get_newly_arrived_tasks(current_sim_time)
total_new_tasks = 0
for dc_id, tasks in newly_arrived_tasks_by_origin.items():
    if dc_id:
        print(f"  {len(tasks)} new tasks originated at DC {dc_id}")
        cluster_manager.nodes[dc_id].add_originating_tasks(tasks)
        total_new_tasks += len(tasks)
    else:
        print("  No new tasks originated at the cloud level (dc_id=0).")
print(f"Total new tasks this step: {total_new_tasks}")


# 2. Prepare observations for all DTA_Managers
print("\n--- Preparing DTA_Manager Observations ---")
manager_observations = cluster_manager._prepare_all_manager_observations(current_sim_time)

# Verify one of the manager observations
test_dc_id = 1
if test_dc_id in manager_observations:
    print_dict(manager_observations[test_dc_id], f"Observation for DTA_Manager at DC {test_dc_id}")
    padded_options = manager_observations[test_dc_id]['obs_all_options_set_padded']
    mask = manager_observations[test_dc_id]['all_options_padding_mask']
    assert padded_options.shape == (MAX_OPTIONS, cluster_manager.D_OPTION_FEAT)
    assert mask.shape == (MAX_OPTIONS,)
    assert np.sum(~mask) == N_DCS_TO_TEST # Number of valid options should match num DCs
    print(f"✅ Observation for Manager {test_dc_id} has correct shapes.")
else:
    print(f"No tasks originated at DC {test_dc_id}, so no manager observation was prepared for it in this test setup.")


# 3. Simulate getting random actions from the policies
print("\n--- Simulating Agent Actions ---")
# --- DTA_Manager Actions (randomly choose a valid destination index) ---
manager_actions = {}
for dc_id, obs_dict in manager_observations.items():
    num_valid_options = np.sum(~obs_dict['all_options_padding_mask'])
    manager_actions[dc_id] = np.random.randint(0, num_valid_options)
print(f"Random Manager Actions: {manager_actions}")

# --- DTA_Worker Actions (randomly choose True/False for Execute Now) ---
worker_actions = {dc_id: np.random.choice([True, False]) for dc_id in cluster_manager.nodes}
print(f"Random Worker Actions: {worker_actions}")


# 4. Use the full `step_marl` method to apply actions and step the simulation
print("\n--- Calling cluster_manager.step_marl() ---")
results = cluster_manager.step_marl(current_sim_time, manager_actions, worker_actions)
print("`step_marl` executed without errors.")


# 5. Inspect the results and the state of the nodes
print("\n--- Inspecting Post-Step State ---")
for dc_id, node in cluster_manager.nodes.items():
    print(f"\n  State of DC {dc_id}:")
    print(f"    Originating Queue Length: {len(node.originating_tasks_queue)}")
    print(f"    Worker Commitment Queue Length: {len(node.worker_commitment_queue)}")
    print(f"    Tasks Currently Running: {len(node.physical_dc_model.running_tasks)}")
    print(f"    Available Cores: {node.physical_dc_model.available_cores:.1f}")

print(f"\nTasks currently in transit: {len(cluster_manager.in_transit_tasks)}")

# We can check if a transfer happened
if any(action > 0 for action in manager_actions.values()):
    # We expect some tasks might be in transit, unless their delay was < 15min
    print("At least one manager chose a remote DC. Check in-transit queue.")

# Let's advance time by 15 mins and see if in-transit tasks arrive
print("\n==================== ADVANCING TO NEXT TIMESTEP ====================")
current_sim_time += pd.Timedelta(minutes=15)
print(f"New Simulation Time: {current_sim_time}")

print("\n--- Processing Arriving Tasks at start of next step ---")
# Manually call the logic that would be at the start of the next `step_marl`
remaining_in_transit = deque()
while cluster_manager.in_transit_tasks:
    arrival_time, task, dest_dc_id = cluster_manager.in_transit_tasks.popleft()
    if arrival_time <= current_sim_time:
        print(f"  Task {task.job_name} has arrived at DC {dest_dc_id}!")
        cluster_manager.nodes[dest_dc_id].add_transferred_tasks([task])
    else:
        remaining_in_transit.append((arrival_time, task, dest_dc_id))
cluster_manager.in_transit_tasks = remaining_in_transit

print("\n--- Final State Inspection ---")
for dc_id, node in cluster_manager.nodes.items():
    print(f"  State of DC {dc_id}:")
    print(f"    Worker Commitment Queue Length: {len(node.worker_commitment_queue)}")

print(f"Tasks still in transit: {len(cluster_manager.in_transit_tasks)}")
print("\n✅ Test script finished.")
# %%
