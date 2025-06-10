# tests/test_datacenter_node.py

# %%
# --- [Cell 1] Setup: Imports and Path Configuration ---
import sys
import os
import numpy as np
import pandas as pd
from collections import deque

# Add the project root to the Python path to allow for package imports
# This is necessary if you run this script from the `tests` directory
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    # If running interactively, assume the current working directory is the project root
    project_root = os.getcwd()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


# Import the classes and functions we want to test
from envs.datacenter_node_ma import DatacenterNodeMA
from rl_components.task import Task
from utils.marl_utils import MANAGER_META_TASK_FEATURE_ORDER, WORKER_META_TASK_FEATURE_ORDER

# --- Helper to pretty print dictionaries ---
def print_dict(d, title=""):
    print(f"--- {title} ---")
    if not d:
        print(" (Empty)")
        return
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}\n    {np.round(v, 3)}")
        else:
            print(f"  {k}: {v}")
    print("-" * (len(title) + 8))

print("Setup complete. Imports successful.")


# %%
# --- [Cell 2] Configuration and Instantiation of DatacenterNodeMA ---

# Create a sample datacenter configuration dictionary (similar to one entry in datacenters.yaml)
sample_dc_config = {
    'dc_id': 1,
    'location': "US-CAL-CISO",
    'timezone_shift': -7,
    'population_weight': 0.18,
    'total_cores': 50000,
    'total_gpus': 1000,
    'total_mem': 80000,
    'dc_config_file': "configs/dcs/dc_config.json",
    # These would be needed by SustainDC's constructor
    'simulation_year': 2023,
    'init_day': 1,
    'init_hour': 0,
    'agents': [], # Not used by SustainDC directly, but good practice
    'max_bat_cap_Mw': 0, # Not testing battery here
    'datacenter_capacity_mw': 1.0, # Placeholder
    'evaluation': False
}

print("Instantiating DatacenterNodeMA for DC 1 (US-CAL-CISO)...")
dc_node = DatacenterNodeMA(dc_config=sample_dc_config, logger=None)

# Reset the node to a specific time
print("Resetting the node...")
init_seed = 42
init_year, init_day, init_hour = 2023, 180, 12
dc_node.reset(init_year, init_day, init_hour, init_seed)

print("\nDatacenterNodeMA instance created and reset successfully.")
print(f"Node ID: {dc_node.dc_id}, Location: {dc_node.location}")
print(f"Initial Resources: {dc_node.physical_dc_model.available_cores} Cores, {dc_node.physical_dc_model.available_gpus} GPUs")


# %%
# --- [Cell 3] Create Mock Tasks ---

current_sim_time = pd.Timestamp(f"{init_year}-01-01 00:00:00", tz='UTC') + pd.Timedelta(days=init_day, hours=init_hour)
print(f"Current Simulation Time (UTC): {current_sim_time}")

# Create tasks for the DTA_Manager's originating queue
originating_tasks = [
    Task(job_name="OrigTask_A", arrival_time=current_sim_time, duration=30, cores_req=16, gpu_req=1, mem_req=32, bandwidth_gb=10),
    Task(job_name="OrigTask_B", arrival_time=current_sim_time, duration=60, cores_req=32, gpu_req=2, mem_req=64, bandwidth_gb=50)
]
for t in originating_tasks: t.origin_dc_id = dc_node.dc_id # Set origin to this DC

# Create tasks for the DTA_Worker's commitment queue (as if they were transferred in or deferred)
worker_tasks = [
    Task(job_name="WorkerTask_C_local", arrival_time=current_sim_time - pd.Timedelta(minutes=15), duration=45, cores_req=8, gpu_req=0, mem_req=16, bandwidth_gb=5),
    Task(job_name="WorkerTask_D_remote", arrival_time=current_sim_time - pd.Timedelta(minutes=30), duration=15, cores_req=4, gpu_req=0.5, mem_req=8, bandwidth_gb=2)
]
# Simulate that these worker tasks have been waiting
worker_tasks[0].wait_intervals = 1 # Waited one step
worker_tasks[1].wait_intervals = 2 # Waited two steps

print(f"\nCreated {len(originating_tasks)} tasks for DTA_Manager.")
print(f"Created {len(worker_tasks)} tasks for DTA_Worker.")

# Add tasks to the node's queues
dc_node.add_originating_tasks(originating_tasks)
dc_node.add_transferred_tasks(worker_tasks) # Using this method to populate worker queue for test

print(f"\nOriginating Queue Length: {len(dc_node.originating_tasks_queue)}")
print(f"Worker Commitment Queue Length: {len(dc_node.worker_commitment_queue)}")


# %%
# --- [Cell 4] Test DTA_Manager Observation Preparation ---

print("\nTesting `prepare_manager_observation`...")
manager_obs_parts = dc_node.prepare_manager_observation(current_sim_time)

# Verify the components of the observation dictionary
print_dict(manager_obs_parts, "Manager Observation Parts")

# Check if the meta-task vector has the correct shape
meta_task_vector_mgr = manager_obs_parts.get("obs_manager_meta_task_i")
assert isinstance(meta_task_vector_mgr, np.ndarray), "Manager meta-task should be a NumPy array"
assert meta_task_vector_mgr.shape == (len(MANAGER_META_TASK_FEATURE_ORDER),), f"Expected manager meta-task shape {(len(MANAGER_META_TASK_FEATURE_ORDER),)}, got {meta_task_vector_mgr.shape}"
print("\n✅ DTA_Manager meta-task vector has correct shape.")

# Check the local destination option features
local_option = manager_obs_parts.get("local_destination_option_features")
assert isinstance(local_option, dict), "Local destination option should be a dictionary"
assert local_option.get("is_local") == 1.0, "is_local flag should be 1.0"
assert local_option.get("transmission_cost") == 0.0, "Local transmission cost should be 0.0"
print("✅ Local destination option features look correct.")


# %%
# --- [Cell 5] Test DTA_Worker Observation Preparation ---

print("\nTesting `prepare_worker_observation`...")
worker_obs_parts = dc_node.prepare_worker_observation(current_sim_time)

# Verify the components
print_dict(worker_obs_parts, "Worker Observation Parts")

# Check meta-task vector shape
meta_task_vector_wrk = worker_obs_parts.get("obs_worker_meta_task_i")
assert isinstance(meta_task_vector_wrk, np.ndarray), "Worker meta-task should be a NumPy array"
assert meta_task_vector_wrk.shape == (len(WORKER_META_TASK_FEATURE_ORDER),), f"Expected worker meta-task shape {(len(WORKER_META_TASK_FEATURE_ORDER),)}, got {meta_task_vector_wrk.shape}"
print("\n✅ DTA_Worker meta-task vector has correct shape.")

# Check local DC state vector shape
local_state_vector = worker_obs_parts.get("obs_local_dc_i_for_worker")
# The length of this vector depends on the features you defined in the method
expected_worker_local_state_len = 5 # cpu, gpu, mem, price, ci
assert local_state_vector.shape == (expected_worker_local_state_len,), f"Expected worker local state shape {(expected_worker_local_state_len,)}, got {local_state_vector.shape}"
print("✅ DTA_Worker local state vector has correct shape.")


# %%
# --- [Cell 6] Test Action Application Logic ---

print("\nTesting action application...")

# --- Test Manager Action: Transfer to remote DC (e.g., DC 2) ---
print("\nSimulating DTA_Manager decision: TRANSFER to remote DC 2")
# Before:
print(f"Originating queue size before: {len(dc_node.originating_tasks_queue)}")
print(f"Worker queue size before: {len(dc_node.worker_commitment_queue)}")
# Apply decision
tasks_to_transfer = dc_node.apply_manager_decision(chosen_destination_dc_id=2)
# After:
print(f"Originating queue size after: {len(dc_node.originating_tasks_queue)}")
print(f"Worker queue size after: {len(dc_node.worker_commitment_queue)}")
print(f"Tasks returned for transfer: {len(tasks_to_transfer)}")
assert len(dc_node.originating_tasks_queue) == 0
assert len(tasks_to_transfer) == 2
assert tasks_to_transfer[0].dest_dc_id == 2
print("✅ Manager 'transfer' action processed correctly.")

# Reset originating queue for the next test
dc_node.add_originating_tasks(originating_tasks)

# --- Test Manager Action: Commit Locally ---
print("\nSimulating DTA_Manager decision: COMMIT to local worker")
# Before:
print(f"Originating queue size before: {len(dc_node.originating_tasks_queue)}")
print(f"Worker queue size before: {len(dc_node.worker_commitment_queue)}")
# Apply decision
tasks_to_transfer = dc_node.apply_manager_decision(chosen_destination_dc_id=dc_node.dc_id)
# After:
print(f"Originating queue size after: {len(dc_node.originating_tasks_queue)}")
print(f"Worker queue size after: {len(dc_node.worker_commitment_queue)}") # Should increase
print(f"Tasks returned for transfer: {len(tasks_to_transfer)}")
assert len(dc_node.originating_tasks_queue) == 0
assert len(dc_node.worker_commitment_queue) == 4 # 2 original worker tasks + 2 new ones
assert len(tasks_to_transfer) == 0
print("✅ Manager 'commit local' action processed correctly.")


# --- Test Worker Action: Defer ---
print("\nSimulating DTA_Worker decision: DEFER")
# Before:
worker_queue_len_before = len(dc_node.worker_commitment_queue)
# Apply decision
newly_scheduled = dc_node.apply_worker_decision(action_execute_now=False, current_time_utc=current_sim_time)
# After:
worker_queue_len_after = len(dc_node.worker_commitment_queue)
print(f"Worker queue size before: {worker_queue_len_before}, after: {worker_queue_len_after}")
print(f"Tasks scheduled: {len(newly_scheduled)}")
assert worker_queue_len_before == worker_queue_len_after # Queue size should not change on defer
assert len(newly_scheduled) == 0
print("✅ Worker 'defer' action processed correctly.")

# --- Test Worker Action: Execute Now ---
print("\nSimulating DTA_Worker decision: EXECUTE NOW")
# Before:
worker_queue_len_before = len(dc_node.worker_commitment_queue)
# Apply decision
newly_scheduled = dc_node.apply_worker_decision(action_execute_now=True, current_time_utc=current_sim_time)
# After:
worker_queue_len_after = len(dc_node.worker_commitment_queue)
print(f"Worker queue size before: {worker_queue_len_before}, after: {worker_queue_len_after}")
print(f"Tasks scheduled: {len(newly_scheduled)}")
assert len(newly_scheduled) > 0 # At least some should be scheduled
assert worker_queue_len_after < worker_queue_len_before # Queue should shrink
print("✅ Worker 'execute' action processed correctly.")
print(f"Physical DC now has {len(dc_node.physical_dc_model.running_tasks)} running tasks.")
# %%
