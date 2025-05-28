import random
import numpy as np
import pandas as pd
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from rewards.base_reward import BaseReward
from torch.utils.tensorboard import SummaryWriter  # if not already imported
from data.network_cost.network_delay import get_transmission_delay
from rl_components.task import Task 

# Define default aggregated observation dimension - this will depend on your aggregation
# For simple average/sum of task features + global features:
# 4 (time) + N_AGG_TASK_FEATURES (e.g., 5 for avg_cpu, avg_gpu, avg_dur, avg_deadline, num_tasks) + 5*N_DC (dc_states)
# Let's define a placeholder and calculate it properly in __init__
DEFAULT_AGGREGATED_OBS_DIM = 64 # Placeholder, will be calculated

class TaskSchedulingEnv(gym.Env):
    """
    RL Environment for global task scheduling across distributed datacenters.

    This environment wraps around DatacenterClusterManager and exposes a
    Gym-compatible interface. It manages task-level actions (assignment or defer),
    computes observations, and tracks rewards via a modular reward function.

    RL agents interact with this class.
    """
    def __init__(self, cluster_manager, start_time, end_time, 
                 reward_fn: BaseReward, writer: SummaryWriter = None,
                 sim_config: dict = None, initial_seed_for_resets=None): # Add sim_config
        
        super().__init__()
        self.cluster_manager = cluster_manager
        self.logger = getattr(self.cluster_manager, "logger", None)
        self.start_time = start_time
        self.end_time = end_time
        self.time_step = pd.Timedelta(minutes=15)
        self.current_time = self.start_time
        self.reward_fn = reward_fn 
        self.writer = writer

        self.pending_tasks = []
        self.deferred_tasks = []
        # queue of (arrival_time: Timestamp, task: Task, dest_dc_name: str)
        self.in_transit_tasks = []

        self.current_task = None
        self.global_step = 0  # Used to track time for TensorBoard logs

        # Set dynamically based on number of DCs
        self.num_dcs = len(self.cluster_manager.datacenters)
        self.sim_config = sim_config
        self.base_seed = initial_seed_for_resets if initial_seed_for_resets is not None else random.randint(0, 1_000_000)
        self.current_episode_count = 0 # Track episodes within this env instance

        
        # --- Read single_action_mode and aggregation_method from sim_config ---
        if sim_config is None:
            # Fallback if sim_config not passed, though it should be
            print("Warning: sim_config not passed to TaskSchedulingEnv. Defaulting single_action_mode to False.")
            self.single_action_mode = False
            self.aggregation_method = "average" # Default aggregation
            self.disable_defer_action = False

        else:
            self.single_action_mode = sim_config.get("single_action_mode", False)
            self.aggregation_method = sim_config.get("aggregation_method", "average")
            self.disable_defer_action = sim_config.get("disable_defer_action", False)


        if self.logger:
            self.logger.info(f"TaskSchedulingEnv initialized with single_action_mode: {self.single_action_mode}, aggregation: {self.aggregation_method}, disable_defer_action: {self.disable_defer_action}")
        
        print(f"TaskSchedulingEnv initialized with single_action_mode: {self.single_action_mode}, aggregation: {self.aggregation_method}, disable_defer_action: {self.disable_defer_action}")

        # --- Define Observation and Action Spaces ---
        # Per-task observation dimension (4 time + 5 task_base + 5*N_dcs)
        # Task features: origin_dc_id, cores_req, gpu_req, duration, time_to_deadline
        self.obs_dim_per_task = 4 + 5 + (5 * self.num_dcs)

        if self.single_action_mode:
            # Calculate aggregated observation dimension
            # Time (4) + Aggregated Task (e.g., 5: num_tasks, avg_cores, avg_gpus, avg_duration, min_deadline) + DC States (5*N)
            # This depends on self._aggregate_task_observations implementation
            self.num_aggregated_task_features = 5 # Example: count, avg_cpu, avg_gpu, avg_dur, min_deadline
            self.obs_dim_aggregated = 4 + self.num_aggregated_task_features + (5 * self.num_dcs)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim_aggregated,), dtype=np.float32
            )
            if self.disable_defer_action:
                self.action_space = spaces.Discrete(self.num_dcs)
                self.agent_output_act_dim = self.num_dcs
                if self.logger: self.logger.info(f"Single action mode, Defer Disabled: Action dim = {self.num_dcs} (maps to DCs 1..N)")
            else:
                self.action_space = spaces.Discrete(self.num_dcs + 1)
                self.agent_output_act_dim = self.num_dcs + 1
                if self.logger: self.logger.info(f"Single action mode, Defer Enabled: Action dim = {self.num_dcs + 1} (0=defer, 1..N=DCs)")
                
            if self.logger: self.logger.info(f"Single action mode: Obs dim = {self.obs_dim_aggregated}, Action dim = {self.num_dcs + 1}")
        else:
            # For multi-task mode, obs is a list of vectors. The Gym space technically describes one vector.
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim_per_task,), dtype=np.float32
            )
            # Action space is complex for variable tasks; agent handles this.
            # For Gym compliance, could use MultiDiscrete with max_tasks, but agent logic is key.
            if self.disable_defer_action:
                self.action_space = spaces.Discrete(self.num_dcs) # Action for one task, mapping 0..N-1 to DCs 1..N
                self.agent_output_act_dim = self.num_dcs # What the agent network should output
                if self.logger: self.logger.info(f"Multi-task mode, Defer Disabled: Action dim per task = {self.num_dcs}")
            else:
                self.action_space = spaces.Discrete(self.num_dcs + 1)
                self.agent_output_act_dim = self.num_dcs + 1
                if self.logger: self.logger.info(f"Multi-task mode, Defer Enabled: Action dim per task = {self.num_dcs + 1}")
            # Or spaces.MultiDiscrete([self.num_dcs + 1] * MAX_TASKS_IN_ENV_CONFIG)
            if self.logger: self.logger.info(f"Multi-task mode: Obs dim per task = {self.obs_dim_per_task}, Action dim per task = {self.num_dcs + 1}")


    def _aggregate_task_observations(self, list_of_per_task_obs: list, current_tasks_list: list) -> np.ndarray:
        """
        Aggregates features from a list of per-task observations and raw tasks
        into a single fixed-size vector.
        """
        if not list_of_per_task_obs: # Should be handled by _get_obs before calling this
            return np.zeros(self.obs_dim_aggregated, dtype=np.float32)

        # 1. Global Time Features (from the first task's observation vector)
        time_features = np.array(list_of_per_task_obs[0][:4], dtype=np.float32)

        # 2. Aggregated Task-Specific Features
        num_tasks = float(len(current_tasks_list))
        if num_tasks > 0:
            avg_cores_req = np.mean([task.cores_req for task in current_tasks_list])
            avg_gpu_req = np.mean([task.gpu_req for task in current_tasks_list])
            avg_duration = np.mean([task.duration for task in current_tasks_list])
            min_time_to_deadline = np.min([
                max(0.0, (task.sla_deadline - self.current_time).total_seconds() / 60.0)
                for task in current_tasks_list
            ])
            # Add more features as needed for self.num_aggregated_task_features
            # Example: sum of bandwidth, max urgency, etc.
        else: # Should not happen if list_of_per_task_obs is not empty, but defensive
            avg_cores_req = 0.0
            avg_gpu_req = 0.0
            avg_duration = 0.0
            min_time_to_deadline = 0.0 # Or a large number if 0 has specific meaning

        aggregated_task_features = np.array([
            num_tasks, avg_cores_req, avg_gpu_req, avg_duration, min_time_to_deadline
            # Ensure this matches self.num_aggregated_task_features
        ], dtype=np.float32)

        # 3. Per-Datacenter Features (from the first task's observation vector, as they are global DC states)
        dc_state_features_start_index = 4 + 5 # Time features + original per-task features
        dc_state_features = np.array(list_of_per_task_obs[0][dc_state_features_start_index:], dtype=np.float32)

        aggregated_obs = np.concatenate([time_features, aggregated_task_features, dc_state_features])

        if aggregated_obs.shape[0] != self.obs_dim_aggregated:
            # This should not happen if dimensions are calculated correctly
            if self.logger: self.logger.error(f"Aggregated obs dim mismatch! Expected {self.obs_dim_aggregated}, got {aggregated_obs.shape[0]}")
            # Fallback: pad or truncate (less ideal)
            if aggregated_obs.shape[0] < self.obs_dim_aggregated:
                padding = np.zeros(self.obs_dim_aggregated - aggregated_obs.shape[0])
                aggregated_obs = np.concatenate([aggregated_obs, padding])
            else:
                aggregated_obs = aggregated_obs[:self.obs_dim_aggregated]
        return aggregated_obs


    def _get_obs(self):
        # Current _get_obs logic generates a list of per-task observations
        # Let's call it _get_per_task_obs_list()
        per_task_obs_list = self._generate_per_task_obs_list()

        if self.single_action_mode:
            if not per_task_obs_list: # No tasks
                return np.zeros(self.obs_dim_aggregated, dtype=np.float32)
            # Pass current_tasks along with their observations for easier aggregation
            return self._aggregate_task_observations(per_task_obs_list, self.current_tasks)
        else:
            return per_task_obs_list # Returns a list of np.arrays
    
    
    def _generate_per_task_obs_list(self):
        obs_list = []

        # === Step 1: Time encoding (sine/cosine of day of year and hour) ===
        day_of_year = self.current_time.dayofyear
        hour_of_day = self.current_time.hour + self.current_time.minute / 60.0
        day_sin = np.sin(2 * np.pi * day_of_year / 365.0)
        day_cos = np.cos(2 * np.pi * day_of_year / 365.0)
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24.0)
        time_features = [day_sin, day_cos, hour_sin, hour_cos]

        # === Step 2: Extract DC resource and sustainability info ===
        dc_infos_list = [] # Store individual dc_info lists
        for dc in self.cluster_manager.datacenters.values():
            dc_infos_list.append([
                dc.available_cores / dc.total_cores if dc.total_cores > 0 else 0,
                dc.available_gpus / dc.total_gpus if dc.total_gpus > 0 else 0,
                dc.available_mem / dc.total_mem_GB if dc.total_mem_GB > 0 else 0,
                float(dc.ci_manager.get_current_ci(norm=False)/1000.0),  # carbon intensity
                float(dc.price_manager.get_current_price())/100.0,       # energy price
            ])
        dc_state_features = [value for dc_info_single in dc_infos_list for value in dc_info_single]

        # === Step 3: Build observation per task ===
        for task in self.current_tasks: # self.current_tasks is the list of task objects
            time_to_deadline = max(0.0, (task.sla_deadline - self.current_time).total_seconds() / 60.0)
            task_features = [
                float(task.origin_dc_id), # Ensure float
                task.cores_req,
                task.gpu_req,
                task.duration,
                time_to_deadline
            ]
            
            full_obs_vector = time_features + task_features + dc_state_features
            obs_list.append(np.array(full_obs_vector, dtype=np.float32))
            
        return obs_list


    def reset(self, seed=None, options=None):
        # --- Seeding Logic ---
        if seed is None:
            # If RLlib doesn't pass a specific seed for this reset,
            # derive one from our base seed and episode count.
            # This ensures different trajectories on subsequent resets within the same worker.
            current_reset_seed = self.base_seed + self.current_episode_count
        else:
            # If RLlib passes a seed (e.g., for evaluation), use that.
            current_reset_seed = seed
        
        self.current_episode_count += 1
        seed = current_reset_seed
        super().reset(seed=seed) # Gymnasium expects options, but we don't use them yet
        random.seed(seed) # Set random seed for reproducibility
        self.current_time = self.start_time
        self.cluster_manager.reset(seed=seed) # Pass seed to cluster manager
        
        # print(f"Resetting environment with seed {seed} at time {self.current_time}")

        self.deferred_tasks.clear()
        self.in_transit_tasks.clear()
        self._load_new_tasks() # Load initial tasks for current_time

        self.global_step = 0 # Reset TensorBoard step counter

        return self._get_obs(), {} # Gymnasium returns obs, info


    def step(self, actions): # actions is a single int if single_action_mode else list
        # ... (existing in_transit_tasks logic) ...
        remaining_in_transit = []
        for arrival_time, task, dc_name in self.in_transit_tasks:
            if arrival_time <= self.current_time:
                self.cluster_manager.datacenters[dc_name].pending_tasks.append(task)
                if self.logger:
                    self.logger.info(f"[{self.current_time}] Task {task.job_name} arrived at {dc_name}")
            else:
                remaining_in_transit.append((arrival_time, task, dc_name))
        self.in_transit_tasks = remaining_in_transit

        dc_list_values = list(self.cluster_manager.datacenters.values()) # For direct indexing

        # --- Adapt action processing based on mode ---
        if self.single_action_mode:
            if not self.current_tasks: # No tasks to act upon
                # if self.logger and actions is not None and actions != []: # Log if action provided for no tasks
                    #  self.logger.warning(f"[{self.current_time}] Received action {actions} but no current tasks.")
                processed_actions_count = 0
            else:
                single_action_taken = actions # actions is a single integer
                processed_actions_count = len(self.current_tasks) # Acted on all current tasks

                # Map agent's action to environment action (defer or DC index)
                if self.disable_defer_action:
                    # Agent outputs 0 to N-1, map to DC 1 to N
                    single_action_taken = single_action_taken + 1
                else:
                    # Agent outputs 0 (defer) or 1 to N (DCs)
                    single_action_taken = single_action_taken

                for task_idx, task in enumerate(self.current_tasks):
                    # Apply SLA check
                    if self.current_time > task.sla_deadline:
                        origin_dc_obj = next(dc for dc in dc_list_values if dc.dc_id == task.origin_dc_id)
                        origin_dc_obj.pending_tasks.append(task)
                        task.dest_dc_id = origin_dc_obj.dc_id; task.dest_dc = origin_dc_obj
                        if self.logger:
                            self.logger.info(f"[{self.current_time}] Task {task.job_name} exceeded SLA. Forced to origin DC{origin_dc_obj.dc_id}.")
                        continue

                    if single_action_taken == 0: # Defer all
                        self.deferred_tasks.append(task)
                        task.temporarily_deferred = True
                        if self.logger:
                            self.logger.info(f"[{self.current_time}] Task {task.job_name} (batch) deferred.")
                            
                    else: # Assign all to the chosen DC
                        dest_dc_chosen_by_agent = dc_list_values[single_action_taken - 1]
                        task.dest_dc_id = dest_dc_chosen_by_agent.dc_id
                        task.dest_dc = dest_dc_chosen_by_agent
                        # Handle transmission delay for this task
                        origin_loc = self.cluster_manager.get_dc_location(task.origin_dc_id)
                        dest_loc = dest_dc_chosen_by_agent.location
                        delay_s = get_transmission_delay(origin_loc, dest_loc, self.cluster_manager.cloud_provider, task.bandwidth_gb)
                        arrival_ts = self.current_time + pd.to_timedelta(delay_s, unit='s')
                        dest_dc_name = next(name for name, dc_obj in self.cluster_manager.datacenters.items() if dc_obj.dc_id == task.dest_dc_id)
                        self.in_transit_tasks.append((arrival_ts, task, dest_dc_name))
                        
                        if self.logger:
                            self.logger.info(f"[{self.current_time}] Task {task.job_name} (batch) routed to DC{task.dest_dc_id}, delay={delay_s:.1f}s.")
        else: # Multi-action mode (original logic)
            if self.cluster_manager.strategy == "manual_rl": # Only assert if RL is driving
                assert len(actions) == len(self.current_tasks), \
                    f"Expected {len(self.current_tasks)} actions, got {len(actions)}"
            processed_actions_count = len(actions)

            for task, action_for_task in zip(self.current_tasks, actions):
                # Map agent's action for this task
                if self.disable_defer_action:
                    # Agent outputs 0 to N-1 for this task, map to DC 1 to N
                    action_for_task = action_for_task + 1
                else:
                    # Agent outputs 0 (defer) or 1 to N (DCs) for this task
                    action_for_task = action_for_task
            
                # ... (existing per-task action processing logic for defer/assign with delay) ...
                if self.current_time > task.sla_deadline:
                    origin_dc_obj = next(dc for dc in dc_list_values if dc.dc_id == task.origin_dc_id)
                    origin_dc_obj.pending_tasks.append(task)
                    task.dest_dc_id = origin_dc_obj.dc_id; task.dest_dc = origin_dc_obj
                    if self.logger:
                        self.logger.info(f"[{self.current_time}] Task {task.job_name} exceeded SLA. Forced to origin DC{origin_dc_obj.dc_id}.")
                    continue
                if action_for_task == 0:
                    self.deferred_tasks.append(task)
                    task.temporarily_deferred = True
                    if self.logger:
                        self.logger.info(f"[{self.current_time}] Task {task.job_name} (individual) deferred.")
                else:
                    dest_dc_chosen_by_agent = dc_list_values[action_for_task - 1]
                    task.dest_dc_id = dest_dc_chosen_by_agent.dc_id
                    task.dest_dc = dest_dc_chosen_by_agent
                    origin_loc = self.cluster_manager.get_dc_location(task.origin_dc_id)
                    dest_loc = dest_dc_chosen_by_agent.location
                    delay_s = get_transmission_delay(origin_loc, dest_loc, self.cluster_manager.cloud_provider, task.bandwidth_gb)
                    arrival_ts = self.current_time + pd.to_timedelta(delay_s, unit='s')
                    dest_dc_name = next(name for name, dc_obj in self.cluster_manager.datacenters.items() if dc_obj.dc_id == task.dest_dc_id)
                    self.in_transit_tasks.append((arrival_ts, task, dest_dc_name))
                    if self.logger:
                        self.logger.info(f"[{self.current_time}] Task {task.job_name} (individual) routed to DC{task.dest_dc_id}, delay={delay_s:.1f}s.")

        # === Step cluster manager, calculate reward, advance time (existing logic) ===
        results = self.cluster_manager.step(self.current_time, logger=self.logger)
        if self.reward_fn:
            reward = self.reward_fn(cluster_info=results, current_tasks=self.current_tasks, current_time=self.current_time)
        else: reward = 0.0

        if self.writer and self.reward_fn:
            # ... (existing TensorBoard logging) ...
            if hasattr(self.reward_fn, "get_last_components"):
                for name, value in self.reward_fn.get_last_components().items():
                    self.writer.add_scalar(f"RewardComponents/{name}", value, self.global_step)
            elif hasattr(self.reward_fn, "get_last_value"): self.writer.add_scalar(f"Reward/{str(self.reward_fn)}", self.reward_fn.get_last_value(), self.global_step)
        self.global_step += 1 # Moved here, to increment once per env step

        self.current_time += self.time_step
        self._load_new_tasks() # Load new and previously deferred tasks for next obs

        done = self.current_time >= self.end_time
        truncated = done # For now, done and truncated are the same

        obs_next = self._get_obs() # Generate next observation based on new self.current_tasks
        info = {
            "total_energy_kwh": results.get("transmission_energy_total_kwh", 0), # Example, adjust keys
            "total_emissions_kg": results.get("transmission_emissions_total_kg", 0), # Example
            "scheduled_tasks_this_step": processed_actions_count, # How many actions were processed
            "datacenter_infos": results["datacenter_infos"],
            "transmission_cost_total_usd" : results["transmission_cost_total_usd"],
        }
        return obs_next, reward, done, truncated, info
    

    def _load_new_tasks(self):
        """Load tasks for the current time step, including previously deferred ones."""
        newly_loaded_tasks = []
        if self.cluster_manager.strategy == "manual_rl": # Only if RL is managing
            newly_loaded_tasks = self.cluster_manager.get_tasks_for_timestep(self.current_time)

        # Combine deferred tasks (first) with newly loaded tasks
        self.current_tasks = self.deferred_tasks + newly_loaded_tasks
        self.deferred_tasks = [] # Clear deferred list for next step

        if self.logger:
            self.logger.info(f"[{self.current_time}] Start of step. Current tasks: {len(self.current_tasks)} (Deferred: {len(self.current_tasks) - len(newly_loaded_tasks)}, New: {len(newly_loaded_tasks)})")
