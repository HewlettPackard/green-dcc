import numpy as np
import pandas as pd
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from rewards.base_reward import BaseReward
from torch.utils.tensorboard import SummaryWriter  # if not already imported
from data.network_cost.network_delay import get_transmission_delay

class TaskSchedulingEnv(gym.Env):
    """
    RL Environment for global task scheduling across distributed datacenters.

    This environment wraps around DatacenterClusterManager and exposes a
    Gym-compatible interface. It manages task-level actions (assignment or defer),
    computes observations, and tracks rewards via a modular reward function.

    RL agents interact with this class.
    """
    def __init__(self, cluster_manager, start_time, end_time, reward_fn: BaseReward, writer: SummaryWriter = None,):
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
        
        # Observation space: [4 sin/cos features, 4 task features features, 5 * num_dcs task features]
        obs_dim = 4 + 5 + 5 * self.num_dcs

        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        self.action_space = None       # Variable-length batch of int (one per task)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = self.start_time

        # Reset the cluster manager and all datacenters
        self.cluster_manager.reset(seed=seed)

        # Load the first batch of tasks
        self._load_new_tasks()

        return self._get_obs(), {}

    def step(self, actions):
        """
        actions: list[int] of length == len(self.current_tasks)
        Each element is the index of the destination datacenter (0-based)
        """
        
        # === Deliver any in‑flight (transmitting) tasks whose arrival time has come ===
        remaining = []
        for arrival_time, task, dc_name in self.in_transit_tasks:
            if arrival_time <= self.current_time:
                # now it appears in the destination DC’s pending queue
                self.cluster_manager.datacenters[dc_name].pending_tasks.append(task)
                if self.logger:
                    self.logger.info(f"[{self.current_time}] Task {task.job_name} arrived at {dc_name}")
            else:
                remaining.append((arrival_time, task, dc_name))
        self.in_transit_tasks = remaining
        
        if self.cluster_manager.strategy == "manual_rl":
            assert len(actions) == len(self.current_tasks), \
                f"Expected {len(self.current_tasks)} actions, got {len(actions)}"
            

        dc_list = list(self.cluster_manager.datacenters.values())

        # === Route each task to its assigned destination DC ===
        for task, action in zip(self.current_tasks, actions):
            
            # Check if the task has exceeded its SLA deadline
            if self.current_time > task.sla_deadline:
                # Enforce computation at origin datacenter
                origin_dc = next(dc for dc in self.cluster_manager.datacenters.values()
                                 if dc.dc_id == task.origin_dc_id)
                origin_dc.pending_tasks.append(task)
                task.dest_dc_id = origin_dc.dc_id
                task.dest_dc = origin_dc
                if self.logger:
                    self.logger.info(
                        f"[{self.current_time}] Task {task.job_name} exceeded SLA deadline. "
                        f"Forced to origin DC{origin_dc.dc_id}."
                    )
                continue
            
            # === Temporal deferral ===
            if action == 0:
                self.deferred_tasks.append(task)
                task.temporarily_deferred = True
                if self.logger:
                    self.logger.info(
                        f"[{self.current_time}] Task {task.job_name}, with origin DC{task.origin_dc_id}, "
                        "has been deferred in time (not assigned destination DC)."
                    )
                continue
            
            # === Geographical routing ===
            dest_dc = dc_list[action - 1]  # Now action ∈ [1..num_dcs]
            # dest_dc.pending_tasks.append(task)
            # Assign the destination to the task info
            task.dest_dc_id = dest_dc.dc_id
            task.dest_dc = dest_dc
            
            # compute network delay
            origin_loc = self.cluster_manager.get_dc_location(task.origin_dc_id)
            dest_loc   = dest_dc.location
            provider   = self.cluster_manager.cloud_provider  # 'aws' or 'azure'
            size_gb    = task.bandwidth_gb

            delay_s = get_transmission_delay(origin_loc, dest_loc, provider, size_gb)
            arrival_ts = self.current_time + pd.to_timedelta(delay_s, unit='s')

            # enqueue for later delivery
            dc_name = next(name for name, dc in self.cluster_manager.datacenters.items()
                           if dc.dc_id == task.dest_dc_id)
            self.in_transit_tasks.append((arrival_ts, task, dc_name))

            if self.logger:
                self.logger.info(
                    f"[{self.current_time}] Routed task {task.job_name} from DC{task.origin_dc_id} to DC{task.dest_dc_id}, requiring a bandwidth of {task.bandwidth_gb:.2f} GB. "
                    f"(delay={delay_s:.1f}s, will arrive at {arrival_ts})"
                )

        # === Step all datacenters (releases, schedules, updates) ===
        results = self.cluster_manager.step(self.current_time, logger=self.logger)

        # === Compute emissions and total energy ===
        emissions_total = 0.0
        energy_total = 0.0

        if self.reward_fn:
            reward = self.reward_fn(
                cluster_info=results,
                current_tasks=self.current_tasks,
                current_time=self.current_time
            )
        else:
            reward = 0.0


        # Log the individual rewards components in the tensorboard
        # === TensorBoard logging ===
        if self.writer and self.reward_fn:
            if hasattr(self.reward_fn, "get_last_components"): # Composite reward
                for name, value in self.reward_fn.get_last_components().items():
                    self.writer.add_scalar(f"RewardComponents/{name}", value, self.global_step)
            elif hasattr(self.reward_fn, "get_last_value"): # Individual reward
                self.writer.add_scalar(f"Reward/{str(self.reward_fn)}", self.reward_fn.get_last_value(), self.global_step)
            self.global_step += 1


        # === Advance time by 15 minutes and load next tasks ===
        self.current_time += pd.Timedelta(minutes=15)
        self._load_new_tasks()

        done = self.current_time >= self.end_time
        truncated = done

        obs = self._get_obs()
        info = {
            "total_energy_kwh": energy_total,
            "total_emissions_kg": emissions_total,
            "scheduled_tasks": len(actions),
            "datacenter_infos": results["datacenter_infos"],
            "transmission_cost_total_usd" : results["transmission_cost_total_usd"],
        }

        return obs, reward, done, truncated, info

    def _load_new_tasks(self):
        """Load tasks for the current time step."""
        
        self.current_tasks = self.deferred_tasks  # first pick leftovers
        self.deferred_tasks = []
        # Only load tasks manually if using RL agent
        if self.cluster_manager.strategy == "manual_rl":
            new_tasks = self.cluster_manager.get_tasks_for_timestep(self.current_time)
            self.current_tasks += new_tasks
            if self.logger:
                self.logger.info(f"[{self.current_time}] Loaded {len(new_tasks)} new tasks + {len(self.current_tasks) - len(new_tasks)} total.")
        else:
            # RBC loads and handles tasks internally
            self.current_tasks = []

    # def _next_task(self):
    #     if self.pending_tasks:
    #         self.current_task = self.pending_tasks.pop(0)
    #     else:
    #         self.current_task = None

    # def _advance_time_if_needed(self):
    #     if not self.pending_tasks:
    #         self.current_time += self.time_step

    # def _check_done(self):
    #     return self.current_time >= self.end_time

    def _get_obs(self):
        obs = []
        dc_infos = []

        # === Step 1: Time encoding (sine/cosine of day of year and hour) ===
        day_of_year = self.current_time.dayofyear
        hour_of_day = self.current_time.hour + self.current_time.minute / 60.0

        day_sin = np.sin(2 * np.pi * day_of_year / 365.0)
        day_cos = np.cos(2 * np.pi * day_of_year / 365.0)
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24.0)

        # === Step 2: Extract current prices ===
        prices = []
        for dc in self.cluster_manager.datacenters.values():
            price = float(dc.price_manager.get_current_price()) / 100  # Normalize
            prices.append(price)

        prices = np.array(prices, dtype=np.float32)
        num_dcs = len(prices)

        # === Step 3: One-hot encode the cheapest DC ===
        # cheapest_idx = int(np.argmin(prices))
        # one_hot_cheapest = np.zeros(num_dcs, dtype=np.float32)
        # one_hot_cheapest[cheapest_idx] = 1.0

        # === Step 4: Extract DC resource and sustainability info ===
        for dc in self.cluster_manager.datacenters.values():
            dc_infos.append([
                dc.available_cores / dc.total_cores,
                dc.available_gpus / dc.total_gpus,
                dc.available_mem / dc.total_mem_GB,
                float(dc.ci_manager.get_current_ci(norm=False)/1000),  # carbon intensity
                float(dc.price_manager.get_current_price())/100,       # energy price
            ])

        dc_state_features = [value for dc_info in dc_infos for value in dc_info]

        # === Step 5: Build observation per task ===
        for task in self.current_tasks:
            time_to_deadline = max(0.0, (task.sla_deadline - self.current_time).total_seconds() / 60.0)

            task_features = [
                task.origin_dc_id,
                task.cores_req,
                task.gpu_req,
                task.duration,
                time_to_deadline
            ]

            full_obs = (
                [day_sin, day_cos, hour_sin, hour_cos] +
                task_features +
                dc_state_features
            )
            obs.append(full_obs)

        return obs