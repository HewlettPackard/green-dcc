import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
from rewards.base_reward import BaseReward

class TaskSchedulingEnv(gym.Env):
    def __init__(self, cluster_manager, start_time, end_time, reward_fn: BaseReward):
        super().__init__()
        self.cluster_manager = cluster_manager
        self.logger = getattr(self.cluster_manager, "logger", None)
        self.start_time = start_time
        self.end_time = end_time
        self.time_step = pd.Timedelta(minutes=15)
        self.current_time = self.start_time
        self.reward_fn = reward_fn 
        
        self.pending_tasks = []
        self.current_task = None

        # Set dynamically based on number of DCs
        self.num_dcs = len(self.cluster_manager.datacenters)
        obs_dim = 3+4*self.num_dcs + self.num_dcs # 1 for task origin DC + 4 features per datacenter + One hot for cheapest DC
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
        if self.cluster_manager.strategy == "manual_rl":
            assert len(actions) == len(self.current_tasks), \
                f"Expected {len(self.current_tasks)} actions, got {len(actions)}"
            

        dc_list = list(self.cluster_manager.datacenters.values())

        # === Route each task to its assigned destination DC ===
        for task, action in zip(self.current_tasks, actions):
            dest_dc = dc_list[action]
            dest_dc.pending_tasks.append(task)
            # Assign the destination to the task info
            task.dest_dc_id = dest_dc.dc_id
            task.dest_dc = dest_dc
            
            if self.logger:
                self.logger.info(f"[{self.current_time}] Routed task {task.job_name} "
                            f"(origin DC{task.origin_dc_id}) → destination DC{dest_dc.dc_id}")

        # === Step all datacenters (releases, schedules, updates) ===
        results = self.cluster_manager.step(self.current_time, logger=self.logger)

        # === Compute emissions and total energy ===
        emissions_total = 0.0
        energy_total = 0.0
        # energy_cost_total = 0.0
        # sla_met = 0
        # sla_violated = 0
        
        # for dc_name, info in results["datacenter_infos"].items():
        #     ci = info["__common__"]["ci"]
        #     energy_kwh = info["__common__"]["energy_consumption_kwh"]
        #     emissions_kg = info["__common__"]["carbon_emissions_kg"]
        #     energy_cost_usd = info["__common__"]["energy_cost_USD"]

        #     energy_total += energy_kwh
        #     emissions_total += emissions_kg
        #     energy_cost_total += energy_cost_usd
            
        #     sla_met += info["__common__"]["__sla__"].get("met", 0)
        #     sla_violated += info["__common__"]["__sla__"].get("violated", 0)
        # # reward = 1.0 - (energy_cost_total/100)

        # total_task_cost = 0.0
        # for task in self.current_tasks:
        #     dest_dc = task.dest_dc  # based on agent action
        #     task_energy = task.cpu_req * task.duration
        #     task_cost = task_energy * dest_dc.price_manager.get_current_price()
        #     total_task_cost += task_cost

        # reward = -total_task_cost/100000 # Normalize reward
        reward = self.reward_fn(
                                cluster_info=results,
                                current_tasks=self.current_tasks,
                                current_time=self.current_time
                            )


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
        }

        return obs, reward, done, truncated, info

    def _load_new_tasks(self):
        """Load tasks for the current time step."""

        # Only load tasks manually if using RL agent
        if self.cluster_manager.strategy == "manual_rl":
            self.current_tasks = self.cluster_manager.get_tasks_for_timestep(self.current_time)
            if self.logger:
                self.logger.info(f"[{self.current_time}] Loaded {len(self.current_tasks)} tasks to use RL agent")
        else:
            # RBC loads and handles tasks internally
            self.current_tasks = []

    def _next_task(self):
        if self.pending_tasks:
            self.current_task = self.pending_tasks.pop(0)
        else:
            self.current_task = None

    def _advance_time_if_needed(self):
        if not self.pending_tasks:
            self.current_time += self.time_step

    def _check_done(self):
        return self.current_time >= self.end_time

    def _get_obs(self):
        obs = []
        dc_infos = []
        # # === Step 1: Extract current prices ===
        prices = []
        for dc in self.cluster_manager.datacenters.values():
            price = float(dc.price_manager.get_current_price()) / 100  # Normalize
            prices.append(price)

        prices = np.array(prices, dtype=np.float32)
        num_dcs = len(prices)

        # === Step 2: One-hot encode the cheapest DC ===
        cheapest_idx = int(np.argmin(prices))
        one_hot_cheapest = np.zeros(num_dcs, dtype=np.float32)
        one_hot_cheapest[cheapest_idx] = 1.0

        # # === Step 3: Relative price differences ===
        # price_diffs = prices - prices[cheapest_idx]

        # # === Step 4: Build observation per task ===
        # for task in self.current_tasks:
        #     task_vec = [task.origin_dc_id]  # You can add more task-level features here

        #     full_obs = (
        #         prices.tolist() +                 # [DC1_price, DC2_price, DC3_price]
        #         one_hot_cheapest.tolist() +       # [0, 1, 0]
        #         price_diffs.tolist()              # [Δprice_1, Δprice_2, Δprice_3]
        #     )

        #     # print(f"Task {task.job_name} observation: {full_obs}")
        #     obs.append(full_obs)
        

        # Extract datacenter states (same for all tasks in the step)
        for dc in self.cluster_manager.datacenters.values():
            dc_infos.append([
                dc.available_cpus / dc.total_cpus,
                dc.available_gpus / dc.total_gpus,
                dc.available_mem / dc.total_mem,
                # float(dc.ci_manager.get_current_ci(norm=False)/1000),  # carbon intensity
                float(dc.price_manager.get_current_price())/100,  # energy price
            ])

        # Flatten all DC features into one list
        flat_dc_info = [value for dc_info in dc_infos for value in dc_info]

        for task in self.current_tasks:
            task_vec = [
                task.origin_dc_id,
                task.cpu_req,
                # task.gpu_req,
                # task.mem_req,
                task.duration,
                # task.bandwidth_gb,
            ]
            obs.append(task_vec + flat_dc_info + one_hot_cheapest.tolist())

        # Print a warning if the observation len is higher tan 32 tasks
        # if len(obs) > 200:
            # print(f"Warning: Observation length exceeds 200 tasks. Current length: {len(obs)}")
        return obs


    def render(self):
        pass

    def close(self):
        pass
