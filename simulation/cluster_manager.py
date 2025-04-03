import random
from collections import defaultdict
from envs.sustaindc.sustaindc_env import SustainDC
from utils.task_assignment_strategies import (distribute_most_available, distribute_random, distribute_priority_order,
                                     distribute_least_pending, distribute_lowest_carbon, distribute_round_robin, distribute_lowest_price)
import numpy as np
import pandas as pd
from rl_components.task import Task


def assign_task_origins(tasks, datacenter_configs, current_time_utc, logger=None):
    """
    Assigns an origin DC to each task based on population-weighted + time-zone-aware probability.

    Args:
        tasks (List[Task]): Tasks to assign origin DC.
        datacenter_configs (List[dict]): DC configurations.
        current_time_utc (datetime): Current simulation time in UTC.
        logger (logging.Logger or None): Optional logger for debug statements.
    """
    def compute_activity_score(local_hour):
        return 1.0 if 8 <= local_hour < 20 else 0.3  # Business hours: high score; otherwise low score

    scores = {}
    for config in datacenter_configs:
        dc_id = config['dc_id']
        # Retrieve the population weight from the config; default to 0.1 if not provided.
        pop_weight = config.get('population_weight', 0.1)
        timezone_shift = config.get('timezone_shift', 0)
        tz_hour = (current_time_utc + pd.Timedelta(hours=timezone_shift)).hour
        time_score = compute_activity_score(tz_hour)
        scores[dc_id] = pop_weight * time_score

    total_score = sum(scores.values())
    probabilities = {dc_id: (score / total_score) for dc_id, score in scores.items()}
    dc_ids = list(probabilities.keys())
    probs = list(probabilities.values())

    for task in tasks:
        chosen_origin = int(np.random.choice(dc_ids, p=probs))
        task.origin_dc_id = chosen_origin
        # if logger:
            # logger.info(f"   assign_task_origins: Set origin DC{chosen_origin} for task {task.job_name}.")


def extract_tasks_from_row(row, scale=1, datacenter_configs=None, current_time_utc=None, logger=None):
    """
    Convert a row from task_df into a list of Task objects, scaling the number of tasks if needed.

    Args:
        row (pd.Series): A row from task_df containing 'tasks_matrix'.
        scale (int): Scaling factor for task duplication.
        datacenter_configs (List[dict]): DC configurations for assigning task origins.
        current_time_utc (datetime): Current simulation time in UTC.
        logger (logging.Logger or None): Optional logger for debug statements.

    Returns:
        List[Task]: A list of Task objects extracted and scaled from the row.
    """
    tasks = []
    for task_data in row['tasks_matrix']:
        job_name = task_data[0]
        arrival_time = current_time_utc  # Task arrival time
        duration = float(task_data[4])
        cpu_req = float(task_data[5]) / 100.0   # Convert percentage to CPU cores.
        gpu_req = float(task_data[6]) / 100.0   # Convert percentage to fraction of GPUs count.
        mem_req = float(task_data[7])           # Memory in GB
        bandwidth_gb = float(task_data[8])      # Bandwidth in GB

        # Create the base task
        task = Task(job_name, arrival_time, duration, cpu_req, gpu_req, mem_req, bandwidth_gb)
        tasks.append(task)

        # Scale the number of tasks
        for i in range(scale - 1):
            # Introduce random variation in CPU, GPU, Memory, and Bandwidth requirements
            varied_cpu = max(0.5, cpu_req * np.random.uniform(0.8, 1.2))  # ±20% variation
            varied_gpu = max(0.0, gpu_req * np.random.uniform(0.8, 1.2))  # Ensure GPU usage isn't negative
            varied_mem = max(0.5, mem_req * np.random.uniform(0.8, 1.2))
            varied_bw = max(0.1, bandwidth_gb * np.random.uniform(0.8, 1.2))

            # Create new task with variations
            new_task = Task(
                f"{job_name}_scaled_{i}",
                arrival_time,
                duration,
                varied_cpu,
                varied_gpu,
                varied_mem,
                varied_bw
            )
            tasks.append(new_task)

    # Assign origin datacenter
    if datacenter_configs and current_time_utc:
        # We'll pass in the same logger for debug prints
        assign_task_origins(tasks, datacenter_configs, current_time_utc, logger=logger)

    if logger:
        logger.info(f"extract_tasks_from_row: Created {len(tasks)} tasks at time {current_time_utc}.")
        for idx, t in enumerate(tasks):
            logger.info(f"   Task[{idx}]: {t.job_name}, origin=DC{t.origin_dc_id}, "
                        f"cpu={t.cpu_req:.2f}, gpu={t.gpu_req:.2f}, mem={t.mem_req:.2f}, "
                        f"bandwidth={t.bandwidth_gb:.2f}, dur={t.duration:.2f}")

    return tasks

class DatacenterClusterManager:
    def __init__(self, config_list, simulation_year, init_day, init_hour, strategy="priority_order", tasks_file_path=None, shuffle_datacenter_order=True):
        """
        Initializes multiple datacenters using SustainDC and loads tasks.

        Args:
            config_list (list): List of datacenter configuration dictionaries.
            simulation_year (int): Simulation year to use.
            init_day (int): Initial day of the simulation.
            init_hour (int): Initial hour of the simulation.
            strategy (str): Task distribution strategy.
            tasks_file_path (str, optional): Path to the tasks pickle file.
        """
        # Inject simulation parameters into each datacenter's configuration
        for config in config_list:
            config['simulation_year'] = simulation_year
            config['init_day'] = init_day
            config['init_hour'] = init_hour

        self.datacenters = {f"DC{i+1}": SustainDC(config) for i, config in enumerate(config_list)}
        self.simulation_year = simulation_year
        self.init_day = init_day
        self.init_hour = init_hour
        self.shuffle_datacenter_order = shuffle_datacenter_order

        # Load tasks if a file path is provided; otherwise, self.tasks remains None
        if tasks_file_path:
            self.tasks = pd.read_pickle(tasks_file_path)
            # Convert the timestamp to UTC (if needed)
            self.tasks['interval_15m'] = self.tasks['interval_15m'].dt.tz_convert('UTC')
        else:
            # Raise an error if no file path is provided
            raise ValueError("No tasks file path provided. Please provide a valid path to the tasks pickle file.")

        self.strategy = strategy
        self.strategy_map = {
            "most_available": distribute_most_available,
            "random": distribute_random,
            "priority_order": distribute_priority_order,
            "least_pending": distribute_least_pending,
            "lowest_carbon": distribute_lowest_carbon,
            "round_robin": distribute_round_robin,
            "lowest_price": distribute_lowest_price,
        }

    def reset(self, seed=None):
        """
        Reset all datacenters in the cluster.

        This method ensures that each datacenter starts fresh for a new episode.
        """
        # for dc_name, dc in self.datacenters.items():
            # print(f"Resetting {dc_name}...")
            # dc.reset()  # Call reset on each SustainDC instance
        # Set the random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            # print(f"Random seed set to {seed} for reproducibility.")
        rng = np.random.default_rng(seed)
        # Limit day to ±7 days around the initial value, wrapping around the year (0-364)
        low_day = max(0, self.init_day - 7)
        high_day = min(364, self.init_day + 7)
        self.random_init_day = int(rng.integers(low_day, high_day + 1))  # +1 because high is exclusive

        self.random_year = self.simulation_year
        # You can keep full 0–23 hour range, or restrict it too if needed
        self.random_init_hour = int(rng.integers(0, 24))

        if self.shuffle_datacenter_order:
            items = list(self.datacenters.items())
            np.random.shuffle(items)
            self.datacenters = dict(items)


        for dc_name, dc in self.datacenters.items():
            # print(f"Resetting {dc_name} with UTC start: Day {self.random_init_day}, Hour {self.random_init_hour}")
            dc.reset(init_year=self.random_year, init_day=self.random_init_day, init_hour=self.random_init_hour, seed=seed)
        # print("All datacenters have been reset.")


    def get_tasks_for_timestep(self, current_time):
        """
        Returns the list of Task objects for the given timestep.

        Args:
            current_time (pd.Timestamp): Current simulation time (UTC).

        Returns:
            list: List of Task objects with their origin assigned.
        """
        logger = self.logger
        if self.tasks is None:
            return []
        # Replace year in current_time with 2020
        adjusted_time = current_time.replace(year=2020)

        tasks_for_time = self.tasks[self.tasks['interval_15m'] == adjusted_time]
        tasks_list = []

        if not tasks_for_time.empty:
            # We'll just take the first row in that timeslot
            row = tasks_for_time.iloc[0]
            tasks_list = extract_tasks_from_row(
                row,
                scale=1,
                datacenter_configs=self.get_config_list(),
                current_time_utc=current_time,
                logger=logger  # pass logger for debug
            )

        if logger:
            logger.info(f"get_tasks_for_timestep: Found {len(tasks_list)} tasks at time {current_time}")
        return tasks_list
    
    def distribute_workload(self, task, current_time, logger):
        if self.strategy == "manual_rl":
            # Let the RL agent handle task assignment
            return None
        elif self.strategy in self.strategy_map:
            assigned_dc_id = self.strategy_map[self.strategy](task, self.datacenters, logger)
            assigned_dc = next((dc for dc in self.datacenters.values() if dc.dc_id == assigned_dc_id), None)
            task.dest_dc = assigned_dc
            task.dest_dc_id = assigned_dc.dc_id
            if not assigned_dc:
                logger.warning(f"[{current_time}] Task {task.job_name} could not be assigned and remains in queue.")
            return assigned_dc_id
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        
    def step(self, current_time, logger=None):
        """
        Advance all datacenters by one simulation step (15 minutes).
        
        Depending on the selected strategy:
            - If strategy == "manual_rl": tasks are already assigned by the agent.
            - Otherwise: use a rule-based strategy to assign tasks before stepping DCs.
        
        Returns:
            dict containing:
                - energy_usage: energy consumption per DC
                - datacenter_infos: detailed info per DC
                - task_distribution: where each task was routed (only in rule-based mode)
                - resource_usage: CPU/GPU/MEM usage per DC
                - pending_task_count: number of pending tasks per DC
        """

        if logger:
            logger.info(f"===== TIMESTEP {current_time} =====")

        # STEP 1. Update simulation clock for all datacenters
        for dc in self.datacenters.values():
            dc._update_current_time_task(current_time)

        # Initialize the results dictionary
        results = {
            "energy_usage": {},
            "datacenter_infos": {},
            "task_distribution": defaultdict(list),  # Only used in rule-based mode
            "resource_usage": defaultdict(list),
            "pending_task_count": {}
            }

        # STEP 2. (Optional) Rule-based task routing
        if self.strategy != "manual_rl":
            # STEP 2.1: Load incoming tasks for this timestep
            tasks = self.get_tasks_for_timestep(current_time)
            if logger:
                logger.info(f"[{current_time}] New tasks fetched: {len(tasks)}")

            # STEP 2.2: Assign tasks to datacenters using rule-based strategy
            dc_id_to_name = {dc.dc_id: name for name, dc in self.datacenters.items()}

            for task in tasks:
                assigned_dc_id = self.distribute_workload(task, current_time, logger)
                if assigned_dc_id is not None:
                    dc_name = dc_id_to_name.get(assigned_dc_id)
                    self.datacenters[dc_name].pending_tasks.append(task)
                    task.dest_dc_id = dc_name
                    task.dest_dc = self.datacenters[dc_name]
                    results["task_distribution"][dc_name].append(task)

                    if logger:
                        logger.info(f"[{current_time}] Task {task.job_name} assigned → {dc_name} (rule-based)")
        else:
            # Manual RL mode: tasks have already been enqueued
            if logger:
                logger.info(f"[{current_time}] Using manual RL mode — skipping internal task assignment")

        # STEP 3. Step each datacenter environment
        for dc_name, dc in self.datacenters.items():
            # STEP 3.1: Advance internal simulation (release finished, schedule pending)
            obs, rew, terminateds, truncateds, info = dc.step(current_time, logger)

            # STEP 3.2: Record energy usage
            results["energy_usage"][dc_name] = info['agent_bat']['bat_total_energy_with_battery_KWh']

            # STEP 3.3: Record full info dict
            results["datacenter_infos"][dc_name] = info

            # STEP 3.4: Track number of pending tasks
            results["pending_task_count"][dc_name] = len(dc.pending_tasks)

            # STEP 3.5: Compute resource utilization snapshot
            resource_snapshot = {
                "cpu": (dc.total_cpus - dc.available_cpus) / dc.total_cpus * 100,
                "gpu": (dc.total_gpus - dc.available_gpus) / dc.total_gpus * 100,
                "mem": (dc.total_mem - dc.available_mem) / dc.total_mem * 100,
            }
            results["resource_usage"][dc_name].append(resource_snapshot)

            if logger:
                logger.info(f"[{current_time}] {dc_name} - Usage: CPU={resource_snapshot['cpu']:.2f}%, "
                            f"GPU={resource_snapshot['gpu']:.2f}%, MEM={resource_snapshot['mem']:.2f}%")
                logger.info(f"[{current_time}] {dc_name} - Running Tasks: {len(dc.running_tasks)}, "
                            f"Pending Tasks: {len(dc.pending_tasks)}")

        # STEP 4. Return aggregated results from all datacenters
        return results



    def get_config_list(self):
        return [dc.env_config for dc in self.datacenters.values()]