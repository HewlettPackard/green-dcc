import numpy as np
import pandas as pd
from rl_components.task import Task

def assign_task_origins(tasks, datacenter_configs, current_time_utc, logger=None):
    """
    Assigns each task an origin datacenter (DC), based on:
    - Population weight of the DC
    - Local time activity (higher weight during 8am–8pm local time)

    Args:
        tasks (List[Task]): List of Task objects to assign.
        datacenter_configs (List[dict]): DC configuration including weights and timezone offsets.
        current_time_utc (datetime): Current simulation time in UTC.
        logger (logging.Logger or None): Optional logger for debug output.
    """
    def compute_activity_score(local_hour):
        # Simulate typical activity pattern: peak between 8am and 8pm local time
        return 1.0 if 8 <= local_hour < 20 else 0.3

    # Step 1: Calculate scores based on population × activity
    scores = {}
    for config in datacenter_configs:
        dc_id = config['dc_id']
        pop_weight = config.get('population_weight', 0.1)
        tz_shift = config.get('timezone_shift', 0)
        local_hour = (current_time_utc + pd.Timedelta(hours=tz_shift)).hour
        scores[dc_id] = pop_weight * compute_activity_score(local_hour)

    # Step 2: Normalize scores into probabilities
    total_score = sum(scores.values())
    probabilities = {dc_id: score / total_score for dc_id, score in scores.items()}

    # Step 3: Assign origin DC to each task
    dc_ids = list(probabilities.keys())
    probs = list(probabilities.values())

    for task in tasks:
        origin_dc = int(np.random.choice(dc_ids, p=probs))
        task.origin_dc_id = origin_dc

        if logger:
            logger.debug(f"Task {task.job_name} assigned origin DC{origin_dc}.")


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

        # Create the original task
        task = Task(job_name, arrival_time, duration, cpu_req, gpu_req, mem_req, bandwidth_gb)
        tasks.append(task)

        # Create scaled/augmented versions of the task (if scale > 1)
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

    # Assign origin datacenter (population × timezone-aware)
    if datacenter_configs and current_time_utc:
        # We'll pass in the same logger for debug prints
        assign_task_origins(tasks, datacenter_configs, current_time_utc, logger=logger)

    if logger:
        logger.info(f"extract_tasks_from_row: Created {len(tasks)} tasks at time {current_time_utc}.")
        for idx, t in enumerate(tasks):
            logger.debug(
                f"  Task[{idx}]: {t.job_name} | origin=DC{t.origin_dc_id} | "
                f"CPU={t.cpu_req:.2f}, GPU={t.gpu_req:.2f}, MEM={t.mem_req:.2f}, "
                f"BW={t.bandwidth_gb:.2f}, duration={t.duration:.2f}"
            )

    return tasks
