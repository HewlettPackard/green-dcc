import random

def distribute_most_available(task, datacenters, logger):
    """
    Assigns the task to the datacenter with the MOST available CPU.
    Now it only selects the datacenter but does NOT schedule the task.
    """    
    best_dc = max(datacenters.values(), key=lambda dc: dc.available_cpus)
    return best_dc.dc_id  # Return selected datacenter ID, do NOT schedule the task



def distribute_random(task, datacenters, logger):
    """
    Randomly assigns the task to one of the eligible datacenters.
    """
    random_dc = random.choice(list(datacenters.values()))
    logger.info(f"Task {task.job_name} added to pending queue of DC{random_dc.dc_id}.")
    return random_dc.dc_id


def distribute_priority_order(task, datacenters, logger, priority_order=["DC1", "DC2", "DC3", "DC4", "DC5"]):
    """
    Assigns tasks following a fixed priority order.
    Now it only selects a datacenter that has available resources.
    """    
    for dc_name in priority_order:
        dc = datacenters.get(dc_name)
        if dc and dc.can_schedule(task):  # Only select if resources are available
            return dc.dc_id  # Return selected datacenter ID

    # No available datacenter found
    logger.warning(f"Task {task.job_name} could not be assigned! No datacenter has enough resources.")
    return None  # Return None if no datacenter can take the task

def distribute_lowest_price(task, datacenters, logger):
    """
    Assigns the task to the datacenter with the lowest current electricity price
    that has enough resources to schedule the task.
    """
    candidates = [
        dc for dc in datacenters.values()
        if dc.can_schedule(task)
    ]

    if not candidates:
        logger.warning(f"Task {task.job_name} could not be assigned! No datacenter has enough resources.")
        return None

    best_dc = min(candidates, key=lambda dc: dc.price_manager.get_current_price())
    logger.info(f"Task {task.job_name} assigned to DC{best_dc.dc_id} (lowest price).")
    return best_dc.dc_id

def distribute_least_pending(task, datacenters, logger):
    """
    Assigns the task to the datacenter with the least pending tasks.
    """
    best_dc = min(datacenters.values(), key=lambda dc: len(dc.pending_tasks))
        
    return best_dc.dc_id


def distribute_lowest_carbon(task, datacenters, logger):
    """
    Assigns the task to the datacenter with the lowest carbon intensity.
    """
    greenest_dc = min(datacenters.values(), key=lambda dc: dc.get_carbon_intensity())
    logger.info(f"Task {task.job_name} added to pending queue of DC{greenest_dc.dc_id}.")
    return greenest_dc.dc_id


def distribute_round_robin(task, datacenters, logger, last_assigned_dc=[-1]):
    """
    Assigns tasks in a round-robin fashion across datacenters.
    """
    datacenter_list = list(datacenters.values())
    last_assigned_dc[0] = (last_assigned_dc[0] + 1) % len(datacenter_list)
    selected_dc = datacenter_list[last_assigned_dc[0]]

    selected_dc.pending_tasks.append(task)
    logger.info(f"Task {task.job_name} added to pending queue of DC{selected_dc.dc_id}.")

    return selected_dc.dc_id

def distribute_lowest_cost(task, datacenters, logger):
    """
    Assigns the task to the datacenter with the lowest electricity cost per kWh.
    """
    cheapest_dc = min(datacenters.values(), key=lambda dc: dc.get_energy_price())
    logger.info(f"Task {task.job_name} assigned to lowest-cost DC{cheapest_dc.dc_id}.")
    return cheapest_dc.dc_id

def distribute_lowest_network_cost(task, datacenters, logger):
    """
    Assigns the task to the datacenter with the lowest network transfer cost.
    """
    best_dc = min(datacenters.values(), key=lambda dc: dc.network_cost_per_gb)
    logger.info(f"Task {task.job_name} assigned to lowest-network-cost DC{best_dc.dc_id}.")
    return best_dc.dc_id

def distribute_lowest_utilization(task, datacenters, logger):
    """
    Assigns the task to the datacenter with the lowest overall utilization (CPU + GPU + Memory).
    """
    best_dc = min(datacenters.values(), key=lambda dc: (dc.available_cpus / dc.total_cpus) +
                                                      (dc.available_gpus / dc.total_gpus) +
                                                      (dc.available_mem / dc.total_mem))
    logger.info(f"Task {task.job_name} assigned to least utilized DC{best_dc.dc_id}.")
    return best_dc.dc_id

def distribute_weighted(task, datacenters, logger, weights={'cost': 0.3, 'carbon': 0.5, 'availability': 0.2}):
    """
    Assigns the task based on a weighted combination of cost, carbon intensity, and resource availability.
    """
    def score(dc):
        cost = dc.get_energy_price() * weights['cost']
        carbon = dc.get_carbon_intensity() * weights['carbon']
        availability = ((dc.available_cpus / dc.total_cpus) + 
                        (dc.available_gpus / dc.total_gpus) + 
                        (dc.available_mem / dc.total_mem)) * weights['availability']
        return cost + carbon - availability  # Lower score is better

    best_dc = min(datacenters.values(), key=score)
    logger.info(f"Task {task.job_name} assigned based on weighted criteria to DC{best_dc.dc_id}.")
    return best_dc.dc_id
