# utils/task_assignment_strategies.py

import random
import numpy as np
import logging # Use standard logging

# --- Base Class ---
class BaseRBCStrategy:
    """Base class for all Rule-Based Controller strategies."""
    def __call__(self, task, datacenters: dict, logger: logging.Logger = None):
        """
        Selects a destination datacenter ID for the given task.

        Args:
            task (Task): The task object to be assigned.
            datacenters (dict): A dictionary mapping DC names (e.g., "DC1")
                                to SustainDC environment objects.
            logger (logging.Logger, optional): Logger instance. Defaults to None.

        Returns:
            int or None: The numerical dc_id of the selected datacenter,
                         or None if no suitable datacenter is found.
        """
        raise NotImplementedError

    def reset(self):
        """Resets any internal state of the strategy (optional)."""
        pass

# --- Concrete Strategy Implementations ---

class DistributeMostAvailable(BaseRBCStrategy):
    """
    Assigns the task to the datacenter with the MOST available CPU cores
    AMONG THOSE THAT CAN SCHEDULE the task.
    """
    def __call__(self, task, datacenters: dict, logger: logging.Logger = None):
        if not datacenters:
             if logger: logger.error("MostAvailable: No datacenters provided.")
             return None

        # --- Filter DCs that can schedule the task ---
        schedulable_dcs = []
        for dc_name, dc in datacenters.items():
            # Check if the DC object has the can_schedule method and if it returns True
            if hasattr(dc, 'can_schedule') and dc.can_schedule(task):
                schedulable_dcs.append(dc)
            # else:
            #     if logger: logger.debug(f"MostAvailable: DC {dc.dc_id} cannot schedule task {task.job_name}. Skipping.")

        # --- If no DC can schedule the task ---
        if not schedulable_dcs:
             if logger: logger.warning(f"MostAvailable: No datacenter can schedule task {task.job_name}. Cannot assign.")
             return None # Indicate no assignment possible

        # --- Find the best DC among the schedulable ones ---
        try:
            # Find the DC with the maximum available cores among the filtered list
            best_dc = max(schedulable_dcs, key=lambda dc: getattr(dc, 'available_cores', -float('inf')))
            if logger: logger.info(f"MostAvailable choice for task {task.job_name}: DC{best_dc.dc_id} ({getattr(best_dc,'available_cores', 0):.1f} cores avail)")
            return best_dc.dc_id
        except Exception as e:
            if logger: logger.error(f"MostAvailable error during max selection: {e}")
            # Fallback: return first schedulable DC's ID if error occurs
            return schedulable_dcs[0].dc_id if schedulable_dcs else None


class DistributeRandom(BaseRBCStrategy):
    """Randomly assigns the task to one of the available datacenters."""
    def __call__(self, task, datacenters: dict, logger: logging.Logger = None):
        if not datacenters:
             if logger: logger.error("Random: No datacenters provided.")
             return None

        # Select a random DC name, then get the object
        random_dc_name = random.choice(list(datacenters.keys()))
        random_dc = datacenters[random_dc_name]
        if logger:
            logger.info(f"Random choice for task {task.job_name}: DC{random_dc.dc_id}")
        return random_dc.dc_id


class DistributePriorityOrder(BaseRBCStrategy):
    """
    Assigns tasks following a fixed priority order of DC names,
    selecting the first one that can schedule the task.
    """
    def __init__(self, priority_order=["DC1", "DC2", "DC3", "DC4", "DC5"]):
         # Default order, can be customized during instantiation if needed
        self.priority_order = priority_order
        super().__init__()

    def __call__(self, task, datacenters: dict, logger: logging.Logger = None):
        if not datacenters:
             if logger: logger.error("PriorityOrder: No datacenters provided.")
             return None

        for dc_name in self.priority_order:
            dc = datacenters.get(dc_name)
            # Check if DC exists and can schedule the task
            if dc and hasattr(dc, 'can_schedule') and dc.can_schedule(task):
                if logger: logger.info(f"PriorityOrder choice for task {task.job_name}: {dc_name} (DC{dc.dc_id})")
                return dc.dc_id # Return numerical ID

        # No available datacenter found in the priority list that can schedule
        if logger:
            logger.warning(f"PriorityOrder: Task {task.job_name} could not be assigned! No suitable DC found in priority list.")
        return None # Indicate no suitable DC found


class DistributeLowestPrice(BaseRBCStrategy):
    """
    Assigns the task to the available datacenter with the lowest current
    electricity price ($/MWh).
    """
    def __call__(self, task, datacenters: dict, logger: logging.Logger = None):
        if not datacenters:
            if logger: logger.error("LowestPrice: No datacenters provided.")
            return None

        candidates = []
        for dc_name, dc in datacenters.items():
            # Check if DC can schedule and has price info
            if hasattr(dc, 'can_schedule') and dc.can_schedule(task) and hasattr(dc, 'price_manager'):
                try:
                    price = dc.price_manager.get_current_price()
                    if price is not None:
                        candidates.append((price, dc)) # Store price and DC object
                    else:
                         if logger: logger.warning(f"LowestPrice: Could not get price for {dc_name}")
                except Exception as e:
                     if logger: logger.error(f"LowestPrice: Error getting price for {dc_name}: {e}")
            # else:
            #      if logger: logger.debug(f"LowestPrice: Skipping {dc_name} (cannot schedule or no price manager)")

        if not candidates:
            if logger:
                logger.warning(f"LowestPrice: Task {task.job_name} could not be assigned! No schedulable DC found with price info.")
            # Fallback: maybe assign randomly or to first available? Or return None? Let's return None.
            return None

        # Find the DC with the minimum price among candidates
        candidates.sort(key=lambda item: item[0]) # Sort by price (first element of tuple)
        best_price, best_dc = candidates[0]

        if logger:
            logger.info(f"LowestPrice choice for task {task.job_name}: DC{best_dc.dc_id} (Price: {best_price:.2f} $/MWh)")
        return best_dc.dc_id


class DistributeLeastPending(BaseRBCStrategy):
    """Assigns the task to the datacenter with the fewest pending tasks."""
    def __call__(self, task, datacenters: dict, logger: logging.Logger = None):
        if not datacenters:
            if logger: logger.error("LeastPending: No datacenters provided.")
            return None

        # Find the DC object with the minimum pending task queue length
        try:
             best_dc = min(datacenters.values(), key=lambda dc: len(getattr(dc, 'pending_tasks', [])))
             pending_count = len(getattr(best_dc, 'pending_tasks', []))
             if logger: logger.info(f"LeastPending choice for task {task.job_name}: DC{best_dc.dc_id} ({pending_count} pending)")
             return best_dc.dc_id
        except Exception as e:
             if logger: logger.error(f"LeastPending error: {e}")
             # Fallback
             return list(datacenters.values())[0].dc_id if datacenters else None


class DistributeLowestCarbon(BaseRBCStrategy):
    """
    Assigns the task to the available datacenter with the lowest current
    carbon intensity (gCO2/kWh).
    """
    def __call__(self, task, datacenters: dict, logger: logging.Logger = None):
        if not datacenters:
            if logger: logger.error("LowestCarbon: No datacenters provided.")
            return None

        candidates = []
        for dc_name, dc in datacenters.items():
             # Check if DC can schedule and has CI info getter
             if hasattr(dc, 'can_schedule') and dc.can_schedule(task) and hasattr(dc, 'get_current_carbon_intensity'):
                try:
                    # Use the helper method that should now exist
                    ci = dc.get_current_carbon_intensity(norm=False) # Get raw gCO2/kWh
                    if ci is not None:
                        candidates.append((ci, dc))
                    else:
                         if logger: logger.warning(f"LowestCarbon: Could not get CI for {dc_name}")
                except Exception as e:
                     if logger: logger.error(f"LowestCarbon: Error getting CI for {dc_name}: {e}")
             # else:
             #      if logger: logger.debug(f"LowestCarbon: Skipping {dc_name} (cannot schedule or no CI getter)")

        if not candidates:
            if logger:
                logger.warning(f"LowestCarbon: Task {task.job_name} could not be assigned! No schedulable DC found with CI info.")
            return None # Indicate no suitable DC

        # Find the DC with the minimum CI among candidates
        candidates.sort(key=lambda item: item[0]) # Sort by CI
        best_ci, best_dc = candidates[0]

        if logger:
            logger.info(f"LowestCarbon choice for task {task.job_name}: DC{best_dc.dc_id} (CI: {best_ci:.2f} gCO2/kWh)")
        return best_dc.dc_id


class DistributeRoundRobin(BaseRBCStrategy):
    """Assigns tasks in a round-robin fashion across datacenters."""
    def __init__(self):
        self.last_assigned_dc_index = -1
        # Consistent order based on sorted numerical IDs
        self._dc_order_ids = []
        super().__init__()

    def reset(self):
        """Resets the round-robin index."""
        self.last_assigned_dc_index = -1
        self._dc_order_ids = [] # Clear the order cache
        if logging.getLogger().isEnabledFor(logging.DEBUG): # Avoid calculation if not debugging
             logging.debug("RoundRobin state reset.")

    def __call__(self, task, datacenters: dict, logger: logging.Logger = None):
        if not datacenters:
            if logger: logger.error("RoundRobin: No datacenters provided.")
            return None

        # Ensure a consistent order (sort by numerical dc_id)
        # Update the order only if the set of keys changes (more robust)
        current_dc_keys = sorted(datacenters.keys()) # Example using names, could sort by dc_id too
        current_dc_ids = sorted([dc.dc_id for dc in datacenters.values()])

        # Rebuild ordered list if it's empty or the IDs have changed
        if not self._dc_order_ids or self._dc_order_ids != current_dc_ids:
            self._dc_order_ids = current_dc_ids
            # Optionally reset index when DC set changes, or just continue cycle
            # self.last_assigned_dc_index = -1 # Uncomment to reset index on change
            if logger: logger.debug(f"RoundRobin order updated: {self._dc_order_ids}")

        if not self._dc_order_ids: # Should not happen if datacenters is not empty
             if logger: logger.error("RoundRobin: Failed to establish DC order.")
             return list(datacenters.values())[0].dc_id # Fallback

        # Increment index and wrap around
        self.last_assigned_dc_index = (self.last_assigned_dc_index + 1) % len(self._dc_order_ids)
        selected_dc_id = self._dc_order_ids[self.last_assigned_dc_index]

        # # Optional: Check if the selected DC can schedule the task
        # selected_dc = next((dc for dc in datacenters.values() if dc.dc_id == selected_dc_id), None)
        # if selected_dc and not selected_dc.can_schedule(task):
        #     if logger: logger.warning(f"RoundRobin selected DC{selected_dc_id} but it cannot schedule task {task.job_name}. Assigning anyway.")
        #     # Policy: Assign anyway, let the DC queue handle it. Or could try next DC.

        if logger:
            logger.info(f"RoundRobin choice for task {task.job_name}: DC{selected_dc_id} (index {self.last_assigned_dc_index})")

        return selected_dc_id # Return numerical ID


class DistributeLocalOnly(BaseRBCStrategy):
    """Assigns the task strictly to its origin datacenter."""
    def __call__(self, task, datacenters: dict, logger: logging.Logger = None):
        if not hasattr(task, 'origin_dc_id') or task.origin_dc_id is None:
            if logger: logger.error(f"LocalOnly: Task {task.job_name} missing valid origin_dc_id.")
            # Decide fallback: Maybe assign randomly? Or return None? Let's return None.
            return None

        origin_id = task.origin_dc_id

        # Optional: Check if the origin DC actually exists in the current cluster setup
        origin_dc_exists = any(dc.dc_id == origin_id for dc in datacenters.values())
        if not origin_dc_exists:
             if logger: logger.error(f"LocalOnly: Origin DC {origin_id} for task {task.job_name} not found in current cluster configuration.")
             # Fallback strategy needed here as well. Assign randomly? Or None?
             # Let's assign randomly among existing DCs as a simple fallback.
             if not datacenters: return None
             fallback_dc = random.choice(list(datacenters.values()))
             if logger: logger.warning(f"LocalOnly: Assigning task {task.job_name} randomly to DC{fallback_dc.dc_id} as origin DC{origin_id} is missing.")
             return fallback_dc.dc_id


        if logger:
            logger.info(f"LocalOnly choice for task {task.job_name}: Assigning to origin DC{origin_id}")

        # Return the numerical origin ID
        return origin_id
# --- Add other strategies similarly ---

class DistributeLowestUtilization(BaseRBCStrategy):
    """Assigns the task to the datacenter with the highest overall average resource availability."""
    def __call__(self, task, datacenters: dict, logger: logging.Logger = None):
        if not datacenters:
             if logger: logger.error("LowestUtilization: No datacenters provided.")
             return None

        def calculate_availability_score(dc):
            cpu_total = getattr(dc, 'total_cores', 0)
            gpu_total = getattr(dc, 'total_gpus', 0)
            mem_total = getattr(dc, 'total_mem_GB', 0)

            # Use getattr with defaults for safety
            cpu_avail = getattr(dc, 'available_cores', 0) / cpu_total if cpu_total > 0 else 0
            gpu_avail = getattr(dc, 'available_gpus', 0) / gpu_total if gpu_total > 0 else 0
            mem_avail = getattr(dc, 'available_mem', 0) / mem_total if mem_total > 0 else 0

            # Average availability - weights could be added
            return (cpu_avail + gpu_avail + mem_avail) / 3.0

        try:
            # Find DC with the maximum availability score
            best_dc = max(datacenters.values(), key=calculate_availability_score)
            if logger:
                score = calculate_availability_score(best_dc)
                logger.info(f"LowestUtilization (Max Avail) choice for task {task.job_name}: DC{best_dc.dc_id} (Score: {score:.3f})")
            return best_dc.dc_id
        except Exception as e:
            if logger: logger.error(f"LowestUtilization error: {e}")
            # Fallback
            return list(datacenters.values())[0].dc_id if datacenters else None


# Example of a Weighted Strategy (Needs helper methods in SustainDC)
# class DistributeWeighted(BaseRBCStrategy):
#     """
#     Assigns the task based on a weighted combination of normalized cost,
#     carbon intensity, and resource availability. Lower score is better.
#     NOTE: Requires normalization and assumes getter methods exist.
#     """
#     def __init__(self, weights={'cost': 0.3, 'carbon': 0.5, 'utilization': 0.2}):
#         self.weights = weights
#         super().__init__()

#     def __call__(self, task, datacenters: dict, logger: logging.Logger = None):
#         if not datacenters: return None

#         dc_scores = []
#         # --- Need to get ranges or use running stats for normalization ---
#         # Example: Placeholder normalization - replace with real stats
#         all_prices = [dc.price_manager.get_current_price() for dc in datacenters.values() if hasattr(dc,'price_manager')]
#         all_cis = [dc.get_current_carbon_intensity(norm=False) for dc in datacenters.values() if hasattr(dc,'get_current_carbon_intensity')]
#         min_price, max_price = min(all_prices) if all_prices else 0, max(all_prices) if all_prices else 1
#         min_ci, max_ci = min(all_cis) if all_cis else 0, max(all_cis) if all_cis else 1
#         price_range = max(1e-6, max_price - min_price)
#         ci_range = max(1e-6, max_ci - min_ci)
#         # --- End Placeholder Normalization ---

#         for dc_name, dc in datacenters.items():
#             try:
#                 norm_cost = (dc.price_manager.get_current_price() - min_price) / price_range if hasattr(dc,'price_manager') else 0.5
#                 norm_ci = (dc.get_current_carbon_intensity(norm=False) - min_ci) / ci_range if hasattr(dc,'get_current_carbon_intensity') else 0.5

#                 cpu_util = 1.0 - (getattr(dc,'available_cores',0) / getattr(dc,'total_cores',1))
#                 gpu_util = 1.0 - (getattr(dc,'available_gpus',0) / getattr(dc,'total_gpus',1))
#                 mem_util = 1.0 - (getattr(dc,'available_mem',0) / getattr(dc,'total_mem_GB',1))
#                 avg_util = (cpu_util + gpu_util + mem_util) / 3.0

#                 # Lower score is better: lower cost, lower ci, lower utilization (higher availability)
#                 score = (norm_cost * self.weights['cost'] +
#                          norm_ci * self.weights['carbon'] +
#                          avg_util * self.weights['utilization']) # Lower utilization = lower score = better? Check logic.

#                 dc_scores.append((score, dc))
#             except Exception as e:
#                  if logger: logger.error(f"Weighted scoring error for {dc_name}: {e}")

#         if not dc_scores:
#             if logger: logger.warning("Weighted: No DCs could be scored.")
#             return list(datacenters.values())[0].dc_id if datacenters else None

#         dc_scores.sort(key=lambda item: item[0]) # Sort by score (ascending)
#         best_score, best_dc = dc_scores[0]

#         if logger:
#             logger.info(f"Weighted choice for task {task.job_name}: DC{best_dc.dc_id} (Score: {best_score:.3f})")
#         return best_dc.dc_id