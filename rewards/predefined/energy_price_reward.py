from rewards.base_reward import BaseReward
from rewards.registry_utils import register_reward
import numpy as np # Import numpy

@register_reward("energy_price")
class EnergyPriceReward(BaseReward):
    """
    Calculates a reward signal based on the total energy cost incurred
    across all datacenters during the simulation timestep.
    Penalizes higher costs.
    """
    def __init__(self, normalize_factor: float = 100.0):
        """
        Args:
            normalize_factor (float): A factor to divide the total cost by,
                                      scaling the reward. Adjust based on expected
                                      cost magnitudes per timestep. Defaults to 100.0.
        """
        super().__init__()
        # Use np.float64 for potentially higher precision if costs can be large
        self.normalize_factor = np.float64(normalize_factor) if normalize_factor != 0 else np.float64(1.0) # Avoid division by zero

    def __call__(self, cluster_info: dict, current_tasks: list, current_time):
        """
        Calculates the reward based on the total energy cost from cluster_info.

        Args:
            cluster_info (dict): Dictionary containing simulation results.
                                 Expected to have cluster_info["datacenter_infos"][dc_name]["__common__"]["energy_cost_USD"].
            current_tasks (list): List of tasks considered in this step (not used by this implementation).
            current_time: Current simulation time (not used by this implementation).

        Returns:
            float: Reward value (typically negative, lower is better).
        """
                
        total_task_cost = np.float64(0.0)
        for task in current_tasks:
            dest_dc = getattr(task, "dest_dc", None)
            if dest_dc:
                task_energy = task.cores_req * task.duration / 10000.0
                task_cost = task_energy * dest_dc.price_manager.get_current_price()
                total_task_cost += task_cost

        # Reward is the negative of the total cost, scaled by the normalization factor
        # Using float64 division
        reward = -total_task_cost / self.normalize_factor
        self.last_reward = float(reward) # Store as standard float
        return self.last_reward

    # def get_last_value(self):
    #     return self.last_reward