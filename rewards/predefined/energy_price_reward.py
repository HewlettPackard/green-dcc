from rewards.base_reward import BaseReward
from rewards.registry_utils import register_reward
import numpy as np # Import numpy

@register_reward("energy_price")
class EnergyPriceReward(BaseReward):
    def __init__(self, normalize_factor: float = 1000.0): # A higher factor might be needed
        super().__init__()
        self.normalize_factor = np.float64(normalize_factor) if normalize_factor != 0 else np.float64(1.0)

    def __call__(self, cluster_info: dict, current_time): # <<< Correct signature
        # Sum the energy cost from the results of all datacenters for this step
        total_cost = np.float64(0.0)
        if "datacenter_infos" in cluster_info:
            for dc_info in cluster_info["datacenter_infos"].values():
                # The info dict from SustainDC.step contains the cost
                total_cost += dc_info["__common__"].get("energy_cost_USD", 0.0)
        
        # Add transmission cost if present at the top level
        total_cost += cluster_info.get("transmission_cost_total_usd", 0.0)

        reward = -total_cost / self.normalize_factor
        self.last_reward = float(reward)
        return self.last_reward