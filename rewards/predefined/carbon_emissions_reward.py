from rewards.base_reward import BaseReward
from rewards.registry_utils import register_reward
import numpy as np

@register_reward("carbon_emissions")
class CarbonEmissionsReward(BaseReward):
    def __init__(self, normalize_factor: float = 100.0):
        super().__init__()
        self.normalize_factor = np.float64(normalize_factor)

    def __call__(self, cluster_info: dict, current_time): # <<< Correct signature
        # Sum carbon emissions from all datacenters' operations
        total_emissions_kg = np.float64(0.0)
        if "datacenter_infos" in cluster_info:
            for dc_info in cluster_info["datacenter_infos"].values():
                total_emissions_kg += dc_info["__common__"].get("carbon_emissions_kg", 0.0)
        
        # Add transmission emissions from the global results
        total_emissions_kg += cluster_info.get("transmission_emissions_total_kg", 0.0)

        reward = -total_emissions_kg / self.normalize_factor
        self.last_reward = float(reward)
        return self.last_reward