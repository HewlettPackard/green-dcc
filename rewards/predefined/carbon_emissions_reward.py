from rewards.base_reward import BaseReward
from rewards.registry_utils import register_reward

@register_reward("carbon_emissions")
class CarbonEmissionsReward(BaseReward):
    def __init__(self, normalize_factor: float = 100.0):
        self.normalize_factor = normalize_factor

    def __call__(self, cluster_info, current_tasks, current_time):
        total_emissions = 0.0
        for dc_info in cluster_info["datacenter_infos"].values():
            total_emissions += dc_info["__common__"].get("carbon_emissions_kg", 0)

        return -total_emissions / self.normalize_factor
