from rewards.base_reward import BaseReward
from rewards.registry_utils import register_reward

@register_reward("energy_consumption")
class EnergyConsumptionReward(BaseReward):
    def __init__(self, normalize_factor: float = 1000.0):
        super().__init__()
        self.normalize_factor = normalize_factor

    def __call__(self, cluster_info, current_tasks, current_time):
        total_energy = 0.0
        for dc_info in cluster_info["datacenter_infos"].values():
            total_energy += dc_info["__common__"].get("energy_consumption_kwh", 0)
            
        reward = -total_energy / self.normalize_factor
        self.last_reward = reward
        return reward

    def get_last_value(self):
        return self.last_reward