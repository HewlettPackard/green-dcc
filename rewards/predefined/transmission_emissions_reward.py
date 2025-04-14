from rewards.base_reward import BaseReward
from rewards.registry_utils import register_reward

@register_reward("transmission_emissions")
class TransmissionEmissionsReward(BaseReward):
    def __init__(self, normalize_factor: float = 1.0):
        super().__init__()
        self.normalize_factor = normalize_factor

    def __call__(self, cluster_info, current_tasks, current_time):
        emissions_kg = cluster_info.get("transmission_emissions_total_kg", 0.0)
        reward = -emissions_kg / self.normalize_factor
        self.last_reward = reward
        return reward

    def get_last_value(self):
        return self.last_reward
