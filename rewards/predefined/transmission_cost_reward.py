from rewards.base_reward import BaseReward
from rewards.registry_utils import register_reward

@register_reward("transmission_cost")
class TransmissionCostReward(BaseReward):
    def __init__(self, normalize_factor: float = 100.0):
        super().__init__()
        self.normalize_factor = normalize_factor

    def __call__(self, cluster_info, current_tasks, current_time):
        cost = cluster_info.get("transmission_cost_total_usd", 0.0)
        reward = -cost / self.normalize_factor
        self.last_reward = reward
        return reward

    def get_last_value(self):
        return self.last_reward
