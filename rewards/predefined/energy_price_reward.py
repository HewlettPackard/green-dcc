from rewards.base_reward import BaseReward
from rewards.registry_utils import register_reward

@register_reward("energy_price")
class EnergyPriceReward(BaseReward):
    def __init__(self, normalize_factor: float = 100000):
        super().__init__()
        self.normalize_factor = normalize_factor

    def __call__(self, cluster_info, current_tasks, current_time):
        total_task_cost = 0.0
        for task in current_tasks:
            dest_dc = getattr(task, "dest_dc", None)
            if dest_dc:
                task_energy = task.cores_req * task.duration
                task_cost = task_energy * dest_dc.price_manager.get_current_price()
                total_task_cost += task_cost

        reward = -total_task_cost / self.normalize_factor
        self.last_reward = reward
        return reward

    def get_last_value(self):
        return self.last_reward