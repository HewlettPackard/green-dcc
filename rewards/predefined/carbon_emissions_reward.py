from rewards.base_reward import BaseReward
from rewards.registry_utils import register_reward

@register_reward("carbon_emissions")
class CarbonEmissionsReward(BaseReward):
    def __init__(self, normalize_factor: float = 100.0):
        super().__init__()
        self.normalize_factor = normalize_factor

    def __call__(self, cluster_info, current_tasks, current_time):
        # total_emissions = 0.0
        # for dc_info in cluster_info["datacenter_infos"].values():
        #     total_emissions += dc_info["__common__"].get("carbon_emissions_kg", 0)
        
        total_task_emissions = 0.0
        for task in current_tasks:
            dest_dc = getattr(task, "dest_dc", None)
            if dest_dc:
                task_energy = task.cores_req * task.duration / 10000.0
                task_emissions = task_energy * dest_dc.ci_manager.get_current_ci(norm=False)
                total_task_emissions += task_emissions

        reward = -total_task_emissions / self.normalize_factor
        self.last_reward = reward
        return reward

    def get_last_value(self):
        return self.last_reward