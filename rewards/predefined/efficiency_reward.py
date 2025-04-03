from rewards.base_reward import BaseReward
from rewards.registry_utils import register_reward

@register_reward("efficiency")
class EfficiencyReward(BaseReward):
    def __call__(self, cluster_info, current_tasks, current_time):
        total_energy = sum(
            dc_info["__common__"].get("energy_consumption_kwh", 0)
            for dc_info in cluster_info["datacenter_infos"].values()
        )
        total_tasks = cluster_info.get("scheduled_tasks", 0)
        if total_tasks == 0:
            return 0.0
        return -total_energy / total_tasks
