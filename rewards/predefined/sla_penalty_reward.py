from rewards.base_reward import BaseReward
from rewards.registry_utils import register_reward

@register_reward("sla_penalty")
class SLAPenaltyReward(BaseReward):
    def __init__(self, penalty_per_violation: float = 10.0):
        super().__init__()
        self.penalty = penalty_per_violation

    def __call__(self, cluster_info, current_tasks, current_time):
        sla_violated = sum(
            dc_info["__common__"]["__sla__"].get("violated", 0)
            for dc_info in cluster_info["datacenter_infos"].values()
        )
        reward = -self.penalty * sla_violated
        self.last_reward = reward
        return reward