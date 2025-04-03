from rewards.registry_utils import get_reward_function, register_reward
from rewards.base_reward import BaseReward

class CompositeReward(BaseReward):
    def __init__(self, components: dict):
        """
        components: {
            "carbon_cost": {"weight": 0.5, "args": {"carbon_price_per_kg": 0.2}},
            "sla_penalty": {"weight": 0.5, "args": {"penalty_per_violation": 20.0}}
        }
        """
        self.components = []
        for name, cfg in components.items():
            weight = cfg.get("weight", 1.0)
            args = cfg.get("args", {})
            reward_fn = get_reward_function(name, **args)
            self.components.append((weight, reward_fn))

    def __call__(self, cluster_info, current_tasks, current_time):
        total = 0.0
        for weight, fn in self.components:
            total += weight * fn(cluster_info, current_tasks, current_time)
        return total

'''
Example of composite reward configuration:

components={
    "energy_price": {
        "weight": 0.5,
        "args": {"normalize_factor": 100000}
    },
    "carbon_cost": {
        "weight": 0.3,
        "args": {"normalize_factor": 100}
    },
    "sla_penalty": {
        "weight": 0.2,
        "args": {"penalty_per_violation": 5.0}
    }
}


'''