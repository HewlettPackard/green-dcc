from rewards.registry_utils import get_reward_function, register_reward
from rewards.base_reward import BaseReward

from collections import defaultdict

class CompositeReward(BaseReward):
    def __init__(self, components: dict, normalize=True, epsilon=1e-8):
        self.normalize = normalize
        self.epsilon = epsilon
        self.running_stats = defaultdict(lambda: {"mean": 0.0, "var": 1.0, "count": 1e-8})

        self.components = []
        self.last_values = {}  # Track latest unnormalized values

        for name, cfg in components.items():
            weight = cfg.get("weight", 1.0)
            args = cfg.get("args", {})
            reward_fn = get_reward_function(name, **args)
            self.components.append((name, weight, reward_fn))

    def update_running_stats(self, name, value):
        stats = self.running_stats[name]
        stats["count"] += 1
        delta = value - stats["mean"]
        stats["mean"] += delta / stats["count"]
        delta2 = value - stats["mean"]
        stats["var"] += delta * delta2  # Bessel’s correction optional

    def normalize_reward(self, name, value):
        stats = self.running_stats[name]
        std = (stats["var"] / stats["count"])**0.5 + self.epsilon
        return (value - stats["mean"]) / std

    def __call__(self, cluster_info, current_tasks, current_time):
        total = 0.0
        self.last_values.clear()
        for name, weight, fn in self.components:
            raw_value = fn(cluster_info, current_tasks, current_time)
            self.last_values[name] = raw_value
            if self.normalize:
                self.update_running_stats(name, raw_value)
                raw_value = self.normalize_reward(name, raw_value)
            total += weight * raw_value
        self.last_reward = total
        return total

    def get_last_components(self):
        return self.last_values

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