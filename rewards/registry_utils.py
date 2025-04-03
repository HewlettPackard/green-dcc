reward_registry = {}

def register_reward(name):
    def decorator(cls):
        reward_registry[name] = cls
        return cls
    return decorator

def get_reward_function(name, **kwargs):
    if name not in reward_registry:
        raise ValueError(f"Reward '{name}' not found in registry.")
    return reward_registry[name](**kwargs)  # Instantiate with kwargs

