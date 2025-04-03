class BaseReward:
    def __init__(self, **kwargs):
        pass

    def __call__(self, cluster_info: dict, current_tasks: list, current_time) -> float:
        raise NotImplementedError("Subclasses must implement this method.")
