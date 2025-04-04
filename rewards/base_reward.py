class BaseReward:
    def __init__(self, **kwargs):
        self.last_reward = None

    def __call__(self, cluster_info: dict, current_tasks: list, current_time):
        raise NotImplementedError("Subclasses must implement this method.")

    def get_last_value(self):
        return self.last_reward

    def __str__(self):
        return self.__class__.__name__
