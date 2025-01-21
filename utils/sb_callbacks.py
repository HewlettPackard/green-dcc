from stable_baselines3.common.callbacks import BaseCallback

class CO2WaterUsageCallback(BaseCallback):
    """
    Custom callback for logging average CO2 footprint and water usage to TensorBoard.
    """

    def __init__(self, verbose=0):
        super(CO2WaterUsageCallback, self).__init__(verbose)
        self.energy_buffer = []
        self.co2_buffer = []
        self.water_buffer = []

    def _on_rollout_start(self) -> None:
        """
        Called before collecting new samples.
        Reset buffers at the start of each rollout.
        """
        self.energy_buffer.clear()
        self.co2_buffer.clear()
        self.water_buffer.clear()

    def _on_step(self) -> bool:
        """
        Called after each environment step in the rollout.
        Accumulate CO2 and water usage from `info`.
        """
        # self.locals["infos"] is a list of info dicts for each parallel env
        infos = self.locals.get("infos", [])
        for info in infos:
            if "co2_footprint" in info:
                self.co2_buffer.append(info["co2_footprint"])
            if "water_usage" in info:
                self.water_buffer.append(info["water_usage"])
        # Return True to continue training
        return True

    def _on_rollout_end(self) -> None:
        """
        Called after each rollout ends (i.e., once `n_steps` per env has been collected).
        Compute averages and log them to TensorBoard.
        """
        if len(self.co2_buffer) > 0:
            avg_co2 = sum(self.co2_buffer) / len(self.co2_buffer)
            # Log it to TensorBoard
            self.logger.record("metrics/avg_co2_footprint", avg_co2)

        if len(self.water_buffer) > 0:
            avg_water = sum(self.water_buffer) / len(self.water_buffer)
            # Log it to TensorBoard
            self.logger.record("metrics/avg_water_usage", avg_water)
            


class TemporalLoadShiftingCallback(BaseCallback):
    """
    Custom callback for logging metrics related to load shifting to TensorBoard.
    Tracks CO2, water usage, energy, and task-related metrics.
    """

    def __init__(self, verbose=0):
        super(TemporalLoadShiftingCallback, self).__init__(verbose)
        # Buffers for tracking metrics
        self.energy_buffer = []
        self.co2_buffer = []
        self.water_buffer = []

        # Task-related accumulative metrics
        self.total_tasks_in_queue = 0
        self.total_tasks_dropped = 0
        self.total_tasks_processed = 0
        self.total_overdue_penalty = 0

        # Task-related max/average metrics
        self.max_oldest_task_age = float('-inf')
        self.total_task_age = 0  # For calculating average
        self.num_task_age_samples = 0

    def _on_rollout_start(self) -> None:
        """
        Reset buffers and accumulators at the start of each rollout.
        """
        self.energy_buffer.clear()
        self.co2_buffer.clear()
        self.water_buffer.clear()

        self.total_tasks_in_queue = 0
        self.total_tasks_dropped = 0
        self.total_tasks_processed = 0
        self.total_overdue_penalty = 0

        self.max_oldest_task_age = float('-inf')
        self.total_task_age = 0
        self.num_task_age_samples = 0

    def _on_step(self) -> bool:
        """
        Collect data at each step from the environment info.
        """
        infos = self.locals.get("infos", [])
        for info in infos:
            ls_info = info.get("agent_ls", {})
            
            # Collect energy, CO2, and water metrics
            if "bat_total_energy_with_battery_KWh" in ls_info:
                self.energy_buffer.append(ls_info["bat_total_energy_with_battery_KWh"])
            if "bat_CO2_footprint" in ls_info:
                self.co2_buffer.append(ls_info["bat_CO2_footprint"])
            if "dc_water_usage" in ls_info:
                self.water_buffer.append(ls_info["dc_water_usage"])

            # Accumulative metrics
            self.total_tasks_in_queue += ls_info.get("ls_tasks_in_queue", 0)
            self.total_tasks_dropped += ls_info.get("ls_tasks_dropped", 0)
            self.total_tasks_processed += ls_info.get("ls_tasks_processed", 0)
            self.total_overdue_penalty += ls_info.get("ls_overdue_penalty", 0)

            # Maximum metric
            self.max_oldest_task_age = max(self.max_oldest_task_age, ls_info.get("ls_oldest_task_age", 0))

            # Average metric calculation
            task_age = ls_info.get("ls_average_task_age", None)
            if task_age is not None:
                self.total_task_age += task_age
                self.num_task_age_samples += 1

        return True  # Continue training

    def _on_rollout_end(self) -> None:
        """
        Calculate averages, maxima, and log metrics to TensorBoard at the end of the rollout.
        """
        # Log energy, CO2, and water usage metrics
        if len(self.energy_buffer) > 0:
            avg_energy = sum(self.energy_buffer) / len(self.energy_buffer)
            self.logger.record("metrics/avg_energy_consumed", avg_energy)
        if len(self.co2_buffer) > 0:
            avg_co2 = sum(self.co2_buffer) / len(self.co2_buffer)
            self.logger.record("metrics/avg_co2_footprint", avg_co2)
        if len(self.water_buffer) > 0:
            avg_water = sum(self.water_buffer) / len(self.water_buffer)
            self.logger.record("metrics/avg_water_usage", avg_water)

        # Log accumulative metrics
        self.logger.record("metrics/total_tasks_in_queue", self.total_tasks_in_queue)
        self.logger.record("metrics/total_tasks_dropped", self.total_tasks_dropped)
        self.logger.record("metrics/total_tasks_processed", self.total_tasks_processed)
        self.logger.record("metrics/total_overdue_penalty", self.total_overdue_penalty)

        # Log max and average task age metrics
        self.logger.record("metrics/max_oldest_task_age", self.max_oldest_task_age)
        if self.num_task_age_samples > 0:
            avg_task_age = self.total_task_age / self.num_task_age_samples
            self.logger.record("metrics/avg_task_age", avg_task_age)