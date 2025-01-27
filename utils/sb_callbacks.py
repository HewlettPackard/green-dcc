from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

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
        
        # Buffers for partial reward metrics
        self.footprint_vals = []
        self.overdue_vals = []
        self.dropped_vals = []
        self.queue_vals = []
        self.util_vals = []
        self.action_vals = []

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
        
        self.footprint_vals.clear()
        self.overdue_vals.clear()
        self.dropped_vals.clear()
        self.queue_vals.clear()
        self.util_vals.clear()
        self.action_vals.clear()

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
            ls_info = info.get("agent_bat", {})
            # Collect energy, CO2, and water metrics
            if "bat_total_energy_with_battery_KWh" in ls_info:
                self.energy_buffer.append(ls_info["bat_total_energy_with_battery_KWh"])
            if "bat_CO2_footprint" in ls_info:
                self.co2_buffer.append(ls_info["bat_CO2_footprint"])
            
            ls_info = info.get("agent_dc", {})
            if "dc_water_usage" in ls_info:
                self.water_buffer.append(ls_info["dc_water_usage"])

            ls_info = info.get("agent_ls", {})
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
            
            ls_info = info.get("partials", {})
            # Gather partial components if present
            if "ls_footprint_reward" in ls_info:
                self.footprint_vals.append(ls_info["ls_footprint_reward"])
            if "ls_overdue_penalty" in ls_info:
                self.overdue_vals.append(ls_info["ls_overdue_penalty"])
            if "ls_dropped_tasks_penalty" in ls_info:
                self.dropped_vals.append(ls_info["ls_dropped_tasks_penalty"])
            if "ls_tasks_in_queue_reward" in ls_info:
                self.queue_vals.append(ls_info["ls_tasks_in_queue_reward"])
            if "ls_over_utilization_penalty" in ls_info:
                self.util_vals.append(ls_info["ls_over_utilization_penalty"])
            if "ls_action_penalty" in ls_info:
                self.action_vals.append(ls_info["ls_action_penalty"])

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

        # Log averages for each partial component
        if len(self.footprint_vals) > 0:
            self.logger.record("ls/footprint_reward_mean", np.mean(self.footprint_vals))
            self.logger.record("ls/footprint_reward_std", np.std(self.footprint_vals))
        if len(self.overdue_vals) > 0:
            self.logger.record("ls/overdue_penalty_mean", np.mean(self.overdue_vals))
            self.logger.record("ls/overdue_penalty_std", np.std(self.overdue_vals))
        if len(self.dropped_vals) > 0:
            self.logger.record("ls/dropped_tasks_penalty_mean", np.mean(self.dropped_vals))
            self.logger.record("ls/dropped_tasks_penalty_std", np.std(self.dropped_vals))
        if len(self.queue_vals) > 0:
            self.logger.record("ls/queue_reward_mean", np.mean(self.queue_vals))
            self.logger.record("ls/queue_reward_std", np.std(self.queue_vals))
        if len(self.util_vals) > 0:
            self.logger.record("ls/over_util_penalty_mean", np.mean(self.util_vals))
            self.logger.record("ls/over_util_penalty_std", np.std(self.util_vals))
        if len(self.action_vals) > 0:
            self.logger.record("ls/action_penalty_mean", np.mean(self.action_vals))
            self.logger.record("ls/action_penalty_std", np.std(self.action_vals))
            
        
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