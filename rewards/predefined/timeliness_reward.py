# rewards/predefined/timeliness_reward.py

from rewards.base_reward import BaseReward
from rewards.registry_utils import register_reward
import numpy as np
import pandas as pd # Required for Timedelta calculations

@register_reward("timeliness_penalty")
class TimelinessReward(BaseReward):
    """
    Calculates a reward signal based on task completion times relative to deadlines.
    Applies a penalty that increases linearly with lateness.
    Optionally, can provide a small positive reward for early completion.

    Goal: Encourage tasks to finish closer to (or before) their deadlines.
    """
    def __init__(self,
                 penalty_factor_per_minute: float = 0.1,
                 max_penalty_per_task: float = 10.0,
                 reward_for_early_completion: bool = False,
                 early_reward_factor_per_minute: float = 0.01,
                 max_early_reward_per_task: float = 1.0):
        """
        Args:
            penalty_factor_per_minute (float): Penalty applied for each minute
                                               a task finishes *after* its deadline.
            max_penalty_per_task (float): Maximum penalty that can be applied to
                                          a single late task.
            reward_for_early_completion (bool): If True, provide a positive reward
                                                for finishing tasks early.
            early_reward_factor_per_minute (float): Reward applied for each minute
                                                    a task finishes *before* its deadline.
            max_early_reward_per_task (float): Maximum positive reward for a single
                                               early task.
        """
        super().__init__()
        self.penalty_factor = penalty_factor_per_minute
        self.max_penalty = max_penalty_per_task
        self.reward_early = reward_for_early_completion
        self.early_factor = early_reward_factor_per_minute
        self.max_early_reward = max_early_reward_per_task

    def __call__(self, cluster_info: dict, current_tasks: list, current_time: pd.Timestamp):
        """
        Calculates the timeliness reward/penalty based on finished tasks in cluster_info.

        Args:
            cluster_info (dict): Dictionary containing simulation results.
                                 Expected to have cluster_info["datacenter_infos"][dc_name]["__common__"]["tasks_finished_this_step"].
            current_tasks (list): List of tasks considered in this step (not used).
            current_time (pd.Timestamp): Current simulation time (used for context).

        Returns:
            float: Reward value (negative for lateness, potentially positive for earliness).
        """
        total_timeliness_signal = 0.0
        tasks_finished_count = 0

        if "datacenter_infos" in cluster_info:
            for dc_info in cluster_info["datacenter_infos"].values():
                common_info = dc_info.get("__common__", {})
                # Access the list of Task objects that finished in this step
                # Assuming the key 'tasks_finished_this_step' holds Task objects
                # Or potentially access via sla_stats if it contains task finish details
                # Let's assume 'finished_tasks' list exists in common_info (needs adding in SustainDC.step)
                finished_tasks_in_dc = common_info.get("tasks_finished_this_step_objects", []) # NEED TO ADD THIS TO SustainDC info

                for task in finished_tasks_in_dc:
                    tasks_finished_count += 1
                    if hasattr(task, 'finish_time') and hasattr(task, 'sla_deadline'):
                        # Calculate lateness or earliness in minutes
                        time_delta_seconds = (task.finish_time - task.sla_deadline).total_seconds()
                        time_delta_minutes = time_delta_seconds / 60.0

                        if time_delta_minutes > 0: # Task is late
                            # Apply linear penalty, capped at max_penalty
                            penalty = min(time_delta_minutes * self.penalty_factor, self.max_penalty)
                            total_timeliness_signal -= penalty
                        elif self.reward_early and time_delta_minutes < 0: # Task is early
                            # Apply linear reward, capped at max_early_reward
                            earliness_minutes = -time_delta_minutes
                            early_reward = min(earliness_minutes * self.early_factor, self.max_early_reward)
                            total_timeliness_signal += early_reward
                        # Else: task finished exactly on time, no penalty/reward for this component

                    # else:
                        # Handle cases where task object might miss expected attributes
                        # print(f"Warning: Task {getattr(task, 'job_name', 'Unknown')} missing finish_time or sla_deadline.")


        # Optional: Average the signal over the number of tasks finished?
        # Or keep it as a total penalty/reward accumulated this step?
        # Let's keep it as total signal for now. Could normalize per task later if needed.
        # if tasks_finished_count > 0:
        #     final_signal = total_timeliness_signal / tasks_finished_count
        # else:
        #     final_signal = 0.0

        final_signal = total_timeliness_signal
        self.last_reward = final_signal # Store the computed signal
        return final_signal
