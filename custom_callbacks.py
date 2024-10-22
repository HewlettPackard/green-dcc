# custom_callbacks.py

from ray.rllib.algorithms.callbacks import DefaultCallbacks
# from ray.rllib.utils.typing import TrainerCallbackContext
# from ray.tune.logger import UnifiedLogger
import numpy as np

class CustomMetricsCallback(DefaultCallbacks):
    """
    Custom callback for logging aggregated variables at the end of each episode to TensorBoard.
    """

    def __init__(self):
        super().__init__()
        # Initialize accumulators and episode lengths for each agent
        self.accumulators = {}
        self.episode_lengths = {}

    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        dc_ids = base_env.envs[0].datacenter_ids # ['DC1', 'DC2', 'DC3']
        for dc_id in dc_ids:
            agent_id = f"{dc_id}_ls_policy"
            self.accumulators[agent_id] = {
                'bat_CO2_footprint_sum': 0.0,
                'ls_tasks_dropped_sum': 0.0,
                'ls_overdue_penalty_sum': 0.0,
                # Add more metrics as needed
            }
            self.episode_lengths[agent_id] = 0
        
        # Accumulated metrics for the high-level policy
        self.accumulators['accumulated'] = {
            'bat_CO2_footprint_sum': 0.0,
            'ls_tasks_dropped_sum': 0.0,
            'ls_overdue_penalty_sum': 0.0,
            # Add more metrics as needed
        }
        self.episode_lengths['accumulated'] = 0


    def on_episode_step(self, *, worker, base_env, policies, episode, **kwargs):
        
        infos = episode._last_infos
        agent_ids = episode.get_agents()
        # if infos is not None:
            # print(infos)
        for agent_id in agent_ids:
            if agent_id in infos and agent_id in self.accumulators.keys():
                info = infos[agent_id]
                self.accumulators[agent_id]['bat_CO2_footprint_sum'] += info['agent_ls']['bat_CO2_footprint']
                self.accumulators[agent_id]['ls_tasks_dropped_sum'] += info['agent_ls']['ls_tasks_dropped']
                self.accumulators[agent_id]['ls_overdue_penalty_sum'] += info['agent_ls']['ls_overdue_penalty']
                # Add more metrics as needed
                self.episode_lengths[agent_id] += 1

        # For the accumulated, sum the metrics from all agents
        self.accumulators['accumulated']['bat_CO2_footprint_sum'] = sum(
            self.accumulators[agent_id]['bat_CO2_footprint_sum'] for agent_id in self.accumulators.keys())
        self.accumulators['accumulated']['ls_tasks_dropped_sum'] = sum(
            self.accumulators[agent_id]['ls_tasks_dropped_sum'] for agent_id in self.accumulators.keys())
        self.accumulators['accumulated']['ls_overdue_penalty_sum'] = sum(
            self.accumulators[agent_id]['ls_overdue_penalty_sum'] for agent_id in self.accumulators.keys())
        
        self.episode_lengths['accumulated'] += 1

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # Log the accumulated metrics to TensorBoard
        for agent_id in self.accumulators.keys():
            for metric, value in self.accumulators[agent_id].items():
                episode.custom_metrics[f'custom_metric/{agent_id}/{metric}'] = value / self.episode_lengths[agent_id]
        # Log the accumulated metrics for the high-level policy
        for metric, value in self.accumulators['accumulated'].items():
            episode.custom_metrics[f'custom_metric/accumulated/{metric}'] = value / sum(self.episode_lengths.values())
        # Reset the accumulators and episode lengths
        self.accumulators = {}
        self.episode_lengths = {}