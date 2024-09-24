from typing import Dict

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import EnvType, PolicyID


class CustomCallbacks(DefaultCallbacks):
    """
    CustomCallbacks class that extends the DefaultCallbacks class and overrides its methods to customize the
    behavior of the callbacks during the RL training process.
    """

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs) -> None:
        """
        Method that is called at the beginning of each episode in the training process. It sets some user_data
        variables to be used later on.

        Args:
            worker (Worker): The worker object that is being used in the training process.
            base_env (BaseEnv): The base environment that is being used in the training process.
            policies (Dict[str, Policy]): The policies that are being used in the training process.
            episode (MultiAgentEpisode): The episode object that is being processed.
            env_index (int): The index of the environment within the worker task.
            **kwargs: additional arguments that can be passed.
        """
        episode.user_data["net_energy_sum"] = 0
        episode.user_data["CO2_footprint_sum"] = 0
        
        episode.user_data["step_count"] = 0
        episode.user_data["instantaneous_net_energy"] = []
        episode.user_data["load_left"] = 0
        episode.user_data["ls_tasks_in_queue"] = 0
        episode.user_data["ls_tasks_dropped"] = 0

        episode.user_data["water_usage"] = 0
    
    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs) -> None:
        """
        Method that is called at each step of each episode in the training process. It updates some user_data
        variables to be used later on.

        Args:
            worker (Worker): The worker object that is being used in the training process.
            base_env (BaseEnv): The base environment that is being used in the training process.
            episode (MultiAgentEpisode): The episode object that is being processed.
            env_index (int): The index of the environment within the worker task.
            **kwargs: additional arguments that can be passed.
        """
        net_energy = base_env.envs[0].bat_info["bat_total_energy_with_battery_KWh"]
        CO2_footprint = base_env.envs[0].bat_info["bat_CO2_footprint"]
        load_left = base_env.envs[0].ls_info["ls_unasigned_day_load_left"]
        
        tasks_in_queue = base_env.envs[0].ls_info["ls_tasks_in_queue"]
        tasks_dropped = base_env.envs[0].ls_info["ls_tasks_dropped"]

        water_usage = base_env.envs[0].dc_info["dc_water_usage"]
        
        episode.user_data["instantaneous_net_energy"].append(net_energy)
        
        episode.user_data["net_energy_sum"] += net_energy
        episode.user_data["CO2_footprint_sum"] += CO2_footprint
        episode.user_data["load_left"] += load_left
        episode.user_data["ls_tasks_in_queue"] += tasks_in_queue
        episode.user_data["ls_tasks_dropped"] += tasks_dropped

        episode.user_data["water_usage"] += water_usage

        episode.user_data["step_count"] += 1
    
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs) -> None:
        """
        Method that is called at the end of each episode in the training process. It calculates some metrics based
        on the updated user_data variables.

        Args:
            worker (Worker): The worker object that is being used in the training process.
            base_env (BaseEnv): The base environment that is being used in the training process.
            policies (Dict[str, Policy]): The policies that are being used in the training process.
            episode (MultiAgentEpisode): The episode object that is being processed.
            env_index (int): The index of the environment within the worker task.
            **kwargs: additional arguments that can be passed.
        """
        if episode.user_data["step_count"] > 0:
            average_net_energy = episode.user_data["net_energy_sum"] / episode.user_data["step_count"]
            average_CO2_footprint = episode.user_data["CO2_footprint_sum"] / episode.user_data["step_count"]
            total_load_left = episode.user_data["load_left"]
            total_tasks_in_queue = episode.user_data["ls_tasks_in_queue"]
            total_tasks_dropped = episode.user_data["ls_tasks_dropped"]

            total_water_usage = episode.user_data["water_usage"]

        else:
            average_net_energy = 0
            average_CO2_footprint = 0
            average_bat_actions = 0
            average_ls_actions = 0
            average_dc_actions = 0
            total_load_left = 0
            total_tasks_in_queue = 0
            total_tasks_dropped = 0
            total_water_usage = 0
        
        episode.custom_metrics["average_total_energy_with_battery"] = average_net_energy
        episode.custom_metrics["average_CO2_footprint"] = average_CO2_footprint
        episode.custom_metrics["load_left"] = total_load_left
        episode.custom_metrics["total_tasks_in_queue"] = total_tasks_in_queue
        episode.custom_metrics["total_tasks_dropped"] = total_tasks_dropped

        episode.custom_metrics["total_water_usage"] = total_water_usage
        
class HierarchicalDCRL_Callback(DefaultCallbacks):
    """
    Callback to log Hierarchical DCRL specific values
    """

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy] | None = None, episode: Episode | EpisodeV2, env_index: int | None = None, **kwargs) -> None:

        episode.custom_metrics["runningstats/mu1"] = base_env.vector_env.envs[0].stats1.mu
        episode.custom_metrics["runningstats/sigma_1"] = base_env.vector_env.envs[0].stats1.stddev
        episode.custom_metrics["runningstats/mu2"] = base_env.vector_env.envs[0].stats2.mu
        episode.custom_metrics["runningstats/sigma_2"] = base_env.vector_env.envs[0].stats2.stddev
        episode.custom_metrics["runningstats/cfp_reward"] = base_env.vector_env.envs[0].cfp_reward
        episode.custom_metrics["runningstats/workload_violation_rwd"] = base_env.vector_env.envs[0].workload_violation_rwd
        episode.custom_metrics["runningstats/combined_reward"] = base_env.vector_env.envs[0].combined_reward
        episode.custom_metrics["runningstats/hysterisis_cost"] = base_env.vector_env.envs[0].cost_of_moving_mw
        ax1,ax2,ax3 = base_env.vector_env.envs[0].action_choice
        episode.custom_metrics["runningstats/ax1"] = ax1
        episode.custom_metrics["runningstats/ax2"] = ax2
        episode.custom_metrics["runningstats/ax3"] = ax3

class CustomMetricsCallback_deprecated(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs) -> None:
        # Initialize the cumulative CFP for the episode
        episode.user_data["cumulative_cfp"] = 0.0
        episode.user_data["cumulative_water"] = 0
        episode.user_data["steps"] = 0  # To count the number of steps in the episode
        episode.user_data["dropped_tasks"] = 0  # To count the number of dropped tasks

    def on_episode_step(self, *, worker, base_env, episode, **kwargs) -> None:
        # This is called at every step 
        episode.user_data["steps"] += 1

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs) -> None:
        # Calculate the average CFP per step in the episode
        if hasattr(base_env, 'vector_env'):
            metrics = base_env.vector_env.envs[0].metrics
        else:
            metrics = base_env.envs[0].metrics
        
        cfp = 0
        water = 0
        dropped_tasks = 0
        for dc in metrics:
            cfp += np.mean(metrics[dc]['bat_CO2_footprint']) / 1e3  # Summing up the CFP from all data centers
            water += np.mean(metrics[dc]['dc_water_usage'])
            dropped_tasks += sum(metrics[dc]['ls_tasks_dropped'])
            
        if episode.user_data["steps"] > 0:
            average_cfp = cfp
            total_dropped_tasks = dropped_tasks
            average_water = water
        else:
            average_cfp = 0.0
            total_dropped_tasks = 0
            average_water = 0
        
        # Record the average CFP in custom metrics
        episode.custom_metrics['custom_metrics/average_CFP'] = average_cfp
        episode.custom_metrics['custom_metrics/total_dropped_tasks'] = total_dropped_tasks
        episode.custom_metrics['custom_metrics/average_water_usage'] = average_water


class CustomMetricsCallback(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs) -> None:
        # Initialize the cumulative metrics for the episode
        episode.user_data["cumulative_cfp"] = 0.0
        episode.user_data["cumulative_water"] = 0
        episode.user_data["steps"] = 0  # To count the number of steps in the episode
        episode.user_data["dropped_tasks"] = 0  # To count the number of dropped tasks
        episode.user_data["net_energy_sum"] = 0
        episode.user_data["ite_power_sum"] = 0
        episode.user_data["ct_power_sum"] = 0
        episode.user_data["chiller_power_sum"] = 0
        episode.user_data["hvac_power_sum"] = 0
        episode.user_data["total_tasks_in_queue"] = 0
        episode.user_data["total_tasks_dropped"] = 0
        episode.user_data["total_tasks_overdue"] = 0
        episode.user_data["total_computed_tasks"] = 0
        episode.user_data["hvac_power_on_used"] = []

    def on_episode_step(self, *, worker, base_env, episode, **kwargs) -> None:
        # This is called at every step
        episode.user_data["steps"] += 1

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs) -> None:
        # Access the metrics from the environment
        if hasattr(base_env, 'vector_env'):
            metrics = base_env.vector_env.envs[0].low_level_infos
        else:
            metrics = base_env.envs[0].low_level_infos

        cfp = 0
        water = 0
        dropped_tasks = 0
        net_energy_sum = 0
        ite_power_sum = 0
        ct_power_sum = 0
        chiller_power_sum = 0
        hvac_power_sum = 0
        total_tasks_in_queue = 0
        total_tasks_dropped = 0
        total_tasks_overdue = 0
        total_computed_tasks = 0
        hvac_power_on_used = []

        # Loop through the metrics for each datacenter and aggregate
        for dc in metrics:
            cfp += np.sum(metrics[dc]['__common__']['bat_CO2_footprint']) / 1e3  # Summing up the CFP from all data centers
            water += np.sum(metrics[dc]['__common__']['dc_water_usage'])
            dropped_tasks += np.sum(metrics[dc]['__common__']['ls_tasks_dropped'])
            net_energy_sum += np.sum(metrics[dc]['__common__']['bat_total_energy_with_battery_KWh'])  # Assuming this is net energy
            ite_power_sum += np.sum(metrics[dc]['__common__']['dc_ITE_total_power_kW'])
            ct_power_sum += np.sum(metrics[dc]['__common__']['dc_CT_total_power_kW'])
            chiller_power_sum += np.sum(metrics[dc]['__common__']['dc_Compressor_total_power_kW'])
            hvac_power_sum += np.sum(metrics[dc]['__common__']['dc_HVAC_total_power_kW'])
            total_tasks_in_queue += np.sum(metrics[dc]['__common__']['ls_tasks_in_queue'])
            total_tasks_dropped += np.sum(metrics[dc]['__common__']['ls_tasks_dropped'])
            total_tasks_overdue += np.sum(metrics[dc]['__common__']['ls_overdue_penalty'])  # Overdue tasks represented as a penalty
            total_computed_tasks += np.sum(metrics[dc]['__common__']['ls_computed_tasks'])  # Number of computed tasks

        # Calculate averages
        if episode.user_data["steps"] > 0:
            average_cfp = cfp
            total_dropped_tasks = dropped_tasks
            average_water = water
            average_net_energy = net_energy_sum / episode.user_data["steps"]
            average_ite_power = ite_power_sum / episode.user_data["steps"]
            average_ct_power = ct_power_sum / episode.user_data["steps"]
            average_chiller_power = chiller_power_sum / episode.user_data["steps"]
            average_hvac_power = hvac_power_sum / episode.user_data["steps"]
            average_CO2_footprint = cfp / episode.user_data["steps"]
            PUE = 1 + average_hvac_power / average_ite_power if average_ite_power != 0 else float('inf')
            if len(hvac_power_on_used) > 0:
                avg_hvac_power_on_used = np.mean(hvac_power_on_used)
                max_hvac_power_on_used = np.max(hvac_power_on_used)
                perc_90_hvac_power_on_used = np.percentile(hvac_power_on_used, 90)
            else:
                avg_hvac_power_on_used = max_hvac_power_on_used = perc_90_hvac_power_on_used = 0
        else:
            average_cfp = total_dropped_tasks = average_water = 0
            average_net_energy = average_ite_power = average_ct_power = 0
            average_chiller_power = average_hvac_power = average_CO2_footprint = 0
            PUE = float('inf')
            avg_hvac_power_on_used = max_hvac_power_on_used = perc_90_hvac_power_on_used = 0

        # Log metrics to custom_metrics for RLlib
        episode.custom_metrics['custom_metrics/average_CFP'] = average_cfp
        episode.custom_metrics['custom_metrics/total_dropped_tasks'] = total_dropped_tasks
        episode.custom_metrics['custom_metrics/average_water_usage'] = average_water
        episode.custom_metrics['custom_metrics/average_net_energy'] = average_net_energy
        episode.custom_metrics['custom_metrics/average_ITE_power'] = average_ite_power
        episode.custom_metrics['custom_metrics/average_CT_power'] = average_ct_power
        episode.custom_metrics['custom_metrics/average_chiller_power'] = average_chiller_power
        episode.custom_metrics['custom_metrics/average_HVAC_power'] = average_hvac_power
        episode.custom_metrics['custom_metrics/average_CO2_footprint'] = average_CO2_footprint
        episode.custom_metrics['custom_metrics/PUE'] = PUE
        episode.custom_metrics['custom_metrics/average_HVAC_power_on_used'] = avg_hvac_power_on_used
        episode.custom_metrics['custom_metrics/max_HVAC_power_on_used'] = max_hvac_power_on_used
        episode.custom_metrics['custom_metrics/percentile_90_HVAC_power_on_used'] = perc_90_hvac_power_on_used
        episode.custom_metrics['custom_metrics/total_tasks_in_queue'] = total_tasks_in_queue
        episode.custom_metrics['custom_metrics/total_tasks_dropped'] = total_tasks_dropped
        episode.custom_metrics['custom_metrics/total_tasks_overdue'] = total_tasks_overdue
        episode.custom_metrics['custom_metrics/total_computed_tasks'] = total_computed_tasks
