# custom_callbacks.py

from ray.rllib.algorithms.callbacks import DefaultCallbacks
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
                episode.custom_metrics[f'custom_metric/{agent_id}/{metric}'] = value
        # Log the accumulated metrics for the high-level policy
        for metric, value in self.accumulators['accumulated'].items():
            episode.custom_metrics[f'custom_metric/accumulated/{metric}'] = value
        
        policies_to_train = worker.config.get('multi_agent', {})().get('policies_to_train', None)
        # Track the actual learning rates of the agents
        for pol_name, policy in policies.items():
            curr_lr = policy.cur_lr
            lr = policy.optimizer()[0].defaults['lr']
            # episode.custom_metrics[f'custom_metric/{pol_name}/learning_rate'] = curr_lr
            episode.custom_metrics[f'custom_metric/{pol_name}/optimizer_learning_rate'] = lr
            
            # Determine if the policy is being trained
            if policies_to_train is None:
                is_training = True  # All policies are being trained
            elif isinstance(policies_to_train, list):
                is_training = pol_name in policies_to_train
            elif callable(policies_to_train):
                # If policies_to_train is a callable, we cannot evaluate it without additional context
                is_training = None
            else:
                is_training = False  # Unknown or unsupported type

            # Log whether the policy is being trained
            if is_training is not None:
                episode.custom_metrics[f'custom_metric/{pol_name}/is_training'] = int(is_training)
            else:
                episode.custom_metrics[f'custom_metric/{pol_name}/is_training'] = -1  # Indicates unknown
                        
        # Reset the accumulators and episode lengths
        self.accumulators = {}
        self.episode_lengths = {}

    # def on_learn_on_batch(
    #     self,
    #     *,
    #     algorithm,
    #     worker,
    #     policies,
    #     train_batch,
    #     result,
    #     **kwargs
    # ):
    #     """
    #     This callback is called before the optimizer step.
    #     We'll zero out the gradients for policies not being trained.
    #     """
    #     policies_to_train = algorithm.config["multi_agent"].get("policies_to_train", [])
    #     # policies_to_train = self.config.get('multi_agent', {})().get('policies_to_train', None)

    #     for pid, policy in policies.items():
    #         if pid not in policies_to_train:
    #             # Zero out gradients for non-trained policies
    #             for param in policy.model.parameters():
    #                 if param.grad is not None:
    #                     param.grad.zero_()
    
    
    
class CustomMetricsCallbackSingleAgent(DefaultCallbacks):
    """
    Custom callback for logging aggregated variables at the end of each episode to TensorBoard.
    """

    def __init__(self):
        super().__init__()
        # Initialize accumulators and episode lengths for the single agent
        self.accumulators = {
            'bat_CO2_footprint_sum': 0.0,
            'ls_tasks_dropped_sum': 0.0,
            'ls_overdue_penalty_sum': 0.0,
        }
        self.episode_length = 0

    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        # Reset accumulators at the start of each episode
        self.accumulators = {
            'bat_CO2_footprint_sum': 0.0,
            'ls_tasks_dropped_sum': 0.0,
            'ls_overdue_penalty_sum': 0.0,
        }
        self.episode_length = 0

    def on_episode_step(self, *, worker, base_env, policies, episode, **kwargs):
        # Collect metrics from the environment at each step
        infos = episode._last_infos
        agent_id = episode.get_agents()[0]  # Single agent ID
        # if agent_id in infos:
        info = infos[agent_id]
        self.accumulators['bat_CO2_footprint_sum'] += info['agent_ls']['bat_CO2_footprint']
        self.accumulators['ls_tasks_dropped_sum'] += info['agent_ls']['ls_tasks_dropped']
        self.accumulators['ls_overdue_penalty_sum'] += info['agent_ls']['ls_overdue_penalty']
        self.episode_length += 1

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # Log the accumulated metrics to TensorBoard
        for metric, value in self.accumulators.items():
            episode.custom_metrics[f'custom_metric/{metric}'] = value
        # Log episode length
        episode.custom_metrics['custom_metric/episode_length'] = self.episode_length

        # Log learning rate and training status for the agent's policy
        for pol_name, policy in policies.items():
            optimizer_lr = policy.optimizer()[0].defaults['lr']
        episode.custom_metrics[f'custom_metric/optimizer_learning_rate'] = optimizer_lr

        # Reset the accumulators and episode length after each episode ends
        self.accumulators = {
            'bat_CO2_footprint_sum': 0.0,
            'ls_tasks_dropped_sum': 0.0,
            'ls_overdue_penalty_sum': 0.0,
        }