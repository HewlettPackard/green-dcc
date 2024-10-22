import os
import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from envs.dcrl_env_harl_partialobs_sb3 import DCRL

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
from datetime import datetime

def exponential_schedule(initial_lr, final_lr, decay_rate):
    """
    Exponential learning rate schedule.
    :param initial_lr: Initial learning rate.
    :param final_lr: Final learning rate.
    :param decay_rate: Decay rate (between 0 and 1).
    :return: function that computes the learning rate based on progress_remaining.
    """
    def lr_schedule(progress_remaining):
        return final_lr + (initial_lr - final_lr) * (decay_rate ** (1 - progress_remaining))
    return lr_schedule

class LRSchedulerCallback(BaseCallback):
    """
    Custom callback for logging the learning rate to TensorBoard.
    """
    def __init__(self, verbose=0):
        super(LRSchedulerCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Get the current learning rate from the optimizer
        lr = self.model.lr_schedule(self.model._current_progress_remaining)
        self.logger.record('train/learning_rate', lr)
        return True


class CustomCallback(BaseCallback):
    """
    Custom callback for logging aggregated variables at the end of each episode to TensorBoard.
    """
    def __init__(self, num_envs):
        super(CustomCallback, self).__init__()
        self.num_envs = num_envs
        # Initialize accumulators and episode lengths for each environment
        self.accumulators = [{} for _ in range(num_envs)]
        self.episode_lengths = [0 for _ in range(num_envs)]

    def _on_step(self) -> bool:
        # Access 'infos' and 'dones' from self.locals
        infos = self.locals.get('infos', None)
        dones = self.locals.get('dones', None)

        if infos is not None and dones is not None:
            for idx, (info, done) in enumerate(zip(infos, dones)):
                if 'agent_ls' in info:
                    agent_ls_info = info['agent_ls']
                    # Extract variables to accumulate
                    bat_CO2_footprint = agent_ls_info.get('bat_CO2_footprint')
                    ls_tasks_dropped = agent_ls_info.get('ls_tasks_dropped')
                    ls_overdue_penalty = agent_ls_info.get('ls_overdue_penalty')
                    # Initialize accumulators if not present
                    if 'bat_CO2_footprint' not in self.accumulators[idx]:
                        self.accumulators[idx]['bat_CO2_footprint'] = []
                    if 'ls_tasks_dropped' not in self.accumulators[idx]:
                        self.accumulators[idx]['ls_tasks_dropped'] = []
                    if 'ls_overdue_penalty' not in self.accumulators[idx]:
                        self.accumulators[idx]['ls_overdue_penalty'] = []
                    # Accumulate variables
                    if bat_CO2_footprint is not None:
                        self.accumulators[idx]['bat_CO2_footprint'].append(bat_CO2_footprint)
                    if ls_tasks_dropped is not None:
                        self.accumulators[idx]['ls_tasks_dropped'].append(ls_tasks_dropped)
                    if ls_overdue_penalty is not None:
                        self.accumulators[idx]['ls_overdue_penalty'].append(ls_overdue_penalty)
                    # Increment episode length
                    self.episode_lengths[idx] += 1
                # Check if episode is done
                if done:
                    # Compute aggregated metrics
                    if self.accumulators[idx]:
                        if 'bat_CO2_footprint' in self.accumulators[idx]:
                            bat_CO2_footprint_array = np.array(self.accumulators[idx]['bat_CO2_footprint'])
                            # bat_CO2_footprint_array_mean = bat_CO2_footprint_array.mean()
                            bat_CO2_footprint_array_sum = bat_CO2_footprint_array.sum()
                            # self.logger.record('episode/bat_CO2_footprint_mean', bat_CO2_footprint_array_mean)
                            self.logger.record('episode/bat_CO2_footprint_sum', bat_CO2_footprint_array_sum)
                        if 'ls_tasks_dropped' in self.accumulators[idx]:
                            ls_tasks_dropped_array = np.array(self.accumulators[idx]['ls_tasks_dropped'])
                            # ls_tasks_dropped_mean = ls_tasks_dropped_array.mean()
                            ls_tasks_dropped_sum = ls_tasks_dropped_array.sum()
                            # self.logger.record('episode/ls_tasks_dropped_mean', ls_tasks_dropped_mean)
                            self.logger.record('episode/ls_tasks_dropped_sum', ls_tasks_dropped_sum)
                        if 'ls_overdue_penalty' in self.accumulators[idx]:
                            ls_overdue_penalty_array = np.array(self.accumulators[idx]['ls_overdue_penalty'])
                            # ls_overdue_penalty_mean = ls_overdue_penalty_array.mean()
                            ls_overdue_penalty_sum = ls_overdue_penalty_array.sum()
                            # self.logger.record('episode/ls_overdue_penalty_mean', ls_overdue_penalty_mean)
                            self.logger.record('episode/ls_overdue_penalty_sum', ls_overdue_penalty_sum)
                        # Log episode length
                        self.logger.record('episode/length', self.episode_lengths[idx])
                        # Dump the log
                        self.logger.dump(self.num_timesteps)
                    # Reset accumulators and episode length
                    self.accumulators[idx] = {}
                    self.episode_lengths[idx] = 0
        return True

# Function to create a new environment
def make_env(env_config, seed=None):
    def _init():
        env = DCRL(env_config)
        env = Monitor(env)  # Wrap with Monitor
        if seed is not None:
            env.set_seed(seed)
        return env
    return _init

# Define your environment using the configuration
config = {
    'location': 'ca',
    'cintensity_file': 'CA_NG_&_avgCI.csv',
    'weather_file': 'USA_NY_New.York-LaGuardia.epw',
    'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
    'dc_config_file': 'dc_config_dc1.json',
    'datacenter_capacity_mw': 1.0,
    'flexible_load': 0.4,
    'timezone_shift': 8,
    'month': 7,
    'days_per_episode': 30,
    'partial_obs': True,
    'nonoverlapping_shared_obs_space': True,
    'debug': False,
    'initialize_queue_at_reset': True,
    'agents': ['agent_ls'],
    'workload_baseline': 0.0,
}


if __name__ == '__main__':
    # Check if CUDA is available
    # print(f"CUDA available: {torch.cuda.is_available()}")
    # if torch.cuda.is_available():
        # print(f"Using device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    # else:
        # print("CUDA not available. The model will be trained on CPU.")

    # Get current time
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Ensure directories exist
    os.makedirs(f"./ppo_agent_ls_tensorboard/{current_time}/", exist_ok=True)
    os.makedirs("./models/", exist_ok=True)

    tensorboard_log = f"./ppo_agent_ls_tensorboard/{current_time}/"
    best_save_path = f"./models/ppo_agent_ls_{current_time}_best_model"
    model_save_path = f"./models/ppo_agent_ls_{current_time}_best_model.zip"
    final_model_save_path = f"./models/ppo_agent_ls_{current_time}_final.zip"


    # Initialize the environment
    env = DCRL(config)
    env = Monitor(env)  # Wrap with Monitor for single environment

    # Check that your environment works with Gym interface
    check_env(env)

    # Define policy_kwargs with the updated net_arch format
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(pi=[64, 64], vf=[128, 64]),
    )

    # Create vectorized environment
    num_envs = 8
    env_configs = [config.copy() for _ in range(num_envs)]
    for i, cfg in enumerate(env_configs):
        cfg['seed'] = i  # Ensure each environment has a unique seed

    env_fns = [make_env(cfg, seed=i) for i, cfg in enumerate(env_configs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)  # Wrap the vectorized environment with VecMonitor

    # Create evaluation environment
    eval_env = DCRL(config)
    eval_env = Monitor(eval_env)  # Wrap with Monitor

    # Initialize PPO with the vectorized environment
    ppo_model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=exponential_schedule(initial_lr=3e-3, final_lr=1e-6, decay_rate=0.99),
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tensorboard_log,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device='cpu',  # Use GPU
    )

    # Verify that the model is on the GPU
    print(f"Model device: {ppo_model.device}")

    # Create an instance of your custom callback
    custom_callback = CustomCallback(num_envs=num_envs)

    # Define evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_save_path,
        log_path='./logs/eval_logs/',
        eval_freq=50000,
        deterministic=True,
        render=False,
    )

    # Create an instance of the custom callback to track the learning rate
    lr_callback = LRSchedulerCallback()

    # Combine callbacks if you have more than one
    callbacks = CallbackList([eval_callback, custom_callback, lr_callback])
    
    # Train the model
    ppo_model.learn(
        total_timesteps=30e6,
        callback=callbacks,
        log_interval=1,
        progress_bar=True
    )

    # Save the trained model with the timestamp
    model_save_path = f"ppo_agent_ls_{timestamp}"
    ppo_model.save(model_save_path)

    print(f"Model saved to {model_save_path}")