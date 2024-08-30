'''Code used to train the High Level Only Baseline for the NeuirsIPS 2024 submission'''
import os
import ray
from ray import air, tune
from ray.rllib.algorithms.sac import SAC, SACConfig
from ray.rllib.utils.filter import MeanStdFilter

from envs.heirarchical_env import (
    HeirarchicalDCRL, 
    DEFAULT_CONFIG
)

from utils.create_trainable import create_wrapped_trainable


import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete, Dict

class HybridActionSpaceEnv(gym.Env):
    def __init__(self, env):
        self.env = HeirarchicalDCRL(DEFAULT_CONFIG)
        
        # The action space will be a continuous space where:
        # - First part corresponds to discrete choices, encoded as continuous variables.
        # - Second part is the continuous action.
        self.action_space = Box(
            low=np.array([0.0, 0.0, 0.0]), 
            high=np.array([1.0, 1.0, 1.0]), 
            dtype=np.float32
        )
        self.observation_space = self.env.observation_space

    def continuous_to_discrete_action(self, value):
        if 0.0 <= value < 1/3:
            return 0
        elif 1/3 <= value < 2/3:
            return 1
        elif 2/3 <= value <= 1.0:
            return 2
        else:
            raise ValueError("Value out of expected range [0, 1].")

    def step(self, action):
        # Split the action into discrete and continuous parts
        discrete_action_1 = self.continuous_to_discrete_action(action[0])
        discrete_action_2 = self.continuous_to_discrete_action(action[1])
        continuous_action = action[2]
        
        # Create the original action dictionary expected by the wrapped environment
        action_dict = {
            'transfer_0': {
                'receiver': discrete_action_1,
                'sender': discrete_action_2,
                'workload_to_move': np.array([continuous_action])
            }
        }

        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        return self.env.reset()


NUM_WORKERS = 4
NAME = "SAC"
RESULTS_DIR = './results/'

CONFIG = (
        SACConfig()
        .environment(
            env=HybridActionSpaceEnv,
            env_config=DEFAULT_CONFIG
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=NUM_WORKERS,
            rollout_fragment_length=2,
            )
        .training(
            # gamma=0.99,
            gamma=0.0, 
            lr=1e-5,
            train_batch_size=1024,
            model={'fcnet_hiddens': [256, 256]}, 
        )
        .resources(num_gpus=0)
        .debugging(seed=0)
    )

if __name__ == '__main__':
    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init(num_cpus=NUM_WORKERS+1, ignore_reinit_error=True)
    # ray.init(local_mode=True, ignore_reinit_error=True)
    
    tune.Tuner(
        create_wrapped_trainable(SAC),
        param_space=CONFIG.to_dict(),
        run_config=air.RunConfig(
            stop={"timesteps_total": 100_000_000},
            verbose=0,
            local_dir=RESULTS_DIR,
            # storage_path=RESULTS_DIR,
            name=NAME,
            checkpoint_config=ray.air.CheckpointConfig(
                checkpoint_frequency=5,
                num_to_keep=5,
                checkpoint_score_attribute="episode_reward_mean",
                checkpoint_score_order="max"
            ),
        )
    ).fit()
