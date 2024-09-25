import os

import ray
from ray import air, tune
from ray.rllib.algorithms.ddpg import DDPG, DDPGConfig

from envs.truly_heirarchical_env import TrulyHeirarchicalDCRL
from envs.heirarchical_env import HeirarchicalDCRL, DEFAULT_CONFIG
from utils.create_trainable import create_wrapped_trainable
from utils.rllib_callbacks import CustomMetricsCallback

import gym
import numpy as np
from gym.spaces import Box, Discrete, Dict

class HybridActionSpaceEnv(gym.Env):
    def __init__(self, env):
        self.env = env
        
        # The action space is a continuous space where:
        # - First part corresponds to discrete choices, encoded as continuous variables.
        # - Second part is the continuous action.
        self.action_space = Box(
            low=np.array([0, 0, 0.0]), 
            high=np.array([2, 2, 1.0]), 
            dtype=np.float32
        )
        self.observation_space = env.observation_space

    def step(self, action):
        # Split the action into discrete and continuous parts
        discrete_action_1 = int(np.round(action[0]))
        discrete_action_2 = int(np.round(action[1]))
        continuous_action = action[2]
        
        # Reconstruct the action dictionary expected by the original environment
        action_dict = {
            'high_level_policy': {
                'sender': discrete_action_1,
                'receiver': discrete_action_2,
                'workload_to_move': np.array([continuous_action])
            }
        }

        # Call the wrapped environment's step function with the reconstructed action dictionary
        obs, reward, done, info = self.env.step(action_dict)
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

# Example usage:
# wrapped_env = HybridActionSpaceEnv(TrulyHeirarchicalDCRL(DEFAULT_CONFIG))
NUM_WORKERS = 1
NAME = "test_ddpg_hybrid"
RESULTS_DIR = './results/'

# Wrap your environment
hdcrl_env = HybridActionSpaceEnv(HeirarchicalDCRL(DEFAULT_CONFIG))

CONFIG = (
    DDPGConfig()
    .environment(
        env=TrulyHeirarchicalDCRL,
        env_config=DEFAULT_CONFIG
    )
    .framework("torch")
    .rollouts(
        num_rollout_workers=NUM_WORKERS,
        rollout_fragment_length=2,
    )
    .training(
        gamma=0.99,
        lr=1e-4,
        tau=0.005,
        train_batch_size=256,
    )
    .multi_agent(
        policies={
            "high_level_policy": (
                None,
                hdcrl_env.observation_space,
                hdcrl_env.action_space,
                DDPGConfig()
            ),
            "DC1_ls_policy": (
                None,
                hdcrl_env.observation_space,
                hdcrl_env.action_space,
                DDPGConfig()
            ),
            "DC2_ls_policy": (
                None,
                hdcrl_env.observation_space,
                hdcrl_env.action_space,
                DDPGConfig()
            ),
            "DC3_ls_policy": (
                None,
                hdcrl_env.observation_space,
                hdcrl_env.action_space,
                DDPGConfig()
            ),
        },
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: agent_id,
    )
    .callbacks(CustomMetricsCallback)
    .resources(num_gpus=0)
    .debugging(seed=0)
)

if __name__ == "__main__":
    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init(ignore_reinit_error=True)
    
    tune.Tuner(
        create_wrapped_trainable(DDPG),
        param_space=CONFIG.to_dict(),
        run_config=air.RunConfig(
            stop={"timesteps_total": 100_000_000},
            verbose=0,
            local_dir=RESULTS_DIR,
            name=NAME,
            checkpoint_config=ray.air.CheckpointConfig(
                checkpoint_frequency=5,
                num_to_keep=5,
                checkpoint_score_attribute="episode_reward_mean",
                checkpoint_score_order="max"
            ),
        )
    ).fit()