import os
import ray
from ray import air, tune
from ray.rllib.algorithms.td3 import TD3, TD3Config
from envs.truly_heirarchical_env import TrulyHeirarchicalDCRL
from envs.heirarchical_env import HeirarchicalDCRL, DEFAULT_CONFIG
from utils.create_trainable import create_wrapped_trainable
from utils.rllib_callbacks import CustomMetricsCallback
import gym
import numpy as np
from gym.spaces import Box, Discrete, Dict

# Custom environment wrappers
class HybridActionSpaceEnv(gym.Env):
    def __init__(self, env):
        self.env = env
        # The action space will be a continuous space where:
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
                'receiver': discrete_action_1,
                'sender': discrete_action_2,
                'workload_to_move': np.array([continuous_action])
            }
        }
        # Call the wrapped environment's step function with the reconstructed action dictionary
        obs, reward, done, info = self.env.step(action_dict)
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

class FlattenedObservationEnv(gym.Env):
    def __init__(self, env):
        self.env = env
        # Flatten the observation space
        self.observation_space = self._flatten_observation_space(env.observation_space)
        self.action_space = env.action_space

    def _flatten_observation_space(self, observation_space):
        # Flatten the observation space by summing the dimensions of all subspaces
        if isinstance(observation_space, Dict):
            flat_dim = sum(np.prod(space.shape) for space in observation_space.spaces.values())
            return Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)
        else:
            return observation_space

    def _flatten_observation(self, observation):
        # Flatten the OrderedDict observation into a single ndarray
        if isinstance(observation, dict):
            return np.concatenate([observation[key].flatten() for key in sorted(observation.keys())])
        else:
            return observation

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._flatten_observation(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self._flatten_observation(obs)

# Main code
NUM_WORKERS = 1
NAME = "test_td3_hybrid"
RESULTS_DIR = './results/'

# Wrap the original environment with both the HybridActionSpaceEnv and FlattenedObservationEnv
wrapped_env = FlattenedObservationEnv(HybridActionSpaceEnv(HeirarchicalDCRL(DEFAULT_CONFIG)))

CONFIG = (
    TD3Config()
    .environment(
        env=wrapped_env  # Pass the wrapped environment directly
    )
    .framework("torch")
    .rollouts(
        num_rollout_workers=NUM_WORKERS,
        rollout_fragment_length=2,
    )
    .training(
        gamma=0.99,
        lr=1e-3,
        actor_hiddens=[400, 300],
        critic_hiddens=[400, 300],
        tau=0.005,
        train_batch_size=100,
        target_noise_clip=0.5,
        target_noise=0.2,
    )
    .multi_agent(
        policies={
            "high_level_policy": (
                None,
                wrapped_env.observation_space,  # Use the observation space from the wrapped environment
                wrapped_env.action_space,       # Use the action space from the wrapped environment
                TD3Config()
            ),
            "DC1_ls_policy": (
                None,
                wrapped_env.observation_space,
                wrapped_env.action_space,
                TD3Config()
            ),
            "DC2_ls_policy": (
                None,
                wrapped_env.observation_space,
                wrapped_env.action_space,
                TD3Config()
            ),
            "DC3_ls_policy": (
                None,
                wrapped_env.observation_space,
                wrapped_env.action_space,
                TD3Config()
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
        create_wrapped_trainable(TD3),
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
