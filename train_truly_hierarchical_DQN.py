import os

import ray
from ray import air, tune
from ray.rllib.algorithms.dqn import DQN, DQNConfig
from gymnasium.spaces import Discrete, Box

from envs.truly_heirarchical_env import TrulyHeirarchicalDCRL
from envs.heirarchical_env import HeirarchicalDCRL, DEFAULT_CONFIG
from utils.create_trainable import create_wrapped_trainable
from utils.rllib_callbacks import CustomMetricsCallback

NUM_WORKERS = 1
NAME = "test_dqn"
RESULTS_DIR = './results/'

# Dummy env to get obs and action space
hdcrl_env = HeirarchicalDCRL()

CONFIG = (
    DQNConfig()
    .environment(
        env=TrulyHeirarchicalDCRL,
        env_config=DEFAULT_CONFIG
    )
    .framework("torch")
    .rollouts(
        num_rollout_workers=NUM_WORKERS,
        rollout_fragment_length=4,  # Typically longer fragments are used in DQN
    )
    .training(
        gamma=0.99,
        lr=1e-4,
        num_steps_sampled_before_learning_starts= 10000,
        replay_buffer_config={"type": "ReplayBuffer",
                              "capacity": 50000},  # Use standard replay buffer
        train_batch_size=32,  # Standard batch size for DQN
        target_network_update_freq=500,  # How often to update the target network
        model={
            "fcnet_hiddens": [256, 256],  # Larger networks are often used with DQN
            "fcnet_activation": "relu",
        },
    )
    .multi_agent(
        policies={
            "high_level_policy": (
                None,
                hdcrl_env.observation_space,
                hdcrl_env.action_space,
                DQNConfig()
            ),
            "DC1_ls_policy": (
                None,
                Box(-1.0, 1.0, (14,)),
                Discrete(3),
                DQNConfig()
            ),
            "DC2_ls_policy": (
                None,
                Box(-1.0, 1.0, (14,)),
                Discrete(3),
                DQNConfig()
            ),
            "DC3_ls_policy": (
                None,
                Box(-1.0, 1.0, (14,)),
                Discrete(3),
                DQNConfig()
            ),
        },
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: agent_id,
    )
    .callbacks(CustomMetricsCallback)
    .resources(num_gpus=0, num_cpus_per_worker=1)  # Allocate only 1 CPU
    .debugging(seed=0)
)


if __name__ == "__main__":
    os.environ["RAY_DEDUP_LOGS"] = "3"
    # ray.init(local_mode=True, ignore_reinit_error=True)
    ray.init(ignore_reinit_error=True)
    
    tune.Tuner(
        create_wrapped_trainable(DQN),
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
