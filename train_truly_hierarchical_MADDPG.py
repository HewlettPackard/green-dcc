import os

import ray
from ray import air, tune
from ray.rllib.algorithms.maddpg import MADDPG, MADDPGConfig
from gymnasium.spaces import Discrete, Box

from envs.truly_heirarchical_env import TrulyHeirarchicalDCRL
from envs.heirarchical_env import HeirarchicalDCRL, DEFAULT_CONFIG
from utils.create_trainable import create_wrapped_trainable
from utils.rllib_callbacks import CustomMetricsCallback

NUM_WORKERS = 1
NAME = "test_maddpg"
RESULTS_DIR = './results/'

# Dummy env to get obs and action space
hdcrl_env = HeirarchicalDCRL()

CONFIG = (
    MADDPGConfig()
    .environment(
        env=TrulyHeirarchicalDCRL,
        env_config=DEFAULT_CONFIG
    )
    .framework("tf")
    .rollouts(
        num_rollout_workers=NUM_WORKERS,
        rollout_fragment_length=2,
    )
    .training(
        gamma=0.99,
        lr=1e-3,  # Typical learning rate for MADDPG
        critic_lr=1e-3,  # Separate learning rate for the critic
        actor_lr=1e-4,  # Learning rate for the actor network
        tau=0.01,  # Target network update rate
        train_batch_size=1024,  # Adjust this according to your environment
    )
    .multi_agent(
        policies={
            "high_level_policy": (
                None,
                hdcrl_env.observation_space,
                Discrete(3),  # MADDPG traditionally uses continuous actions, adapt as needed
                MADDPGConfig()
            ),
            "DC1_ls_policy": (
                None,
                Box(-1.0, 1.0, (14,)),
                Discrete(3),
                MADDPGConfig()
            ),
            "DC2_ls_policy": (
                None,
                Box(-1.0, 1.0, (14,)),
                Discrete(3),
                MADDPGConfig()
            ),
            "DC3_ls_policy": (
                None,
                Box(-1.0, 1.0, (14,)),
                Discrete(3),
                MADDPGConfig()
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
        create_wrapped_trainable(MADDPG),
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
