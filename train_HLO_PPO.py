'''Code used to train the High Level Only Baseline for the NeuirsIPS 2024 submission'''
import os
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.utils.filter import MeanStdFilter
from envs.heirarchical_env_cont import HeirarchicalDCRL, DEFAULT_CONFIG

from utils.create_trainable import create_wrapped_trainable

NUM_WORKERS = 8
NAME = "PPO"
RESULTS_DIR = './results/'

CONFIG = (
        PPOConfig()
        .environment(
            env=HeirarchicalDCRL,
            env_config=DEFAULT_CONFIG
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=NUM_WORKERS,
            # rollout_fragment_length=1,
            )
        .training(
            # gamma=0.99,
            gamma=0.90, 
            lr=1e-5,
            kl_coeff=0.2,
            clip_param=0.2,
            grad_clip = 0.5,
            entropy_coeff=0.01,
            use_gae=True,
            train_batch_size=4096,
            sgd_minibatch_size=128,
            num_sgd_iter=15,
            model={'fcnet_hiddens': [64, 64]}
            )
        .resources(num_gpus=0)
        .debugging(seed=2)
    )

if __name__ == '__main__':
    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init(num_cpus=NUM_WORKERS+1, ignore_reinit_error=True)
    # ray.init(local_mode=True, ignore_reinit_error=True)
    
    tune.Tuner(
        create_wrapped_trainable(PPO),
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
