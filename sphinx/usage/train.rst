=================================
Train 
=================================

To train and evaluate an RL algorithm using Ray, use the appropriate training script. Here are the commands for different configurations:


**HRL (Hierarchical Reinforcement Learning) Configuration**

.. code-block:: bash
      
      python train_truly_hierarchical.py


**HL+LLP (High Level + Low-Level Pretrained) Configuration**

.. code-block:: bash
      
      python baselines/train_geo_dcrl.py


**HLO (High Level Only) Configuration**

.. code-block:: bash
      
      python baselines/train_hierarchical.py


Training Script
-------------------

The provided training script :code:`train_truly_hierarchical.py` uses Ray for distributed training. Here's a brief overview of the script for PPO of HRL configuration:

.. code-block:: python

    import os
    import ray
    from ray import air, tune
    from ray.rllib.algorithms.ppo import PPO, PPOConfig
    from gymnasium.spaces import Discrete, Box
    from ray.rllib.algorithms.ppo import PPOConfig
    
    from envs.truly_heirarchical_env import TrulyHeirarchicalDCRL
    from envs.heirarchical_env import HeirarchicalDCRL, DEFAULT_CONFIG
    from create_trainable import create_wrapped_trainable
    
    NUM_WORKERS = 1
    NAME = "test"
    RESULTS_DIR = './results/'
    
    # Dummy env to get obs and action space
    hdcrl_env = HeirarchicalDCRL()
    
    CONFIG = (
            PPOConfig()
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
                lr=1e-5,
                kl_coeff=0.2,
                clip_param=0.1,
                entropy_coeff=0.0,
                use_gae=True,
                train_batch_size=4096,
                num_sgd_iter=10,
                model={'fcnet_hiddens': [64, 64]}, 
                shuffle_sequences=True
            )
            .multi_agent(
            policies={
                "high_level_policy": (
                    None,
                    hdcrl_env.observation_space,
                    hdcrl_env.action_space,
                    PPOConfig()
                ),
                "DC1_ls_policy": (
                    None,
                    Box(-1.0, 1.0, (14,)),
                    Discrete(3),
                    PPOConfig()
                ),
                "DC2_ls_policy": (
                    None,
                    Box(-1.0, 1.0, (14,)),
                    Discrete(3),
                    PPOConfig()
                ),
                "DC3_ls_policy": (
                    None,
                    Box(-1.0, 1.0, (14,)),
                    Discrete(3),
                    PPOConfig()
                ),
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: agent_id,
            )
            .resources(num_gpus=0)
            .debugging(seed=0)
        )


    if __name__ == "__main__":
        os.environ["RAY_DEDUP_LOGS"] = "0"
        ray.init(ignore_reinit_error=True)
        
        tune.Tuner(
            create_wrapped_trainable(PPO),
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


This example assumes a DCC with three data centers. To use a different algorithm, such as A2C, you need to replace the :code:`PPOConfig` with :code:`A2CConfig` (or the appropriate config class for the algorithm) and adjust the hyperparameters accordingly. For example:


.. code-block:: python

    from ray.rllib.algorithms.a2c import A2C, A2CConfig

    CONFIG = (
            A2CConfig()
            .environment(
                env=TrulyHeirarchicalMSDCRL,
                env_config=DEFAULT_CONFIG
            )
            .framework("torch")
            .rollouts(
                num_rollout_workers=NUM_WORKERS,
                rollout_fragment_length=2,
                )
            .training(
                gamma=0.99,
                lr=1e-5,
                kl_coeff=0.2,
                clip_param=0.1,
                entropy_coeff=0.0,
                use_gae=True,
                train_batch_size=4096,
                num_sgd_iter=10,
                model={'fcnet_hiddens': [64, 64]}, 
            )
            .multi_agent(
            policies={
                "high_level_policy": (
                    None,
                    hdcrl_env.observation_space,
                    hdcrl_env.action_space,
                    A2CConfig()
                ),
                "DC1_ls_policy": (
                    None,
                    Box(-1.0, 1.0, (14,)),
                    Discrete(3),
                    A2CConfig()
                ),
                "DC2_ls_policy": (
                    None,
                    Box(-1.0, 1.0, (14,)),
                    Discrete(3),
                    A2CConfig()
                ),
                "DC3_ls_policy": (
                    None,
                    Box(-1.0, 1.0, (14,)),
                    Discrete(3),
                    A2CConfig()
                ),
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: agent_id,
            )
            .resources(num_gpus=0)
            .debugging(seed=1)
        )


    if __name__ == "__main__":
        os.environ["RAY_DEDUP_LOGS"] = "0"
        ray.init(ignore_reinit_error=True)
        
        tune.Tuner(
            create_wrapped_trainable(A2C),
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
