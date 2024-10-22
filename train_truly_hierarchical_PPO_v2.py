import os
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from gymnasium.spaces import Box
from ray.tune.registry import register_env
from envs.truly_heirarchical_env import TrulyHeirarchicalDCRL
from envs.heirarchical_env_cont import DEFAULT_CONFIG
from custom_callbacks import CustomMetricsCallback
import datetime
import gymnasium as gym

NAME = "TrulyHierarchicalPPO"
# RESULTS_DIR = './results/'

# 1. Define and Register the Environment
def env_creator(config):
    return TrulyHeirarchicalDCRL(config)

register_env("truly_hierarchical_dcrl", env_creator)

# 2. Initialize Ray
debug = False
if debug:
    NUM_WORKERS = 0
    ray.init(num_cpus=NUM_WORKERS+1, ignore_reinit_error=True, logging_level="DEBUG")
else:
    NUM_WORKERS = 8
    ray.init(num_cpus=NUM_WORKERS+1, ignore_reinit_error=True, logging_level="WARN")


# 3. Create a Dummy Environment to Extract Observation and Action Spaces
dummy_env = env_creator(DEFAULT_CONFIG)
assert isinstance(dummy_env.observation_space, gym.spaces.Dict), "The environment's observation_space must be a gym.spaces.Dict for multi-agent setups."

# 4. Define Policies with hierarchical levels
# You can adjust the observation and action space based on your hierarchical policies
policies = {
    "high_level_policy": (
        None,
        dummy_env.observation_space["high_level_policy"],
        dummy_env.action_space["high_level_policy"],
        PPOConfig().training(gamma=0.9)  # Example: a lower gamma for high-level policy
    ),
    "DC1_ls_policy": (
        None,
        dummy_env.observation_space["DC1_ls_policy"],
        dummy_env.action_space["DC1_ls_policy"],
        PPOConfig().training(gamma=0.99)  # Example: a different gamma for DC1 low-level policy
    ),
    "DC2_ls_policy": (
        None,
        dummy_env.observation_space["DC2_ls_policy"],
        dummy_env.action_space["DC2_ls_policy"],
        PPOConfig().training(gamma=0.99)  # Example: a different gamma for DC2 low-level policy
    ),
    "DC3_ls_policy": (
        None,
        dummy_env.observation_space["DC3_ls_policy"],
        dummy_env.action_space["DC3_ls_policy"],
        PPOConfig().training(gamma=0.99)  # Example: a different gamma for DC3 low-level policy
    ),
}

# 5. Define the Policy Mapping Function
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return agent_id  # Each agent uses its own policy

# 6. Configure PPO with hierarchical setup
CONFIG = (
    PPOConfig()
    .environment("truly_hierarchical_dcrl", env_config=DEFAULT_CONFIG)
    .framework("torch")
    .rollouts(
        num_rollout_workers=NUM_WORKERS,  # Number of parallel workers
    )
    .training(
        gamma=0.99,               # Learning rate and other PPO-specific configs
        lr=1e-5,
        kl_coeff=0.2,
        clip_param=0.2,
        grad_clip=2.0,
        entropy_coeff=0.00001,
        use_gae=True,
        train_batch_size=1024,
        minibatch_size=128,
        num_sgd_iter=5,
        model={'fcnet_hiddens': [64, 64]}  # Neural network architecture
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,  # Each agent follows its corresponding policy
    )
    .callbacks(CustomMetricsCallback)  # Custom callback for logging metrics
    .resources(num_gpus=0)  # Adjust this according to your hardware
)

# 7. Set Up Tune for training with checkpoints and logging
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join("ray_results", f"truly_hierarchical_dcrl_{current_datetime}")

# Convert log_dir to an absolute path
log_dir = os.path.abspath(log_dir)

# Use Ray Tune to train and checkpoint the PPO algorithm
tune.Tuner(
    PPO,
    param_space=CONFIG.to_dict(),
    run_config=air.RunConfig(
        stop={"timesteps_total": 100_000_000},  # Stop after this many timesteps
        verbose=2 if debug else 0,
        storage_path=log_dir,
        name=NAME,
        checkpoint_config=ray.air.CheckpointConfig(
            checkpoint_frequency=5,  # Save a checkpoint every 5 iterations
            num_to_keep=5,           # Keep up to 5 checkpoints
            checkpoint_score_attribute="episode_reward_mean",  # Metric to optimize when saving checkpoints
            checkpoint_score_order="max"
        ),
    )
).fit()

# 8. Shutdown Ray after training
ray.shutdown()
