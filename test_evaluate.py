#%%
import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.evaluation import RolloutWorker
from envs.heirarchical_env_cont import HeirarchicalDCRL, DEFAULT_CONFIG
import glob
import numpy as np
#%%
# Set paths
FOLDER = 'results/PPO/PPO_HeirarchicalDCRL_98449_00000_0_2024-09-02_05-38-34'
CHECKPOINT_DIR = sorted(glob.glob(FOLDER + '/checkpoint_*'))[-1]
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, 'rllib_checkpoint.json')  # Point to the rllib_checkpoint.json file

# Evaluation parameters
EVAL_EPISODES = 1  # Number of episodes for evaluation

# Load trained algorithm
config = (
    PPOConfig()
    .environment(
        env=HeirarchicalDCRL,
        env_config=DEFAULT_CONFIG
    )
    .framework("torch")
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .debugging(seed=10)
)

algo = PPO(config=config.to_dict())
algo.restore(CHECKPOINT_FILE)  # Use the full path to the rllib_checkpoint.json file

# Initialize environment for evaluation
env = HeirarchicalDCRL(DEFAULT_CONFIG)
policy = algo.get_policy()
#%%
# Run evaluation loop
episode_rewards = []
episode_lengths = []
for _ in range(EVAL_EPISODES):
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0

    while not done:
        action = policy.compute_single_action(obs)
        # obs, reward, done, info = env.step(action)
        obs, reward, terminated, done, info = env.step(action)
        episode_reward += reward
        episode_length += 1

    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)

# Calculate evaluation metrics
mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
mean_length = np.mean(episode_lengths)
std_length = np.std(episode_lengths)

print(f"Evaluation over {EVAL_EPISODES} episodes:")
print(f"Mean Reward: {mean_reward} ± {std_reward}")
print(f"Mean Episode Length: {mean_length} ± {std_length}")
#%%