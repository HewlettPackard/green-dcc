#%%
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from envs.dcrl_env_harl_partialobs import DCRL, EnvConfig
from tqdm import tqdm
#%%
def evaluate_agent(env, agent, n_episodes=10):
    """
    Evaluate the performance of an agent in the given environment.
    
    Args:
        env: The environment to evaluate in.
        agent: The agent to evaluate (callable, returns actions).
        n_episodes: Number of episodes to evaluate.
    
    Returns:
        A dictionary with rewards and metrics.
    """
    rewards = []
    obss = []
    infos = []
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset(seed=episode)
        episode_reward = 0
        done = False
        while not done:
            obss.append(obs)
            infos.append(info)
            action = agent(obs)  # Get action from the agent
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)
    return {"mean_reward": np.mean(rewards), "rewards": rewards, "observations": obss, "infos": infos}

# Baseline agent: Always predict 1
class AlwaysPredictOneAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, obs):
        return np.ones(self.action_space.shape, dtype=np.float32)

#%%
# Load environment configuration
config = EnvConfig.DEFAULT_CONFIG.copy()
config["agents"] = ["agent_ls"]  # Only load-shifting agent active
config['flexible_load'] = 0.4
config['location'] = 'ca'
config['month'] = 6

# Initialize the environment
env = DCRL(config)
env.seed = 0

# Load the trained model
model_path = "checkpoints/load_shifting/KOYY6O/ls_ppo_6840000_steps.zip"  # Adjust path as necessary
trained_model = PPO.load(model_path, env)

# Initialize the baseline agent
baseline_agent = AlwaysPredictOneAgent(env.action_space)

# Evaluate the trained agent
def trained_agent(obs):
    return trained_model.predict(obs, deterministic=True)[0]

trained_results = evaluate_agent(env, trained_agent, n_episodes=1)
print(f"Trained Agent Mean Reward: {trained_results['mean_reward']}")

# Evaluate the baseline agent
baseline_results = evaluate_agent(env, baseline_agent, n_episodes=1)
print(f"Baseline Agent Mean Reward: {baseline_results['mean_reward']}")

# Compare results
print("Evaluation Results:")
print(f"Trained Agent Rewards: {trained_results['rewards']}")
print(f"Baseline Agent Rewards: {baseline_results['rewards']}")

# Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(trained_results["rewards"], label="Trained Agent")
plt.plot(baseline_results["rewards"], label="Baseline Agent")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Agent Performance Comparison")
plt.legend()
plt.grid()
plt.show()
#%% Plot the original workload and the shifted workload of the first week on the first episode fron the infos['agent_ls']
# Extract workload information from the first episode
original_workload = []
shifted_workload = []
carbon_intensity = []
for info in trained_results['infos']:
    if 'agent_ls' in info:
        original_workload.append(info['agent_ls']['ls_original_workload'])
        shifted_workload.append(info['agent_ls']['ls_shifted_workload'])
        carbon_intensity.append(info['agent_bat']['bat_avg_CI'])

# Plot the original and shifted workload for the first week in one y-axis and the carbon intensity in another y-axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot original and shifted workload
ax1.plot(original_workload[1:24*4*7], label="Original Workload", color='tab:blue')
ax1.plot(shifted_workload[1:24*4*7], label="Shifted Workload", color='tab:orange')
ax1.set_xlabel("Time (hours)")
ax1.set_ylabel("Workload")
ax1.legend(loc="upper left")
ax1.grid()

# Create a second y-axis for carbon intensity
ax2 = ax1.twinx()
ax2.plot(carbon_intensity[1:24*4*7], label="Carbon Intensity", color='tab:green')
ax2.set_ylabel("Carbon Intensity (gCO2/kWh)")
ax2.legend(loc="upper right")

plt.title("Original vs Shifted Workload and Carbon Intensity (First Week)")
plt.show()
# %%
