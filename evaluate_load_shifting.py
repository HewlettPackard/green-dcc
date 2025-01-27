#%%
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from envs.dcrl_env_harl_partialobs_sb3 import DCRL, EnvConfig
from tqdm import tqdm

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
config['initialize_queue_at_reset'] = False
config['random_init_day_at_reset'] = False

# Initialize the environment
env = DCRL(config)
env.seed = 0

# Load the trained model
model_path = "ls_ppo_model_2ND6YU"  # Adjust path as necessary
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
import matplotlib.pyplot as plt
import numpy as np

# Assuming original_workload, shifted_workload, and carbon_intensity are already populated
original_workload = []
shifted_workload = []
carbon_intensity = []
for info in trained_results['infos']:
    if 'agent_ls' in info:
        original_workload.append(info['agent_ls']['ls_original_workload'])
        shifted_workload.append(info['agent_ls']['ls_shifted_workload'])
        carbon_intensity.append(info['agent_bat']['bat_avg_CI'])

# Define parameters
days = 4  # Number of simulated days to display
samples_per_day = 96  # 96 samples per day (15-minute intervals)

# Plot original and shifted workload
fig, ax1 = plt.subplots(figsize=(4.5, 2.9))

x_values = np.arange(len(original_workload[:samples_per_day * days-1])) / samples_per_day

# Plot original and shifted workload
ax1.plot(x_values, np.array(original_workload[1:samples_per_day * days]) * 100, label="Original Workload", 
         color='tab:blue', linestyle='-', linewidth=2, alpha=0.9)
ax1.plot(x_values, np.array(shifted_workload[1:samples_per_day * days]) * 100, label="Shifted Workload", 
         color='tab:orange', linestyle='-', linewidth=2, alpha=0.9)
ax1.set_xlabel("Simulated Days")
ax1.set_ylabel("Data Center Utilization (%)")
ax1.legend(loc="upper left")
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_ylim(0, 110)
ax1.set_xlim(0, days)

ax1.set_xticks(np.arange(0, days + 1, 1))  # Whole numbers from 0 to `days`

# Create a second y-axis for carbon intensity
ax2 = ax1.twinx()
ax2.plot(x_values, carbon_intensity[1:samples_per_day * days], label="CI", color='tab:green', linestyle='-', linewidth=2, alpha=0.9)
ax2.set_ylabel("Carbon Intensity (gCO2/kWh)")
ax2.legend(loc="upper right")
ax2.set_ylim(100, 350)

plt.title("Original vs Temporal Shifted Workload and CI")
plt.tight_layout()

# Save and show
# plt.savefig('Figures/Original_vs_temporal_loadshifting.pdf', format='pdf')
plt.show()

#%% Plot the action taken by the agent for the first week on the first episode from the infos['agent_ls'] and the carbon intensity in another y-axis
# Extract actions taken by the agent from the first episode
actions = []
for info in trained_results['infos']:
    if 'agent_ls' in info:
        actions.append(info['agent_ls']['ls_action'])

# Plot the actions taken by the agent for the first week
fig, ax1 = plt.subplots(figsize=(4.5, 2.6))

# Plot actions
ax1.plot(np.tanh(actions[1:24*4*7]), label="Actions Taken", color='tab:red')
ax1.set_xlabel("Time (hours)")
ax1.set_ylabel("Actions")
ax1.legend(loc="upper left")
ax1.grid()

# Create a second y-axis for carbon intensity
ax2 = ax1.twinx()
ax2.plot(carbon_intensity[1:24*4*7], label="Carbon Intensity", color='tab:green')
ax2.set_ylabel("Carbon Intensity (gCO2/kWh)")
ax2.legend(loc="upper right")

plt.title("Actions Taken by Agent and Carbon Intensity (First Week)")
plt.show()
# %% Obtain the sum of the carbon emissions for the first week and compare both
# Calculate the carbon emissions for the first week
trained_carbon_emmisions = []
for info in trained_results['infos']:
    if 'agent_ls' in info:
        trained_carbon_emmisions.append(info['agent_bat']['bat_CO2_footprint'])

baseline_carbon_emmisions = []
for info in baseline_results['infos']:
    if 'agent_ls' in info:
        baseline_carbon_emmisions.append(info['agent_bat']['bat_CO2_footprint'])
        
print(f"Trained Agent Total Carbon Emissions: {np.sum(trained_carbon_emmisions)}")
print(f"Baseline Agent Total Carbon Emissions: {np.sum(baseline_carbon_emmisions)}")
print(f"Carbon Emissions Difference: {np.sum(trained_carbon_emmisions) - np.sum(baseline_carbon_emmisions)}")
print(f"Carbon Emissions Reduccion (%) : {100 * (np.sum(baseline_carbon_emmisions) - np.sum(trained_carbon_emmisions)) / np.sum(baseline_carbon_emmisions)}")
# %%
