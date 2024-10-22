#%%
import os
import torch
import numpy as np
import pandas as pd
from hierarchical_ppo import HierarchicalPPO as HRLPPO  # Adjust the import if needed
from greendcc_env import GreenDCC_Env
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import socket
from utils.utils_cf import generate_node_connections

from envs.heirarchical_env_cont import DEFAULT_CONFIG
# update default config where initialize_queue_at_reset is set to False
DEFAULT_CONFIG['config1']['initialize_queue_at_reset'] = False
DEFAULT_CONFIG['config2']['initialize_queue_at_reset'] = False
DEFAULT_CONFIG['config3']['initialize_queue_at_reset'] = False
#%%
def save_traces_to_csv(dc1_traces, dc2_traces, dc3_traces, dc_action_traces, save_path='results.csv'):
    """
    Save traces to CSV for further analysis.
    """
    df1 = pd.DataFrame(dc1_traces)
    df2 = pd.DataFrame(dc2_traces)
    df3 = pd.DataFrame(dc3_traces)
    df4 = pd.DataFrame(dc_action_traces)
    df = pd.concat([df1, df2, df3, df4], axis=1)
    df.to_csv(save_path, index=False)
    print(f"Traces saved to {save_path}")

def evaluate_trained_agents(env_config, ppo_agent_params, num_episodes=10):
    """
    Evaluates the performance of the trained agents over a set of episodes.
    """

    # Initialize the environment
    env = GreenDCC_Env(default_config=DEFAULT_CONFIG)

    # Initialize the trained PPO agent
    ppo_agent = HRLPPO(**ppo_agent_params)
    env_name = "GreenDCC_Env"  # environment name

    run_datetime = '20241017_234619'
    chkpt_directory = f"HRL_PPO_preTrained/{env_name}/"
    random_seed = 0         # set random seed if required (0 = no random seed)
    run_num_pretrained = 0      #### read which weights to load

    hl_checkpoint_path = chkpt_directory + f"HL_PPO_{env_name}_{random_seed}_{run_num_pretrained}_{run_datetime}.pth"
    ll_checkpoint_path = chkpt_directory + f"LL_PPO_{env_name}_{random_seed}_{run_num_pretrained}_{run_datetime}.pth"

    # Load pre-trained models
    ppo_agent.load(hl_checkpoint_path, ll_checkpoint_path)

    # TensorBoard setup for evaluation
    hostname = socket.gethostname()
    current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'HRL_eval_runs/{current_time_str}_{hostname}'

    # Traces dictionaries
    dc1_traces = {key: [] for key in ['Original Workload', 'Spatial Shifted Workload', 'Temporal Shifted Workload']}
    dc2_traces = {key: [] for key in ['Original Workload', 'Spatial Shifted Workload', 'Temporal Shifted Workload']}
    dc3_traces = {key: [] for key in ['Original Workload', 'Spatial Shifted Workload', 'Temporal Shifted Workload']}
    dc_action_traces = {key: [] for key in ['HL Action 1', 'HL Action 2', 'HL Action 3']}

    disable_top = True
    if disable_top:
        print("High-level policy is disabled for evaluation.")
        
    # Evaluation loop
    for episode in range(num_episodes):
        state = env.reset()
        done = {'high_level_done': False, 'low_level_done_DC1': False, 'low_level_done_DC2': False, 'low_level_done_DC3': False}
        
        episode_hl_reward = 0
        episode_ll_rewards = [0 for _ in range(env_config['num_ll_policies'])]
        episode_length = 0

        # Metrics to collect
        CO2_footprint_per_step = []
        bat_total_energy_with_battery_KWh_per_step = []
        ls_tasks_dropped_per_step = []

        while not done['high_level_done']:
            # High-level action
            high_level_action = ppo_agent.high_policy.select_action(state['high_level_obs'], evaluate=True)
            if disable_top:
                high_level_action = np.zeros_like(high_level_action)
                
            ppo_agent.goal = high_level_action

            # Prepare low-level goals from high-level action
            node_connections = generate_node_connections(N=ppo_agent.ll_policy_ids, E=ppo_agent.goal)
            goal_mapping = {dc_id: [edge[1] for edge in edges]
                            for dc_id, edges in node_connections.items()}

            actions = {'high_level_action': np.clip(ppo_agent.goal, -1.0, 1.0)}
            
            # Low-level action selection (evaluation mode)
            for dc_id in ppo_agent.ll_policy_ids:
                policy = ppo_agent.low_policies[dc_id]
                goal = goal_mapping[dc_id]

                # Extract workloads of other data centers
                other_dc_workloads = []
                for dc_id2, infos in env.low_level_infos.items():
                    if dc_id != dc_id2:
                        other_dc_workloads.append(infos['agent_ls']['ls_original_workload'])

                # Concatenate low-level observation with goal and other DC workloads
                state_ll = np.concatenate([state['low_level_obs_' + dc_id], goal, other_dc_workloads])
                low_level_action = policy.select_action(state_ll, evaluate=True)
                
                # Change the low_level_action to a random action inside the action space
                # low_level_action = np.random.choice(policy.action_space.n)
                actions['low_level_action_' + dc_id] = low_level_action

            # Step the environment
            next_state, reward, done, info = env.step(actions)
            episode_length += 1

            # Collect rewards
            episode_hl_reward += reward['high_level_rewards']
            for idx in range(env_config['num_ll_policies']):
                episode_ll_rewards[idx] += reward[f'low_level_rewards_DC{idx+1}']

            # Collect workload traces
            # print(info['low_level_info_DC1'].keys())
            dc1_traces['Original Workload'].append(info['low_level_info_DC1']['Original Workload'])
            dc1_traces['Spatial Shifted Workload'].append(info['low_level_info_DC1']['Spatial Shifted Workload'])
            dc1_traces['Temporal Shifted Workload'].append(info['low_level_info_DC1']['Temporal Shifted Workload'])

            dc2_traces['Original Workload'].append(info['low_level_info_DC2']['Original Workload'])
            dc2_traces['Spatial Shifted Workload'].append(info['low_level_info_DC2']['Spatial Shifted Workload'])
            dc2_traces['Temporal Shifted Workload'].append(info['low_level_info_DC2']['Temporal Shifted Workload'])

            dc3_traces['Original Workload'].append(info['low_level_info_DC3']['Original Workload'])
            dc3_traces['Spatial Shifted Workload'].append(info['low_level_info_DC3']['Spatial Shifted Workload'])
            dc3_traces['Temporal Shifted Workload'].append(info['low_level_info_DC3']['Temporal Shifted Workload'])

            # Collect actions
            # print(actions['high_level_action'])
            dc_action_traces['HL Action 1'].append(actions['high_level_action']['transfer_0']['workload_to_move'][0])
            dc_action_traces['HL Action 2'].append(actions['high_level_action']['transfer_1']['workload_to_move'][0])
            dc_action_traces['HL Action 3'].append(actions['high_level_action']['transfer_2']['workload_to_move'][0])

            # Update state
            state = next_state

    # Save traces to CSV
    save_traces_to_csv(dc1_traces, dc2_traces, dc3_traces, dc_action_traces, save_path='evaluation_traces.csv')

    env.close()


env_config = {
    'random_seed': 45,
    'num_ll_policies': 3,
    'datacenter_ids': ['DC1', 'DC2', 'DC3']  # Example; adjust as needed
}

env = GreenDCC_Env(default_config=DEFAULT_CONFIG)

obs_space_hl = env.observation_space_hl
action_space_hl = env.action_space_hl
obs_space_ll = env.observation_space_ll
action_space_ll = env.action_space_ll
goal_dim_ll = env.goal_dimension_ll

num_ll_policies = len(action_space_ll)

hl_K_epochs = 5               # update policy for K epochs in one PPO update for high level network
ll_K_epochs = 5               # update policy for K epochs in one PPO update for low level network

eps_clip = 0.2             # clip parameter for PPO
hl_gamma = 0.50            # discount factor for high level network
ll_gamma = 0.99            # discount factor for low level network

hl_lr_actor = 0.0003       # learning rate for high level actor network
hl_lr_critic = 0.001       # learning rate for high level critic network
ll_lr_actor = 0.0003       # learning rate for low level actor network(s)
ll_lr_critic = 0.001       # learning rate for low level critic network(s)

# PPO agent parameters (ensure these match your model)
ppo_agent_params = {
    'num_ll_policies': num_ll_policies,
    'obs_dim_hl': obs_space_hl.shape[0],
    'obs_dim_ll': [i.shape[0] for i in obs_space_ll],
    'action_dim_hl': action_space_hl.shape[0],
    # 'action_dim_ll': [i.shape[0] for i in action_space_ll],
    'action_dim_ll': [i.n for i in action_space_ll],  # Use .n for discrete action spaces

    'goal_dim_ll': [i for i in goal_dim_ll],
    'hl_lr_actor': hl_lr_actor,
    'hl_lr_critic': hl_lr_critic,
    'll_lr_actor': ll_lr_actor,
    'll_lr_critic': ll_lr_critic,
    'hl_gamma': hl_gamma,
    'll_gamma': ll_gamma,
    'hl_K_epochs': hl_K_epochs,
    'll_K_epochs': ll_K_epochs,
    'eps_clip': eps_clip,
    'hl_has_continuous_action_space': True,
    'll_has_continuous_action_space': False,
    'action_std_init': 0.01,
    'high_policy_action_freq': 1,
    'll_policy_ids': env.datacenter_ids
}

# Number of episodes for evaluation
num_episodes = 1

# Call the evaluation function
evaluate_trained_agents(env_config, ppo_agent_params, num_episodes)


#%% Now let's analyze the results from the saved CSV
df = pd.read_csv('evaluation_traces.csv')

#%% Let's plot the workloads for each data center
import matplotlib.pyplot as plt

# Plot DC1 Workloads
fig, ax1 = plt.subplots(figsize=(12, 5))
# ax1.plot(df['Original Workload'][:1000], label='DC1 Original Workload', linewidth=2)
# ax1.plot(df['Spatial Shifted Workload'][:1000], linestyle='--', label='DC1 Spatial Shifted Workload', linewidth=2)
# ax1.plot(df['Temporal Shifted Workload'][:1000], linestyle='-.', label='DC1 Temporal Shifted Workload', linewidth=2)
# ax1.set_title('DC1 Original vs Spatial vs Temporal Shifted Workload')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Workload')

# Plot DC2 Workloads in the same figure
ax1.plot(df['Original Workload.1'][:1000], label='DC2 Original Workload', linewidth=2)
# ax1.plot(df['Spatial Shifted Workload.1'][:1000], linestyle='--', label='DC2 Spatial Shifted Workload', linewidth=2)
ax1.plot(df['Temporal Shifted Workload.1'][:1000], linestyle='-.', label='DC2 Temporal Shifted Workload', linewidth=2)

# In other axis plot the carbon intensity
# ax2 = ax1.twinx()
# ax2.plot(df['Carbon Intensity'][:1000], color='red', label='Carbon Intensity', linewidth=2)

# Plot DC3 Workloads in the same figure
# ax1.plot(df['Original Workload.2'][:1000], label='DC3 Original Workload', linewidth=2)
# ax1.plot(df['Spatial Shifted Workload.2'][:1000], linestyle='--', label='DC3 Spatial Shifted Workload', linewidth=2)
# ax1.plot(df['Temporal Shifted Workload.2'][:1000], linestyle='-.', label='DC3 Temporal Shifted Workload', linewidth=2)

# Add the legend and grid
ax1.legend()
ax1.grid(linestyle='--')

# Customize the layout
plt.tight_layout()

# Show the plot
plt.show()
#%%
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('results.csv')
# %%
df.head()
# %% Datacenter 1
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df['DC1 Original Workload'][:1000], label='DC1 Original Workload', linewidth=2)
ax1.plot(df['DC1 Spatial Shifted Workload'][:1000], linestyle='--', label='DC1 Spatial Shifted Workload', linewidth=2)
ax1.plot(df['DC1 Temporal Shifted Workload'][:1000], linestyle='-.', label='DC1 Temporal Shifted Workload', linewidth=2)
ax1.set_title('DC1 Original vs Spatial vs Temporal Shifted Workload')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Workload')
# %% Datacenter 2
fig, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(df['DC2 Original Workload'][:1000], label='DC2 Original Workload', linewidth=2)
ax2.plot(df['DC2 Spatial Shifted Workload'][:1000], linestyle='--', label='DC2 Spatial Shifted Workload', linewidth=2)
ax2.plot(df['DC2 Temporal Shifted Workload'][:1000], linestyle='-.', label='DC2 Temporal Shifted Workload', linewidth=2)
ax2.set_title('DC2 Original vs Spatial vs Temporal Shifted Workload')


#%% Datacenter 3
fig, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(df['DC3 Original Workload'][:1000], label='DC3 Original Workload', linewidth=2)
ax3.plot(df['DC3 Spatial Shifted Workload'][:1000], linestyle='--', label='DC3 Spatial Shifted Workload', linewidth=2)
ax3.plot(df['DC3 Temporal Shifted Workload'][:1000], linestyle='-.', label='DC3 Temporal Shifted Workload', linewidth=2)
ax3.set_title('DC3 Original vs Spatial vs Temporal Shifted Workload')

# %%
