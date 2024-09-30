#%%
import os
import sys
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

file_dir = os.path.dirname(__file__)
# append one level upper directory to python path
sys.path.append(file_dir + '/..')
from HRL_withcooling.hierarchical_ppo import HierarchicalPPO as HRLPPO  # pylint: disable=C0413,E0611,E0001
from HRL_withcooling.greendcc_env import GreenDCC_Env  # pylint: disable=C0413
from envs.heirarchical_env_cont import DEFAULT_CONFIG
# update default config where initialize_queue_at_reset is set to False
DEFAULT_CONFIG['config1']['initialize_queue_at_reset'] = False
DEFAULT_CONFIG['config2']['initialize_queue_at_reset'] = False
DEFAULT_CONFIG['config3']['initialize_queue_at_reset'] = False
# pylint: disable=missing-function-docstring,C0303,C0301,C0103,C0209,C0116,C0413
def save_traces_to_csv(dc1_traces, dc2_traces, dc3_traces, dc_action_traces, save_path = 'results.csv'):
    # save traces to csv file
    
    df1 = pd.DataFrame(dc1_traces)
    df2 = pd.DataFrame(dc2_traces)
    df3 = pd.DataFrame(dc3_traces)
    df4 = pd.DataFrame(dc_action_traces)
    df = pd.concat([df1, df2, df3, df4], axis=1)
    df.to_csv(save_path, index=False)
    print(f"Traces saved to {save_path}")
#%%
################################### Evaluation ###################################
print("============================================================================================")

####### initialize environment hyperparameters ######
env_name = "GreenDCC_Env"  # environment name

hl_has_continuous_action_space = True  # continuous action space
ll_has_continuous_action_space = True  # continuous action space
dc_has_continuous_action_space = False  # Discrete action space

max_ep_len = 96*7                  # max timesteps in one evaluation episode
#####################################################
action_std = 0.0001                    # starting std for action distribution (Multivariate Normal)
# action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
# min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
# action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
#####################################################

## Note : print/log frequencies should be > than max_ep_len

################ PPO hyperparameters ################
hl_K_epochs = 5               # update policy for K epochs in one PPO update for high level network
ll_K_epochs = 5               # update policy for K epochs in one PPO update for low level network
dc_K_epochs = 5               # update policy for K epochs in one PPO update for DC network

eps_clip = 0.2             # clip parameter for PPO
hl_gamma = 0.50            # discount factor for high level network
ll_gamma = 0.99            # discount factor for low level network
dc_gamma = 0.99            # discount factor for DC network

hl_lr_actor = 0.0003       # learning rate for high level actor network
hl_lr_critic = 0.001       # learning rate for high level critic network
ll_lr_actor = 0.0003       # learning rate for low level actor network(s)
ll_lr_critic = 0.001       # learning rate for low level critic network(s)
dc_lr_actor = 0.0003       # learning rate for DC actor network(s)
dc_lr_critic = 0.001       # learning rate for DC critic network(s)

#####################################################

print("evaluating environment name : " + env_name)
env = GreenDCC_Env(default_config=DEFAULT_CONFIG)  # initialize environment

obs_space_hl = env.observation_space_hl
action_space_hl = env.action_space_hl

# Include the LL agent
obs_space_ll = env.observation_space_ll
action_space_ll = env.action_space_ll
goal_dim_ll = env.goal_dimension_ll
num_ll_policies = len(action_space_ll)

# Include the DC agent
obs_space_dc = env.observation_space_dc
action_space_dc = env.action_space_dc
goal_dim_dc = env.goal_dimension_dc
num_dc_policies = len(action_space_dc)

################### checkpointing ###################
random_seed = 43         # set random seed if required (0 = no random seed)
run_num_pretrained = 0      #### read which weights to load
run_datetime = '20240928_161210'
chkpt_directory = f"HRL_PPO_preTrained/{env_name}/"

# pylint: disable=consider-using-f-string
hl_checkpoint_path = chkpt_directory + f"HL_PPO_{env_name}_{random_seed}_{run_num_pretrained}_{run_datetime}.pth"
ll_checkpoint_path = chkpt_directory + f"LL_PPO_{env_name}_{random_seed}_{run_num_pretrained}_{run_datetime}.pth"
dc_checkpoint_path = chkpt_directory + f"DC_PPO_{env_name}_{random_seed}_{run_num_pretrained}_{run_datetime}.pth"

print("top level policy save checkpoint path : " + hl_checkpoint_path)
print("low level policy save checkpoint path (partial) : " + ll_checkpoint_path)
print("dc policy save checkpoint path (partial) : " + dc_checkpoint_path)

#####################################################


############# print all hyperparameters #############
print("--------------------------------------------------------------------------------------------")
if hl_has_continuous_action_space:
    print("Initializing a continuous action space policy for top level agent")
    print("--------------------------------------------------------------------------------------------")
else:
    print("Initializing a discrete action space High Level policy")
if ll_has_continuous_action_space:
    print("Initializing a continuous action space policy for low level agent(s)")
    print("--------------------------------------------------------------------------------------------")
else:
    print("Initializing a discrete action space Low Level policy/policies")

if random_seed:
    print("--------------------------------------------------------------------------------------------")
    print("setting random seed to ", random_seed)
    torch.manual_seed(random_seed)
    env.seed(random_seed)  # pylint: disable=no-member
    np.random.seed(random_seed)
#####################################################

print("============================================================================================")

################# evaluation procedure ################

# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Started evaluation at (GMT) : ", start_time)

# initialize a PPO agent
ppo_agent = HRLPPO(
    num_ll_policies=num_ll_policies,
    num_dc_policies=num_dc_policies,
    obs_dim_hl=obs_space_hl.shape[0],
    obs_dim_ll=[i.shape[0] for i in obs_space_ll],
    obs_dim_dc=[i.shape[0] for i in obs_space_dc],
    action_dim_hl=action_space_hl.shape[0],
    action_dim_ll=[i.shape[0] for i in action_space_ll],
    action_dim_dc=[i.n for i in action_space_dc],  # Use .n for discrete action spaces
    goal_dim_ll=[i for i in goal_dim_ll],
    goal_dim_dc=[i for i in goal_dim_dc],
    hl_lr_actor=hl_lr_actor,
    hl_lr_critic=hl_lr_critic,
    ll_lr_actor=ll_lr_actor,
    ll_lr_critic=ll_lr_critic,
    dc_lr_actor=dc_lr_actor,
    dc_lr_critic=dc_lr_critic,
    hl_gamma=hl_gamma,
    ll_gamma=ll_gamma,
    dc_gamma=dc_gamma,
    hl_K_epochs=hl_K_epochs,
    ll_K_epochs=ll_K_epochs,
    dc_K_epochs=dc_K_epochs,
    eps_clip=eps_clip,
    hl_has_continuous_action_space=hl_has_continuous_action_space,
    ll_has_continuous_action_space=ll_has_continuous_action_space,
    dc_has_continuous_action_space=dc_has_continuous_action_space,
    action_std_init=action_std,
    high_policy_action_freq=1,
    ll_policy_ids=env.datacenter_ids
)
    

# assert previous weights if exist
assert os.path.exists(hl_checkpoint_path), "Error: High Level weights path does not exist"

for i in range(num_ll_policies):
    path, ext = ll_checkpoint_path.rsplit('.', 1)
    assert os.path.exists(f"{path}_{i}.{ext}"), f"Error: Low Level weights path {i} does not exist"
    
for i in range(num_dc_policies):
    path, ext = dc_checkpoint_path.rsplit('.', 1)
    assert os.path.exists(f"{path}_{i}.{ext}"), f"Error: DC weights path {i} does not exist"

ppo_agent.load(hl_checkpoint_path, ll_checkpoint_path, dc_checkpoint_path)
print("loaded High Level policy weights from path : ", hl_checkpoint_path) 
print("loaded Low Level policy weights from path : ", ll_checkpoint_path)
print("loaded DC policy weights from path : ", dc_checkpoint_path)
print("============================================================================================")

state = env.reset()

time_step = 0
i_episode = 0

# printing and logging variables
print_running_reward = [0 for i in range(1 + num_ll_policies)]

hl_current_ep_reward = 0
ll_current_ep_reward = [0 for _ in range(num_ll_policies)]

# variables to log for results
# cfp_traces,energy_traces,wrkld_left_traces,ci_traces,ext_temp_traces,original_wrkld_traces,
# spatial_shifted_traces,temporal_shifted_traces,water_use_traces,queue_tasks_traces,avg_age_task_traces,
# dropped_tasks_traces,action_traces
dc1_traces = {key: [] for key in ['DC1 CO2_footprint_per_step', 'DC1 bat_total_energy_with_battery_KWh', 'DC1 Carbon Intensity', 'DC1 External Temperature',
                                    'DC1 Original Workload', 'DC1 Spatial Shifted Workload', 'DC1 Temporal Shifted Workload', 'DC1 Water Consumption',
                                    'DC1 Queue Tasks', 'DC1 Avg Age Task in Queue', 'DC1 ls_tasks_dropped', 'DC1 ls_overdue_penalty']} # pylint: disable=consider-using-dict-comprehension
dc2_traces = {key: [] for key in ['DC2 CO2_footprint_per_step', 'DC2 bat_total_energy_with_battery_KWh', 'DC2 Carbon Intensity', 'DC2 External Temperature',
                                    'DC2 Original Workload', 'DC2 Spatial Shifted Workload', 'DC2 Temporal Shifted Workload', 'DC2 Water Consumption',
                                    'DC2 Queue Tasks', 'DC2 Avg Age Task in Queue', 'DC2 ls_tasks_dropped', 'DC2 ls_overdue_penalty']} # pylint: disable=consider-using-dict-comprehension
dc3_traces = {key: [] for key in ['DC3 CO2_footprint_per_step', 'DC3 bat_total_energy_with_battery_KWh', 'DC3 Carbon Intensity', 'DC3 External Temperature',
                                    'DC3 Original Workload', 'DC3 Spatial Shifted Workload', 'DC3 Temporal Shifted Workload', 'DC3 Water Consumption',
                                    'DC3 Queue Tasks', 'DC3 Avg Age Task in Queue', 'DC3 ls_tasks_dropped', 'DC3 ls_overdue_penalty']} # pylint: disable=consider-using-dict-comprehension
dc_action_traces = {key: [] for key in ['HL Action 1', 'HL Action 2', 'HL Action 3', 'DC1 LS Action', 'DC2 LS Action', 'DC3 LS Action']} # pylint: disable=consider-using-dict-comprehension
# run one evaluation episode
with tqdm(total=max_ep_len) as pbar:
    for _ in range(1, max_ep_len+1):
        # select action with policy
        action = ppo_agent.select_action(state)
        action_raw = action.copy()
        state, reward, done, info = env.step(action)
        
        hl_current_ep_reward += reward['high_level_rewards']
        for i in range(num_ll_policies):
            ll_current_ep_reward[i] += reward[f'low_level_rewards_DC{i+1}']
        
        time_step +=1
        
        # log traces from info dictionary
        for dc_id, dc in zip(env.datacenter_ids, [dc1_traces, dc2_traces, dc3_traces]):
            
            for key,val in info[f'low_level_info_{dc_id}'].items():
                dc[dc_id + " " + key].append(val)
                
        dc_action_traces['HL Action 1'].append(action_raw['high_level_action'][0])
        dc_action_traces['HL Action 2'].append(action_raw['high_level_action'][1])
        dc_action_traces['HL Action 3'].append(action_raw['high_level_action'][2])
        dc_action_traces['DC1 LS Action'].append(action_raw['low_level_action_DC1'][0])
        dc_action_traces['DC2 LS Action'].append(action_raw['low_level_action_DC2'][0])
        dc_action_traces['DC3 LS Action'].append(action_raw['low_level_action_DC3'][0])

        # update progress bar
        pbar.update(1)
        
        # break; if the episode is over
        done = done['low_level_done_DC1']
        if done:
            env.reset()
    
    i_episode += 1
    # print average reward till last episode
    print_running_reward[0] += hl_current_ep_reward
    for i in range(num_ll_policies):
        print_running_reward[i+1] += ll_current_ep_reward[i]
    print_avg_reward = [round(print_running_reward[i] / i_episode, 2) for i in range(1 + num_ll_policies)]    
            

print("Episode : {} \t\t Timestep : {} \t\t Average Reward (HL + LL(s)): {}".format(i_episode, time_step, print_avg_reward))  # pylint: disable=consider-using-f-string

env.close()

# print total evaluation time
print("============================================================================================")
end_time = datetime.now().replace(microsecond=0)
print("Started evaluation at (GMT) : ", start_time)
print("Finished evaluation at (GMT) : ", end_time)
print("Total evalutation time  : ", end_time - start_time)
print("============================================================================================")

# save traces to csv file
save_traces_to_csv(dc1_traces, dc2_traces, dc3_traces, dc_action_traces, save_path = 'results.csv')
#%% Now let's analyze the results from the saved csv
df = pd.read_csv('results.csv')

#%% Let's plot the DC1 original workload, DC1 Spatial Shifted Workload, DC2 original workload, DC2 Spatial Shifted Workload, DC3 original workload, DC3 Spatial Shifted Workload
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(12, 5))  # Adjust height as necessary

ax1.plot(df['DC1 Original Workload'][:100], label='DC1 Original Workload', linewidth=2)
ax1.plot(df['DC1 Spatial Shifted Workload'][:100], linestyle='--', label='DC1 Spatial Shifted Workload', linewidth=2)
ax1.plot(df['DC1 Temporal Shifted Workload'][:100], linestyle='-.', label='DC1 Temporal Shifted Workload', linewidth=2)
ax1.set_title('Original Workload vs Spatial Shifted Workload vs Temporal Shifted Workload')

ax1.plot(df['DC2 Original Workload'][:100], label='DC2 Original Workload', linewidth=2)
ax1.plot(df['DC2 Spatial Shifted Workload'][:100], linestyle='--',label='DC2 Spatial Shifted Workload', linewidth=2)
ax1.plot(df['DC2 Temporal Shifted Workload'][:100], linestyle='-.',label='DC2 Temporal Shifted Workload', linewidth=2)

ax1.plot(df['DC3 Original Workload'][:100], label='DC3 Original Workload', linewidth=2)
ax1.plot(df['DC3 Spatial Shifted Workload'][:100], linestyle='--',label='DC3 Spatial Shifted Workload', linewidth=2)
ax1.plot(df['DC3 Temporal Shifted Workload'][:100], linestyle='-.',label='DC3 Temporal Shifted Workload', linewidth=2)

ax1.set_title('DC3 Original vs Spatial Shifted Workload')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Workload')

# Add the legend
ax1.legend()

# Add grid and limits
ax1.grid(linestyle='--')

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Show the plot
plt.show()
# %% Plot the 'HL Action 1'  

fig, ax1 = plt.subplots(figsize=(12, 5))  # Adjust height as necessary
ax1.plot(df['HL Action 1'][:100], label='HL Action 1', linewidth=2)

ax1.set_title('HL Action 1')

# Plot the 'DC1 Carbon Intensity

ax2 = ax1.twinx()
ax2.plot(df['DC1 Carbon Intensity'][:100], label='DC1 Carbon Intensity', linewidth=2, color='red')
ax2.set_ylabel('Carbon Intensity')

# Add grid and limits
ax1.grid(linestyle='--')

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Show the plot
plt.show()
#%% Now plot the Queue tasks for each DC
fig, ax1 = plt.subplots(figsize=(12, 5))  # Adjust height as necessary

ax1.plot(pd['DC1 Queue Tasks'][:], label='DC1 Queue Tasks', linewidth=2)
ax1.plot(pd['DC2 Queue Tasks'][:], label='DC2 Queue Tasks', linewidth=2)
ax1.plot(pd['DC3 Queue Tasks'][:], label='DC3 Queue Tasks', linewidth=2)
ax1.set_title('Queue Tasks for each DC')

ax1.set_xlabel('Time Step')
ax1.set_ylabel('Queue Tasks')

# Add the legend
ax1.legend()

# Add grid and limits
ax1.grid(linestyle='--')

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Show the plot
plt.show()
# %%
