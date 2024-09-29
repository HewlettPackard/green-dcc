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
from HRL.hierarchical_ppo import HierarchicalPPO as HRLPPO  # pylint: disable=C0413,E0611,E0001
from HRL.greendcc_env import GreenDCC_Env  # pylint: disable=C0413
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
    df = pd.concat([df1,df2,df3,df4], axis=1)
    df.to_csv(save_path, index=False)
    print(f"Traces saved to {save_path}")
            
################################### Evaluation ###################################
def evaluate():  # pylint: disable=missing-function-docstring
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "GreenDCC_Env"  # environment name

    hl_has_continuous_action_space = True  # continuous action space
    ll_has_continuous_action_space = True  # continuous action space

    max_ep_len = 10000                  # max timesteps in one evaluation episode
    #####################################################
    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    # action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    # min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    # action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    hl_K_epochs = 50               # update policy for K epochs in one PPO update for high level network
    ll_K_epochs = 50               # update policy for K epochs in one PPO update for low level network

    eps_clip = 0.2             # clip parameter for PPO
    hl_gamma = 0.90            # discount factor for high level network
    ll_gamma = 0.50            # discount factor for low level network

    hl_lr_actor = 0.0003       # learning rate for high level actor network
    hl_lr_critic = 0.001       # learning rate for high level critic network
    ll_lr_actor = 0.0003       # learning rate for low level actor network(s)
    ll_lr_critic = 0.001       # learning rate for low level critic network(s)

    random_seed = 123         # set random seed if required (0 = no random seed)
    #####################################################

    print("evaluating environment name : " + env_name)
    env = GreenDCC_Env(default_config=DEFAULT_CONFIG)  # initialize environment

    obs_space_hl = env.observation_space_hl
    action_space_hl = env.action_space_hl
    obs_space_ll = env.observation_space_ll
    action_space_ll = env.action_space_ll
    goal_dim_ll = env.goal_dimension_ll
    
    num_ll_policies = len(action_space_ll)

    ################### checkpointing ###################
    run_num_pretrained = 0      #### read which weights to load

    chkpt_directory = f"HRL_PPO_preTrained/{env_name}/"
    
    # pylint: disable=consider-using-f-string
    hl_checkpoint_path = chkpt_directory + "HL_PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    ll_checkpoint_path = chkpt_directory + "LL_PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)  # incomplete path, policy ids are appended later

    print("top level policy save checkpoint path : " + hl_checkpoint_path)
    print("low level policy save checkpoint path (partial) : " + ll_checkpoint_path)
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
    ppo_agent = HRLPPO(num_ll_policies=num_ll_policies, 
                       obs_dim_hl=obs_space_hl.shape[0], obs_dim_ll=[i.shape[0] for i in obs_space_ll],
                       action_dim_hl=action_space_hl.shape[0], action_dim_ll=[i.shape[0] for i in action_space_ll],
                       goal_dim_ll=[i for i in goal_dim_ll],
                       hl_lr_actor=hl_lr_actor, hl_lr_critic=hl_lr_critic, ll_lr_actor=ll_lr_actor, ll_lr_critic=ll_lr_critic,
                       hl_gamma=hl_gamma, ll_gamma=ll_gamma, hl_K_epochs=hl_K_epochs, ll_K_epochs=ll_K_epochs,
                       eps_clip=eps_clip,
                       hl_has_continuous_action_space=hl_has_continuous_action_space, ll_has_continuous_action_space=ll_has_continuous_action_space,
                       action_std_init=action_std,
                       high_policy_action_freq = 1,
                       ll_policy_ids=env.datacenter_ids)

    # assert previous weights if exist
    assert os.path.exists(hl_checkpoint_path), "Error: High Level weights path does not exist"
    
    for i in range(num_ll_policies):
        path, ext = ll_checkpoint_path.rsplit('.', 1)
        assert os.path.exists(f"{path}_{i}.{ext}"), f"Error: Low Level weights path {i} does not exist"

    ppo_agent.load(hl_checkpoint_path, ll_checkpoint_path)
    print("loaded High Level policy weights from path : ", hl_checkpoint_path) 
    print("loaded Low Level policy weights from path : ", ll_checkpoint_path)
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
                                      'DC1 Queue Tasks', 'DC1 Avg Age Task in Queue', 'DC1 ls_tasks_dropped']} # pylint: disable=consider-using-dict-comprehension
    dc2_traces = {key: [] for key in ['DC2 CO2_footprint_per_step', 'DC2 bat_total_energy_with_battery_KWh', 'DC2 Carbon Intensity', 'DC2 External Temperature',
                                        'DC2 Original Workload', 'DC2 Spatial Shifted Workload', 'DC2 Temporal Shifted Workload', 'DC2 Water Consumption',
                                        'DC2 Queue Tasks', 'DC2 Avg Age Task in Queue', 'DC2 ls_tasks_dropped']} # pylint: disable=consider-using-dict-comprehension
    dc3_traces = {key: [] for key in ['DC3 CO2_footprint_per_step', 'DC3 bat_total_energy_with_battery_KWh', 'DC3 Carbon Intensity', 'DC3 External Temperature',
                                        'DC3 Original Workload', 'DC3 Spatial Shifted Workload', 'DC3 Temporal Shifted Workload', 'DC3 Water Consumption',
                                        'DC3 Queue Tasks', 'DC3 Avg Age Task in Queue', 'DC3 ls_tasks_dropped']} # pylint: disable=consider-using-dict-comprehension
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
            for dc_id, dc in zip(env.datacenter_ids,[dc1_traces, dc2_traces, dc3_traces]):
                
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
            if  done['high_level_done']:
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
    


if __name__ == '__main__':

    evaluate()
